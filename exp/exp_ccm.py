from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.ccm.patchtst import PatchTSTC
from models.ccm.tsmixer import TSMixerC
from models.ccm.Dlinear import DLinearC
from models.ccm.timesnet import TimesNetC
import torch.nn.functional as F
from utils.ccm.tools import EarlyStopping, adjust_learning_rate
from utils.ccm.metrics import metric, MSE_dim

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
# import wandb
import os
import time
import json
import pickle
from torchinfo import summary



class Exp_CCM(Exp_Basic):
    def __init__(self, args):
        super(Exp_CCM, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'PatchTSTC': PatchTSTC,
            'TSMixerC': TSMixerC,
            'DLinearC': DLinearC,
            'TimesNetC': TimesNetC
        }
        model = model_dict[self.args.model](self.args).float()
        summary(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        loss_f_list = []
        loss_s_list = []
            
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, if_update=False)
                loss_f = nn.MSELoss()(pred.detach().cpu(), true.detach().cpu())
                if self.args.individual == "c":
                    simMatrix = self.get_similarity_matrix(batch_x)
                    loss_s = self.similarity_loss_batch(self.model.cluster_prob, simMatrix)
                else: 
                    loss_s = torch.tensor(0).to(self.device)
                loss = loss_f + self.args.beta * loss_s
                total_loss.append(loss.detach().item())
                loss_f_list.append(loss_f.detach().item())
                loss_s_list.append(loss_s.detach().item())
                
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
        total_loss = np.average(total_loss)
        loss_f = np.average(loss_f_list)
        loss_s = np.average(loss_s_list)
        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num
        mae, mse, rmse, mape, mspe = metrics_mean
        self.model.train()
        return mse, total_loss, loss_f, loss_s, mae
    

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean_, 'std': train_data.scaler.scale_}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)
        # s_type = "DTW" if self.args.data in ["ILI", "ETTm2", "ETTm1", "EXR", "ETTh2"] else "EUC"
        # similarity_matrix = self._get_similarity_matrix(s_type=s_type)
        # self.simMatrix = torch.from_numpy(similarity_matrix).to(self.device).float()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion_ts =  nn.MSELoss()
        
        
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            tl_f = []
            tl_s = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, if_update=True)
                loss_f = criterion_ts(pred, true)
                if self.args.individual == "c":
                    simMatrix = self.get_similarity_matrix(batch_x)
                    loss_s = self.similarity_loss_batch(self.model.cluster_prob, simMatrix)
                else: 
                    loss_s = torch.tensor(0).to(self.device)
                loss = loss_f + self.args.beta * loss_s
                train_loss.append(loss.item())
                tl_f.append(loss_f.item())
                
                tl_s.append(loss_s.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
                
                # if epoch % 10 == 0:
            # self.vis_linear_weight(epoch)
            
            train_loss = np.average(train_loss)
            train_loss_f = np.average(tl_f)
            train_loss_s = np.average(tl_s)
            vali_mse, vali_loss, vali_loss_f, vali_loss_s, vali_mae = self.vali(vali_data, vali_loader)
            test_mse, test_loss, test_loss_f, test_loss_s, test_mae = self.vali(test_data, test_loader)
            print("prob", self.model.cluster_prob)

            print("Epoch: {0}, Steps: {1}, Cost time: {2:.3f} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f} Test MSE: {6:.3f} Test MAE: {7:.3f}".format(
                epoch + 1, train_steps, time.time()-epoch_time, train_loss, vali_loss, test_loss, test_mse, test_mae))
            
            
            # wandb.log({"Train_loss":train_loss, "Train_forecast_loss":train_loss_f ,"Train_similarity_loss": train_loss_s,
            #     "Vali_loss": vali_loss, "Vali_forecast_loss":vali_loss_f , "Vali_similarity_loss": vali_loss_s, "Vali_mse": vali_mse,
            #         "Test_loss": test_loss ,"Test_forecast_loss": test_loss_f,"Test_similarity_loss":test_loss_s, "Test_mse": test_mse,
            #         "Test_mae": test_mae, "Vali_mae": vali_mae,})
            # wandb.log({"Cluster_prob": wandb.Histogram(self.model.cluster_prob.detach().cpu().numpy())})

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')
        
        return self.model

    def test(self, setting, save_pred = True, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, inverse=False, if_update=False)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, inverse=False, if_update=False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        # For CCM models, we may only need the batch_x and batch_y
        # Shuffle the channels if needed
        if self.args.features == 'M':
            channel_dim = -1
            num_channels = batch_x.shape[channel_dim]
            shuffled_indices = torch.randperm(num_channels)
            batch_x = batch_x[:, :, shuffled_indices]
            batch_y = batch_y[:, :, shuffled_indices]
        
        # Get predictions
        outputs = self.model(batch_x, if_update=if_update)

        # Handle inverse transform if needed
        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        # Only use the prediction length portion of batch_y
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        return outputs, batch_y
    
    def _similarity_loss_batch(self, prob, batch_x):
        membership = self.concrete_bern(prob)  #[n_vars, n_clusters]
        # membership = prob
        temp_1 = torch.mm(membership.t(), self.simMatrix) 
        SAS = torch.mm(temp_1, membership)
        _SS = 1 - torch.mm(membership, membership.t())
        loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, self.simMatrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss
    
    
    def _get_similarity_matrix(self, s_type):
        SimMatrixDict = np.load(f"temp_store/SimilarityMatrix_{self.args.data}.npy", allow_pickle=True).item()
        SimMatrix = SimMatrixDict[s_type]
        return SimMatrix
        
    
    def get_similarity_matrix(self, batch_x):
        # batch_x shape: [batch_size, seq_len, n_variables]
        # We want similarity between variables, so we'll use the mean over batch and time
        batch_x = batch_x.permute(2, 0, 1)  # [n_variables, batch_size, seq_len]
        batch_x = batch_x.reshape(batch_x.shape[0], -1)  # [n_variables, batch_size * seq_len]
        
        # Compute pairwise distances between variables
        diff = batch_x.unsqueeze(1) - batch_x.unsqueeze(0)  # [n_variables, n_variables, batch_size * seq_len]
        dist_squared = torch.sum(diff ** 2, dim=-1)  # [n_variables, n_variables]
        param = torch.max(dist_squared)
        euc_similarity = torch.exp(-5 * dist_squared / param)
        return euc_similarity.to(self.device)
        
     
    def similarity_loss_batch(self, prob, simMatrix):
        def concrete_bern(prob, temp = 0.07):
            random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
            prob_bern = ((prob + random_noise) / temp).sigmoid()
            return prob_bern
        
        # Ensure simMatrix is on the same device and is 2D
        simMatrix = simMatrix.to(self.device)
        
        # prob shape should be [n_variables, n_clusters]
        membership = concrete_bern(prob)  # [n_variables, n_clusters]
        
        # Matrix multiplications
        temp_1 = torch.mm(membership.t(), simMatrix)  # [n_clusters, n_variables]
        SAS = torch.mm(temp_1, membership)  # [n_clusters, n_clusters]
        _SS = 1 - torch.mm(membership, membership.t())  # [n_variables, n_variables]
        
        loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, simMatrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        
        return loss + ent_loss
        
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern