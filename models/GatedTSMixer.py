import torch
import torch.nn as nn
from models.Rev_in import RevIN

class Model(nn.Module):
    """
    Gated Time series model with Channel mixing
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_blocks = configs.num_blocks
        self.seq_len = configs.seq_len
        #Mixer block including gating
        self.mixer_block = MixerBlock(configs.enc_in, configs.hidden_size,
                                      configs.seq_len, configs.dropout,
                                        configs.activation, configs.single_layer_mixer)
        self.channels = configs.enc_in
        self.pred_len = configs.pred_len
        self.rev_norm = RevIN(self.channels, affine=configs.affine)
        #Individual layer for each variate(if true) otherwise, one shared linear
        self.individual_linear_layers = configs.individual
        if(self.individual_linear_layers) :
            self.output_linear_layers = nn.ModuleList()
            for _ in range(self.channels):
                self.output_linear_layers.append(nn.Linear(configs.seq_len, configs.pred_len))
        else :
            self.output_linear_layers = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x) :
        x = self.rev_norm(x, 'norm')
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)
        #Final linear layer applied on the transpoed mixers' output
        x = torch.swapaxes(x, 1, 2)
        #Preparing tensor output with the correct prediction length
        y = torch.zeros([x.size(0), x.size(1), self.pred_len],dtype=x.dtype).to(x.device)
        if self.individual_linear_layers :
            for c in range(self.channels): 
                y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
        else :
            y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, 'denorm')
        return y

class GatingLayer(nn.Module) :
    """Gating Layer for timesteps importance sampling"""
    def __init__(self, seq_len) :
        super(GatingLayer, self).__init__()
        self.linear_gating = nn.Linear(seq_len, seq_len)
        self.sigmoid_activation = nn.Sigmoid()
    
    def forward(self, x) :
        x = torch.swapaxes(x, 1, 2)
        """Expecting single channel input"""
        gating_array = self.linear_gating(x)
        gating_array = self.sigmoid_activation(gating_array)
        y = torch.mul(x, gating_array)
        y = torch.swapaxes(y, 1, 2)
        return y

class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
        self.single_layer_mixer = single_layer_mixer
        if self.single_layer_mixer :
            self.linear_layer1 = nn.Linear(channels, channels)
        else :
            self.linear_layer1 = nn.Linear(channels, mlp_dim)
            if activation=="gelu" :
                self.activation_layer = nn.GELU()
            elif activation=="relu" :
                self.activation_layer = nn.ReLU()
            else :
                self.activation_layer = None
            self.linear_layer2 = nn.Linear(mlp_dim, channels)
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        y = torch.swapaxes(x, 1, 2)
        y = self.normalization_layer(y)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer1(y)
        if not(self.single_layer_mixer) :
            if self.activation_layer is not None :
                y = self.activation_layer(y)
            y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        return x + y

class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, seq_len, dropout_factor, activation):
        super(MlpBlockTimesteps, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)


    def forward(self, x) :
        y = self.normalization_layer(x)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer(y)
        y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2)
        return x + y
    
class MixerBlock(nn.Module):
    """Mixer block layer only mixing channels in this model"""
    def __init__(self, channels, features_block_mlp_dims, seq_len, dropout_factor, activation, single_layer_mixer) :
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        #Gating unit based on the seq len
        self.gating_unit = GatingLayer(self.seq_len)
        #Timesteps mixing block 
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)
        #Features mixing block 
        self.channels_mixer = MlpBlockFeatures(channels, features_block_mlp_dims, dropout_factor, activation, single_layer_mixer)

    def forward(self, x) :
        #Gating step
        y = self.gating_unit(x)
        y = self.timesteps_mixer(y)   
        y = self.channels_mixer(y)
        return y