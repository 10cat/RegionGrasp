from copy import deepcopy
import torch
import torch.nn as nn
from option import MyOptions as cfg

class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, condition_size=1024):
        super(VAE, self).__init__()

        if conditional:
            assert condition_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        
        # import pdb; pdb.set_trace()

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, condition_size)

        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, condition_size)
        
        # import pdb; pdb.set_trace()

    
    def forward(self, x, c=None):
        batch_size = x.size(0)
        means, log_var = self.encoder(x, c)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size], device=c.device)
        recon_x = self.decoder(z, c)

        return recon_x, z

class Encoder(nn.Module):

    def __init__(self, layer_sizes_set, latent_size, conditional, condition_size):
        
        super().__init__()

        layer_sizes = deepcopy(layer_sizes_set) # 发现：必须要在这一步用deepcopy才不会将某一次init中layer_sizes[0] += condition_size的变化传递到cfg.VAE_encoder_sizes上，任何之前一步做deepcopy都无济于事（？？？）
        # layer_sizes = layer_sizes_set

        self.conditional = conditional
        if self.conditional:
            # import pdb; pdb.set_trace()
            layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()
        self.MLP = nn.Sequential()
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        # for decoding the hand params based on mano framework from otaheri
        pose_dim, trans_dim = layer_sizes[-1]
        feat_dim = layer_sizes[-2]
        self.dec_pose = nn.Linear(feat_dim, pose_dim)
        self.dec_trans = nn.Linear(feat_dim, trans_dim)

    def forward(self, z, c):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)
        
        x = self.MLP(z)
        pose = self.dec_pose(x)
        trans = self.dec_trans(x)

        return [pose, trans]


