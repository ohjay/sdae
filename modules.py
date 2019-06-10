import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """Stacked autoencoder."""

    def __init__(self):
        super(SAE, self).__init__()

        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        self.num_trained_blocks = 0
        self.num_blocks = len(self.encoders)

    def forward(self, x, ae_idx=None):
        x = self.encode(x, ae_idx)
        x = self.decode(x, ae_idx)
        return x

    def get_first_layer_weights(self, as_tensor=False):
        if as_tensor:
            return self.encoders[0][0].weight.data.cpu()
        return self.encoders[0][0].weight.data.cpu().numpy()

    def get_block_parameters(self, ae_idx):
        """Get parameters corresponding to a particular block."""
        return list(self.encoders[ae_idx].parameters()) + \
               list(self.decoders[self.num_blocks-ae_idx-1].parameters())

    def get_enc_out_features(self, ae_idx):
        """Get the output dimensionality of the "AE_IDX"-th encoder.
        To get the output dimensionality of the final encoder, pass -1 as AE_IDX."""
        enc_out_features = None
        for module in self.encoders[ae_idx]:
            if hasattr(module, 'out_features'):
                enc_out_features = module.out_features
        return enc_out_features

    def encode(self, x, ae_idx=None):
        """Encode the input. If AE_IDX is provided,
        encode with that particular encoder only."""
        if ae_idx is None:
            for i in range(self.num_trained_blocks):
                x = self.encoders[i](x)
        else:
            x = self.encoders[ae_idx](x)
        return x

    def decode(self, x, ae_idx=None):
        """Decode the input. If AE_IDX is provided,
        decode with that particular decoder only."""
        if ae_idx is None:
            start = self.num_blocks - self.num_trained_blocks
            for i in range(start, self.num_blocks):
                x = self.decoders[i](x)
        else:
            x = self.decoders[self.num_blocks-ae_idx-1](x)
        return x


class OlshausenAE(SAE):
    """Olshausen autoencoder."""

    def __init__(self):
        super(OlshausenAE, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=12*12, out_features=120),
                nn.Sigmoid(),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=120, out_features=12*12),
            ),
        ])
        self.num_trained_blocks = 0
        self.num_blocks = len(self.encoders)


class MNISTAE(SAE):
    """MNIST autoencoder."""

    def __init__(self):
        super(MNISTAE, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=28*28, out_features=120),
                nn.Sigmoid(),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=120, out_features=28*28),
                nn.Sigmoid(),
            ),
        ])
        self.num_trained_blocks = 0
        self.num_blocks = len(self.encoders)


class MNISTSAE2(SAE):
    """MNIST stacked autoencoder (two blocks)."""

    def __init__(self):
        super(MNISTSAE2, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=28*28, out_features=1000),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=1000, out_features=1000),
                nn.Sigmoid(),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=1000, out_features=1000),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=1000, out_features=28*28),
                nn.Sigmoid(),
            ),
        ])
        self.num_trained_blocks = 0
        self.num_blocks = len(self.encoders)


class OlshausenSAE3(SAE):
    """Olshausen stacked autoencoder (three blocks)."""

    def __init__(self):
        super(OlshausenSAE3, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=12*12, out_features=500),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=500, out_features=750),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=750, out_features=1000),
                nn.Sigmoid(),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=1000, out_features=750),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=750, out_features=500),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=500, out_features=12*12),
            ),
        ])
        self.num_trained_blocks = 0
        self.num_blocks = len(self.encoders)


class MNISTDenseClassifier2(nn.Module):
    """MNIST classifier (two dense blocks)."""

    def __init__(self, enc_out_features):
        super(MNISTDenseClassifier2, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(enc_out_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.classifier(x)


class VAE(nn.Module):
    """Variational autoencoder."""

    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential()
        self.mean_estimator = nn.Sequential()
        self.log_var_estimator = nn.Sequential()
        self.decoder = nn.Sequential()

        self.num_blocks = 1  # currently only one encoder/decoder pair

    def forward(self, x):
        mean, log_var = self.encode(x)

        # reparameterization trick
        # ------------------------
        # MEAN and LOG_VAR are "mu" and log("sigma" ^2) using
        # the notation from eq. 10 in Auto-Encoding Variational Bayes
        # -----------------------------------------------------------
        stdev = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(stdev)
        z = mean + stdev * epsilon  # sampled latent vector

        return self.decode(z), mean, log_var

    def get_enc_out_features(self, ae_idx):
        enc_out_features = None
        for module in self.mean_estimator:
            if hasattr(module, 'out_features'):
                enc_out_features = module.out_features
        return enc_out_features

    def encode(self, x):
        x = self.encoder(x)
        return self.mean_estimator(x), self.log_var_estimator(x)

    def decode(self, z):
        return self.decoder(z)


class VAELoss(nn.Module):
    def __init__(self, reduction='sum', reconstruction_loss_type='mse'):
        super(VAELoss, self).__init__()
        self.reduction = reduction
        self.reconstruction_loss_type = reconstruction_loss_type.lower()
        assert self.reconstruction_loss_type in {'mse', 'bce', 'binary_cross_entropy'}

    def forward(self, input_, target, mean, log_var):
        # reconstruction
        if self.reconstruction_loss_type == 'mse':
            reconstruction_loss = F.mse_loss(input_, target, reduction=self.reduction)
        else:
            reconstruction_loss = F.binary_cross_entropy(input_, target, reduction=self.reduction)

        # regularization
        kl_div = 0.5 * torch.sum(mean * mean + torch.exp(log_var) - log_var - 1, dim=1)
        kl_div = torch.mean(kl_div) if self.reduction == 'mean' else torch.sum(kl_div)

        return reconstruction_loss + kl_div


class MNISTVAE(VAE):
    """MNIST variational autoencoder."""

    def __init__(self):
        super(MNISTVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=400),
            nn.ReLU(),
        )
        self.mean_estimator = nn.Sequential(
            nn.Linear(in_features=400, out_features=20),
        )
        self.log_var_estimator = nn.Sequential(
            nn.Linear(in_features=400, out_features=20),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=20, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=28*28),
            nn.Sigmoid(),
        )
