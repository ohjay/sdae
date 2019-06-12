import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import is_iterable, product


class Flatten(nn.Module):
    """Source: https://bit.ly/2I8PJyH."""
    def forward(self, x):
        return x.view(x.size(0), -1)


# ====================
# REGULAR AUTOENCODERS
# ====================


class SAE(nn.Module):
    """Stacked autoencoder."""

    def __init__(self):
        super(SAE, self).__init__()

        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        self.num_trained_blocks = 0
        self.is_convolutional = False

    def forward(self, x, ae_idx=None):
        x = self.encode(x, ae_idx)
        x = self.decode(x, ae_idx)
        return x

    @property
    def num_blocks(self):
        return len(self.encoders)

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


class MNISTCAE(SAE):
    """MNIST convolutional autoencoder."""

    def __init__(self):
        super(MNISTCAE, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=2, stride=1),  # shape: (batch, 8, 2, 2)
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=2, stride=2, padding=1),
                nn.Sigmoid(),
            ),
        ])
        self.is_convolutional = True

    def get_enc_out_features(self, ae_idx):
        _enc_out_features = [(8, 2, 2)]
        return _enc_out_features[ae_idx]


class MNISTCAE2(SAE):
    """MNIST stacked convolutional autoencoder (two blocks)."""

    def __init__(self):
        super(MNISTCAE2, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 16, 14, 14)
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 16, 7, 7)
                nn.Sigmoid(),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid(),
            ),
        ])
        self.is_convolutional = True

    def get_enc_out_features(self, ae_idx):
        _enc_out_features = [(16, 14, 14), (16, 7, 7)]
        return _enc_out_features[ae_idx]


class CUBCAE2(SAE):
    """CUB stacked convolutional autoencoder (two blocks)."""

    def __init__(self):
        super(CUBCAE2, self).__init__()

        # shape: (batch, 3, 128, 128)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 8, 64, 64)
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 16, 32, 32)
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 32, 16, 16)
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 64, 8, 8)
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 128, 4, 4)
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 256, 2, 2)
                nn.ReLU(),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),  # shape: (batch, 128, 4, 4)
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),  # shape: (batch, 64, 8, 8)
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),  # shape: (batch, 32, 16, 16)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),  # shape: (batch, 16, 32, 32)
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),  # shape: (batch, 8, 64, 64)
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_channels=8, out_channels=3, kernel_size=5, stride=1, padding=0),
                # shape: (batch, 3, 128, 128)
            ),
        ])
        self.is_convolutional = True

    def get_enc_out_features(self, ae_idx):
        _enc_out_features = [(32, 16, 16), (256, 2, 2)]
        return _enc_out_features[ae_idx]


class CIFARCAE(SAE):
    """CIFAR-10 stacked convolutional autoencoder."""

    def __init__(self):
        super(CIFARCAE, self).__init__()

        # shape: (batch, 3, 32, 32)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 8, 16, 16)
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 16, 8, 8)
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),  # shape: (batch, 32, 4, 4)
                nn.ReLU(),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),  # shape: (batch, 16, 8, 8)
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),  # shape: (batch, 8, 16, 16)
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=0),
                # shape: (batch, 3, 32, 32)
            ),
        ])
        self.is_convolutional = True

    def get_enc_out_features(self, ae_idx):
        _enc_out_features = [(32, 4, 4)]
        return _enc_out_features[ae_idx]


# ===========
# CLASSIFIERS
# ===========


class Classifier(nn.Module):
    """Classifier."""

    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential()
        self.is_convolutional = False

    def forward(self, x):
        return self.classifier(x)


class MNISTDenseClassifier2(Classifier):
    """MNIST classifier (two dense layers)."""

    def __init__(self, enc_out_features):
        super(MNISTDenseClassifier2, self).__init__()

        if is_iterable(enc_out_features):
            enc_out_features = product(enc_out_features)

        self.classifier = nn.Sequential(
            nn.Linear(enc_out_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
            nn.LogSoftmax(dim=1),
        )


class MNISTConvClassifier4(Classifier):
    """MNIST convolutional classifier (four conv layers)."""

    def __init__(self, enc_out_features):
        super(MNISTConvClassifier4, self).__init__()

        # input dimensions
        c, h, w = enc_out_features

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=9, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c*4, out_channels=10, kernel_size=(h-6, w-6), stride=1, padding=0),
            Flatten(),
            nn.LogSoftmax(dim=1),
        )
        self.is_convolutional = True


class CUBDenseClassifier3(Classifier):
    """CUB classifier (three dense layers)."""

    def __init__(self, enc_out_features):
        super(CUBDenseClassifier3, self).__init__()

        if is_iterable(enc_out_features):
            enc_out_features = product(enc_out_features)

        self.classifier = nn.Sequential(
            nn.Linear(enc_out_features, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.LogSoftmax(dim=1),
        )


# ========================
# VARIATIONAL AUTOENCODERS
# ========================


class SVAE(SAE):
    """Stacked variational autoencoder."""

    def __init__(self):
        super(SVAE, self).__init__()

        self.mean_estimators = nn.ModuleList([])
        self.log_var_estimators = nn.ModuleList([])

    def forward(self, x, ae_idx=None):
        z, mean, log_var = self.encode(x, ae_idx)
        return self.decode(z, ae_idx), mean, log_var

    def get_enc_out_features(self, ae_idx):
        enc_out_features = None
        for module in self.mean_estimators[ae_idx]:
            if hasattr(module, 'out_features'):
                enc_out_features = module.out_features
        return enc_out_features

    def encode(self, x, ae_idx=None):
        mean = x
        log_var = x
        if ae_idx is None:
            for i in range(self.num_trained_blocks):
                x = self.encoders[i](x)
                mean = self.mean_estimators[i](x)
                log_var = self.log_var_estimators[i](x)
                x = self.sample_latent_vector(mean, log_var)
        else:
            x = self.encoders[ae_idx](x)
            mean = self.mean_estimators[ae_idx](x)
            log_var = self.log_var_estimators[ae_idx](x)
            x = self.sample_latent_vector(mean, log_var)
        return x, mean, log_var

    @staticmethod
    def sample_latent_vector(mean, log_var):
        # ------------------------
        # reparameterization trick
        # ------------------------
        # MEAN and LOG_VAR are "mu" and log("sigma" ^2) using
        # the notation from eq. 10 in Auto-Encoding Variational Bayes
        # -----------------------------------------------------------
        stdev = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(stdev)
        return mean + stdev * epsilon  # sampled latent vector


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
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        kl_div = 0.5 * torch.sum(mean * mean + torch.exp(log_var) - log_var - 1, dim=1)
        kl_div = torch.mean(kl_div) if self.reduction == 'mean' else torch.sum(kl_div)

        return reconstruction_loss + kl_div


class MNISTVAE(SVAE):
    """MNIST variational autoencoder."""

    def __init__(self):
        super(MNISTVAE, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=28*28, out_features=400),
                nn.ReLU(),
            ),
        ])
        self.mean_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=400, out_features=20),
            ),
        ])
        self.log_var_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=400, out_features=20),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=20, out_features=400),
                nn.ReLU(),
                nn.Linear(in_features=400, out_features=28*28),
                nn.Sigmoid(),
            ),
        ])


class MNISTSVAE(SVAE):
    """MNIST stacked variational autoencoder."""

    def __init__(self):
        super(MNISTSVAE, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=28*28, out_features=500),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(in_features=50, out_features=25),
                nn.ReLU(),
            ),
        ])
        self.mean_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=500, out_features=50),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=25, out_features=10),
            ),
        ])
        self.log_var_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=500, out_features=50),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=25, out_features=10),
            ),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=10, out_features=25),
                nn.ReLU(),
                nn.Linear(in_features=25, out_features=50),
                nn.Sigmoid(),
            ),
            nn.Sequential(
                nn.Linear(in_features=50, out_features=500),
                nn.ReLU(),
                nn.Linear(in_features=500, out_features=28*28),
                nn.Sigmoid(),
            ),
        ])
