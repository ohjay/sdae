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
