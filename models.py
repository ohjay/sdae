import torch
import torch.nn as nn
import torch.nn.functional as F


class OlshausenAE(nn.Module):

    def __init__(self):
        super(OlshausenAE, self).__init__()

        self.encoder_dense1 = nn.Linear(in_features=12*12, out_features=120)
        self.decoder_dense1 = nn.Linear(in_features=120, out_features=12*12)

        self.encoder = nn.Sequential(
            self.encoder_dense1,
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            self.decoder_dense1,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_first_layer_weights(self, as_tensor=False):
        if as_tensor:
            return self.encoder_dense1.weight.data.cpu()
        return self.encoder_dense1.weight.data.cpu().numpy()


class MNISTAE(nn.Module):

    def __init__(self):
        super(MNISTAE, self).__init__()

        self.encoder_dense1 = nn.Linear(in_features=28*28, out_features=120)
        self.decoder_dense1 = nn.Linear(in_features=120, out_features=28*28)

        self.encoder = nn.Sequential(
            self.encoder_dense1,
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            self.decoder_dense1,
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_first_layer_weights(self, as_tensor=False):
        if as_tensor:
            return self.encoder_dense1.weight.data.cpu()
        return self.encoder_dense1.weight.data.cpu().numpy()
