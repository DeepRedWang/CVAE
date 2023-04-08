from Model.CVbasicmodel2 import *
import torch
import torch.nn as nn

class SAE321(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE321, self).__init__()
        self.encoder = Encoder3(in_features)
        self.decoder = Decoder21(out_features)


    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)
        return x3

class SAE421(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE421, self).__init__()
        self.encoder = Encoder4(in_features)
        self.decoder = Decoder21(out_features)


    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)
        return x3

class SAE521(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE521, self).__init__()
        self.encoder = Encoder5(in_features)
        self.decoder = Decoder21(out_features)

    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)
        return x3


class SAE411(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE411, self).__init__()
        self.encoder = Encoder4(in_features)
        self.decoder = Decoder11(out_features)

    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)
        return x3

class SAE431(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE431, self).__init__()
        self.encoder = Encoder4(in_features)
        self.decoder = Decoder31(out_features)

    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)
        return x3
