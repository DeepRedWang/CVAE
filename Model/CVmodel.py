import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.CVbasicmodule import *
import torch
import torch.nn as nn
# device3 = "cuda:3" if torch.cuda.is_available() else "cpu"
class SAE(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE, self).__init__()
        self.encoder = Encoder1(in_features)
        D = torch.load('./Data/D.pth').type(torch.complex64)
        # FFT = torch.load('./Data/FFT.pth').type(torch.complex64)
        self.register_buffer('downsample', D)
        self.decoder = Decoder1(out_features)


    def forward(self, x):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, self.downsample)
        x3 = self.decoder(x2)


        return x3

class SAERes6(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAERes6, self).__init__()
        self.encoder = Encoder1(in_features)
        self.decoder = DecoderRes2(out_features)


    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)
        return x3


class SAE1(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE1, self).__init__()
        self.encoder = Encoder1(in_features)
        # D = torch.load('./Data/D.pth').type(torch.complex64)
        # FFT = torch.load('./Data/FFT.pth').type(torch.complex64)
        # self.register_buffer('downsample', D)
        self.decoder = Decoder1(out_features)


    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)


        return x3

class SAE42(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE42, self).__init__()
        self.encoder = Encoder1(in_features)
        # D = torch.load('./Data/D.pth').type(torch.complex64)
        # FFT = torch.load('./Data/FFT.pth').type(torch.complex64)
        # self.register_buffer('downsample', D)
        self.decoder = Decoder2blcok(out_features)


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
        self.encoder = Encoder1(in_features)
        # D = torch.load('./Data/D.pth').type(torch.complex64)
        # FFT = torch.load('./Data/FFT.pth').type(torch.complex64)
        # self.register_buffer('downsample', D)
        self.decoder = Decoder21(out_features)


    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)

        return x3

class RSAE421(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(RSAE421, self).__init__()
        self.encoder = REncoder1(in_features)
        # D = torch.load('./Data/D.pth').type(torch.complex64)
        # FFT = torch.load('./Data/FFT.pth').type(torch.complex64)
        # self.register_buffer('downsample', D)
        self.decoder = RDecoder21(out_features)


    def forward(self, x, D):
        x1 = self.encoder(x)
        # print(x1.shape)

        x1 = x1[:,0,:].squeeze_().type(torch.complex64) + 1j * x1[:,1,:].squeeze_().type(torch.complex64)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x2 = torch.stack((x2.real, x2.imag), -2)
        x3 = self.decoder(x2)

        return x3


class SAERes4Trans2(nn.Module):
    """

    """
    def __init__(self, in_features=32, embedding_dim=8):
        super(SAERes4Trans2, self).__init__()
        self.encoder = Encoder1(in_features)
        self.cvlinear1 = CVLinear(1, embedding_dim)
        # D = torch.load('./Data/D.pth').type(torch.complex64)
        # FFT = torch.load('./Data/FFT.pth').type(torch.complex64)
        # self.register_buffer('downsample', D)
        self.decoder = CVTransformerDecoder(embedding_dim, embedding_dim, embedding_dim,
                                            embedding_dim, embedding_dim, 2*embedding_dim, embedding_dim, embedding_dim, 0.1)
        self.cvlinear2 = CVLinear(embedding_dim, 1)


    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D).unsqueeze(dim=-1)
        x2 = self.cvlinear1(x2)
        x3 = self.decoder(x2)
        x3 = self.cvlinear2(x3).squeeze(dim=-1)

        return x3



class SAE511(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=256):
        super(SAE511, self).__init__()
        self.encoder = Encoder5(in_features)
        # D = torch.load('./Data/D.pth').type(torch.complex64)
        # FFT = torch.load('./Data/FFT.pth').type(torch.complex64)
        # self.register_buffer('downsample', D)
        self.decoder = Decoder1(out_features)


    def forward(self, x, D):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, D)
        x3 = self.decoder(x2)

        return x3

class SAE2(nn.Module):
    """

    """
    def __init__(self, in_features=32, out_features=128):
        super(SAE2, self).__init__()
        self.encoder = Encoder(in_features)
        D = torch.load('./Data/D.pth').type(torch.complex64)
        FFT = torch.load('./Data/FFTori.pth').type(torch.complex64)
        self.register_buffer('downsample', D)
        self.register_buffer('FFTori', FFT)
        self.decoder = Decoder(out_features)
        self.threshold = CVSoftThreshold()


    def forward(self, x):
        x1 = self.encoder(x)
        x2 = torch.einsum('ik,kj->ij', x1, self.downsample)
        x3 = self.decoder(x2)
        x4 = torch.einsum('ik,kj->ij', x3, self.FFTori)
        x4 = self.threshold(x4)
        return x4
