import torch
import torch.nn as nn
import torch.nn.functional as F

class CVLinear(nn.Module):
    """

    """
    def __init__(self, in_features, out_features, bias=False):
        super(CVLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weights_real = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weights_imag = nn.Linear(self.in_features, self.out_features, bias=False)
        # self.weights_real = nn.Parameter(nn.init.xavier_normal_(torch.ones(in_features, out_features)))
        # self.weights_imag = nn.Parameter(nn.init.xavier_normal_(torch.ones(in_features, out_features)))
        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_features))
            self.bias_imag = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        AC = self.weights_real(input.real)
        BD = self.weights_imag(input.imag)
        AD = self.weights_imag(input.real)
        BC = self.weights_real(input.imag)
        AC_sub_BD = AC - BD
        AD_add_BC = AD + BC
        if self.bias:
            AC_sub_BD += self.bias_real
            AD_add_BC += self.bias_imag

        return AC_sub_BD.type(torch.complex64) + 1j * AD_add_BC.type(torch.complex64)

class CVSoftThreshold(nn.Module):
    """

        :param R_x: real part of input data
        :param I_x: image part of input data
        :param theta: soft threshold
        :return: real part and image part of input data operated by soft-threshold function

        note:: You can also compare the return with the MATLAB.
    """

    def __init__(self):
        super(CVSoftThreshold, self).__init__()
        self.theta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        R_x_square = torch.einsum('ij,ij->ij', x.real, x.real)
        I_x_square = torch.einsum('ij,ij->ij', x.imag, x.imag)
        abs_x = torch.sqrt(R_x_square+I_x_square)
        B = F.relu(abs_x - self.theta)
        R_out = torch.einsum('ij,ij->ij', torch.div(x.real, abs_x), B)
        I_out = torch.einsum('ij,ij->ij', torch.div(x.imag, abs_x), B)

        return R_out.type(torch.complex64) + 1j * I_out.type(torch.complex64)

class CVLayerNorm(nn.Module):
    """

    """
    def __init__(self , in_features):
        super(CVLayerNorm, self).__init__()
        self.layernorm_real = nn.LayerNorm(in_features)
        self.layernorm_imag = nn.LayerNorm(in_features)

    def forward(self, x):
        out_real = self.layernorm_real(x.real)
        out_imag = self.layernorm_imag(x.imag)
        return out_real.type(torch.complex64) + 1j * out_imag.type(torch.complex64)

class CVResblock(nn.Module):
    """

    """
    def __init__(self, in_features=32):
        super(CVResblock, self).__init__()
        self.resblock = nn.Sequential(
            CVLinear(in_features, in_features),
            CVLayerNorm(in_features),
            CVSoftThreshold(),
            CVLinear(in_features, in_features),
            CVLayerNorm(in_features),
        )
        self.threshold = CVSoftThreshold()

    def forward(self, x):
        identity = x
        out = self.resblock(x)
        out = out + identity
        out = self.threshold(out)
        return out

class Encoder3(nn.Module):
    """

    """
    def __init__(self, in_features=32):
        super(Encoder3, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features)
        )

    def forward(self, x):
        out = self.resblock(x)
        return out

class Encoder4(nn.Module):
    """

    """
    def __init__(self, in_features=32):
        super(Encoder4, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features)
        )

    def forward(self, x):
        out = self.resblock(x)
        return out

class Encoder5(nn.Module):
    """

    """
    def __init__(self, in_features=32):
        super(Encoder5, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features)
        )

    def forward(self, x):
        out = self.resblock(x)
        return out



class Decoder11(nn.Module):
    """

    """
    def __init__(self, out_features):
        super(Decoder11, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(out_features),
            CVLinear(out_features, out_features)
        )

    def forward(self, x):

        out = self.resblock(x)
        return out

class Decoder21(nn.Module):
    """

    """
    def __init__(self, out_features):
        super(Decoder21, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(out_features),
            CVResblock(out_features),
            CVLinear(out_features, out_features)
        )

    def forward(self, x):
        out = self.resblock(x)
        return out

class Decoder31(nn.Module):
    """

    """
    def __init__(self, out_features):
        super(Decoder31, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(out_features),
            CVResblock(out_features),
            CVResblock(out_features),
            CVLinear(out_features, out_features)
        )

    def forward(self, x):
        out = self.resblock(x)
        return out

