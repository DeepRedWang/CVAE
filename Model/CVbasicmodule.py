import torch
import torch.nn as nn
import torch.nn.functional as F
#
# def apply_complex(fr, fi, input, dtype = torch.complex64):
#     return (fr(input.real)-fi(input.imag)).type(dtype) \
#             + 1j*(fr(input.imag)+fi(input.real)).type(dtype)
# def complex_dropout(input, p=0.5, training=True):
#     # need to have the same dropout mask for real and imaginary part,
#     # this not a clean solution!
#     #mask = torch.ones_like(input).type(torch.float32)
#     mask = torch.ones(input.shape, dtype=torch.float32)
#     mask = F.dropout(mask, p, training)
#     mask.type(input.dtype)
#     return mask*input

class CVDropout(nn.Module):
    def __init__(self, p=0.1):
        super(CVDropout, self).__init__()
        self.p = p
        # self.dropout = nn.Dropout(p)
        # self.register_buffer('mask', torch.ones((2048,256,256), dtype=torch.float32))


    def forward(self, input):
        if self.training:
            mask = input/input
            # mask = self.dropout(self.mask)
            mask = F.dropout(mask.real, self.p, self.training)
            # mask.type_as(input)
            # print(mask.device)
            # print(mask.dtype)
            return mask*input
        else:
            return input

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

class RCVLinear(nn.Module):
    """

    """
    def __init__(self, in_features, out_features, bias=False):
        super(RCVLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weights_real = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weights_imag = nn.Linear(self.in_features, self.out_features, bias=False)
        # self.weights_real = nn.Parameter(nn.init.xavier_normal_(torch.ones(in_features, out_features)))
        # self.weights_imag = nn.Parameter(nn.init.xavier_normal_(torch.ones(in_features, out_features)))
        # if bias:
        #     self.bias_real = nn.Parameter(torch.randn(out_features))
        #     self.bias_imag = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        b = input[:,0,:].squeeze()
        c = input[:,1,:].squeeze()
        AC = self.weights_real(b)
        BD = self.weights_imag(c)
        AD = self.weights_imag(b)
        BC = self.weights_real(c)
        AC_sub_BD = AC - BD
        AD_add_BC = AD + BC
        # if self.bias==True:
        #     AC_sub_BD += self.bias_real
        #     AD_add_BC += self.bias_imag

        return torch.stack((AC_sub_BD,AC_sub_BD), -2)
        # return AC_sub_BD.type(torch.complex64) + 1j * AD_add_BC.type(torch.complex64)



class CVSoftThreshold2D(nn.Module):
    """

        :param R_x: real part of input data
        :param I_x: image part of input data
        :param theta: soft threshold
        :return: real part and image part of input data operated by soft-threshold function

        note:: You can also compare the return with the MATLAB.
    """

    def __init__(self):
        super(CVSoftThreshold2D, self).__init__()
        self.theta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        R_x_square = torch.einsum('ijk,ijk->ijk', x.real, x.real)
        I_x_square = torch.einsum('ijk,ijk->ijk', x.imag, x.imag)
        abs_x = torch.sqrt(R_x_square+I_x_square)
        B = F.relu(abs_x - self.theta)
        R_out = torch.einsum('ijk,ijk->ijk', torch.div(x.real, abs_x), B)
        I_out = torch.einsum('ijk,ijk->ijk', torch.div(x.imag, abs_x), B)

        return R_out.type(torch.complex64) + 1j * I_out.type(torch.complex64)


class CVDotProductAttention(nn.Module):
    """

    """
    def __init__(self, dropout, **kwargs):
        super(CVDotProductAttention, self).__init__(**kwargs)
        self.cvdropout = CVDropout(dropout)

    def forward(self, queries, keys, values):
        d_k = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(d_k))
        attention_weights = nn.functional.softmax(torch.abs(scores), dim=-1)
        cv_attention_weights = attention_weights + 1j*torch.zeros_like(attention_weights)
        return torch.bmm(self.cvdropout(cv_attention_weights), values)

class CVMultiHeadAttention(nn.Module):
    """

    """
    def __init__(self, d_q, d_k, d_v, num_heads, dropout, **kwargs):
        super(CVMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = CVDotProductAttention(dropout)
        self.W_q = CVLinear(d_q, d_q*num_heads)
        self.W_k = CVLinear(d_k, d_k*num_heads)
        self.W_v = CVLinear(d_v, d_v*num_heads)
        self.W_o = CVLinear(d_v*num_heads, d_v)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        #input
        #(num_batch, num_sequence, num_head*d_v)

        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        #(num_batch, num_sequence, num_head, d_v)

        X = X.permute(0,2,1,3)
        #(num_batch, num_head, num_sequence, d_v)

        return X.reshape(-1, X.shape[2], X.shape[3])
        #(num_batch * num_head, num_sequence, d_v)
    def transpose_output(self, X, num_heads):
        """
        Invese the operation of transpose_qkv

        """
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        # (num_batch, num_head, num_sequence, d_v)

        X = X.permute(0,2,1,3)
        # (num_batch, num_sequence, num_head, d_v)

        return X.reshape(X.shape[0], X.shape[1], -1)
        # (num_batch, num_sequence, num_head*d_v)

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

class RCVLayerNorm(nn.Module):
    """

    """
    def __init__(self , in_features):
        super(RCVLayerNorm, self).__init__()
        self.layernorm_real = nn.LayerNorm(in_features)
        self.layernorm_imag = nn.LayerNorm(in_features)

    def forward(self, x):

        out_real = self.layernorm_real(x[:,0,:].squeeze())
        out_imag = self.layernorm_imag(x[:,1,:].squeeze())

        return torch.stack((out_real, out_imag), -2)

class CVAddLyNorm(nn.Module):
    """

    """
    def __init__(self, d_v, dropout, **kwargs):
        super(CVAddLyNorm, self).__init__(**kwargs)
        self.dropout = CVDropout(dropout)
        self.cvlayernorm = CVLayerNorm(d_v)

    def forward(self, X, Y):
        return self.cvlayernorm(X + self.dropout(Y))

class CVMLP(nn.Module):
    """
    Feedforward Complexed-Value Neural Network based position.
    """
    def __init__(self, cvmlp_num_inputs, cvmlp_num_hiddens, cvmlp_num_outputs, **kwargs):
        super(CVMLP, self).__init__(**kwargs)
        self.cvmlp = nn.Sequential(
            CVLinear(cvmlp_num_inputs, cvmlp_num_hiddens, True),
            CVReLU(),
            CVLinear(cvmlp_num_hiddens, cvmlp_num_outputs, True)
        )

    def forward(self, X):
        return self.cvmlp(X)

class CVTransformerDecoderBlock(nn.Module):
    """
    Created a Decoder mechanism based on Transformer decoder architecture.
    """
    def __init__(self, d_q, d_k, d_v, feature, mlp_num_inputs, mlp_num_hiddens, mlp_num_outputs,
                 num_heads, dropout, **kwargs):
        super(CVTransformerDecoderBlock, self).__init__(**kwargs)
        self.attention = CVMultiHeadAttention(d_q, d_k, d_v, num_heads, dropout)
        self.add_lynorm1 = CVAddLyNorm(feature, dropout)
        self.mlp = CVMLP(mlp_num_inputs, mlp_num_hiddens, mlp_num_outputs)
        self.add_lynorm2 = CVAddLyNorm(feature, dropout)

    def forward(self, X):
        Y = self.add_lynorm1(X, self.attention(X, X, X))
        return self.add_lynorm2(Y, self.mlp(Y))

class CVTransformerDecoder(nn.Module):
    """

    """
    def __init__(self, d_q, d_k, d_v, feature, mlp_num_inputs, mlp_num_hiddens, mlp_num_outputs,
                 num_heads, dropout, **kwargs):
        super(CVTransformerDecoder, self).__init__(**kwargs)
        """

        :param num_block: The num of block.
        """
        self.transformerdecoder = nn.Sequential(
            CVTransformerDecoderBlock(d_q, d_k, d_v, feature, mlp_num_inputs, mlp_num_hiddens, mlp_num_outputs,
                 num_heads, dropout),
            CVTransformerDecoderBlock(d_q, d_k, d_v, feature, mlp_num_inputs, mlp_num_hiddens, mlp_num_outputs,
                 num_heads, dropout)
        )
    def forward(self, X):
        return self.transformerdecoder(X)

class Encoder(nn.Module):
    """

    """
    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.resblock = nn.Sequential(
            CVLinear(in_features=in_features, out_features=in_features),
            CVSoftThreshold(),
            CVLinear(in_features=in_features, out_features=in_features),
        )

    def forward(self, x):

        out = self.resblock(x)
        return out

class Decoder(nn.Module):
    """

    """
    def __init__(self, out_features):
        super(Decoder, self).__init__()
        self.resblock = nn.Sequential(
            CVLinear(in_features=out_features, out_features=out_features),
            CVSoftThreshold(),
            CVLinear(in_features=out_features, out_features=out_features)
        )

    def forward(self, x):
        out = self.resblock(x)
        return out


class RCVResblock(nn.Module):
    """

    """
    def __init__(self, in_features=32):
        super(RCVResblock, self).__init__()
        self.resblock = nn.Sequential(
            RCVLinear(in_features, in_features),
            RCVLayerNorm(in_features),
            RCVSoftThreshold(),
            RCVLinear(in_features, in_features),
            RCVLayerNorm(in_features),
        )
        self.threshold = RCVSoftThreshold()

    def forward(self, x):
        identity = x
        out = self.resblock(x)
        out = out + identity
        out = self.threshold(out)
        return out
#    ----------------------------backup-------------------------
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


class Encoder1(nn.Module):
    """

    """
    def __init__(self, in_features=32):
        super(Encoder1, self).__init__()
        self.resblock = nn.Sequential(
            # ComplexBatchNorm1d(in_features),
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features)
            # ComplexBatchNorm1d(in_features)
        )


    def forward(self, x):

        out = self.resblock(x)

        return out


class REncoder1(nn.Module):
    """

    """
    def __init__(self, in_features=32):
        super(REncoder1, self).__init__()
        self.resblock = nn.Sequential(
            # ComplexBatchNorm1d(in_features),
            RCVResblock(in_features),
            RCVResblock(in_features),
            RCVResblock(in_features),
            RCVResblock(in_features)
            # ComplexBatchNorm1d(in_features)
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
            # ComplexBatchNorm1d(in_features),
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features),
            CVResblock(in_features)

            # ComplexBatchNorm1d(in_features)
        )


    def forward(self, x):

        out = self.resblock(x)

        return out



class Decoder1(nn.Module):
    """

    """
    def __init__(self, out_features):
        super(Decoder1, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(out_features),
            CVLinear(out_features, out_features)
        )


    def forward(self, x):

        out = self.resblock(x)
        return out

class Decoder2blcok(nn.Module):
    """

    """
    def __init__(self, out_features):
        super(Decoder2blcok, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(out_features),
            CVResblock(out_features)
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

#real-valued mode
class RDecoder21(nn.Module):
    """

    """
    def __init__(self, out_features):
        super(RDecoder21, self).__init__()
        self.resblock = nn.Sequential(
            RCVResblock(out_features),
            RCVResblock(out_features),
            RCVLinear(out_features, out_features)
        )


    def forward(self, x):

        out = self.resblock(x)
        return out

class RCVSoftThreshold(nn.Module):
    """

        :param R_x: real part of input data
        :param I_x: image part of input data
        :param theta: soft threshold
        :return: real part and image part of input data operated by soft-threshold function

        note:: You can also compare the return with the MATLAB.
    """

    def __init__(self):
        super(RCVSoftThreshold, self).__init__()
        self.theta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        a = x[:, 0, :].squeeze()
        b = x[:, 1, :].squeeze()
        R_x_square = torch.einsum('ij,ij->ij', a, a)
        I_x_square = torch.einsum('ij,ij->ij', b, b)
        abs_x = torch.sqrt(R_x_square+I_x_square)
        B = F.relu(abs_x - self.theta)
        R_out = torch.einsum('ij,ij->ij', torch.div(a, abs_x), B)
        I_out = torch.einsum('ij,ij->ij', torch.div(b, abs_x), B)

        return torch.stack((R_out, I_out), -2)


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


class CVReLU(nn.Module):
    """

    """
    def __init__(self):
        super(CVReLU, self).__init__()
        self.ReLU = nn.ReLU()

    def forward(self, input):
        input_real = self.ReLU(input.real)
        input_imag = self.ReLU(input.imag)
        return input_real.type(torch.complex64) + 1j * input_imag.type(torch.complex64)


class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)

class DecoderRes2(nn.Module):
    """

    """
    def __init__(self, out_features=256):
        super(DecoderRes2, self).__init__()
        self.resblock = nn.Sequential(
            CVResblock(out_features),
            CVResblock(out_features),
            CVLinear(out_features, out_features)
        )

    def forward(self, x):

        out = self.resblock(x)
        return out

class CVConv2d(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(CVConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.conv_imag = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bias = bias
        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_channels))
            self.bias_imag = nn.Parameter(torch.randn(out_channels))


    def forward(self, input):
        AC = self.conv_real(input.real)
        BD = self.conv_imag(input.imag)
        AD = self.conv_imag(input.real)
        BC = self.conv_real(input.imag)
        AC_sub_BD = AC - BD
        AD_add_BC = AD + BC

        if self.bias:
            AC_sub_BD += self.bias_real
            AD_add_BC += self.bias_imag
        return AC_sub_BD.type(torch.complex64) + 1j * AD_add_BC.type(torch.complex64)



class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = input.real.mean(dim=0).type(torch.complex64)
            mean_i = input.imag.mean(dim=0).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + \
                                    (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, ...]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = input.real.var(dim=0, unbiased=False) + self.eps
            Cii = input.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) + \
                                       (1 - exponential_average_factor) * self.running_covar[:, 0]

            self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) + \
                                       (1 - exponential_average_factor) * self.running_covar[:, 1]

            self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) + \
                                       (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :] * input.real + Rri[None, :] * input.imag).type(torch.complex64) + \
                1j * (Rii[None, :] * input.imag + Rri[None, :] * input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0] * input.real + self.weight[None, :, 2] * input.imag +
                     self.bias[None, :, 0]).type(torch.complex64) + \
                    1j * (self.weight[None, :, 2] * input.real + self.weight[None, :, 1] * input.imag +
                          self.bias[None, :, 1]).type(torch.complex64)

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input

class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + \
                                    (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1. / n * input.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) + \
                                           (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) + \
                                           (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) + \
                                           (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :, None, None] * input.real +
                 Rri[None, :, None, None] * input.imag).type(torch.complex64) + \
                1j * (Rii[None, :, None, None] * input.imag +
                      Rri[None, :, None, None] * input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0, None, None] * input.real +
                     self.weight[None, :, 2, None, None] * input.imag +
                     self.bias[None, :, 0, None, None]).type(torch.complex64) + \
                    1j * (self.weight[None, :, 2, None, None] * input.real +
                          self.weight[None, :, 1, None, None] * input.imag +
                          self.bias[None, :, 1, None, None]).type(torch.complex64)

        return input
















