import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import DataEmbedding
from layers.dilated_conv import DilatedConvEncoder


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=2,
                 hidden_layers=2,  # 默认是2
                 dropout=0.25,
                 activation='relu'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = f_in * hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(self.hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class Attention(nn.Module):
    def __init__(self, h_num, dropout=0.25):
        super(Attention, self).__init__()
        self.head = h_num
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, v):
        B, L_pre, D = q.shape
        # print(q.shape)
        q = q.reshape(B, self.head, -1, D)
        scale = 1. / math.sqrt(D)
        k = torch.nn.Parameter(torch.FloatTensor([[1] * D] * L_pre), requires_grad=True).to('cuda')
        k = k.reshape(self.head, -1, D)

        v = v.reshape(B, -1, k.shape[1], D)
        scores = torch.einsum("bhnd,hld->bnhl", q, k)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bnhn,blnd->bnhd", A, v).reshape(B, -1, D)
        return V, A


def FFT_for_Period(x, k=1):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    value, top_list = torch.topk(frequency_list, k)
    # top_list = top_list.detach().cpu().numpy()

    zeros = torch.zeros_like(xf[:, 0, :], device=xf.device).float()
    # res = torch.zeros_like(xf, device=xf.device).float()
    xf[:, 0, :] = zeros
    # res[:, 0, :] = xf[:, 0, :]
    for i in top_list:
        xf[:, i, :] = zeros
        # res[:, i, :] = xf[:, i, :]
    # xf = torch.fft.irfft(xf, dim=1)
    res = torch.fft.irfft(xf, dim=1)
    return res


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class fft_decom(nn.Module):
    def __init__(self, top_k):
        super(fft_decom, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        long = FFT_for_Period(x, self.top_k)
        short = x - long
        return short, long


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_dims = configs.d_model
        self.repr_dims = configs.d_ff
        self.depth = configs.e_layers

        self.AutoCon = configs.AutoCon

        self.AutoCon_wnorm = configs.AutoCon_wnorm
        self.AutoCon_multiscales = configs.AutoCon_multiscales
        self.top_k = configs.top_k

        self.enc_embedding = DataEmbedding(configs.enc_in, self.hidden_dims, configs.embed, configs.freq,
                                           dropout=configs.dropout)
        self.feature_extractor = DilatedConvEncoder(
            self.hidden_dims,
            [self.hidden_dims] * self.depth + [self.repr_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)
        self.attn = Attention(configs.n_heads, dropout=0.25)

        self.ch_mlps = nn.ModuleList([nn.Linear(self.repr_dims, self.c_out) for _ in self.AutoCon_multiscales])
        self.length_mlp = nn.Linear(self.seq_len, self.pred_len)
        self.s_dm_mlp = nn.Linear(self.repr_dims, self.c_out)
        if configs.Auto_decomp == 'mv':
            self.trend_decoms = nn.ModuleList(
                [series_decomp(kernel_size=dlen + 1) for dlen in self.AutoCon_multiscales])
        elif configs.Auto_decomp == 'fft':
            self.trend_decoms = nn.ModuleList([fft_decom(top_k=k) for k in configs.top_k_multi])

        # self.trend_decoms = nn.ModuleList([series_decomp(kernel_size=dlen + 1) for dlen in self.AutoCon_multiscales])

        self.fft_decomp = fft_decom(configs.top_k_decomp)
        # self.fft_decomps = nn.ModuleList([fft_decom(top_k=k) for k in configs.top_k_multi])

        self.input_decom = series_decomp(25)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.mlp = MLP(self.seq_len, self.pred_len, 2, 2, 0.1)
        self.pos_linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        # x: [Batch, Input length, Channel]
        # x_enc = self.attn(x_enc, x_enc)

        # x_pos = self.pos_linear(x_enc.transpose(1, -1)).transpose(1, -1)

        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:, -1:, :].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'FFT':
            short_x, long_x = self.fft_decomp(x_enc)
        else:
            raise Exception(
                f'Not Supported Window Normalization:{self.AutoCon_wnorm}. Use {"{ReVIN | Mean | LastVal | Decomp}"}.')

        enc_out = self.enc_embedding(long_x, x_mark_enc)
        enc_out = enc_out.transpose(1, 2)  # B x Ch x T
        repr = self.repr_dropout(self.feature_extractor(enc_out)).transpose(1, 2)  # B x Co x T

        # repr, _ = self.repr_dropout(self.attn(enc_out, enc_out))
        # print(repr.shape)
        # exit()

        if onlyrepr:
            return None, repr

        len_out = F.gelu(repr.transpose(1, 2))  # B x O x T
        len_out = self.length_mlp(len_out).transpose(1, 2)  # (B, I, C) > (B, O, C)

        trend_outs = []
        for trend_decom, ch_mlp in zip(self.trend_decoms, self.ch_mlps):
            # res_ = 0
            # for trend_decom, ch_mlp in zip(self.fft_decomps, self.ch_mlps):
            # 之前的版本 只保留trend（但实际上问题是不使用decomp，或许并不是trend项
            _, dec_out = trend_decom(len_out)

            dec_out = F.gelu(dec_out)
            dec_out = ch_mlp(dec_out)  # (B, I, D) > (B, I, C)
            _, trend_out = trend_decom(dec_out)
            trend_outs.append(trend_out)

            # 4.18新加对于short_x的处理
            # _, dec_out = trend_decom(len_out)
            # res_ = _ + res_
            # dec_out = F.gelu(dec_out)
            # dec_out = ch_mlp(dec_out)
            # _, trend_out = trend_decom(dec_out)
            # trend_outs.append(trend_out)
            # 多出了res_

        trend_outs = torch.stack(trend_outs, dim=-1).sum(dim=-1)

        # Seasonal Prediction: NLinear
        # season_out = self.Linear(short_x.permute(0, 2, 1)).permute(0, 2, 1)
        # short_x = self.Linear(short_x.permute(0, 2, 1)).permute(0, 2, 1)

        # 4.18 添加剩余的short
        # print(short_x.shape)
        # print(res_.shape)
        # res_, _ = self.attn(res_, res_)
        # res_ = self.s_dm_mlp(res_)

        # short_x = short_x + res_

        # season_out = self.mlp(short_x.permute(0, 2, 1)).permute(0, 2, 1) + res_
        # season_out = self.mlp(short_x.permute(0, 2, 1)).permute(0, 2, 1)
        season_out = self.Linear(short_x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.AutoCon_wnorm == 'ReVIN':
            pred = (season_out + trend_outs) * (seq_std + 1e-5) + seq_mean
        elif self.AutoCon_wnorm == 'Mean':
            pred = season_out + trend_outs + seq_mean
        elif self.AutoCon_wnorm == 'Decomp':
            pred = season_out + trend_outs
        elif self.AutoCon_wnorm == 'LastVal':
            pred = season_out + trend_outs + seq_last
        elif self.AutoCon_wnorm == 'FFT':
            pred = season_out + trend_outs
        else:
            raise Exception()

        if self.AutoCon:
            return pred, repr, enc_out.transpose(1,-1)  # [Batch, Output length, Channel]
        else:
            return pred

    def get_embeddings(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                       enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):

        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:, -1:, :].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception()

        if self.ablation != 2:
            enc_out = self.enc_embedding(long_x, x_mark_enc)
            enc_out = enc_out.transpose(1, 2)  # B x Ch x T
            repr = self.repr_dropout(self.feature_extractor(enc_out)).transpose(1, 2)  # B x Co x T
        else:
            repr = None

        return repr
