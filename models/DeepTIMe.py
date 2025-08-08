# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import gin
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat, reduce

from models.modules.inr import INR
from models.modules.regressors import RidgeRegressor

from layers.dilated_conv import DilatedConvEncoder
from layers.Embed import DataEmbedding


@gin.configurable()
def deeptime(datetime_feats: int, layer_size: int, inr_layers: int, n_fourier_feats: int, scales: float):
    return Model(datetime_feats, layer_size, inr_layers, n_fourier_feats, scales)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        datetime_feats = args.datetime_feats
        layer_size = args.layer_size
        inr_layers = args.inr_layers
        n_fourier_feats = args.n_fourier_feats
        scales = args.scales

        self.channels = args.enc_in
        self.c_out = args.c_out
        self.hidden_dims = args.d_model
        self.repr_dims = args.d_ff

        self.inr = INR(in_feats=datetime_feats + 1, layers=inr_layers, layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats, scales=scales)
        self.adaptive_weights = RidgeRegressor()
        self.depth = args.e_layers

        self.AutoCon_wnorm = args.AutoCon_wnorm
        self.AutoCon_multiscales = args.AutoCon_multiscales

        self.enc_embedding = DataEmbedding(args.enc_in, self.hidden_dims, args.embed, args.freq, dropout=args.dropout)
        self.feature_extractor = DilatedConvEncoder(
            self.hidden_dims,
            [self.hidden_dims] * self.depth + [self.repr_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.AutoCon = args.AutoCon

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        tgt_horizon_len = y_time.shape[1]
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)

        if y_time.shape[-1] != 0:
            time = torch.cat([x_time, y_time], dim=1)
            coords = repeat(coords, '1 t 1 -> b t 1', b=time.shape[0])
            coords = torch.cat([coords, time], dim=-1)
            time_reprs = self.inr(coords)
        else:
            time_reprs = repeat(self.inr(coords), '1 t d -> b t d', b=batch_size)

        lookback_reprs = time_reprs[:, :-tgt_horizon_len]
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
        w, b = self.adaptive_weights(lookback_reprs, x)
        preds = self.forecast(horizon_reprs, w, b)

        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x.mean(dim=1, keepdim=True).detach()
            seq_std = x.std(dim=1, keepdim=True).detach()
            short_x = (x - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x.mean(dim=1, keepdim=True).detach()
            short_x = (x - seq_mean)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x[:, -1:, :].detach()
            short_x = (x - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception(
                f'Not Supported Window Normalization:{self.AutoCon_wnorm}. Use {"{ReVIN | Mean | LastVal | Decomp}"}.')

        enc_out = self.enc_embedding(long_x, x_time)

        repr = self.repr_dropout(self.feature_extractor(enc_out.transpose(1, 2))).transpose(1, 2)
        if self.AutoCon:
            return preds, repr
        return preds

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, 't -> 1 t 1')
