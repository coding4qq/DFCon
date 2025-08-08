import torch
import torch.nn as nn
from models.modules.inr import INR


class gs_embedding(nn.Module):
    def __init__(self, args):
        super(gs_embedding, self).__init__()
        datetime_feats = args.datetime_feats
        layer_size = args.layer_size
        inr_layers = args.inr_layers
        n_fourier_feats = args.n_fourier_feats
        scales = args.scales
        d_model = args.d_model
        self.inr = INR(in_feats=datetime_feats, layers=inr_layers, layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats, scales=scales, dropout=0.1)

    def forward(self, x, x_time):
        emb_time = self.inr(x_time)
        # print(emb_time.shape)
        emb = torch.cat([x, emb_time], dim=-1)
        # print(emb.shape)
        return emb

