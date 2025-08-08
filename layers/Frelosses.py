import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from layers.Dominant_Estimation import GetDominant


def autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf)  # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=1, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask


def dominant_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    if reduction:
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    else:
        loss = (logits[:, t, T + t - 1].mean(dim=1) + logits[:, T + t, t].mean(dim=1)) / 2
    return loss  # , a_loss


class DominantLoss(nn.Module):
    def __init__(self, args, acf_values, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(DominantLoss, self).__init__()
        self.get_dominant_repr = GetDominant(args.local_k)
        self.get_dominant_pos = GetDominant(args.pos_k)

        self.get_res = GetDominant(args.global_k)
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.acf_values = torch.from_numpy(acf_values)

    def local_fre_contrastive_loss(self, predict_data, pos_data):
        pre_dom, _ = self.get_dominant_repr(predict_data)
        tar_dom, _ = self.get_dominant_pos(pos_data)
        loss = dominant_contrastive_loss(pre_dom, tar_dom)
        return loss

    def avg_global_contrastive_loss(self, features, labels):
        _, features = self.get_res(features)

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        B, T, C = features.shape

        pooled_features = F.max_pool1d(features.permute(0, 2, 1), kernel_size=T).squeeze(-1)

        global_features = pooled_features.clone()

        anchor_dot_contrast = torch.div(
            torch.matmul(global_features, global_features.transpose(1, 0)),  # (B, C) X (C, B) > (B, B)
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        global_logits = anchor_dot_contrast - logits_max.detach()  # subtract most large value

        # Global Distance map shape: (B, B)
        global_distmap = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()

        global_distmap = self.acf_values[global_distmap.abs().long()].float().to(device)

        neg_mask = torch.scatter(
            torch.ones_like(global_distmap),
            1,
            torch.arange(B).view(-1, 1).to(device),
            0
        )

        self_mask = (global_distmap == 1.0)
        pos_mask = autocorr_mask(global_distmap, self_mask) + (neg_mask * self_mask)

        exp_global_logits = torch.exp(global_logits) * neg_mask  # denominator

        log_global_prob = global_logits - torch.log(exp_global_logits.sum(1, keepdim=True))  # (B, B) > (B ,1)

        mean_log_global_prob_pos = (global_distmap * pos_mask * log_global_prob).sum(1) \
                                   / pos_mask.sum(1)

        global_loss = - (self.temperature / self.base_temperature) * mean_log_global_prob_pos

        return global_loss

    def forward(self, repr, pos, labels):
        local_loss = self.local_fre_contrastive_loss(repr, pos)
        global_loss = self.avg_global_contrastive_loss(repr, labels)
        global_loss = global_loss.mean(dim=0)
        return local_loss, global_loss


if __name__ == '__main__':
    batch_size = 5
    n_view = 2
    seq_len = 4
    dim = 3
    # features = torch.rand((batch_size, n_view, seq_len, dim)) # (Batch_size, N_view, T, Dim)
    features = torch.rand((batch_size, seq_len, dim))  # (Batch_size, T, Dim)
    labels = torch.tensor([0, 2, 3, 4, 6])
    # distmap = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()
    #
    # acf_values = np.array([1.0, 0.3, 0.1, -0.3, 0.2, 0.4, 0.1, 0.0])
    # print(acf_values[distmap])
    # supcreg = AutoCon(batch_size, seq_len, acf_values, contrast_mode='all')  # .cuda()
    # local_loss, global_loss = supcreg(features.cuda(), labels.cuda())
    # print(local_loss.shape, global_loss.shape)
