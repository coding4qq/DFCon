import torch
import torch.nn as nn

def add_gaussian_noise(signal, target_snr_db, noise_axis=1):
    
    signal_power = torch.mean(signal**2, dim=(1,2))
    
    snr_linear = 10 ** (target_snr_db / 10)
    
    noise_var = signal_power / snr_linear
    
    noise_shape = list(signal.shape)
    
    noise_shape[noise_axis] = 1
    
    noise = torch.randn_like(signal) * torch.sqrt(noise_var[:, None, None])
    
    noisy_signal = signal + noise
    
    noise_power = torch.mean(noise ** 2, dim=(1,2))
    
    actual_snr = 10 * torch.log10(signal_power / noise_power).mean().item()
    
#     return noisy_signal, actual_snr
    return noisy_signal



def FFT_for_Period(x, k=1, first_freq=0):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    dominant_list = xf.clone()
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    value, top_list = torch.topk(frequency_list, k)
    # top_list = top_list.detach().cpu().numpy()

    zeros = torch.zeros_like(xf[:, 0, :], device=xf.device).float()
    # res = torch.zeros_like(xf, device=xf.device).float()
    if first_freq == 0:
        xf[:, 0, :] = zeros
    # res[:, 0, :] = xf[:, 0, :]
    for i in top_list:
        xf[:, i, :] = zeros
        # res[:, i, :] = xf[:, i, :]
    # xf = torch.fft.irfft(xf, dim=1)
    dominant_list = dominant_list - xf
    res = torch.fft.irfft(xf, dim=1)

    dominant_res = torch.fft.irfft(dominant_list, dim=1)
    return dominant_res, res


class GetDominant(nn.Module):
    def __init__(self, top_k, first_freq=0):
        super(GetDominant, self).__init__()
        self.k = top_k
        self.first_freq = first_freq

    def forward(self, x):
        dominant, res = FFT_for_Period(x, self.k, self.first_freq)
        return dominant, res
