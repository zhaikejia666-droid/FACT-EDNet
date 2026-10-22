import torch
from torch import nn
from torch.nn import functional as F

from .model import time_intervals


def transient_spectrum(sequence, dt=0.01, eps=1e-8):
    if sequence.ndim != 3 or sequence.shape[1] < 4:
        raise ValueError("Spectrum requires [B,T,C] with T >= 4")
    intervals = time_intervals(sequence, dt)
    if not torch.allclose(intervals[:, 1:], intervals[:, 1:2].expand_as(intervals[:, 1:]), rtol=1e-4, atol=1e-7):
        raise ValueError("FFT objective requires uniformly sampled data; resample irregular data first")
    variation = (sequence[:, 1:] - sequence[:, :-1]) / intervals[:, 1:, None]
    window = torch.hann_window(variation.shape[1], periodic=False, dtype=sequence.dtype, device=sequence.device)
    power = torch.fft.rfft(variation * window[None, :, None], dim=1).abs().square()
    power = power[:, 1:].mean(dim=-1)
    return power / (power.sum(dim=-1, keepdim=True) + eps)


def spectral_consistency_loss(hidden, pressure, dt=0.01):
    if hidden.shape[:2] != pressure.shape[:2]:
        raise ValueError("Pressure and encoder features must share batch and temporal dimensions")
    return F.mse_loss(transient_spectrum(hidden, dt), transient_spectrum(pressure, dt))


class HybridObjective(nn.Module):
    def __init__(self, lambda_freq=0.1, lambda_sparse=1e-5):
        super().__init__()
        if min(lambda_freq, lambda_sparse) < 0:
            raise ValueError("Loss weights cannot be negative")
        self.lambda_freq = lambda_freq
        self.lambda_sparse = lambda_sparse

    def forward(self, model, logits, hidden, pressure, target, dt=0.01):
        classification = F.cross_entropy(logits, target)
        frequency = spectral_consistency_loss(hidden, pressure, dt)
        sparse = model.sparsity_loss()
        total = classification + self.lambda_freq * frequency + self.lambda_sparse * sparse
        return total, {"classification": classification.detach(), "frequency": frequency.detach(), "sparsity": sparse.detach()}
