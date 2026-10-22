import math

import torch
from torch import nn
from torch.nn import functional as F


def time_intervals(x, dt):
    dt = torch.as_tensor(dt, dtype=x.dtype, device=x.device)
    if dt.ndim == 0:
        dt = dt.expand(x.shape[0], x.shape[1])
    elif dt.ndim == 1 and dt.shape[0] == x.shape[1]:
        dt = dt.unsqueeze(0).expand(x.shape[0], -1)
    elif dt.ndim == 3 and dt.shape[-1] == 1:
        dt = dt.squeeze(-1)
    if dt.shape != x.shape[:2] or not torch.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError("dt must be positive and finite, with shape scalar, [T], or [B,T]")
    return dt


class MWKAN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=64, num_basis=4):
        super().__init__()
        self.input_projection = nn.Linear(in_features, hidden_features)
        self.translation = nn.Parameter(torch.linspace(-2, 2, num_basis).repeat(hidden_features, 1))
        self.raw_scale = nn.Parameter(torch.full((hidden_features, num_basis), math.log(math.expm1(1.0))))
        self.coefficients = nn.Parameter(torch.ones(hidden_features, num_basis) / num_basis)
        self.output_projection = nn.Linear(hidden_features, out_features)
        self.base_projection = nn.Linear(in_features, out_features)

    @property
    def scale(self):
        return F.softplus(self.raw_scale) + 1e-5

    def forward(self, x):
        hidden = self.input_projection(x)
        coordinate = (hidden.unsqueeze(-1) - self.translation) / self.scale
        square = coordinate.square()
        wavelet = (1 - square) * torch.exp(-0.5 * square)
        mixed = (wavelet * self.coefficients).sum(dim=-1)
        return self.output_projection(mixed) + self.base_projection(F.silu(x))


class ContinuousTimeCell(nn.Module):
    def __init__(self, input_size, hidden_size=64, wavelet_hidden=64, num_basis=4, dropout=0.1):
        super().__init__()
        self.dynamics = MWKAN(input_size + hidden_size, 2 * hidden_size, wavelet_hidden, num_basis)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transition(self, x, state, dt):
        rate, equilibrium = self.dynamics(torch.cat((x, state), dim=-1)).chunk(2, dim=-1)
        rate = F.softplus(rate) + 1e-5
        update = -torch.expm1(-rate * dt)
        return state + update * (torch.tanh(equilibrium) - state)

    def forward(self, x, state, dt):
        state = self.transition(x, state, dt)
        return state, self.dropout(self.norm(state))


class DynamicsEncoder(nn.Module):
    def __init__(self, hidden_size=64, depth=2, wavelet_hidden=64, num_basis=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([
            ContinuousTimeCell(1 if i == 0 else hidden_size, hidden_size, wavelet_hidden, num_basis, dropout)
            for i in range(depth)
        ])

    def forward(self, x, dt):
        states = [x.new_zeros(x.shape[0], self.hidden_size) for _ in self.cells]
        outputs = []
        for t in range(x.shape[1]):
            value = x[:, t]
            for i, cell in enumerate(self.cells):
                states[i], value = cell(value, states[i], dt[:, t:t + 1])
            outputs.append(value)
        return torch.stack(outputs, dim=1)


class GroupedSparseChannelAttention(nn.Module):
    def __init__(self, seq_len, heads=4, kv_heads=1, dropout=0.1):
        super().__init__()
        if seq_len % heads or heads % kv_heads:
            raise ValueError("seq_len must be divisible by heads and heads by kv_heads")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = seq_len // heads
        self.query = nn.Linear(seq_len, seq_len, bias=False)
        self.key = nn.Linear(seq_len, self.head_dim * kv_heads, bias=False)
        self.value = nn.Linear(seq_len, self.head_dim * kv_heads, bias=False)
        self.output = nn.Linear(seq_len, seq_len, bias=False)
        self.raw_temperature = nn.Parameter(torch.full((heads, 1, 1), math.log(math.expm1(1.0))))
        self.gate = nn.Sequential(nn.Linear(seq_len, max(1, seq_len // 4)), nn.GELU(), nn.Linear(max(1, seq_len // 4), 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, channels, length = x.shape
        query = self.query(x).reshape(batch, channels, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(x).reshape(batch, channels, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).reshape(batch, channels, self.kv_heads, self.head_dim).transpose(1, 2)
        key = key.repeat_interleave(self.heads // self.kv_heads, dim=1)
        value = value.repeat_interleave(self.heads // self.kv_heads, dim=1)
        importance = self.gate(x.abs()).squeeze(-1)
        counts = (channels * importance.detach().mean(dim=-1)).floor().long().clamp(min=max(1, channels // 4), max=channels)
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores * (F.softplus(self.raw_temperature) + 1e-5)
        order = scores.argsort(dim=-1, descending=True)
        rank = order.argsort(dim=-1)
        mask = rank < counts[:, None, None, None]
        weights = self.dropout(scores.masked_fill(~mask, -torch.inf).softmax(dim=-1))
        value = value * importance[:, None, :, None]
        output = (weights @ value).transpose(1, 2).reshape(batch, channels, length)
        return self.output(output)


class GhostFFN(nn.Module):
    def __init__(self, seq_len, dropout=0.1):
        super().__init__()
        self.primary = nn.Sequential(nn.Linear(seq_len, seq_len, bias=False), nn.LayerNorm(seq_len), nn.GELU())
        self.cheap = nn.Sequential(nn.Conv1d(seq_len, seq_len, 3, padding=1, groups=seq_len, bias=False), nn.BatchNorm1d(seq_len), nn.GELU())
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * seq_len, seq_len), nn.Dropout(dropout))

    def forward(self, x):
        primary = self.primary(x)
        cheap = self.cheap(primary.transpose(1, 2)).transpose(1, 2)
        return self.output(torch.cat((primary, cheap), dim=-1))


class DecoderBlock(nn.Module):
    def __init__(self, seq_len, heads=4, kv_heads=1, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(seq_len)
        self.norm2 = nn.LayerNorm(seq_len)
        self.attention = GroupedSparseChannelAttention(seq_len, heads, kv_heads, dropout)
        self.ffn = GhostFFN(seq_len, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.ffn(self.norm2(x))


class FACTEDNet(nn.Module):
    def __init__(self, seq_len=64, num_classes=5, hidden_size=64, encoder_depth=2, decoder_depth=2, wavelet_hidden=64, num_basis=4, heads=4, kv_heads=1, dropout=0.1):
        super().__init__()
        if min(seq_len, num_classes, hidden_size, encoder_depth, decoder_depth, wavelet_hidden, num_basis, heads, kv_heads) < 1 or seq_len < 4:
            raise ValueError("Invalid model dimensions")
        self.seq_len = seq_len
        self.encoder = DynamicsEncoder(hidden_size, encoder_depth, wavelet_hidden, num_basis, dropout)
        self.decoder = nn.Sequential(*[DecoderBlock(seq_len, heads, kv_heads, dropout) for _ in range(decoder_depth)], nn.LayerNorm(seq_len))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.SiLU(), nn.Dropout(0.2), nn.Linear(hidden_size // 2, num_classes))

    def forward(self, x, dt=0.01, return_features=False):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, 1):
            raise ValueError(f"Expected [B,{self.seq_len},1] input")
        hidden = self.encoder(x, time_intervals(x, dt))
        decoded = self.decoder(hidden.transpose(1, 2))
        logits = self.classifier(decoded.mean(dim=-1))
        return (logits, hidden) if return_features else logits

    def sparsity_loss(self):
        return torch.cat([module.coefficients.flatten() for module in self.modules() if isinstance(module, MWKAN)]).abs().mean()
