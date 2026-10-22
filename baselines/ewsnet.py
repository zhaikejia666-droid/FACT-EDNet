import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

def Laplace(p):
    w = 2 * pi * 80
    q = (1 - 0.03 ** 2) ** 0.5
    return 0.08 * torch.exp(-0.03 / q * (w * (p - 0.1))) * torch.sin(w * (p - 0.1))

class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, stride=1):
        super(Laplace_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.a_ = nn.Parameter(torch.linspace(1, 100, out_channels).view(-1, 1))
        self.b_ = nn.Parameter(torch.linspace(0, 100, out_channels).view(-1, 1))

    def forward(self, waveforms):
        time_disc = torch.linspace(0, 1, steps=int(self.kernel_size), device=waveforms.device, dtype=waveforms.dtype)
        p1 = (time_disc - self.b_) / (self.a_ + 1e-05)
        laplace_filter = Laplace(p1)
        filters = laplace_filter.view(self.out_channels, 1, self.kernel_size)
        padding = self.kernel_size // 2
        return F.conv1d(waveforms, filters, stride=self.stride, padding=padding)

class Laplace_fastv2:

    def __init__(self, out_channels, kernel_size, eps=0.001):
        super(Laplace_fastv2, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(1, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, 1, steps=int(kernel_size))

    def forward(self):
        p1 = (self.time_disc - self.b_) / (self.a_ + self.eps)
        filter = Laplace(p1).view(self.out_channels, 1, self.kernel_size)
        return filter

class Shrinkagev3ppp2(nn.Module):

    def __init__(self, gap_size, channel):
        super(Shrinkagev3ppp2, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.48]))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(nn.Linear(channel, channel), nn.BatchNorm1d(channel), nn.ReLU(inplace=True), nn.Linear(channel, channel), nn.Sigmoid())

    def forward(self, x):
        x_raw = x
        x_abs = x.abs()
        x = self.gap(x_abs)
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        sub = torch.max(x_abs - x, torch.zeros_like(x_abs - x))
        mask = sub.clone()
        mask[mask > 0] = 1
        x = sub + (1 - self.a.clamp(0, 1)) * x
        x = torch.mul(torch.sign(x_raw), torch.mul(x, mask))
        return x

class Mish1(nn.Module):

    def __init__(self):
        super(Mish1, self).__init__()
        self.mish = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.mish(x)

class Net(nn.Module):

    def __init__(self, in_channels=1, num_classes=5, seq_len=20):
        super(Net, self).__init__()
        self.seq_len = seq_len
        k1_0 = min(7, seq_len)
        self.p1_0 = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=k1_0, stride=1, padding=k1_0 // 2, bias=True), nn.BatchNorm1d(64), Mish1())
        k1_1 = min(5, seq_len // 2)
        self.p1_1 = nn.Sequential(nn.Conv1d(64, 16, kernel_size=k1_1, stride=1, padding=k1_1 // 2, bias=True), nn.BatchNorm1d(16), Mish1())
        k1_2 = min(3, seq_len // 4)
        self.p1_2 = nn.Sequential(nn.Conv1d(16, 10, kernel_size=k1_2, stride=1, padding=k1_2 // 2, bias=True), nn.BatchNorm1d(10), Mish1())
        self.p1_3 = nn.AdaptiveAvgPool1d(1)
        k2_1 = min(3, seq_len // 4)
        self.p2_1 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=k2_1, stride=1, padding=k2_1 // 2, bias=True), nn.BatchNorm1d(32), Mish1())
        k2_2 = min(3, seq_len // 4)
        self.p2_2 = nn.Sequential(nn.Conv1d(32, 16, kernel_size=k2_2, stride=1, padding=k2_2 // 2, bias=True), nn.BatchNorm1d(16), Mish1())
        self.p2_3 = nn.AdaptiveAvgPool1d(1)
        self.p2_4 = nn.Sequential(nn.Conv1d(16, 10, kernel_size=1, stride=1, bias=True), nn.BatchNorm1d(10), Mish1())
        self.p3_0 = Shrinkagev3ppp2(channel=64, gap_size=1)
        k3_1 = min(5, seq_len)
        self.p3_1 = nn.Sequential(nn.Conv1d(64, 10, kernel_size=k3_1, stride=1, padding=k3_1 // 2, bias=True), nn.BatchNorm1d(10), Mish1())
        self.p3_2 = nn.AdaptiveAvgPool1d(1)
        self.p4 = nn.Linear(10, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size[0] == min(7, self.seq_len) and m.in_channels == 1:
                    wavelet_init = Laplace_fastv2(out_channels=64, kernel_size=m.kernel_size[0])
                    with torch.no_grad():
                        m.weight.copy_(wavelet_init.forward())
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        x = x.transpose(1, 2)
        shared = self.p1_0(x)
        x1 = shared
        x1 = self.p1_1(x1)
        x1 = self.p1_2(x1)
        p1 = self.p1_3(x1)
        x2 = shared
        x2 = self.p2_1(x2)
        x2 = self.p2_2(x2)
        x2 = self.p2_3(x2)
        p2 = self.p2_4(x2)
        x3 = shared
        x3 = self.p3_1(x3 + self.p3_0(x3))
        p3 = self.p3_2(x3)
        x = p1 + p2 + p3
        x = x.squeeze(-1)
        x = self.p4(x)
        return x

def EWSNet(in_channel, out_channel, seq_len=20):
    model = Net(in_channels=in_channel, num_classes=out_channel, seq_len=seq_len)
    return model
