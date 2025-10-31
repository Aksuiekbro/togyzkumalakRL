
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaZero1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        channels: int = 128,
        num_blocks: int = 8,
        board_len: int = 18,
        num_actions: int = 9,
    ) -> None:
        super().__init__()
        self.board_len = board_len
        padding = 1
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels, kernel_size=3) for _ in range(num_blocks)])
        # Policy head
        self.policy_conv = nn.Conv1d(channels, 2, kernel_size=3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm1d(2)
        self.policy_fc = nn.Linear(2 * board_len, num_actions)
        # Value head
        self.value_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm1d(channels)
        self.value_fc1 = nn.Linear(channels, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, 18)
        h = self.stem(x)
        h = self.blocks(h)
        # policy
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)
        p_logits = self.policy_fc(p)
        # value
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.mean(dim=2)  # (B, channels)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return p_logits, v
