from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.togyzkumalak import TogyzKumalakState
from src.game.encoding import encode_features, canonicalize


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels: int = 7, channels: int = 128, num_blocks: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head: produce 9 logits
        self.policy_head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(channels * 18, 9)

        # Value head: scalar in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(channels * 18, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, 18]
        h = self.stem(x)
        h = self.blocks(h)

        p = self.policy_head(h)
        p = p.reshape(p.shape[0], -1)
        p = self.policy_fc(p)  # [B, 9]

        v = self.value_head(h)
        v = v.reshape(v.shape[0], -1)
        v = self.value_fc(v).squeeze(-1)  # [B]
        return p, v

    @torch.no_grad()
    def infer(self, state: TogyzKumalakState) -> Tuple[List[float], float]:
        self.eval()
        s = canonicalize(state)
        feats = encode_features(s)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # [1, C, 18]
        p_logits, v = self.forward(x)
        p = F.softmax(p_logits, dim=-1)[0].tolist()
        return p, float(v.item())


