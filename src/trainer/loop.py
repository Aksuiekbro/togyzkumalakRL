from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.game.encoding import encode_features
from src.game.togyzkumalak import TogyzKumalakState
from src.nn.model import AlphaZeroNet


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs_per_iter: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000) -> None:
        self.capacity = capacity
        self._storage: List[Tuple[TogyzkumalakState, List[float], int]] = []

    def add_many(self, samples: Sequence[Tuple[TogyzkumalakState, List[float], int]]) -> None:
        for s in samples:
            self._storage.append(s)
        if len(self._storage) > self.capacity:
            overflow = len(self._storage) - self.capacity
            self._storage = self._storage[overflow:]

    def sample_batch(self, batch_size: int) -> List[Tuple[TogyzkumalakState, List[float], int]]:
        return random.sample(self._storage, min(batch_size, len(self._storage)))

    def __len__(self) -> int:
        return len(self._storage)


def _collate(samples: Sequence[Tuple[TogyzkumalakState, List[float], int]]):
    feats = [encode_features(s) for (s, _, _) in samples]
    pi = [p for (_, p, _) in samples]
    z = [v for (_, _, v) in samples]
    x = torch.tensor(feats, dtype=torch.float32)
    pi = torch.tensor(pi, dtype=torch.float32)
    z = torch.tensor(z, dtype=torch.float32)
    return x, pi, z


def train_one_iteration(
    model: AlphaZeroNet,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    cfg: TrainConfig,
    steps: int = 1000,
) -> float:
    model.train()
    device = cfg.device
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.startswith("cuda"))
    avg_loss = 0.0

    for step in range(steps):
        batch = buffer.sample_batch(cfg.batch_size)
        if not batch:
            break
        x, pi_t, z_t = _collate(batch)
        x, pi_t, z_t = x.to(device), pi_t.to(device), z_t.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp and device.startswith("cuda")):
            p_logits, v = model(x)
            policy_loss = -(pi_t * F.log_softmax(p_logits, dim=-1)).sum(dim=-1).mean()
            value_loss = F.mse_loss(v, z_t)
            loss = policy_loss + value_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        avg_loss = 0.9 * avg_loss + 0.1 * float(loss.item()) if step > 0 else float(loss.item())

    return avg_loss


def save_checkpoint(path: str, model: AlphaZeroNet, optimizer: optim.Optimizer, step: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(path: str, model: AlphaZeroNet, optimizer: optim.Optimizer) -> int:
    if not os.path.exists(path):
        return 0
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["opt"])
    return int(data.get("step", 0))


