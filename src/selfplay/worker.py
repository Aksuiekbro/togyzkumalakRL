from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import torch

from src.game.togyzkumalak import TogyzKumalakState, WHITE, BLACK
from src.game.encoding import canonicalize, state_key
from src.mcts.search import MCTS
from src.nn.model import AlphaZeroNet


def softmax_temperature(visits: List[float], tau: float) -> List[float]:
    if tau <= 1e-6:
        # Almost argmax: 1.0 on best, tie-broken randomly
        m = max(visits)
        best = [i for i, v in enumerate(visits) if v == m]
        pi = [0.0] * len(visits)
        pi[random.choice(best)] = 1.0
        return pi
    x = [v ** (1.0 / tau) for v in visits]
    s = sum(x)
    return [xi / s if s > 0 else 0.0 for xi in x]


@dataclass
class SelfPlayConfig:
    simulations: int = 160
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temp_moves: int = 10  # τ=1 until this move; then τ=0.1


def play_episode(model: AlphaZeroNet, cfg: SelfPlayConfig) -> List[Tuple[TogyzKumalakState, List[float], int]]:
    """Play one self-play game and return training samples (s, π, z).

    z is from the perspective of the stored state (current player at that time).
    """
    def inference_fn(s: TogyzKumalakState):
        return model.infer(s)

    mcts = MCTS(
        inference_fn,
        c_puct=cfg.c_puct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_frac=cfg.dirichlet_frac,
        num_simulations=cfg.simulations,
    )

    state = TogyzKumalakState.initial()
    history: List[Tuple[TogyzKumalakState, List[float]]] = []

    while True:
        pi = mcts.search(state)
        move_index = state.move_number - 1
        tau = 1.0 if move_index < cfg.temp_moves else 0.1
        # Convert visit counts distribution into temperature policy
        visits = [v * 1.0 for v in pi]
        pi_tau = softmax_temperature(visits, tau)

        history.append((state, pi_tau))

        # Sample move according to π^τ
        legal = state.legal_moves()
        if not legal:
            break
        # Normalize to legal-only sampling
        mass = sum(pi_tau[a] for a in legal)
        if mass <= 0:
            action = random.choice(legal)
        else:
            r = random.random() * mass
            cum = 0.0
            action = legal[0]
            for a in legal:
                cum += pi_tau[a]
                if r <= cum:
                    action = a
                    break

        state = state.apply_move(action)
        if state.is_terminal():
            break

    # Game ended; assign outcomes
    winner = state.outcome()  # WHITE/BLACK or None for draw
    samples: List[Tuple[TogyzKumalakState, List[float], int]] = []
    for s, pi_s in history:
        if winner is None:
            z = 0
        elif winner == s.player_to_move:
            z = 1
        else:
            z = -1
        samples.append((s, pi_s, z))

    return samples


