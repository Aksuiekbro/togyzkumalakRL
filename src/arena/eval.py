from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from src.game.togyzkumalak import TogyzKumalakState, WHITE, BLACK
from src.mcts.search import MCTS
from src.nn.model import AlphaZeroNet


@dataclass
class ArenaConfig:
    simulations: int = 160
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    games: int = 50


def play_game(model_white: AlphaZeroNet, model_black: AlphaZeroNet, cfg: ArenaConfig) -> int:
    """Play one game: returns 1 if White wins, 0 for draw, -1 if Black wins."""
    mcts_w = MCTS(model_white.infer, cfg.c_puct, cfg.dirichlet_alpha, cfg.dirichlet_frac, cfg.simulations)
    mcts_b = MCTS(model_black.infer, cfg.c_puct, cfg.dirichlet_alpha, cfg.dirichlet_frac, cfg.simulations)

    state = TogyzKumalakState.initial()
    while not state.is_terminal():
        if state.player_to_move == WHITE:
            pi = mcts_w.search(state)
        else:
            pi = mcts_b.search(state)
        # Greedy at evaluation time
        legal = state.legal_moves()
        if not legal:
            break
        action = max(legal, key=lambda a: pi[a])
        state = state.apply_move(action)

    outcome = state.outcome()
    if outcome is None:
        return 0
    return 1 if outcome == WHITE else -1


def arena(model_new: AlphaZeroNet, model_ref: AlphaZeroNet, cfg: ArenaConfig) -> Tuple[int, int, int, float]:
    """Play matches alternating colors; return (wins, draws, losses, win_rate)."""
    wins = draws = losses = 0
    for i in range(cfg.games):
        if i % 2 == 0:
            res = play_game(model_new, model_ref, cfg)
        else:
            res = -play_game(model_ref, model_new, cfg)
        if res > 0:
            wins += 1
        elif res < 0:
            losses += 1
        else:
            draws += 1
    win_rate = (wins + 0.5 * draws) / max(1, cfg.games)
    return wins, draws, losses, win_rate


def elo_update(rating: float, score: float, expected: float, k: float = 20.0) -> float:
    """Single-step Elo update."""
    return rating + k * (score - expected)


