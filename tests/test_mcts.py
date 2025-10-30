from typing import List, Tuple

from src.game.togyzkumalak import TogyzKumalakState
from src.mcts.search import MCTS


def dummy_inference(state: TogyzKumalakState) -> Tuple[List[float], float]:
    # Uniform prior on legal moves, zero value
    priors = [0.0] * 9
    legal = state.legal_moves()
    if legal:
        p = 1.0 / len(legal)
        for a in legal:
            priors[a] = p
    return priors, 0.0


def test_mcts_runs_and_returns_distribution():
    s = TogyzKumalakState.initial()
    mcts = MCTS(dummy_inference, num_simulations=8)
    pi = mcts.search(s)
    assert len(pi) == 9
    assert abs(sum(pi) - 1.0) < 1e-6 or sum(pi) == 0.0


