
from typing import Tuple
import numpy as np
from game.togyzkumalak import TogyzKumalakState, NUM_PITS_PER_SIDE, TOTAL_PITS, TOTAL_SEEDS


def to_canonical(state: TogyzKumalakState, include_phase: bool = True) -> np.ndarray:
    """Encode state into (C, 18) canonical tensor.
    Channels: [my_pits, opp_pits, my_tuz, opp_tuz, my_kazan, opp_kazan, move_phase]
    """
    pits = np.array(state.pits, dtype=np.float32)
    if state.player == 1:
        # Flip perspective: bring current player to front
        pits = np.concatenate([pits[NUM_PITS_PER_SIDE:], pits[:NUM_PITS_PER_SIDE]], axis=0)
        t0, t1 = state.tuzdyk[1], state.tuzdyk[0]
    else:
        t0, t1 = state.tuzdyk

    my = np.zeros(18, dtype=np.float32)
    opp = np.zeros(18, dtype=np.float32)
    my[:NUM_PITS_PER_SIDE] = pits[:NUM_PITS_PER_SIDE]
    opp[NUM_PITS_PER_SIDE:] = pits[NUM_PITS_PER_SIDE:]

    my_tuz = np.zeros(18, dtype=np.float32)
    opp_tuz = np.zeros(18, dtype=np.float32)
    if t0 is not None:
        # my tuz lives on opponent row (right half in canonical)
        opp_idx = NUM_PITS_PER_SIDE + t0
        opp_tuz[opp_idx] = 1.0
    if t1 is not None:
        # opponent tuz on my row (left half)
        my_idx = t1
        my_tuz[my_idx] = 1.0

    my_k = np.full(18, float(state.kazan[state.player] / TOTAL_SEEDS), dtype=np.float32)
    opp_k = np.full(18, float(state.kazan[1 - state.player] / TOTAL_SEEDS), dtype=np.float32)

    if include_phase:
        # crude phase heuristic: normalize by 200 moves
        phase = np.full(18, min(1.0, state.move_count / 200.0), dtype=np.float32)
        feats = np.stack([my, opp, my_tuz, opp_tuz, my_k, opp_k, phase], axis=0)
    else:
        feats = np.stack([my, opp, my_tuz, opp_tuz, my_k, opp_k], axis=0)
    return feats
