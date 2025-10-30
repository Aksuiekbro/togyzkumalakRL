"""
State encoding, symmetry transforms, hashing, and serialization for Togyz Kumalak.

The encoder produces simple channel-first lists to avoid a hard dependency on
NumPy/PyTorch at this layer. Higher layers can convert to tensors as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .togyzkumalak import TogyzKumalakState, WHITE, BLACK, opponent_of


def canonicalize(state: TogyzKumalakState) -> TogyzKumalakState:
    """Return a state oriented from the current player's perspective.

    We keep the internal representation absolute, but for canonicalization we
    mirror the board if the mover is BLACK so that the mover always perceives
    themselves as "bottom" (WHITE index).
    """
    if state.player_to_move == WHITE:
        return state
    return mirror_state(state)


def mirror_state(state: TogyzKumalakState) -> TogyzKumalakState:
    """Swap sides and mirror pits so that roles are exchanged.

    - White pits become Black pits and vice versa
    - Pit order is preserved relative to each player's right-to-left indexing
    - Tuzdyk indices swap owners unchanged (same index number)
    - Player to move flips
    """
    return TogyzKumalakState(
        pits=[state.pits[BLACK][:], state.pits[WHITE][:]],
        kazans=[state.kazans[BLACK], state.kazans[WHITE]],
        tuzdyk_indices=[state.tuzdyk_indices[BLACK], state.tuzdyk_indices[WHITE]],
        player_to_move=opponent_of(state.player_to_move),
        move_number=state.move_number,
    )


def encode_features(state: TogyzKumalakState) -> List[List[float]]:
    """Encode a state into channel-first features of shape [C, 18].

    Channels:
      0: mover pits (9)
      1: opponent pits (9)
      2: mover tuzdyk one-hot across opponent row (9)
      3: opponent tuzdyk one-hot across mover row (9)
      4: mover kazan / 162 (scalar broadcast)
      5: opponent kazan / 162 (scalar broadcast)
      6: normalized move number (optional context), broadcast
    """
    s = canonicalize(state)
    mover = s.player_to_move  # after canonicalize, mover == WHITE
    assert mover == WHITE

    features: List[List[float]] = [[0.0] * 18 for _ in range(7)]

    # Pits
    for i in range(9):
        features[0][i + 9] = float(s.pits[WHITE][i])  # mover at bottom indices 9..17 for clarity
        features[1][i] = float(s.pits[BLACK][i])

    # Tuzdyk one-hots
    if s.tuzdyk_indices[WHITE] is not None:
        features[2][s.tuzdyk_indices[WHITE]] = 1.0  # on opponent row
    if s.tuzdyk_indices[BLACK] is not None:
        features[3][9 + s.tuzdyk_indices[BLACK]] = 1.0  # on mover row

    total_stones = 162.0
    mover_k = s.kazans[WHITE] / total_stones
    opp_k = s.kazans[BLACK] / total_stones
    for j in range(18):
        features[4][j] = mover_k
        features[5][j] = opp_k
        features[6][j] = min(1.0, s.move_number / 400.0)

    return features


def state_key(state: TogyzKumalakState) -> Tuple:
    """Hashable key for transposition tables and repetition detection.

    Includes player to move so that (s, player) is unique.
    """
    return (
        tuple(state.pits[WHITE]),
        tuple(state.pits[BLACK]),
        tuple(state.kazans),
        state.tuzdyk_indices[WHITE],
        state.tuzdyk_indices[BLACK],
        state.player_to_move,
    )


def serialize_fen(state: TogyzKumalakState) -> str:
    """Serialize to a compact FEN-like string for debugging and storage.

    Format: W|w_pits/b_pits|kzW,kzB|tuzW,tuzB|m
      pits are dash-separated counts, tuz indices are - for None or 0..8
    """
    tuz = lambda x: "-" if x is None else str(x)
    parts = [
        "W" if state.player_to_move == WHITE else "B",
        ",".join(
            [
                "-".join(str(x) for x in state.pits[WHITE]),
                "-".join(str(x) for x in state.pits[BLACK]),
            ]
        ),
        f"{state.kazans[WHITE]},{state.kazans[BLACK]}",
        f"{tuz(state.tuzdyk_indices[WHITE])},{tuz(state.tuzdyk_indices[BLACK])}",
        str(state.move_number),
    ]
    return "|".join(parts)


def deserialize_fen(s: str) -> TogyzKumalakState:
    side, pits_s, kaz_s, tuz_s, mv_s = s.split("|")
    white_pits_s, black_pits_s = pits_s.split(",")
    white_pits = [int(x) for x in white_pits_s.split("-")]
    black_pits = [int(x) for x in black_pits_s.split("-")]
    kz_w, kz_b = kaz_s.split(",")
    tuz_w, tuz_b = tuz_s.split(",")
    tuz_w_v = None if tuz_w == "-" else int(tuz_w)
    tuz_b_v = None if tuz_b == "-" else int(tuz_b)
    return TogyzKumalakState(
        pits=[white_pits, black_pits],
        kazans=[int(kz_w), int(kz_b)],
        tuzdyk_indices=[tuz_w_v, tuz_b_v],
        player_to_move=WHITE if side == "W" else BLACK,
        move_number=int(mv_s),
    )


