"""
Core rules engine for Togyz Kumalak (Togyzkumalak).

This module implements:
- Board state and turn handling
- Legal move generation and move application (sowing rules)
- Captures by even number
- Tuzdyk creation with official restrictions
- Terminal detection including atsyrau and early win by 82+

All indices are 0-based internally. Pit index 8 corresponds to the 9th pit (#9).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple


WHITE: int = 0
BLACK: int = 1


def opponent_of(player: int) -> int:
    return BLACK if player == WHITE else WHITE


@dataclass(frozen=True)
class TogyzKumalakState:
    """Immutable game state for Togyz Kumalak.

    Attributes:
        pits: 2×9 pit counts. pits[WHITE][i] is White's i-th pit (0..8).
        kazans: [white_kazan, black_kazan].
        tuzdyk_indices: Tuzdyk index for each player on opponent's side, or None.
            Example: if tuzdyk_indices[WHITE] == 2, White owns a tuzdyk in Black's pit #3.
        player_to_move: WHITE or BLACK.
        move_number: Increments after every move (1-based).
    """

    pits: List[List[int]]
    kazans: List[int]
    tuzdyk_indices: List[Optional[int]]
    player_to_move: int = WHITE
    move_number: int = 1

    # ------------------------- Construction helpers ------------------------- #
    @staticmethod
    def initial() -> "TogyzKumalakState":
        return TogyzKumalakState(
            pits=[[9 for _ in range(9)], [9 for _ in range(9)]],
            kazans=[0, 0],
            tuzdyk_indices=[None, None],
            player_to_move=WHITE,
            move_number=1,
        )

    # ----------------------------- Query methods ---------------------------- #
    def legal_moves(self) -> List[int]:
        """Return list of legal pit indices (0..8) for the current player.

        A move is legal if the selected pit on the mover's side contains at least one stone.
        """
        side = self.player_to_move
        return [i for i, stones in enumerate(self.pits[side]) if stones > 0]

    def has_legal_move(self) -> bool:
        return any(stones > 0 for stones in self.pits[self.player_to_move])

    def is_terminal(self) -> bool:
        # Early win by score
        if self.kazans[WHITE] >= 82 or self.kazans[BLACK] >= 82:
            return True
        # Atsyrau: current player has no stones to move
        if not self.has_legal_move():
            return True
        return False

    def outcome(self) -> Optional[int]:
        """Return WHITE, BLACK, or None for ongoing; handles draws via 81–81.

        If terminal by atsyrau, opponent collects remaining stones before scoring.
        """
        if not self.is_terminal():
            return None
        white_score, black_score = self.final_scores()
        if white_score > black_score:
            return WHITE
        if black_score > white_score:
            return BLACK
        return None  # draw

    def final_scores(self) -> Tuple[int, int]:
        """Compute final scores if the position is terminal.

        For atsyrau (no legal move for side to move), opponent captures all remaining stones
        on the board to their kazan.
        """
        white_kazan, black_kazan = self.kazans
        if self.kazans[WHITE] >= 82 or self.kazans[BLACK] >= 82:
            return white_kazan, black_kazan

        # Atsyrau check
        if not self.has_legal_move():
            mover = self.player_to_move
            opp = opponent_of(mover)
            # Opponent collects all remaining stones
            remaining_opp = sum(self.pits[opp])
            remaining_mover = sum(self.pits[mover])
            # If mover has no legal move, their side is empty (by definition of atsyrau),
            # but we compute both sides robustly.
            if opp == WHITE:
                return white_kazan + remaining_opp, black_kazan + remaining_mover
            else:
                return white_kazan + remaining_mover, black_kazan + remaining_opp

        return white_kazan, black_kazan

    # --------------------------- Move application --------------------------- #
    def apply_move(self, pit_index: int) -> "TogyzKumalakState":
        """Apply a legal move and return the next state.

        Sowing rules:
        - If chosen pit has >1 stones: drop one back into the starting pit, then continue.
        - If chosen pit has exactly 1 stone: drop it into the next pit.
        - Stones dropped into any tuzdyk are immediately moved to that tuzdyk's owner's kazan.
        - After sowing, if the last stone landed in opponent's pit and count becomes exactly 3
          and all restrictions allow, create a tuzdyk and capture those 3 to mover's kazan.
        - Else if last stone landed in opponent's pit and count becomes even, capture them all.
        """
        mover = self.player_to_move
        opp = opponent_of(mover)
        pits = [self.pits[WHITE][:], self.pits[BLACK][:]]
        kazans = self.kazans[:]
        tuzdyk_indices = self.tuzdyk_indices[:]

        if pits[mover][pit_index] <= 0:
            raise ValueError("Illegal move: selected pit is empty")

        stones_in_hand = pits[mover][pit_index]
        pits[mover][pit_index] = 0

        # Helper to determine tuzdyk owner at a specific board square
        def tuzdyk_owner_at(side_idx: int, local_index: int) -> Optional[int]:
            if tuzdyk_indices[WHITE] is not None and side_idx == BLACK and tuzdyk_indices[WHITE] == local_index:
                return WHITE
            if tuzdyk_indices[BLACK] is not None and side_idx == WHITE and tuzdyk_indices[BLACK] == local_index:
                return BLACK
            return None

        # Step to the next (side, index) around the ring of 18 pits
        def next_pos(side_idx: int, local_index: int) -> Tuple[int, int]:
            if local_index == 8:
                return (opponent_of(side_idx), 0)
            return (side_idx, local_index + 1)

        # Determine starting drop position and pre-drop for >1 stones
        if stones_in_hand > 1:
            # Drop one back into the starting pit (never a tuzdyk)
            pits[mover][pit_index] += 1
            stones_to_sow = stones_in_hand - 1
            cur_side, cur_idx = next_pos(mover, pit_index)
        else:
            stones_to_sow = stones_in_hand
            cur_side, cur_idx = next_pos(mover, pit_index)

        last_side, last_idx = cur_side, cur_idx
        while stones_to_sow > 0:
            owner = tuzdyk_owner_at(cur_side, cur_idx)
            if owner is not None:
                kazans[owner] += 1
            else:
                pits[cur_side][cur_idx] += 1
                last_side, last_idx = cur_side, cur_idx

            stones_to_sow -= 1
            if stones_to_sow > 0:
                cur_side, cur_idx = next_pos(cur_side, cur_idx)

        # After sowing, evaluate captures or tuzdyk creation
        if last_side == opp:
            # If last landed in a tuzdyk, nothing else happens (already credited above)
            if tuzdyk_owner_at(last_side, last_idx) is None:
                stones_there = pits[opp][last_idx]
                # Tuzdyk creation check (exactly 3)
                if stones_there == 3 and self._can_create_tuzdyk(mover, last_idx):
                    tuzdyk_indices[mover] = last_idx
                    kazans[mover] += 3
                    pits[opp][last_idx] = 0
                # Even capture (if not tuzdyk)
                elif stones_there % 2 == 0 and stones_there > 0:
                    kazans[mover] += stones_there
                    pits[opp][last_idx] = 0

        next_player = opp
        return TogyzKumalakState(
            pits=pits,
            kazans=kazans,
            tuzdyk_indices=tuzdyk_indices,
            player_to_move=next_player,
            move_number=self.move_number + 1,
        )

    # ------------------------------ Internal API ---------------------------- #
    def _can_create_tuzdyk(self, mover: int, opp_pit_index: int) -> bool:
        """Return True if mover may create a tuzdyk in opponent's pit at opp_pit_index.

        Restrictions:
          - mover has no existing tuzdyk
          - cannot create in opponent's #9 (index 8)
          - cannot create symmetrical to opponent's current tuzdyk (same index)
        """
        opp = opponent_of(mover)
        if self.tuzdyk_indices[mover] is not None:
            return False
        if opp_pit_index == 8:
            return False
        if self.tuzdyk_indices[opp] is not None and self.tuzdyk_indices[opp] == opp_pit_index:
            return False
        return True


# Convenience alias
GameState = TogyzKumalakState


