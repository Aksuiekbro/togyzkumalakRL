
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

NUM_PITS_PER_SIDE = 9
TOTAL_PITS = NUM_PITS_PER_SIDE * 2
TOTAL_SEEDS = 162

@dataclass(frozen=True)
class TogyzKumalakState:
    pits: Tuple[int, ...]  # length 18
    kazan: Tuple[int, int]
    player: int  # 0 or 1; player to move
    tuzdyk: Tuple[Optional[int], Optional[int]]  # per-side tuzdyk on opponent row (0..8)
    move_count: int = 0

    def legal_moves(self) -> List[int]:
        start = 0 if self.player == 0 else NUM_PITS_PER_SIDE
        end = start + NUM_PITS_PER_SIDE
        return [i - start for i in range(start, end) if self.pits[i] > 0]

    def is_terminal(self) -> bool:
        if self.kazan[0] >= 82 or self.kazan[1] >= 82:
            return True
        start = 0 if self.player == 0 else NUM_PITS_PER_SIDE
        end = start + NUM_PITS_PER_SIDE
        if sum(self.pits[start:end]) == 0:
            return True
        return False

    def winner(self) -> Optional[int]:
        if not self.is_terminal():
            return None
        if self.kazan[0] >= 82:
            return 0
        if self.kazan[1] >= 82:
            return 1
        k0 = self.kazan[0] + sum(self.pits[0:NUM_PITS_PER_SIDE])
        k1 = self.kazan[1] + sum(self.pits[NUM_PITS_PER_SIDE:TOTAL_PITS])
        if k0 > k1:
            return 0
        if k1 > k0:
            return 1
        return None  # draw

    def to_fen(self) -> str:
        pits_s = ",".join(map(str, self.pits))
        kaz_s = f"{self.kazan[0]}|{self.kazan[1]}"
        t0 = -1 if self.tuzdyk[0] is None else self.tuzdyk[0]
        t1 = -1 if self.tuzdyk[1] is None else self.tuzdyk[1]
        return f"{pits_s} {kaz_s} {self.player} {t0}:{t1} {self.move_count}"

    def apply_move(self, action: int) -> "TogyzKumalakState":
        pits = list(self.pits)
        k0, k1 = self.kazan
        p = self.player
        t0, t1 = self.tuzdyk

        start = 0 if p == 0 else 9
        src = start + action
        stones = pits[src]
        if stones <= 0:
            raise ValueError("Illegal move from empty pit")
        pits[src] = 0

        # Absolute tuzdyk indices
        tuz_abs0 = (9 + t0) if t0 is not None else None  # P0's tuz on P1 row
        tuz_abs1 = (0 + t1) if t1 is not None else None  # P1's tuz on P0 row

        last_idx: Optional[int] = None
        pos = src
        if stones > 1:
            # Drop first stone into starting pit
            pits[pos] += 1
            remaining = stones - 1
            while remaining > 0:
                pos = (pos + 1) % 18
                if tuz_abs0 is not None and pos == tuz_abs0:
                    k0 += 1
                elif tuz_abs1 is not None and pos == tuz_abs1:
                    k1 += 1
                else:
                    pits[pos] += 1
                    last_idx = pos
                remaining -= 1
        else:
            # single stone → place in next pit
            pos = (src + 1) % 18
            if tuz_abs0 is not None and pos == tuz_abs0:
                k0 += 1
                last_idx = None
            elif tuz_abs1 is not None and pos == tuz_abs1:
                k1 += 1
                last_idx = None
            else:
                pits[pos] += 1
                last_idx = pos

        new_t0, new_t1 = t0, t1

        # Captures / tuzdyk creation only if last stone ended in opponent pit
        if last_idx is not None:
            if p == 0 and 9 <= last_idx <= 17:
                local = last_idx - 9
                cnt = pits[last_idx]
                if cnt == 3 and new_t0 is None and local != 8 and (t1 is None or local != t1):
                    k0 += 3
                    pits[last_idx] = 0
                    new_t0 = local
                elif cnt % 2 == 0 and cnt > 0:
                    k0 += cnt
                    pits[last_idx] = 0
            elif p == 1 and 0 <= last_idx <= 8:
                local = last_idx - 0
                cnt = pits[last_idx]
                if cnt == 3 and new_t1 is None and local != 8 and (t0 is None or local != t0):
                    k1 += 3
                    pits[last_idx] = 0
                    new_t1 = local
                elif cnt % 2 == 0 and cnt > 0:
                    k1 += cnt
                    pits[last_idx] = 0

        return TogyzKumalakState(
            pits=tuple(pits),
            kazan=(k0, k1),
            player=1 - p,
            tuzdyk=(new_t0, new_t1),
            move_count=self.move_count + 1,
        )


def initial_state() -> TogyzKumalakState:
    pits = tuple([9] * TOTAL_PITS)
    return TogyzKumalakState(pits=pits, kazan=(0, 0), player=0, tuzdyk=(None, None), move_count=0)
