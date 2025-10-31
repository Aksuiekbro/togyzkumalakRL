
from typing import Optional
import numpy as np


def select_move(policy: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> int:
    """Select argmax over masked policy."""
    p = policy.copy()
    if valid_mask is not None:
        p *= valid_mask
        s = p.sum()
        if s > 0:
            p /= s
    return int(np.argmax(p))
