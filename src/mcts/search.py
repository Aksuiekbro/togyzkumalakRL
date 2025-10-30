"""
PUCT Monte Carlo Tree Search for AlphaZero-style training.

This implementation is framework-agnostic. It relies on a provided inference
function to obtain (policy, value) for a given state.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from src.game.togyzkumalak import TogyzKumalakState, WHITE, BLACK, opponent_of
from src.game.encoding import canonicalize, state_key


PolicyValueFn = Callable[[TogyzKumalakState], Tuple[List[float], float]]


@dataclass
class Node:
    prior: float
    state: Optional[TogyzKumalakState] = None  # only set at root/expansion time
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)  # action -> child

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        inference_fn: PolicyValueFn,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        num_simulations: int = 160,
    ) -> None:
        self.inference_fn = inference_fn
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.num_simulations = num_simulations
        self._table: Dict[Tuple, Node] = {}

    # ----------------------------- Public API ------------------------------ #
    def search(self, root_state: TogyzKumalakState) -> List[float]:
        """Run MCTS simulations from root and return improved policy π over 9 actions.

        π is proportional to visit counts of root children, normalized.
        """
        root = self._get_or_create_node(root_state)
        self._ensure_expanded(root, root_state)
        self._add_root_dirichlet(root)

        for _ in range(self.num_simulations):
            self._simulate(root_state)

        # Build policy from visit counts
        counts = [0.0] * 9
        for action, child in root.children.items():
            counts[action] = float(child.visit_count)
        total = sum(counts)
        if total <= 0:
            # Fallback to uniform over legal moves
            legal = root_state.legal_moves()
            for a in legal:
                counts[a] = 1.0 / len(legal)
            return counts
        return [c / total for c in counts]

    # ---------------------------- Core internals --------------------------- #
    def _simulate(self, root_state: TogyzKumalakState) -> None:
        path: List[Tuple[Node, int]] = []  # (node, action)
        node = self._get_or_create_node(root_state)
        state = root_state

        # Selection
        while True:
            self._ensure_expanded(node, state)

            if not node.children:
                break  # terminal

            action = self._select_child(node)
            path.append((node, action))
            state = state.apply_move(action)
            node = self._get_or_create_node(state)

            if state.is_terminal():
                break

        # Evaluate leaf
        value = self._evaluate(state)

        # Backpropagate (value is from perspective of state.player_to_move at leaf)
        # We need to flip signs along the path alternately.
        player = state.player_to_move
        for parent, action in reversed(path):
            parent_child = parent.children[action]
            parent_child.visit_count += 1
            # If the child state had mover == player, then backprop value as-is to its parent
            parent_value = value
            parent_child.value_sum += parent_value
            value = -value  # flip for the next step up the tree

    def _evaluate(self, state: TogyzKumalakState) -> float:
        if state.is_terminal():
            # terminal value from perspective of player_to_move
            outcome = state.outcome()
            if outcome is None:
                return 0.0
            return 1.0 if outcome == state.player_to_move else -1.0

        policy, value = self.inference_fn(canonicalize(state))

        # Expand current node with priors if not yet expanded
        node = self._get_or_create_node(state)
        if not node.children:
            legal = set(state.legal_moves())
            for a in range(9):
                if a in legal:
                    node.children.setdefault(a, Node(prior=float(policy[a])))
        return float(value)

    def _select_child(self, node: Node) -> int:
        total_visits = 1 + sum(child.visit_count for child in node.children.values())
        best_score = -1e9
        best_action = None
        for action, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            score = child.value + u
            if score > best_score:
                best_score = score
                best_action = action
        assert best_action is not None
        return best_action

    def _add_root_dirichlet(self, root: Node) -> None:
        if not root.children:
            return
        actions = list(root.children.keys())
        if not actions:
            return
        # Sample Dirichlet noise over actions
        alpha = self.dirichlet_alpha
        noise = _dirichlet(len(actions), alpha)
        for a, eps in zip(actions, noise):
            child = root.children[a]
            child.prior = (1 - self.dirichlet_frac) * child.prior + self.dirichlet_frac * eps

    # --------------------------- Node management --------------------------- #
    def _get_or_create_node(self, state: TogyzKumalakState) -> Node:
        key = state_key(canonicalize(state))
        n = self._table.get(key)
        if n is None:
            n = Node(prior=1.0, state=state)
            self._table[key] = n
        return n

    def _ensure_expanded(self, node: Node, state: TogyzKumalakState) -> None:
        if node.children or state.is_terminal():
            return
        policy, _ = self.inference_fn(canonicalize(state))
        for a in state.legal_moves():
            node.children.setdefault(a, Node(prior=float(policy[a])))


def _dirichlet(k: int, alpha: float) -> List[float]:
    # Simple Dirichlet sampler via Gamma; relies on Python's random.gammavariate
    xs = [random.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(xs)
    if s == 0:
        return [1.0 / k for _ in range(k)]
    return [x / s for x in xs]


