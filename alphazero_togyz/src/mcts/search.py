
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import math, random
import numpy as np
import torch

@dataclass
class MCTSNode:
    prior: float
    visit_count: int = 0
    total_value: float = 0.0
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    def __init__(self, model, encoder_fn, c_puct: float = 1.5, device: Optional[torch.device] = None):
        self.model = model
        self.encoder_fn = encoder_fn
        self.c_puct = c_puct
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _eval(self, state) -> Tuple[np.ndarray, float]:
        x = self.encoder_fn(state)  # (C, 18)
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        logits, v = self.model(xt)
        pi = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return pi, float(v.item())

    def search(self, root_state, num_simulations: int = 100, dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        root = MCTSNode(prior=1.0)
        priors, value = self._eval(root_state)
        legal = root_state.legal_moves()
        mask = np.zeros_like(priors)
        for a in legal:
            mask[a] = 1.0
        priors = priors * mask
        s = priors.sum()
        if s > 0:
            priors = priors / s
        # Dirichlet noise at root
        noise = np.random.dirichlet([dirichlet_alpha] * len(priors))
        priors = (1 - dirichlet_eps) * priors + dirichlet_eps * noise
        for a in range(len(priors)):
            if priors[a] > 0:
                root.children[a] = MCTSNode(prior=float(priors[a]))

        for _ in range(num_simulations):
            path: List[Tuple[MCTSNode, int]] = []
            node = root
            state = root_state
            # Selection
            while node.children:
                best_score = -1e9
                best_action = None
                total_visits = sum(c.visit_count for c in node.children.values())
                for a, child in node.children.items():
                    u = self.c_puct * child.prior * math.sqrt(total_visits + 1e-8) / (1 + child.visit_count)
                    score = child.value + u
                    if score > best_score:
                        best_score = score
                        best_action = a
                path.append((node, best_action))
                # Apply move
                try:
                    state = state.apply_move(best_action)
                except NotImplementedError:
                    break
                node = node.children[best_action]

            # Expansion
            try:
                priors, value = self._eval(state)
            except Exception:
                value = 0.0
                priors = np.ones(9, dtype=np.float32) / 9.0
            legal = state.legal_moves() if hasattr(state, 'legal_moves') else list(range(9))
            mask = np.zeros_like(priors)
            for a in legal:
                mask[a] = 1.0
            priors = priors * mask
            s = priors.sum()
            if s > 0:
                priors = priors / s
            leaf = MCTSNode(prior=1.0)
            for a in range(len(priors)):
                if priors[a] > 0:
                    leaf.children[a] = MCTSNode(prior=float(priors[a]))

            # Backprop
            for parent, action in reversed(path):
                child = parent.children[action]
                child.visit_count += 1
                child.total_value += value

        visit_counts = np.zeros(9, dtype=np.float32)
        for a, child in root.children.items():
            visit_counts[a] = child.visit_count
        if visit_counts.sum() > 0:
            policy = visit_counts / visit_counts.sum()
        else:
            policy = np.ones(9, dtype=np.float32) / 9.0
        return policy
