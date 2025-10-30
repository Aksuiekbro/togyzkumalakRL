from __future__ import annotations

import argparse
import json
from typing import List

import torch

from src.game.togyzkumalak import TogyzKumalakState
from src.game.encoding import deserialize_fen
from src.mcts.search import MCTS
from src.nn.model import AlphaZeroNet


def load_model(ckpt_path: str | None) -> AlphaZeroNet:
    net = AlphaZeroNet()
    if ckpt_path:
        data = torch.load(ckpt_path, map_location="cpu")
        state_dict = data.get("model", data)
        net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net


def choose_move(fen: str, ckpt_path: str | None = None, simulations: int = 160) -> int:
    state = deserialize_fen(fen)
    net = load_model(ckpt_path)
    mcts = MCTS(net.infer, num_simulations=simulations)
    pi = mcts.search(state)
    legal = state.legal_moves()
    if not legal:
        return -1
    action = max(legal, key=lambda a: pi[a])
    return action


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Togyz Kumalak agent")
    parser.add_argument("fen", type=str, help="FEN-like position string")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--sims", type=int, default=160, help="MCTS simulations")
    args = parser.parse_args()
    action = choose_move(args.fen, args.ckpt, args.sims)
    print(json.dumps({"action": action}))


if __name__ == "__main__":
    main()


