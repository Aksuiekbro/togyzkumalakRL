from src.agent import choose_move
from src.game.togyzkumalak import TogyzKumalakState
from src.game.encoding import serialize_fen


def test_agent_choose_move_returns_index():
    fen = serialize_fen(TogyzKumalakState.initial())
    action = choose_move(fen, ckpt_path=None, simulations=4)
    assert isinstance(action, int)
    assert -1 <= action <= 8


