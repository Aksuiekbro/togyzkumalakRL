import torch

from src.nn.model import AlphaZeroNet
from src.game.togyzkumalak import TogyzKumalakState


def test_model_forward_shapes():
    net = AlphaZeroNet(in_channels=7, channels=32, num_blocks=2)
    x = torch.randn(2, 7, 18)
    p, v = net(x)
    assert p.shape == (2, 9)
    assert v.shape == (2,)


def test_infer_runs():
    net = AlphaZeroNet(in_channels=7, channels=16, num_blocks=1)
    s = TogyzKumalakState.initial()
    pi, val = net.infer(s)
    assert len(pi) == 9
    assert -1.0 <= val <= 1.0


