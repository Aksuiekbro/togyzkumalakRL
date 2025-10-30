from src.nn.model import AlphaZeroNet
from src.selfplay.worker import play_episode, SelfPlayConfig


def test_selfplay_generates_samples():
    net = AlphaZeroNet(in_channels=7, channels=16, num_blocks=1)
    cfg = SelfPlayConfig(simulations=4, temp_moves=2)
    samples = play_episode(net, cfg)
    assert isinstance(samples, list)
    assert all(len(x) == 3 for x in samples)


