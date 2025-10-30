import torch

from src.nn.model import AlphaZeroNet
from src.trainer.loop import ReplayBuffer, TrainConfig, train_one_iteration
from src.game.togyzkumalak import TogyzKumalakState


def test_training_step_smoke():
    # Minimal buffer with few random samples (uniform pi, random z)
    buf = ReplayBuffer(capacity=100)
    s = TogyzKumalakState.initial()
    uniform_pi = [1 / 9.0] * 9
    for _ in range(32):
        buf.add_many([(s, uniform_pi, 0)])

    net = AlphaZeroNet(in_channels=7, channels=16, num_blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    cfg = TrainConfig(batch_size=16, epochs_per_iter=1, amp=False, device="cpu")
    loss = train_one_iteration(net, opt, buf, cfg, steps=2)
    assert isinstance(loss, float)


