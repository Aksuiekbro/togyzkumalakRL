from src.nn.model import AlphaZeroNet
from src.arena.eval import arena, ArenaConfig


def test_arena_smoke():
    a = AlphaZeroNet(in_channels=7, channels=16, num_blocks=1)
    b = AlphaZeroNet(in_channels=7, channels=16, num_blocks=1)
    cfg = ArenaConfig(games=2, simulations=4)
    wins, draws, losses, wr = arena(a, b, cfg)
    assert wins + draws + losses == 2
    assert 0.0 <= wr <= 1.0


