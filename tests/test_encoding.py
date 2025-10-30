from src.game.togyzkumalak import TogyzKumalakState, WHITE
from src.game.encoding import canonicalize, encode_features, serialize_fen, deserialize_fen, state_key


def test_canonicalize_white_is_identity():
    s = TogyzKumalakState.initial()
    c = canonicalize(s)
    assert c == s


def test_fen_roundtrip():
    s = TogyzKumalakState.initial()
    fen = serialize_fen(s)
    s2 = deserialize_fen(fen)
    assert s2 == s


def test_encode_shape():
    s = TogyzKumalakState.initial()
    feats = encode_features(s)
    assert len(feats) == 7
    assert all(len(ch) == 18 for ch in feats)


def test_state_key_changes_after_move():
    s = TogyzKumalakState.initial()
    k1 = state_key(s)
    s2 = s.apply_move(0)
    k2 = state_key(s2)
    assert k1 != k2


