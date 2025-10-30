import pytest
from dataclasses import replace

from src.game import TogyzKumalakState, WHITE, BLACK


def make_empty_state():
    return TogyzKumalakState(
        pits=[[0] * 9, [0] * 9], kazans=[0, 0], tuzdyk_indices=[None, None]
    )


def test_initial_legal_moves():
    s = TogyzKumalakState.initial()
    assert len(s.legal_moves()) == 9


def test_single_stone_move_advances():
    s = make_empty_state()
    s.pits[WHITE][0] = 1
    # White moves from pit 0: single stone goes to next pit (index 1)
    s = TogyzKumalakState(
        pits=[s.pits[WHITE][:], s.pits[BLACK][:]],
        kazans=s.kazans[:],
        tuzdyk_indices=[None, None],
        player_to_move=WHITE,
    )
    ns = s.apply_move(0)
    assert ns.pits[WHITE][0] == 0
    assert ns.pits[WHITE][1] == 1


def test_even_capture():
    s = make_empty_state()
    # White side pit 0 has 1 stone, Black pit 0 has 1 stone.
    s.pits[WHITE][0] = 1
    s.pits[BLACK][0] = 1
    s = TogyzKumalakState(
        pits=[s.pits[WHITE][:], s.pits[BLACK][:]],
        kazans=[0, 0],
        tuzdyk_indices=[None, None],
        player_to_move=WHITE,
    )
    ns = s.apply_move(0)  # last stone to Black pit 0 -> becomes 2 -> capture
    assert ns.kazans[WHITE] == 2
    assert ns.pits[BLACK][0] == 0


def test_create_tuzdyk_when_three_and_allowed():
    s = make_empty_state()
    # White moves from pit 0 with 1 stone to Black pit 0 where 2 already -> tuzdyk
    s.pits[WHITE][0] = 1
    s.pits[BLACK][0] = 2
    s = TogyzKumalakState(
        pits=[s.pits[WHITE][:], s.pits[BLACK][:]],
        kazans=[0, 0],
        tuzdyk_indices=[None, None],
        player_to_move=WHITE,
    )
    ns = s.apply_move(0)
    assert ns.tuzdyk_indices[WHITE] == 0
    assert ns.kazans[WHITE] == 3
    assert ns.pits[BLACK][0] == 0


def test_cannot_create_tuzdyk_in_ninth_pit():
    s = make_empty_state()
    s.pits[WHITE][8] = 1
    s.pits[BLACK][0] = 0
    # Arrange so last stone lands in Black pit 8 which already has 2
    s.pits[BLACK][8] = 2
    s = TogyzKumalakState(
        pits=[s.pits[WHITE][:], s.pits[BLACK][:]],
        kazans=[0, 0],
        tuzdyk_indices=[None, None],
        player_to_move=WHITE,
    )
    # Sow from pit 8: single stone goes to Black pit 0 (not 8). Instead, craft a multi-stone move
    # Make pit 7 contain 2 so one returns to 7, last to 8
    s = TogyzKumalakState(
        pits=[[0, 0, 0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2]],
        kazans=[0, 0],
        tuzdyk_indices=[None, None],
        player_to_move=WHITE,
    )
    ns = s.apply_move(6)  # drop to 6 then 7 -> last on Black pit 8 (index 8)
    assert ns.tuzdyk_indices[WHITE] is None
    assert ns.pits[BLACK][8] == 3  # remains as 3, no tuzdyk created


def test_cannot_create_symmetrical_tuzdyk():
    s = make_empty_state()
    # Black already has tuzdyk on index 2 (White's side)
    s = TogyzKumalakState(
        pits=[s.pits[WHITE][:], s.pits[BLACK][:]],
        kazans=[0, 0],
        tuzdyk_indices=[None, 2],
        player_to_move=WHITE,
    )
    # White tries to create tuzdyk at Black index 2
    s = replace(s, pits=[[1] + [0] * 8, [0, 0, 2, 0, 0, 0, 0, 0, 0]])
    ns = s.apply_move(0)
    assert ns.tuzdyk_indices[WHITE] is None
    assert ns.pits[BLACK][2] == 3


def test_stones_fall_into_tuzdyk_are_captured_immediately():
    s = make_empty_state()
    # White owns a tuzdyk on Black pit 1
    s = TogyzKumalakState(
        pits=[[0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        kazans=[0, 0],
        tuzdyk_indices=[1, None],
        player_to_move=WHITE,
    )
    # Move from White pit 8 (2 stones): one back to 8, one to Black pit 0, next would be 1 but we only had 2.
    ns = s.apply_move(8)
    # No stone fell into tuzdyk yet. Now Black moves a stone so that a stone falls into White's tuzdyk at index 1
    ns = replace(ns, pits=[ns.pits[WHITE][:], [1, 0, 0, 0, 0, 0, 0, 0, 0]], player_to_move=BLACK)
    ns2 = ns.apply_move(0)
    assert ns2.kazans[WHITE] >= 1  # captured by tuzdyk owner


def test_atsyrau_terminal_and_scoring():
    s = make_empty_state()
    # White to move with no stones on White side → terminal; Black collects remaining
    s = TogyzKumalakState(
        pits=[[0] * 9, [1, 2, 3, 4, 0, 0, 0, 0, 0]],
        kazans=[10, 20],
        tuzdyk_indices=[None, None],
        player_to_move=WHITE,
    )
    assert s.is_terminal()
    w, b = s.final_scores()
    assert w == 10
    assert b == 20 + sum([1, 2, 3, 4])


