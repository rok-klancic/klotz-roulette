import random

import pytest

from roulette_wheels import EuropeanRoulette, KlotzRoulette


def test_european_red_black_win(monkeypatch):
    game = EuropeanRoulette()
    monkeypatch.setattr(game, "_spin", lambda: (7, "red"))

    payout, result = game.get_payout("red_black", 10, "red")

    assert payout == 10
    assert result == (7, "red")


def test_european_red_black_loss(monkeypatch):
    game = EuropeanRoulette()
    monkeypatch.setattr(game, "_spin", lambda: (8, "black"))

    payout, result = game.get_payout("red_black", 15, "red")

    assert payout == -15
    assert result == (8, "black")


def test_european_red_black_invalid_selection():
    game = EuropeanRoulette()

    with pytest.raises(ValueError):
        game.get_payout("red_black", 5, "blue")


def test_european_straight_up_win(monkeypatch):
    game = EuropeanRoulette()
    monkeypatch.setattr(game, "_spin", lambda: (12, "red"))

    payout, result = game.get_payout("straight_up", [5, 5], [12, 13])

    assert payout == 170
    assert result == (12, "red")


def test_european_straight_up_loss(monkeypatch):
    game = EuropeanRoulette()
    monkeypatch.setattr(game, "_spin", lambda: (14, "black"))

    payout, result = game.get_payout("straight_up", [5, 5], [1, 2])

    assert payout == -10
    assert result == (14, "black")


def test_european_straight_up_invalid_selection():
    game = EuropeanRoulette()

    with pytest.raises(ValueError):
        game.get_payout("straight_up", 5, 12)


def test_klotz_spin_uses_weights(monkeypatch):
    game = KlotzRoulette()
    chosen = game.numbers[5]

    def fake_choices(seq, weights, k):
        assert seq is game.numbers
        assert weights == game.probabilities
        assert k == 1
        return [chosen]

    monkeypatch.setattr(random, "choices", fake_choices)

    result = game.spin()

    assert result == chosen


def test_klotz_red_black_win(monkeypatch):
    game = KlotzRoulette()
    monkeypatch.setattr(game, "spin", lambda: (20, "black"))

    payout, result = game.get_payout("red_black", 25, "black")

    assert payout == 25
    assert result == (20, "black")


def test_klotz_red_black_loss(monkeypatch):
    game = KlotzRoulette()
    monkeypatch.setattr(game, "spin", lambda: (21, "red"))

    payout, result = game.get_payout("red_black", 30, "black")

    assert payout == -30
    assert result == (21, "red")


def test_klotz_straight_up_win(monkeypatch):
    game = KlotzRoulette()
    monkeypatch.setattr(game, "spin", lambda: (2, "black"))

    payout, result = game.get_payout("straight_up", [10, 10, 10], [2, 3, 4])

    assert payout == 330
    assert result == (2, "black")


def test_klotz_straight_up_loss(monkeypatch):
    game = KlotzRoulette()
    monkeypatch.setattr(game, "spin", lambda: (30, "red"))

    payout, result = game.get_payout("straight_up", [10, 10, 10], [1, 2, 3])

    assert payout == -30
    assert result == (30, "red")


def test_klotz_invalid_selection():
    game = KlotzRoulette()

    with pytest.raises(ValueError):
        game.get_payout("red_black", 10, "green")


def test_klotz_invalid_bet_type():
    game = KlotzRoulette()

    with pytest.raises(ValueError):
        game.get_payout("dozen", 10, [])

