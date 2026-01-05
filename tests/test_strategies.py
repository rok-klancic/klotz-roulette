import pytest
import numpy as np
from strategies import (
    ConstantBetStrategy,
    MartingaleBetStrategy,
    HotNumbersBetStrategy,
    KellyBetStrategy,
    FrequencyAnalysisBetStrategy
)

# ConstantBetStrategy Tests

def test_constant_bet_strategy_initialization():
    strategy = ConstantBetStrategy(10, "straight_up", 17)
    assert strategy.bet_amount == 10
    assert strategy.bet_type == "straight_up"
    assert strategy.selection == 17

def test_constant_bet_strategy_next_bet():
    strategy = ConstantBetStrategy(10, "straight_up", 17)
    bet = strategy.next_bet()
    assert bet == {
        "bet_amount": 10,
        "bet_type": "straight_up",
        "selection": 17
    }

# MartingaleBetStrategy Tests

def test_martingale_bet_strategy_initialization():
    strategy = MartingaleBetStrategy(10, "red_black", "red")
    assert strategy.initial_bet_amount == 10
    assert strategy.bet_amount == 10
    assert strategy.bet_type == "red_black"
    assert strategy.selection == "red"

def test_martingale_bet_strategy_loss_doubles():
    strategy = MartingaleBetStrategy(10, "red_black", "red")
    # First bet is always initial
    bet = strategy.next_bet(0) # previous_payout = 0 (loss)
    assert bet["bet_amount"] == 20
    
    # Second loss doubles again
    bet = strategy.next_bet(0)
    assert bet["bet_amount"] == 40

def test_martingale_bet_strategy_win_resets():
    strategy = MartingaleBetStrategy(10, "red_black", "red")
    # Loss
    strategy.next_bet(0) # bet_amount becomes 20
    # Win
    bet = strategy.next_bet(20) # previous_payout > 0
    assert bet["bet_amount"] == 10

def test_martingale_bet_strategy_sequence():
    strategy = MartingaleBetStrategy(5, "red_black", "black")
    assert strategy.next_bet(0)["bet_amount"] == 10
    assert strategy.next_bet(0)["bet_amount"] == 20
    assert strategy.next_bet(20)["bet_amount"] == 5
    assert strategy.next_bet(0)["bet_amount"] == 10

# HotNumbersBetStrategy Tests

def test_hot_numbers_strategy_invalid_bet_type():
    with pytest.raises(ValueError, match="HotNumbersBetStrategy only supports straight_up bets"):
        HotNumbersBetStrategy(10, 2, 5, "red_black")

def test_hot_numbers_strategy_selection_logic():
    # N=3, K=2. Track last 3 spins, pick top 2.
    strategy = HotNumbersBetStrategy(N=3, K=2, bet_amount=10, bet_type="straight_up")
    
    # First spin: 1
    bet = strategy.next_bet([(1, "red")])
    # History: [1]. Hot: 1 (count 1), others (count 0).
    assert 1 in bet["selection"]
    assert len(bet["selection"]) == 2
    assert bet["bet_amount"] == [10, 10]

    # Second spin: 2
    bet = strategy.next_bet([(1, "red"), (2, "black")])
    # History: [1, 2]. Hot: 1, 2 (counts 1).
    assert set(bet["selection"]) == {1, 2}

    # Third spin: 1 again
    bet = strategy.next_bet([(1, "red"), (2, "black"), (1, "red")])
    # History: [1, 2, 1]. Hot: 1 (count 2), 2 (count 1).
    assert bet["selection"] == [1, 2]

    # Fourth spin: 3. Window of 3 means first '1' is pushed out.
    # History: [1, 2, 1, 3] -> window is [2, 1, 3]
    bet = strategy.next_bet([(1, "red"), (2, "black"), (1, "red"), (3, "red")])
    # Counts in window [2, 1, 3]: 2:1, 1:1, 3:1.
    # Top 2 will be from {1, 2, 3}.
    assert len(bet["selection"]) == 2
    for s in bet["selection"]:
        assert s in {1, 2, 3}

def test_hot_numbers_strategy_multiple_bet_amounts():
    strategy = HotNumbersBetStrategy(N=5, K=2, bet_amount=[10, 20], bet_type="straight_up")
    bet = strategy.next_bet([(1, "red")])
    assert len(bet["selection"]) == 2
    assert bet["bet_amount"] == [10, 20]

def test_hot_numbers_strategy_invalid_bet_amount_length():
    strategy = HotNumbersBetStrategy(N=5, K=2, bet_amount=[10], bet_type="straight_up")
    with pytest.raises(ValueError, match="Selection and bet_amount must have same length"):
        strategy.next_bet([(1, "red")])

# KellyBetStrategy Tests

def test_kelly_strategy_invalid_bet_type():
    with pytest.raises(ValueError, match="KellyBetStrategy only supports straight_up bets"):
        KellyBetStrategy(bet_type="red_black")

def test_kelly_strategy_zero_bankroll():
    strategy = KellyBetStrategy(bet_type="straight_up")
    bet = strategy.next_bet(bankroll=0, previous_spin=None)
    assert bet["selection"] == []
    assert bet["bet_amount"] == []

def test_kelly_strategy_initial_bets():
    # With alpha=1.0, initially all p_hat = (0+1)/(0+37) = 1/37
    # Threshold R for the first selection: (1-0)/(36-0) = 1/36
    # Since 1/37 < 1/36, no numbers should be selected initially.
    
    strategy = KellyBetStrategy(bet_type="straight_up", alpha=1.0, kelly_fraction=1.0, max_fraction=0.2, min_bet=0.1)
    bankroll = 100
    bet = strategy.next_bet(bankroll, None)
    
    assert len(bet["selection"]) == 0
    assert len(bet["bet_amount"]) == 0

def test_kelly_strategy_updates_and_edge():
    # If one number wins many times, it should get more stake
    strategy = KellyBetStrategy(bet_type="straight_up", alpha=1.0, kelly_fraction=1.0, max_fraction=0.2)
    bankroll = 100
    
    # Simulate number 5 winning 10 times in 10 trials
    for _ in range(10):
        strategy.next_bet(bankroll, (5, "red"))
    
    # trials = 10, wins[5] = 10, others = 0
    # p_hat[5] = (10+1)/(10 + 37*1) = 11/47 ~= 0.234
    # Threshold R for first selection (S={}): (1-0)/(36-0) = 0.027
    # 0.234 > 0.027, so 5 is selected.
    # New R for S={5}: (1 - 0.234)/(36-1) = 0.766 / 35 ~= 0.0218
    # p_hat[others] = (0+1)/(10+47) = 1/47 ~= 0.0212
    # Since 0.0212 < 0.0218, no other numbers are selected.
    
    bet = strategy.next_bet(bankroll, None)
    
    # Only number 5 should be selected
    assert bet["selection"] == [5]
    
    # Fraction for 5: p_hat[5] - R_final = 11/47 - (1 - 11/47)/35
    # = 11/47 - (36/47)/35 = (385 - 36) / (47 * 35) = 349 / 1645 ~= 0.212
    # Capped at max_fraction = 0.2
    assert bet["bet_amount"][0] == pytest.approx(0.2 * bankroll)

def test_kelly_strategy_min_bet():
    # Set min_bet very high so only high edge numbers are picked
    strategy = KellyBetStrategy(bet_type="straight_up", alpha=1.0, min_bet=50)
    bankroll = 100
    # Initially all stakes are 100/37 ~= 2.7, which is < 50
    bet = strategy.next_bet(bankroll, None)
    assert bet["selection"] == []

# FrequencyAnalysisBetStrategy Tests

def test_frequency_analysis_strategy_invalid_bet_type():
    with pytest.raises(ValueError, match="FrequencyAnalysisBetStrategy only supports straight_up bets"):
        FrequencyAnalysisBetStrategy(bet_type="red_black")

def test_frequency_analysis_strategy_initial_state():
    strategy = FrequencyAnalysisBetStrategy(bet_type="straight_up")
    bet = strategy.next_bet(None)
    assert bet["selection"] == []
    assert bet["bet_amount"] == []

def test_frequency_analysis_strategy_hot_number():
    # alpha=0.05. p_fair=1/37. p_profit=1/36.
    strategy = FrequencyAnalysisBetStrategy(bet_type="straight_up", alpha=0.05, bet_amount=5.0)
    
    # Simulate a very hot number
    # If 7 hits 50 times in 100 spins, it's definitely hot
    for _ in range(100):
        # We need to call next_bet to update state, or call _update_wins_per_number directly
        # next_bet(previous_spin)
        strategy.next_bet((7, "red"))
    
    bet = strategy.next_bet(None)
    
    # Number 7 should be selected
    assert 7 in bet["selection"]
    # Bet amount should be the constant 5.0
    idx_7 = bet["selection"].index(7)
    assert bet["bet_amount"][idx_7] == 5.0

def test_frequency_analysis_strategy_no_selection_for_fair_wheel():
    strategy = FrequencyAnalysisBetStrategy(bet_type="straight_up", alpha=0.01)
    
    # Simulate 37 spins where each number hits once (perfectly fair)
    for i in range(37):
        strategy.next_bet((i, "red" if i % 2 == 0 else "black"))
        
    bet = strategy.next_bet(None)
    # No number should be significantly better than fair with such low alpha and small sample
    assert bet["selection"] == []

