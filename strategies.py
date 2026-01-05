import numpy as np
from scipy.stats import binomtest, binom
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

class ConstantBetStrategy:
    # Class representing a constant bet strategy
    # It places a constant bet amount
    def __init__(self, bet_amount, bet_type, selection):
        self.bet_amount = bet_amount
        self.bet_type = bet_type
        self.selection = selection

    def next_bet(self):
        return {
            "bet_amount": self.bet_amount, 
            "bet_type": self.bet_type, 
            "selection": self.selection
        }

class MartingaleBetStrategy:
    # Class representing a Martingale bet strategy
    # It doubles the bet amount after each loss
    # It resets the bet amount to the initial bet amount after a win
    def __init__(self, initial_bet_amount, bet_type, selection):
        self.initial_bet_amount = initial_bet_amount
        self.bet_amount = initial_bet_amount
        self.bet_type = bet_type
        self.selection = selection

    def next_bet(self, previous_payout):
        if previous_payout > 0:
            self.bet_amount = self.initial_bet_amount
        else:
            self.bet_amount *= 2
        return {
            "bet_amount": self.bet_amount, 
            "bet_type": self.bet_type, 
            "selection": self.selection
        }


class HotNumbersBetStrategy:
    # Class representing a Hot Numbers bet strategy
    # It places a straight-up bet on the hot numbers
    def __init__(self, N, K, bet_amount, bet_type):
        if bet_type != "straight_up":
            raise ValueError("HotNumbersBetStrategy only supports straight_up bets")
        self.N = N
        self.K = K
        self.bet_amount = bet_amount
        self.bet_type = bet_type
        self.hot_numbers = [[i, 0] for i in range(37)]

    def next_bet(self, previous_spins):

        self._update_hot_numbers(previous_spins)

        selection = self._select_hot_numbers()

        # Supply a stake per selected number
        if isinstance(self.bet_amount, list):
            if len(self.bet_amount) != len(selection):
                raise ValueError("Selection and bet_amount must have same length")
            bet_amount = self.bet_amount
        else:
            bet_amount = [self.bet_amount] * len(selection)
        
        return {
            "bet_amount": bet_amount,
            "bet_type": self.bet_type, 
            "selection": selection
        }
    
    def _update_hot_numbers(self, previous_spins):
        if not previous_spins:
            return

        number_to_add = -1
        number_to_remove = -1

        # Take last N + 1 spins
        if len(previous_spins) >= self.N + 1:
            previous_spins = previous_spins[-(self.N + 1):]
            number_to_remove = previous_spins[0][0]
            number_to_add = previous_spins[-1][0]

        else:
            number_to_add = previous_spins[-1][0]

        for number in self.hot_numbers:
            if number[0] == number_to_add:
                number[1] += 1
            
            elif number[0] == number_to_remove:
                number[1] -= 1

        self.hot_numbers.sort(key=lambda x: x[1], reverse=True)

    def _select_hot_numbers(self):
        return [number[0] for number in self.hot_numbers[:self.K]]


class KellyBetStrategy:
    # Class representing a Kelly bet strategy
    # It places a bet according to the Kelly criterion
    def __init__(self, bet_type, alpha=1.0, kelly_fraction=1.0, max_fraction=0.2, min_bet=1.0):
        if bet_type != "straight_up":
            raise ValueError("KellyBetStrategy only supports straight_up bets")
        self.bet_type = bet_type
        self.wins_per_number = {number: 0 for number in range(37)}
        self.trials = 0
        self.alpha = alpha
        self.kelly_fraction = kelly_fraction  # allow fractional Kelly for risk control
        self.max_fraction = max_fraction      # cap per-number exposure
        self.min_bet = min_bet                # table minimum per number

    def next_bet(self, bankroll, previous_spin):
        # Update observations with the last spin result
        if previous_spin is not None:
            self._update_wins_per_number(previous_spin)

        if bankroll <= 0:
            return {"bet_amount": [], "bet_type": self.bet_type, "selection": []}

        p_hats = self._calculate_p_hats()
        stakes = self._calculate_stakes(bankroll, p_hats)
        if not stakes:
            return {"bet_amount": [], "bet_type": self.bet_type, "selection": []}

        selection = [number for number, _ in stakes]
        bet_amount = [amount for _, amount in stakes]

        return {"bet_amount": bet_amount, "bet_type": self.bet_type, "selection": selection}
    
    def _calculate_p_hats(self):
        return {number: (self.wins_per_number[number] + self.alpha) / (self.trials + 37 * self.alpha) for number in range(37)}

    def _calculate_stakes(self, bankroll, p_hats):
        # 1. Order by p_hat descending
        p_hats_sorted = sorted(p_hats.items(), key=lambda x: x[1], reverse=True)

        selected = []
        current_p_sum = 0.0
        
        # 2. Selection Loop
        for number, p_hat in p_hats_sorted:
            # Safety: Cannot bet on more than 35 numbers with this math
            if len(selected) >= 35:
                break
                
            R = (1.0 - current_p_sum) / (36.0 - len(selected))
            
            if p_hat > R:
                selected.append((number, p_hat))
                current_p_sum += p_hat
            else:
                # CRITICAL: Stop as soon as the condition isn't met
                break

        if not selected:
            return []

        # 3. Calculation Loop
        stakes = []
        # Final reserve rate based on the full set S
        final_R = (1.0 - current_p_sum) / (36.0 - len(selected))
        
        for number, p_hat in selected:
            # Calculate optimal fraction
            fraction = p_hat - final_R
            
            # Apply fractional Kelly (risk control)
            fraction *= self.kelly_fraction
            
            # Apply per-number cap
            fraction = min(fraction, self.max_fraction)
            
            stake = fraction * bankroll
            
            # Apply table minimum
            if stake >= self.min_bet:
                stakes.append((number, stake))

        return stakes

    def _update_wins_per_number(self, previous_spin):
        self.wins_per_number[previous_spin[0]] += 1
        self.trials += 1



class FrequencyAnalysisBetStrategy:
    # Class representing a Frequency Analysis bet strategy
    # It places a bet according to the Frequency Analysis criterion
    def __init__(self, bet_type, alpha=0.05, bet_amount=1.0):
        if bet_type != "straight_up":
            raise ValueError("FrequencyAnalysisBetStrategy only supports straight_up bets")
        self.bet_type = bet_type
        self.wins_per_number = np.zeros(37, dtype=int)
        self.trials = 0
        self.p_fair = 1/37
        self.p_profit = 1/36
        self.alpha = alpha
        self.bet_amount = bet_amount

    def next_bet(self, previous_spin):
        # Update observations with the last spin result
        if previous_spin is not None:
            self._update_wins_per_number(previous_spin)

        if self.trials == 0:
            return {"bet_amount": [], "bet_type": self.bet_type, "selection": []}

        p_values, p_lows = self._calculate_p_and_p_low_values()
        selection = self._calculate_selection(p_values, p_lows)
        
        if not selection:
            return {"bet_amount": [], "bet_type": self.bet_type, "selection": []}

        bet_amount = [self.bet_amount] * len(selection)

        return {"bet_amount": bet_amount, "bet_type": self.bet_type, "selection": selection}
    
    def _calculate_p_and_p_low_values(self):
        # binom.sf(k-1, n, p) calculates P(X >= k), equivalent to binomtest(alternative="greater")
        # This one call replaces the loop of 37 binomtest calls
        p_values = binom.sf(self.wins_per_number - 1, self.trials, self.p_fair)
        
        # statsmodels.stats.proportion.proportion_confint is already vectorized
        p_lows, _ = proportion_confint(
            self.wins_per_number, 
            self.trials, 
            alpha=self.alpha * 2, 
            method="beta"
        )
        return p_values, p_lows

    def _calculate_selection(self, p_values, p_lows):
        # multipletests is already vectorized
        rejected, _, _, _ = multipletests(p_values, alpha=self.alpha, method="fdr_bh")

        # Use NumPy boolean indexing to find indices where both conditions are True
        selection = np.where(rejected & (p_lows > self.p_profit))[0]

        return selection.tolist()

    def _update_wins_per_number(self, previous_spin):
        self.wins_per_number[previous_spin[0]] += 1
        self.trials += 1


