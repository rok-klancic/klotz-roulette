import json
import os
import sys
import time
from itertools import product
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from roulette_wheels import EuropeanRoulette, KlotzRoulette
from strategies import (
    ConstantBetStrategy,
    MartingaleBetStrategy,
    HotNumbersBetStrategy,
    KellyBetStrategy,
    FrequencyAnalysisBetStrategy
)

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
STARTING_BANKROLL = 1000
MIN_BET = 1
N_SPINS_OPTIONS = [50, 200, 1000]
N_SESSIONS = 1000
PROFIT_TARGET = 2000

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class SessionResult:
    final_br: float
    ruined: bool
    hit_target: bool
    drawdown: float
    spins: int
    total_wagered: float


@dataclass
class SimulationResult:
    metadata: Dict[str, Any]
    aggregated: Dict[str, float]
    sessions: List[Dict[str, Any]]
    first_session_history: List[float]
    bias_detection: Optional[Dict[str, List[float]]] = None


# ============================================================================
# PARAMETER GENERATORS
# ============================================================================
def generate_constant_bet_configs():
    """Generate all ConstantBet parameter combinations."""
    bet_amounts = [1, 10, 50]
    configs = []
    for bet_amount in bet_amounts:
        configs.append({
            'bet_amount': bet_amount,
            'bet_type': 'red_black',
            'selection': 'red'
        })
    return configs


def generate_martingale_configs():
    """Generate all Martingale parameter combinations."""
    initial_bets = [1, 5]
    configs = []
    for initial_bet in initial_bets:
        configs.append({
            'initial_bet_amount': initial_bet,
            'bet_type': 'red_black',
            'selection': 'red'
        })
    return configs


def generate_hot_numbers_configs():
    """Generate all HotNumbers parameter combinations (N x K Cartesian product)."""
    N_values = [50, 100, 200]
    K_values = [1, 5, 10]
    configs = []
    for N, K in product(N_values, K_values):
        configs.append({
            'N': N,
            'K': K,
            'bet_amount': 1,
            'bet_type': 'straight_up'
        })
    return configs


def generate_kelly_configs():
    """Generate all Kelly parameter combinations (alpha x kelly_fraction x max_fraction)."""
    alphas = [0.1, 1.0]
    kelly_fractions = [0.5, 1.0]
    max_fractions = [0.2, 0.4]
    configs = []
    for alpha, kf, mf in product(alphas, kelly_fractions, max_fractions):
        configs.append({
            'bet_type': 'straight_up',
            'alpha': alpha,
            'kelly_fraction': kf,
            'max_fraction': mf,
            'min_bet': MIN_BET
        })
    return configs


def generate_frequency_configs():
    """Generate all FrequencyAnalysis parameter combinations."""
    alphas = [0.01, 0.05, 0.1]
    configs = []
    for alpha in alphas:
        configs.append({
            'bet_type': 'straight_up',
            'alpha': alpha,
            'bet_amount': 1.0
        })
    return configs


def generate_all_configs():
    """
    Generate all strategy-wheel-n_spins combinations.
    Returns list of tuples: (strategy_name, strategy_params, wheel_name, n_spins)
    """
    all_configs = []
    wheel_names = ['EuropeanRoulette', 'KlotzRoulette']
    
    strategy_configs = {
        'ConstantBet': generate_constant_bet_configs(),
        'Martingale': generate_martingale_configs(),
        'HotNumbers': generate_hot_numbers_configs(),
        'Kelly': generate_kelly_configs(),
        'FrequencyAnalysis': generate_frequency_configs()
    }
    
    for strategy_name, configs in strategy_configs.items():
        for params in configs:
            for wheel_name in wheel_names:
                for n_spins in N_SPINS_OPTIONS:
                    all_configs.append((strategy_name, params, wheel_name, n_spins))
    
    return all_configs


# ============================================================================
# STRATEGY FACTORY
# ============================================================================
def create_strategy(strategy_name: str, params: Dict[str, Any]):
    """Create a fresh strategy instance from name and parameters."""
    if strategy_name == 'ConstantBet':
        return ConstantBetStrategy(**params)
    elif strategy_name == 'Martingale':
        return MartingaleBetStrategy(**params)
    elif strategy_name == 'HotNumbers':
        return HotNumbersBetStrategy(**params)
    elif strategy_name == 'Kelly':
        return KellyBetStrategy(**params)
    elif strategy_name == 'FrequencyAnalysis':
        return FrequencyAnalysisBetStrategy(**params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# ============================================================================
# WHEEL FACTORY
# ============================================================================
def create_wheel(wheel_name: str):
    """Create a fresh wheel instance from name."""
    if wheel_name == 'EuropeanRoulette':
        return EuropeanRoulette()
    elif wheel_name == 'KlotzRoulette':
        return KlotzRoulette()
    else:
        raise ValueError(f"Unknown wheel: {wheel_name}")


# ============================================================================
# SESSION RUNNER
# ============================================================================
def _distribute_last_ditch_bets(bankroll, bet_amount, selection, min_bet):
    """
    Handles scaling and redistributing bets when the bankroll is insufficient.
    Ensures no individual bet is below min_bet.
    """
    current_total = sum(bet_amount)
    # First scale the values
    scaled_bets = [b * (bankroll / current_total) for b in bet_amount]
    
    # Distribute funds to those >= min_bet
    valid_indices = [i for i, b in enumerate(scaled_bets) if b >= min_bet]
    
    if not valid_indices:
        # If none of them had more than min bet, give everything to the first selection
        return [bankroll], [selection[0]]
    
    invalid_total = sum(scaled_bets[i] for i in range(len(scaled_bets)) if i not in valid_indices)
    valid_total = sum(scaled_bets[i] for i in valid_indices)
    final_bets = []
    final_selection = []
    
    for i in valid_indices:
        # Each valid bet gets its original scaled amount plus its share of the invalid funds
        share = scaled_bets[i] / valid_total
        final_bets.append(scaled_bets[i] + (invalid_total * share))
        final_selection.append(selection[i])
        
    return final_bets, final_selection


def run_session(
    strategy_name: str,
    strategy,
    wheel,
    n_spins: int,
    starting_bankroll: float,
    profit_target: float,
    min_bet: float,
    record_history: bool = False
) -> tuple:
    """
    Run a single session simulation.
    
    Returns:
        SessionResult, bankroll_history (if record_history), estimated_probs (for Kelly/Frequency)
    """
    bankroll = starting_bankroll
    total_wagered = 0.0
    peak_bankroll = starting_bankroll
    max_drawdown = 0.0
    spins_played = 0
    is_ruined = False
    hit_profit_target = False
    
    bankroll_history = [bankroll] if record_history else None
    
    # State tracking for different strategies
    previous_payout = 0
    previous_spins = []
    previous_spin = None
    
    for spin_num in range(n_spins):
        # Check stopping conditions
        if bankroll < MIN_BET:
            is_ruined = True
            break
        if bankroll >= profit_target:
            hit_profit_target = True
            break
        
        # Get next bet based on strategy type
        bet = _get_next_bet(
            strategy_name, strategy, bankroll, 
            previous_payout, previous_spins, previous_spin
        )
        
        bet_amount = bet['bet_amount']
        bet_type = bet['bet_type']
        selection = bet['selection']
        
        # Calculate total bet for this spin
        if isinstance(bet_amount, list):
            total_bet = sum(bet_amount)
        else:
            total_bet = bet_amount
        
        # Skip if no bet
        if total_bet <= 0:
            # Still need to observe the spin for learning strategies
            if bet_type == 'straight_up':
                _, spin_result = wheel.get_payout('straight_up', [0], [0])
            else:
                # For red_black, we need to make a dummy bet to get spin result
                _, spin_result = wheel.get_payout('red_black', 0, 'red')
            previous_spin = spin_result
            previous_spins.append(spin_result)
            spins_played += 1
            if record_history:
                bankroll_history.append(bankroll)
            continue
        
        # Check if bet exceeds bankroll
        if total_bet > bankroll:
            # Use remaining bankroll
            total_bet = bankroll
            if isinstance(bet_amount, list):
                bet_amount, selection = _distribute_last_ditch_bets(
                    bankroll, bet_amount, selection, min_bet
                )
            else:
                bet_amount = bankroll
        
        # Track total wagered
        total_wagered += total_bet
        
        # Execute bet
        payout, spin_result = wheel.get_payout(bet_type, bet_amount, selection)
        
        # Update bankroll
        bankroll += payout
        previous_payout = payout
        previous_spin = spin_result
        previous_spins.append(spin_result)
        spins_played += 1
        
        # Update peak and drawdown
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        current_drawdown = peak_bankroll - bankroll
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
        
        if record_history:
            bankroll_history.append(bankroll)
    
    # Ruin check
    if bankroll <= 0:
        is_ruined = True

    # Profit target check
    if bankroll >= profit_target:
        hit_profit_target = True
    
    # Get estimated probabilities for Kelly/Frequency strategies
    estimated_probs = None
    if strategy_name == 'Kelly':
        # Kelly stores wins_per_number and trials
        estimated_probs = strategy._calculate_p_hats()
        estimated_probs = [estimated_probs[i] for i in range(37)]
    elif strategy_name == 'FrequencyAnalysis':
        # Frequency stores wins_per_number and trials
        if strategy.trials > 0:
            estimated_probs = (strategy.wins_per_number / strategy.trials).tolist()
    
    session_result = SessionResult(
        final_br=bankroll,
        ruined=is_ruined,
        hit_target=hit_profit_target,
        drawdown=max_drawdown,
        spins=spins_played,
        total_wagered=total_wagered
    )
    
    return session_result, bankroll_history, estimated_probs


def _get_next_bet(strategy_name, strategy, bankroll, previous_payout, previous_spins, previous_spin):
    """Get next bet based on strategy type and its specific interface."""
    if strategy_name == 'ConstantBet':
        return strategy.next_bet()
    elif strategy_name == 'Martingale':
        return strategy.next_bet(previous_payout)
    elif strategy_name == 'HotNumbers':
        return strategy.next_bet(previous_spins)
    elif strategy_name == 'Kelly':
        return strategy.next_bet(bankroll, previous_spin)
    elif strategy_name == 'FrequencyAnalysis':
        return strategy.next_bet(previous_spin)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================
def calculate_aggregates(sessions: List[SessionResult], starting_bankroll: float) -> Dict[str, float]:
    """Calculate aggregated statistics from session results."""
    n = len(sessions)
    
    profits = [s.final_br - starting_bankroll for s in sessions]
    total_wagered = [s.total_wagered for s in sessions]
    
    avg_profit_loss = np.mean(profits)
    std_profit = np.std(profits)
    
    # ROI relative to starting capital
    avg_bankroll_roi = avg_profit_loss / starting_bankroll
    
    # ROI relative to total volume
    total_p_l = sum(profits)
    total_volume = sum(total_wagered)
    avg_turnover_roi = total_p_l / total_volume if total_volume > 0 else 0
    
    ruin_count = sum(1 for s in sessions if s.ruined)
    ruin_prob = ruin_count / n
    
    win_count = sum(1 for s in sessions if s.final_br > starting_bankroll)
    win_rate = win_count / n
    
    return {
        'avg_bankroll_roi': round(avg_bankroll_roi, 6),
        'avg_turnover_roi': round(avg_turnover_roi, 6),
        'ruin_prob': round(ruin_prob, 4),
        'win_rate': round(win_rate, 4),
        'avg_profit_loss': round(avg_profit_loss, 2),
        'std_profit': round(std_profit, 2)
    }


# ============================================================================
# SIMULATION RUNNER
# ============================================================================
def run_simulation(
    strategy_name: str,
    strategy_params: Dict[str, Any],
    wheel_name: str,
    n_spins: int,
    n_sessions: int = N_SESSIONS,
    starting_bankroll: float = STARTING_BANKROLL,
    profit_target: float = PROFIT_TARGET,
    min_bet: float = MIN_BET
) -> SimulationResult:
    """
    Run all sessions for a single configuration.
    """
    sessions = []
    first_session_history = None
    bias_detection = None
    
    # Create the wheel instance once to be reused across all sessions
    wheel = create_wheel(wheel_name)
    actual_probs = wheel.get_probabilities() if wheel_name == 'KlotzRoulette' else None
    
    estimated_probs_first = None
    
    for session_idx in range(n_sessions):
        # Create fresh strategy for each session
        strategy = create_strategy(strategy_name, strategy_params)
        
        record_history = (session_idx == 0)
        
        session_result, history, estimated_probs = run_session(
            strategy_name=strategy_name,
            strategy=strategy,
            wheel=wheel,
            n_spins=n_spins,
            starting_bankroll=starting_bankroll,
            profit_target=profit_target,
            min_bet=min_bet,
            record_history=record_history
        )
        
        sessions.append(session_result)
        
        if session_idx == 0:
            first_session_history = history
            estimated_probs_first = estimated_probs
    
    # Calculate aggregates
    aggregated = calculate_aggregates(sessions, starting_bankroll)
    
    # Prepare bias detection data
    if strategy_name in ['Kelly', 'FrequencyAnalysis'] and wheel_name == 'KlotzRoulette':
        if estimated_probs_first is not None and actual_probs is not None:
            bias_detection = {
                'estimated_probs': [round(p, 6) for p in estimated_probs_first],
                'actual_probs': [round(p, 6) for p in actual_probs]
            }
    
    # Build metadata
    metadata = {
        'strategy': strategy_name,
        'wheel': wheel_name,
        'params': {**strategy_params, 'n_spins': n_spins}
    }
    
    # Convert sessions to dicts
    sessions_dicts = [asdict(s) for s in sessions]
    
    return SimulationResult(
        metadata=metadata,
        aggregated=aggregated,
        sessions=sessions_dicts,
        first_session_history=first_session_history,
        bias_detection=bias_detection
    )


# ============================================================================
# JSON OUTPUT
# ============================================================================
def generate_filename(strategy_name: str, wheel_name: str, params: Dict[str, Any], n_spins: int) -> str:
    """Generate a unique filename for the results."""
    # Create a params string for the filename
    param_parts = []
    for key, value in sorted(params.items()):
        if key not in ['bet_type', 'selection']:  # Skip non-varying params
            param_parts.append(f"{key}={value}")
    
    params_str = "_".join(param_parts) if param_parts else "default"
    filename = f"{strategy_name}_{wheel_name}_nspins={n_spins}_{params_str}.json"
    # Clean filename
    filename = filename.replace(" ", "_")
    return filename


def save_results(result: SimulationResult, output_dir: str = "results"):
    """Save simulation results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = generate_filename(
        result.metadata['strategy'],
        result.metadata['wheel'],
        {k: v for k, v in result.metadata['params'].items() if k != 'n_spins'},
        result.metadata['params']['n_spins']
    )
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert to dict
    output = {
        'metadata': result.metadata,
        'aggregated': result.aggregated,
        'sessions': result.sessions,
        'first_session_history': result.first_session_history
    }
    
    if result.bias_detection is not None:
        output['bias_detection'] = result.bias_detection
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    return filepath


# ============================================================================
# MAIN RUNNER
# ============================================================================
def run_all_simulations(output_dir: str = "results", verbose: bool = True):
    """
    Run all 150 strategy-wheel-parameter combinations.
    """
    all_configs = generate_all_configs()
    total_configs = len(all_configs)
    
    if verbose:
        print(f"Running {total_configs} configurations, {N_SESSIONS} sessions each...")
        print(f"Total sessions to simulate: {total_configs * N_SESSIONS:,}")
        print("-" * 60)
    
    for idx, (strategy_name, params, wheel_name, n_spins) in enumerate(all_configs):
        if verbose:
            print(f"[{idx + 1}/{total_configs}] {strategy_name} on {wheel_name} "
                  f"(n_spins={n_spins}, params={params})")
        
        result = run_simulation(
            strategy_name=strategy_name,
            strategy_params=params,
            wheel_name=wheel_name,
            n_spins=n_spins
        )
        
        filepath = save_results(result, output_dir)
        
        if verbose:
            agg = result.aggregated
            print(f"    -> ROI: {agg['avg_bankroll_roi']:.4f}, Ruin: {agg['ruin_prob']:.2%}, "
                  f"Win Rate: {agg['win_rate']:.2%}")
            print(f"    -> Saved to: {filepath}")
    
    if verbose:
        print("-" * 60)
        print(f"Completed all {total_configs} simulations!")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    start_time = time.time()
    run_all_simulations()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

