import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Constants
RESULTS_DIR = "results"
VIS_DIR = "visualisations"
os.makedirs(VIS_DIR, exist_ok=True)
STARTING_BANKROLL = 1000

STRATEGY_MAP = {
    'ConstantBet': 'Konstantna stava',
    'FrequencyAnalysis': 'Frekvenčna analiza',
    'HotNumbers': 'Hot Numbers',
    'Kelly': 'Kellyev kriterij',
    'Martingale': 'Martingale'
}

WHEEL_MAP = {
    'EuropeanRoulette': 'Evropska ruleta',
    'KlotzRoulette': 'Klotz ruleta'
}

def load_all_results():
    all_records = []
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue
            
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, 'r') as f:
            data = json.load(f)
        metadata = data['metadata']
        aggregated = data['aggregated']
        params = metadata['params']
        
        # Calculate avg drawdown from sessions
        sessions = data['sessions']
        avg_drawdown = np.mean([s['drawdown'] for s in sessions])
        avg_drawdown_pct = (avg_drawdown / STARTING_BANKROLL) * 100
        
        # Build record
        record = {
            "strategy": metadata['strategy'],
            "wheel": metadata['wheel'],
            "n_spins": params.get('n_spins'),
            "avg_roi": aggregated['avg_bankroll_roi'],
            "win_rate": aggregated['win_rate'],
            "ruin_prob": aggregated['ruin_prob'],
            "avg_drawdown_pct": avg_drawdown_pct,
            "params_str": str({k: v for k, v in params.items() if k not in ['n_spins', 'bet_amount', 'bet_type']}),
            "filename": filename
        }
        
        # Add all params for sensitivity analysis
        for k, v in params.items():
            record[f"param_{k}"] = v
            
        all_records.append(record)
        
    return pd.DataFrame(all_records)

def v1_global_leaderboard(df):
    """Create Global Leaderboard tables."""
    # Create a copy and map values
    df_mapped = df.copy()
    df_mapped['strategy'] = df_mapped['strategy'].map(STRATEGY_MAP)
    df_mapped['wheel'] = df_mapped['wheel'].map(WHEEL_MAP)
    
    for wheel_raw in df['wheel'].unique():
        wheel_mapped = WHEEL_MAP.get(wheel_raw, wheel_raw)
        wheel_df = df_mapped[df_mapped['wheel'] == wheel_mapped].copy()
        
        # Find best params for each strategy (highest ROI)
        best_idx = wheel_df.groupby('strategy')['avg_roi'].idxmax()
        leaderboard = wheel_df.loc[best_idx].sort_values('avg_roi', ascending=False)
        
        column_map = {
            'strategy': 'Strategija',
            'params_str': 'Parametri',
            'wheel': 'Ruleta',
            'avg_roi': 'Pov. ROI',
            'win_rate': 'Delež zmag',
            'ruin_prob': 'Verjetnost bankrota'
        }
        
        leaderboard_table = leaderboard[list(column_map.keys())].rename(columns=column_map)
        
        # Save as CSV
        leaderboard_table.to_csv(os.path.join(VIS_DIR, f"leaderboard_{wheel_raw}.csv"), index=False)
        
        print(f"\nLeaderboard za {wheel_mapped}:")
        print(leaderboard_table.to_string(index=False))

def v2_bias_exploitation(df):
    """Bar chart: ROI difference between Klotz and European."""
    # Pivot to get ROI for each (strategy, params) across wheels
    # Only use N=1000 for bias exploitation to allow learning
    pivot_df = df[df['n_spins'] == 1000].pivot_table(
        index=['strategy', 'params_str'], 
        columns='wheel', 
        values='avg_roi'
    ).dropna()
    
    if pivot_df.empty:
        return
        
    pivot_df['delta_roi'] = pivot_df['KlotzRoulette'] - pivot_df['EuropeanRoulette']
    
    # Get max delta per strategy
    strategy_delta = pivot_df.groupby('strategy')['delta_roi'].max().sort_values(ascending=False)
    strategy_delta.index = strategy_delta.index.map(STRATEGY_MAP)
    
    plt.figure()
    strategy_delta.plot(kind='bar', color='skyblue')
    plt.title("Izkoristek pristranosti: $\Delta$ ROI (Klotz - Evropska)")
    plt.ylabel("$\Delta$ ROI")
    plt.xlabel("Strategija")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "v2_bias_exploitation.png"))
    plt.close()

def v3_parameter_sensitivity(df):
    """Heatmaps for HotNumbers and Kelly."""
    # 1. HotNumbers: N vs K
    hn_df = df[(df['strategy'] == 'HotNumbers') & (df['wheel'] == 'KlotzRoulette') & (df['n_spins'] == 1000)]
    if not hn_df.empty:
        pivot_hn = hn_df.pivot_table(index='param_N', columns='param_K', values='avg_roi')
        plt.figure()
        sns.heatmap(pivot_hn, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'ROI'})
        plt.title("Občutljivost strategije hot numbers: N (zgodovina) vs K (stave) – ROI")
        plt.xlabel("K")
        plt.ylabel("N")
        plt.savefig(os.path.join(VIS_DIR, "v3_sensitivity_hotnumbers.png"))
        plt.close()

    # 2. Kelly: Alpha vs KellyFraction
    kelly_df = df[(df['strategy'] == 'Kelly') & (df['wheel'] == 'KlotzRoulette') & (df['n_spins'] == 1000)]
    if not kelly_df.empty:
        pivot_kelly = kelly_df.pivot_table(index='param_alpha', columns='param_kelly_fraction', values='avg_roi')
        plt.figure()
        sns.heatmap(pivot_kelly, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'ROI'})
        plt.title("Občutljivost strategije kelly: alfa (Laplace) vs delež kelly (agresivnost) – ROI")
        plt.xlabel("Delež kelly")
        plt.ylabel("Alfa")
        plt.savefig(os.path.join(VIS_DIR, "v3_sensitivity_kelly.png"))
        plt.close()

def v4_risk_reward(df):
    """Scatter plot: ROI vs. Max Drawdown."""
    plt.figure(figsize=(14, 7))
    
    # Filter to 1000 spins for consistency
    plot_df = df[df['n_spins'] == 1000].copy()
    
    # Mapping for translation
    wheel_titles = {
        'EuropeanRoulette': 'Tveganje vs zaslžek: evropska ruleta',
        'KlotzRoulette': 'Tveganje vs zaslužek: Klotz ruleta'
    }
    
    plot_df['strategy'] = plot_df['strategy'].map(STRATEGY_MAP)
    plot_df = plot_df.rename(columns={'strategy': 'Strategija'})
    
    wheels = ['EuropeanRoulette', 'KlotzRoulette']
    for i, wheel in enumerate(wheels):
        plt.subplot(1, 2, i+1)
        wheel_df = plot_df[plot_df['wheel'] == wheel]
        sns.scatterplot(
            data=wheel_df, 
            x='avg_drawdown_pct', 
            y='avg_roi', 
            hue='Strategija', 
            style='Strategija',
            s=100,
            alpha=0.7
        )
        plt.title(wheel_titles[wheel])
        plt.xlabel("Povprečen maksimalni padec (%)")
        plt.ylabel("Povprečen ROI")
        
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "v4_risk_reward.png"))
    plt.close()

def v5_personality_paths(df):
    """Grid of path plots: Fair vs Biased."""
    strategies = df['strategy'].unique()
    n_strats = len(strategies)
    
    fig, axes = plt.subplots(n_strats, 1, figsize=(10, 4 * n_strats), sharex=True)
    if n_strats == 1: axes = [axes]
    
    for i, strat in enumerate(strategies):
        # Find best configuration for this strategy on Klotz at 1000 spins
        klotz_runs = df[(df['strategy'] == strat) & (df['wheel'] == 'KlotzRoulette') & (df['n_spins'] == 1000)]
        if klotz_runs.empty: continue
        
        best_klotz = klotz_runs.sort_values('avg_roi', ascending=False).iloc[0]
        
        # Load histories
        with open(os.path.join(RESULTS_DIR, best_klotz['filename']), 'r') as f:
            klotz_history = json.load(f)['first_session_history']
            
        # Find corresponding European run (same params)
        euro_match = df[
            (df['strategy'] == strat) & 
            (df['wheel'] == 'EuropeanRoulette') & 
            (df['params_str'] == best_klotz['params_str']) &
            (df['n_spins'] == best_klotz['n_spins'])
        ]
        
        if not euro_match.empty:
            with open(os.path.join(RESULTS_DIR, euro_match.iloc[0]['filename']), 'r') as f:
                euro_history = json.load(f)['first_session_history']
            
            axes[i].plot(klotz_history, label='Klotz (pristranska)', linestyle='-', color='blue')
            axes[i].plot(euro_history, label='Evropska (poštena)', linestyle='--', color='red')
            axes[i].set_title(f"Strategija: {STRATEGY_MAP.get(strat, strat)}")
            axes[i].set_ylabel("Kapital")
            axes[i].legend()
            
    plt.xlabel("Vrtljaji")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "v5_personality_paths.png"))
    plt.close()

def v6_learning_accuracy(df):
    """Scatter: Predicted vs Actual Probability for biased wheel."""
    # Find a good example run (Kelly or FrequencyAnalysis on Klotz)
    learning_strats = ['Kelly', 'FrequencyAnalysis']
    
    for strat in learning_strats:
        strat_df = df[(df['strategy'] == strat) & (df['wheel'] == 'KlotzRoulette') & (df['n_spins'] == 1000)]
        if strat_df.empty: continue
        
        # Pick the one with best ROI
        best_run = strat_df.sort_values('avg_roi', ascending=False).iloc[0]
        
        with open(os.path.join(RESULTS_DIR, best_run['filename']), 'r') as f:
            data = json.load(f)
            if 'bias_detection' in data:
                bd = data['bias_detection']
                
                plt.figure(figsize=(8, 8))
                plt.scatter(bd['actual_probs'], bd['estimated_probs'], alpha=0.6)
                
                # 45-degree line
                lims = [0, max(max(bd['actual_probs']), max(bd['estimated_probs'])) * 1.1]
                plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
                plt.xlabel("Resnična verjetnost")
                plt.ylabel("Ocenjena verjetnost")
                plt.title(f"Natančnost učenja pristranosti: {STRATEGY_MAP.get(strat, strat)}")
                plt.savefig(os.path.join(VIS_DIR, f"v6_learning_{strat}.png"))
                plt.close()

def v7_time_sensitivity(df):
    """Line chart: ROI/Ruin vs N spins."""
    # Group by strategy and n_spins, using best ROI configs
    plt.figure(figsize=(12, 6))
    
    for strat in df['strategy'].unique():
        strat_data = df[df['strategy'] == strat]
        # For each N, get the best configuration
        n_results = strat_data.groupby('n_spins')['avg_roi'].max()
        plt.plot(n_results.index, n_results.values, marker='o', label=STRATEGY_MAP.get(strat, strat))
        
    plt.title("Občutljivost na število vrtljajev")
    plt.xlabel("Število vrtljajev")
    plt.ylabel("Maksimalni povprečni ROI")
    plt.xticks([50, 200, 1000])
    plt.legend()
    plt.savefig(os.path.join(VIS_DIR, "v7_time_sensitivity.png"))
    plt.close()

def main():
    print("Loading results...")
    df = load_all_results()
    
    print("Generating visualisations...")
    v1_global_leaderboard(df)
    v2_bias_exploitation(df)
    v3_parameter_sensitivity(df)
    v4_risk_reward(df)
    v5_personality_paths(df)
    v6_learning_accuracy(df)
    v7_time_sensitivity(df)
    
    print(f"All visualisations saved to '{VIS_DIR}' folder.")

if __name__ == "__main__":
    main()

