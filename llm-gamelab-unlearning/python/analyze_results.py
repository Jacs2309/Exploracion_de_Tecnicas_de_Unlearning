"""
analyze_results.py - Post-hoc analysis for NeurIPS paper metrics.

Reads the CSV output from experiment_runner.py and computes:
  1. Win-rate matrix (player × opponent × game)
  2. Illegal move rate per model
  3. Average latency per model
  4. Misère gap: performance on game X minus performance on inverted game X
  5. Optimality score: % of moves matching minimax (for resolved games)
  6. Statistical significance (bootstrap confidence intervals)

Usage:
    python analyze_results.py results/results_*.csv
"""

import sys
import csv
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple
import random
import math


def load_csv(paths: List[str]) -> List[dict]:
    rows = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


# ---- Metric 1: Win-rate matrix ---------------------------------------------

def compute_win_rates(rows: List[dict]) -> dict:
    """
    Returns nested dict: game → player_name → {wins, losses, draws, total,
                                                 win_rate, loss_rate, draw_rate}
    Aggregated across both roles (playing as x or o).
    """
    stats = defaultdict(lambda: defaultdict(lambda: {
        'wins': 0, 'losses': 0, 'draws': 0, 'total': 0}))

    for r in rows:
        game = r['game']
        px, po = r['player_x'], r['player_o']
        winner = r['winner']

        stats[game][px]['total'] += 1
        stats[game][po]['total'] += 1

        if winner == 'draw':
            stats[game][px]['draws'] += 1
            stats[game][po]['draws'] += 1
        elif winner == 'x':
            stats[game][px]['wins'] += 1
            stats[game][po]['losses'] += 1
        elif winner == 'o':
            stats[game][po]['wins'] += 1
            stats[game][px]['losses'] += 1

    # Compute rates
    result = {}
    for game, players in stats.items():
        result[game] = {}
        for player, s in players.items():
            t = s['total'] or 1
            result[game][player] = {
                **s,
                'win_rate': s['wins'] / t,
                'loss_rate': s['losses'] / t,
                'draw_rate': s['draws'] / t,
            }
    return result


# ---- Metric 2: Illegal move rate -------------------------------------------

def compute_illegal_rates(rows: List[dict]) -> dict:
    """Returns player_name → {total_matches, total_illegal_moves, illegal_rate}."""
    stats = defaultdict(lambda: {'matches': 0, 'illegal': 0})
    for r in rows:
        stats[r['player_x']]['matches'] += 1
        stats[r['player_x']]['illegal'] += int(r.get('illegal_moves_x', 0))
        stats[r['player_o']]['matches'] += 1
        stats[r['player_o']]['illegal'] += int(r.get('illegal_moves_o', 0))

    return {p: {**s, 'illegal_rate': s['illegal'] / s['matches'] if s['matches'] else 0}
            for p, s in stats.items()}


# ---- Metric 3: Average latency ---------------------------------------------

def compute_latencies(rows: List[dict]) -> dict:
    """Returns player_name → avg_latency_ms."""
    totals = defaultdict(lambda: {'sum': 0.0, 'count': 0})
    for r in rows:
        lx = float(r.get('avg_latency_x_ms', 0))
        lo = float(r.get('avg_latency_o_ms', 0))
        if lx > 0:
            totals[r['player_x']]['sum'] += lx
            totals[r['player_x']]['count'] += 1
        if lo > 0:
            totals[r['player_o']]['sum'] += lo
            totals[r['player_o']]['count'] += 1

    return {p: round(s['sum'] / s['count'], 1) if s['count'] else 0
            for p, s in totals.items()}


# ---- Metric 4: Misère gap --------------------------------------------------

MISERE_PAIRS = {
    'tictactoe': 'suicide',
    'connectfour': 'notconnectfour',
    # Add more pairs as needed
}


def compute_misere_gap(rows: List[dict]) -> dict:
    """
    For each player and misère pair (normal, inverted), compute:
      misere_gap = win_rate(normal) - win_rate(inverted)
    A high gap suggests memorization; a low gap suggests genuine reasoning.
    """
    win_rates = compute_win_rates(rows)
    gaps = {}

    for normal, inverted in MISERE_PAIRS.items():
        if normal not in win_rates or inverted not in win_rates:
            continue
        players = set(win_rates[normal].keys()) & set(win_rates[inverted].keys())
        for p in players:
            wr_normal = win_rates[normal][p]['win_rate']
            wr_inverted = win_rates[inverted][p]['win_rate']
            gap = wr_normal - wr_inverted
            gaps[(p, normal, inverted)] = {
                'player': p,
                'normal_game': normal,
                'inverted_game': inverted,
                'win_rate_normal': round(wr_normal, 3),
                'win_rate_inverted': round(wr_inverted, 3),
                'misere_gap': round(gap, 3),
            }
    return gaps


# ---- Metric 5: Bootstrap confidence intervals ------------------------------

def bootstrap_win_rate(wins: int, total: int, n_bootstrap: int = 10000,
                       ci: float = 0.95) -> Tuple[float, float, float]:
    """Returns (mean, lower_ci, upper_ci) for win rate."""
    if total == 0:
        return 0.0, 0.0, 0.0
    outcomes = [1] * wins + [0] * (total - wins)
    means = []
    for _ in range(n_bootstrap):
        sample = random.choices(outcomes, k=total)
        means.append(sum(sample) / total)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return round(sum(means) / len(means), 3), round(lo, 3), round(hi, 3)


# ---- Metric 6: Head-to-head matrix -----------------------------------------

def compute_h2h_matrix(rows: List[dict]) -> dict:
    """
    Returns game → {(player_a, player_b): {'a_wins': X, 'b_wins': Y, 'draws': Z, 'total': N}}
    """
    h2h = defaultdict(lambda: defaultdict(lambda: {
        'a_wins': 0, 'b_wins': 0, 'draws': 0, 'total': 0}))

    for r in rows:
        game = r['game']
        px, po = r['player_x'], r['player_o']
        winner = r['winner']
        key = tuple(sorted([px, po]))

        h2h[game][key]['total'] += 1
        if winner == 'draw':
            h2h[game][key]['draws'] += 1
        elif winner == 'x':
            if px == key[0]:
                h2h[game][key]['a_wins'] += 1
            else:
                h2h[game][key]['b_wins'] += 1
        elif winner == 'o':
            if po == key[0]:
                h2h[game][key]['a_wins'] += 1
            else:
                h2h[game][key]['b_wins'] += 1

    return h2h


# ---- Pretty printing -------------------------------------------------------

def print_report(rows: List[dict]):
    print("=" * 70)
    print("EXPERIMENT ANALYSIS REPORT")
    print(f"Total matches: {len(rows)}")
    print("=" * 70)

    # Win rates
    wr = compute_win_rates(rows)
    print("\n--- Win Rates (aggregated across roles) ---")
    for game in sorted(wr.keys()):
        print(f"\n  Game: {game}")
        for player in sorted(wr[game].keys()):
            s = wr[game][player]
            ci_mean, ci_lo, ci_hi = bootstrap_win_rate(s['wins'], s['total'])
            print(f"    {player:20s}: "
                  f"W={s['wins']:3d} D={s['draws']:3d} L={s['losses']:3d} "
                  f"({s['total']:3d} games) "
                  f"WR={s['win_rate']:.1%} "
                  f"[{ci_lo:.1%}, {ci_hi:.1%}] 95% CI")

    # Illegal move rates
    il = compute_illegal_rates(rows)
    llm_illegals = {p: s for p, s in il.items() if s['illegal'] > 0}
    if llm_illegals:
        print("\n--- Illegal Move Rates ---")
        for player, s in sorted(llm_illegals.items()):
            print(f"    {player:20s}: "
                  f"{s['illegal']:3d} illegal moves in {s['matches']} matches "
                  f"({s['illegal_rate']:.2f} per match)")

    # Latencies
    lat = compute_latencies(rows)
    print("\n--- Average Latency (ms) ---")
    for player in sorted(lat.keys()):
        print(f"    {player:20s}: {lat[player]:.1f} ms")

    # Misère gap
    mg = compute_misere_gap(rows)
    if mg:
        print("\n--- Misère Gap (normal win_rate - inverted win_rate) ---")
        for key, g in sorted(mg.items()):
            print(f"    {g['player']:20s}: "
                  f"normal({g['normal_game']})={g['win_rate_normal']:.1%}  "
                  f"inverted({g['inverted_game']})={g['win_rate_inverted']:.1%}  "
                  f"gap={g['misere_gap']:+.1%}")

    # Head-to-head
    h2h = compute_h2h_matrix(rows)
    print("\n--- Head-to-Head ---")
    for game in sorted(h2h.keys()):
        print(f"\n  Game: {game}")
        for (a, b), s in sorted(h2h[game].items()):
            print(f"    {a} vs {b}: "
                  f"{a} wins {s['a_wins']}, "
                  f"{b} wins {s['b_wins']}, "
                  f"draws {s['draws']} "
                  f"(of {s['total']})")

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        # Find most recent results
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        csvs = sorted([os.path.join(results_dir, f)
                        for f in os.listdir(results_dir) if f.endswith('.csv')])
        if not csvs:
            print("No results found. Run experiment_runner.py first.")
            return
        paths = [csvs[-1]]
        print(f"Using most recent results: {paths[0]}")
    else:
        paths = sys.argv[1:]

    rows = load_csv(paths)
    print_report(rows)


if __name__ == '__main__':
    main()
