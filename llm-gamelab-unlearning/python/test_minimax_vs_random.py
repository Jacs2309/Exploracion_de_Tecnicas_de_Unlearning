"""Run minimax vs random on tic-tac-toe. Expected: minimax never loses."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arbiter_client import ArbiterClient
from players import GGPPlayer
from match_orchestrator import MatchOrchestrator


def run_one_match(game, x_algo, o_algo, match_idx):
    with ArbiterClient(game) as arbiter:
        p_x = GGPPlayer(x_algo, game)
        p_o = GGPPlayer(o_algo, game)
        try:
            orch = MatchOrchestrator(
                game=game, arbiter=arbiter,
                players={'x': p_x, 'o': p_o},
                verbose=False,
            )
            result = orch.run()
            return result
        finally:
            p_x.close()
            p_o.close()


def main():
    game = 'tictactoe'
    n_matches = 5
    results = []
    print(f"Running {n_matches} matches: minimax (x) vs random (o) on {game}\n")
    for i in range(n_matches):
        r = run_one_match(game, 'minimax', 'random', i)
        results.append(r)
        print(f"  Match {i+1}: {r.status:10s} winner={r.winner or 'draw':5s} "
              f"rewards={r.rewards} turns={r.total_turns} "
              f"dur={r.total_duration_s:.2f}s")

    wins_x = sum(1 for r in results if r.winner == 'x')
    wins_o = sum(1 for r in results if r.winner == 'o')
    draws = sum(1 for r in results if r.winner is None
                and r.status == 'terminal')
    print(f"\nSummary (minimax as x, random as o):")
    print(f"  x (minimax) wins: {wins_x}/{n_matches}")
    print(f"  o (random)  wins: {wins_o}/{n_matches}")
    print(f"  draws:            {draws}/{n_matches}")
    print(f"  (minimax should never lose)")


if __name__ == '__main__':
    main()
