"""
Validate all GGP players work against random on tic-tac-toe.
Expected rankings (approximate):
  - minimax >= greedy >= mcs > random >= legal
  - minimax should never lose (optimal)
  - legal is usually worse than random (deterministic dumb player)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arbiter_client import ArbiterClient
from players import GGPPlayer
from match_orchestrator import MatchOrchestrator


def run_match(game, x_algo, o_algo, playclock=3):
    """Run one match and return the result."""
    with ArbiterClient(game) as arbiter:
        p_x = GGPPlayer(x_algo, game)
        p_o = GGPPlayer(o_algo, game)
        try:
            # Override playclock for faster tests
            orch = MatchOrchestrator(
                game=game, arbiter=arbiter,
                players={'x': p_x, 'o': p_o},
                verbose=False,
            )
            return orch.run()
        finally:
            p_x.close()
            p_o.close()


def main():
    game = 'tictactoe'
    algorithms = ['random', 'legal', 'mcs', 'greedy', 'minimax']

    print(f"Testing all GGP players on {game}")
    print(f"Each algorithm plays 3 matches vs random (as x)\n")

    for algo in algorithms:
        wins, draws, losses = 0, 0, 0
        total_turns = 0
        for i in range(3):
            r = run_match(game, algo, 'random')
            total_turns += r.total_turns
            if r.winner == 'x':
                wins += 1
            elif r.winner == 'o':
                losses += 1
            else:
                draws += 1
        print(f"  {algo:10s} as x vs random: "
              f"W={wins} D={draws} L={losses}  "
              f"avg_turns={total_turns/3:.1f}")


if __name__ == '__main__':
    main()
