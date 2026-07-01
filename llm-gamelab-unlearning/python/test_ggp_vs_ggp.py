"""
Smoke test: run a full match between two GGP random players on Tic-Tac-Toe.
Validates the entire stack and saves results to JSON files (or MongoDB if available).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arbiter_client import ArbiterClient
from players import GGPPlayer
from match_orchestrator import MatchOrchestrator, create_db_backend


def main():
    game = 'tictactoe'

    # Use JSON file backend by default (no MongoDB needed)
    db = create_db_backend(use_mongo=False, output_dir='../results')

    with ArbiterClient(game) as arbiter:
        p_x = GGPPlayer('random', game)
        p_o = GGPPlayer('random', game)

        try:
            orch = MatchOrchestrator(
                game=game,
                arbiter=arbiter,
                players={'x': p_x, 'o': p_o},
                db=db,

                verbose=True,
            )
            result = orch.run()

            print("\n--- MatchResult summary ---")
            print(f"Match ID: {result.match_id}")
            print(f"Status: {result.status}")
            print(f"Winner: {result.winner}")
            print(f"Rewards: {result.rewards}")
            print(f"Total turns: {result.total_turns}")
            print(f"Total duration: {result.total_duration_s:.2f}s")
        finally:
            p_x.close()
            p_o.close()
            db.close()


if __name__ == '__main__':
    main()
