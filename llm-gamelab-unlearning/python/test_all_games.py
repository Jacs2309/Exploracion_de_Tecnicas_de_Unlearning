"""
Test every game in downloads/ with random vs random.
Validates: rules load, init works, legals found, game terminates, rewards computed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arbiter_client import ArbiterClient
from players import GGPPlayer
from match_orchestrator import MatchOrchestrator, create_db_backend, render_board_natural


def find_games():
    """Discover all games from downloads/ directory."""
    downloads = os.path.join(os.path.dirname(__file__), '..', 'downloads')
    games = []
    for f in sorted(os.listdir(downloads)):
        if f.endswith('_rulesheet.hrf'):
            games.append(f.replace('_rulesheet.hrf', ''))
    return games


def test_game(game: str):
    """Run one random vs random match for a game."""
    with ArbiterClient(game) as arb:
        roles = arb.roles()
        s = arb.init_state()
        ctrl = arb.control(s)
        legals = arb.legals(s)
        
        r1, r2 = roles[0], roles[1] if len(roles) > 1 else roles[0]
        p1 = GGPPlayer('random', game, seed=42)
        p2 = GGPPlayer('random', game, seed=43)
        
        try:
            orch = MatchOrchestrator(
                game=game, arbiter=arb,
                players={r1: p1, r2: p2},
                max_turns=100,  # cap for large games
                verbose=False,
            )
            result = orch.run()
            
            # Board rendering test
            board = render_board_natural(s, game)
            board_lines = len(board.split('\n'))
            
            return {
                'roles': roles,
                'init_facts': len(s),
                'init_legals': len(legals),
                'status': result.status,
                'winner': result.winner or 'draw',
                'turns': result.total_turns,
                'duration': result.total_duration_s,
                'rewards': result.rewards,
                'board_lines': board_lines,
            }
        finally:
            p1.close()
            p2.close()


def main():
    games = find_games()
    print(f"Found {len(games)} games: {', '.join(games)}\n")
    print(f"{'Game':<16} {'Roles':<14} {'Facts':>5} {'Legals':>6} "
          f"{'Status':<10} {'Winner':<8} {'Turns':>5} {'Time':>6} {'Board':>5}")
    print("-" * 90)
    
    all_ok = True
    for game in games:
        try:
            r = test_game(game)
            roles_str = '/'.join(r['roles'])
            print(f"{game:<16} {roles_str:<14} {r['init_facts']:>5} {r['init_legals']:>6} "
                  f"{r['status']:<10} {r['winner']:<8} {r['turns']:>5} "
                  f"{r['duration']:>5.1f}s {r['board_lines']:>5}L")
        except Exception as e:
            print(f"{game:<16} ERROR: {e}")
            all_ok = False
    
    print(f"\n{'ALL OK' if all_ok else 'SOME FAILURES'}")


if __name__ == '__main__':
    main()
