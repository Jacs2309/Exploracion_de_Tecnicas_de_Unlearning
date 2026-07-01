"""
experiment_runner.py - Batch experiment runner for NeurIPS paper.

Runs a matrix of: game × player_x × player_o × n_matches
Supports both GGP and LLM players. Results are saved incrementally
so crashed runs can be resumed.

Usage:
    python experiment_runner.py --config experiments.json
    python experiment_runner.py --quick   # quick smoke test

Config format (experiments.json):
{
  "games": ["tictactoe", "suicide", "connectfour"],
  "ggp_opponents": ["random", "minimax", "mcs"],
  "llm_models": [
    {"model": "gpt-4o", "provider": "openai"},
    {"model": "o3-mini", "provider": "openai"},
    {"model": "deepseek/deepseek-r1:free", "provider": "openrouter"}
  ],
  "n_matches_per_config": 10,
  "playclock": 10,
  "startclock": 10,
  "max_retries": 10,
  "output_dir": "./results",
  "log_dir": "./logs"
}
"""

import sys, os, json, time, argparse, csv
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Dict, List, Optional, Any

# Load .env from project root (one level above python/)
from dotenv import load_dotenv
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
load_dotenv(os.path.join(_project_root, '.env'))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arbiter_client import ArbiterClient
from players import GGPPlayer, LLMPlayer, Player
from match_orchestrator import MatchOrchestrator, MatchResult, create_db_backend


# ---- Rules prompts per game (natural language for LLMs) --------------------
# These can be customized or auto-generated. Kept here for simplicity.
RULES_PROMPTS = {
    "tictactoe": """
Game: Tic-Tac-Toe on a 3x3 board.
Players: "x" and "o". The board has cells at (row, column) with rows 1-3, columns 1-3.
Each cell is "x", "o", or empty ("b"). Mark an empty cell on your turn.
Win by completing a row, column, or diagonal. Full board with no winner = draw.
""",
    "suicide": """
Game: Suicide Tic-Tac-Toe (Misère) on a 3x3 board.
Players: "x" and "o". Same mechanics as Tic-Tac-Toe EXCEPT the goal is INVERTED:
you LOSE if you complete a row, column, or diagonal with your symbol.
Avoid making three in a row. Force your opponent into it.
""",
    "connectfour": """
Game: Connect Four on a 6-row × 8-column board.
Players: "red" and "black". Drop a piece into a column; it falls to the lowest empty row.
Win by connecting 4 in a row (horizontal, vertical, or diagonal).
""",
    "notconnectfour": """
Game: Not Connect Four (Misère) on a 6-row × 8-column board.
Players: "red" and "black". Same mechanics as Connect Four EXCEPT:
If you connect 4 of your pieces in a row, you LOSE.
Avoid making a line of 4. Force your opponent into making one.
""",
    "breakthrough": """
Game: Breakthrough on an 8x8 board.
Players: "white" and "black". Each starts with 2 rows of pawns.
Pawns move forward one square: straight ahead or diagonally forward.
Capture is diagonal forward only (like chess pawns).
Win by reaching the opponent's back row with any pawn.
""",
    "hex7x7": """
Game: Hex on a 7x7 rhombus-shaped board.
Players: "white" and "black". Place one stone per turn on an empty cell.
White connects top-to-bottom; Black connects left-to-right.
No draws possible.
""",
    "lines": """
Game: Lines on a hexagonal board of 55 playable cells (9 rows × 9 cols with gaps).
Players: "red" and "blue". Place one stone per turn on any empty cell.
The board has 27 lines: 9 letter-rows, 9 number-columns, 9 vertical diagonals.
Each line has a majority threshold. You win a line by placing more stones than
your opponent in it (meeting the majority requirement).
Your score is proportional to the number of lines you control. First to 15 lines wins.
""",
    "centralis": """
Game: Centralis on a radial board with 16 sections, 3 rings, and 1 center cell.
Players: "x" and "o". Place your mark on any empty peripheral cell (section 1-16, ring 1-3).
The CENTER cell (c,c) is LOCKED at the start. To unlock it, you must form a valid path:
three of your marks, one in each ring (1,2,3), aligned either:
  - Radially: same section number across all 3 rings (e.g. section 5 rings 1,2,3)
  - Diagonally clockwise: consecutive sections stepping outward (e.g. sec 3 ring 1, sec 2 ring 2, sec 1 ring 3)
  - Diagonally counter-clockwise: the reverse pattern
First player to legally occupy the center WINS. If the peripheral board fills
with no one unlocking the center, it is a DRAW.
""",
}


def get_rules_prompt(game: str) -> str:
    if game in RULES_PROMPTS:
        return RULES_PROMPTS[game]
    return f"Game: {game}. Follow the rules and choose legal moves."


# ---- Experiment config ------------------------------------------------------

class ExperimentConfig:
    def __init__(self, config_dict: dict):
        self.games = config_dict.get('games', ['tictactoe'])
        self.ggp_opponents = config_dict.get('ggp_opponents', ['random', 'minimax'])
        self.llm_models = config_dict.get('llm_models', [])
        self.n_matches = config_dict.get('n_matches_per_config', 5)
        self.playclock = config_dict.get('playclock', 10)
        self.startclock = config_dict.get('startclock', 10)
        self.max_retries = config_dict.get('max_retries', 10)
        self.output_dir = config_dict.get('output_dir', './results')
        self.log_dir = config_dict.get('log_dir', './logs')
        self.ggp_vs_ggp = config_dict.get('ggp_vs_ggp', True)
        self.swap_roles = config_dict.get('swap_roles', True)
        self.use_mongo = config_dict.get('use_mongo', False)
        self.mongo_uri = config_dict.get('mongo_uri', 'mongodb://localhost:27017/')
        self.db_name = config_dict.get('db_name', 'db_games')
        # For LLMs: whether to show legal moves in prompt (False = test reasoning)
        self.show_legals = config_dict.get('show_legals', False)


# ---- Results tracking -------------------------------------------------------

class ResultsTracker:
    """Incrementally writes match results to a CSV + JSONL pair."""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(output_dir, f'results_{ts}.csv')
        self.jsonl_path = os.path.join(output_dir, f'results_{ts}.jsonl')

        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'match_id', 'game', 'player_x', 'kind_x', 'player_o', 'kind_o',
            'winner', 'reward_x', 'reward_o', 'status', 'total_turns',
            'duration_s', 'forfeit_by',
        ])

        self.jsonl_file = open(self.jsonl_path, 'w', encoding='utf-8')

    def record(self, result: MatchResult):
        roles = list(result.players.keys())
        r_x = roles[0] if roles else 'x'
        r_o = roles[1] if len(roles) > 1 else 'o'

        self.csv_writer.writerow([
            result.match_id, result.game,
            result.players.get(r_x, ''), result.player_kinds.get(r_x, ''),
            result.players.get(r_o, ''), result.player_kinds.get(r_o, ''),
            result.winner or 'draw',
            result.rewards.get(r_x, 0), result.rewards.get(r_o, 0),
            result.status, result.total_turns, f"{result.total_duration_s:.2f}",
            result.forfeit_by or '',
        ])
        self.csv_file.flush()

        compact = {
            'match_id': result.match_id,
            'game': result.game,
            'players': result.players,
            'player_kinds': result.player_kinds,
            'rewards': result.rewards,
            'status': result.status,
            'winner': result.winner,
            'forfeit_by': result.forfeit_by,
            'total_turns': result.total_turns,
            'total_duration_s': result.total_duration_s,
            'started_at': result.started_at,
            'ended_at': result.ended_at,
        }
        self.jsonl_file.write(json.dumps(compact) + '\n')
        self.jsonl_file.flush()

    def close(self):
        self.csv_file.close()
        self.jsonl_file.close()
        print(f"\nResults saved to:\n  CSV:   {self.csv_path}\n  JSONL: {self.jsonl_path}")


# ---- Main experiment loop ---------------------------------------------------

def run_single_match(game: str, player_x: Player, player_o: Player,
                     db: Optional[Any] = None, match_num: int = 0,
                     verbose: bool = False) -> MatchResult:
    """Run one match between two already-constructed players."""
    with ArbiterClient(game) as arbiter:
        roles = arbiter.roles()
        role_x = roles[0]
        role_o = roles[1] if len(roles) > 1 else roles[0]

        orch = MatchOrchestrator(
            game=game, arbiter=arbiter,
            players={role_x: player_x, role_o: player_o},
            db=db, match_num=match_num, verbose=verbose,
        )
        return orch.run()


def make_ggp_player(algo: str, game: str, seed: Optional[int] = None) -> GGPPlayer:
    return GGPPlayer(algo, game, seed=seed)


def run_ggp_vs_ggp_experiments(config: ExperimentConfig, tracker: ResultsTracker,
                                db: Optional[Any] = None):
    """Run all GGP-vs-GGP pairings."""
    if not config.ggp_vs_ggp:
        return

    algos = config.ggp_opponents
    total = len(config.games) * len(algos) * (len(algos) - 1) * config.n_matches
    done = 0

    print(f"\n{'='*60}")
    print(f"GGP vs GGP experiments: {total} matches total")
    print(f"{'='*60}\n")

    for game in config.games:
        for algo_x in algos:
            for algo_o in algos:
                if algo_x == algo_o:
                    continue
                for match_i in range(config.n_matches):
                    done += 1
                    label = (f"[{done}/{total}] {game}: "
                             f"{algo_x} vs {algo_o} (#{match_i+1})")
                    print(f"{label}...", end=" ", flush=True)

                    p_x = make_ggp_player(algo_x, game, seed=match_i)
                    p_o = make_ggp_player(algo_o, game, seed=match_i + 10000)
                    try:
                        result = run_single_match(
                            game, p_x, p_o, db=db,
                            match_num=match_i, verbose=False)
                        tracker.record(result)
                        w = result.winner or 'draw'
                        print(f"{result.status} w={w} "
                              f"t={result.total_turns} "
                              f"{result.total_duration_s:.1f}s")
                    except Exception as e:
                        print(f"ERROR: {e}")
                    finally:
                        p_x.close()
                        p_o.close()


def run_llm_vs_ggp_experiments(config: ExperimentConfig, tracker: ResultsTracker,
                                db: Optional[Any] = None):
    """Run all LLM-vs-GGP pairings."""
    if not config.llm_models:
        print("No LLM models configured; skipping LLM experiments.")
        return

    from openai import OpenAI

    clients = {}
    for m in config.llm_models:
        provider = m.get('provider', 'openai')
        if provider not in clients:
            if provider == 'openai':
                clients[provider] = OpenAI(
                    api_key=os.getenv('API_KEY_OPENAI'))
            elif provider == 'openrouter':
                clients[provider] = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv('API_KEY_OPENROUTER'))
            elif provider == 'google':
                clients[provider] = OpenAI(
                    api_key=os.getenv('API_KEY_GOOGLE'),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    done = 0

    print(f"\n{'='*60}")
    print(f"LLM vs GGP experiments")
    print(f"{'='*60}\n")

    for game in config.games:
        for m_cfg in config.llm_models:
            model = m_cfg['model']
            provider = m_cfg.get('provider', 'openai')
            client = clients[provider]

            for algo in config.ggp_opponents:
                roles_configs = [('llm_first', True)]
                if config.swap_roles:
                    roles_configs.append(('ggp_first', False))

                for role_label, llm_is_x in roles_configs:
                    for match_i in range(config.n_matches):
                        done += 1
                        label = (f"[{done}] {game}: "
                                 f"{model} vs {algo} ({role_label} #{match_i+1})")
                        print(f"{label}...", end=" ", flush=True)

                        llm = LLMPlayer(
                            model=model, client=client,
                            game_name=game,
                            rules_prompt=get_rules_prompt(game),
                            max_retries=config.max_retries,
                        )
                        ggp = make_ggp_player(algo, game, seed=match_i)

                        try:
                            if llm_is_x:
                                result = run_single_match(
                                    game, llm, ggp, db=db,
                                    match_num=match_i)
                            else:
                                result = run_single_match(
                                    game, ggp, llm, db=db,
                                    match_num=match_i)
                            tracker.record(result)
                            w = result.winner or 'draw'
                            print(f"{result.status} w={w} "
                                  f"t={result.total_turns} "
                                  f"{result.total_duration_s:.1f}s")
                        except Exception as e:
                            print(f"ERROR: {e}")
                        finally:
                            ggp.close()


# ---- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run batch experiments')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to experiments.json config file')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test: 2 games, 2 algos, 1 match each')
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
    elif args.quick:
        cfg_dict = {
            'games': ['tictactoe'],
            'ggp_opponents': ['random', 'minimax'],
            'llm_models': [],
            'n_matches_per_config': 2,
            'ggp_vs_ggp': True,
            'swap_roles': False,
            'output_dir': '../results',
            'log_dir': '../logs',
        }
    else:
        cfg_dict = {
            'games': ['tictactoe', 'suicide'],
            'ggp_opponents': ['random', 'minimax', 'mcs'],
            'llm_models': [],
            'n_matches_per_config': 3,
            'output_dir': '../results',
            'log_dir': '../logs',
        }

    config = ExperimentConfig(cfg_dict)
    tracker = ResultsTracker(config.output_dir)
    db = create_db_backend(
        use_mongo=config.use_mongo,
        mongo_uri=config.mongo_uri,
        db_name=config.db_name,
        output_dir=config.output_dir,
    )

    try:
        run_ggp_vs_ggp_experiments(config, tracker, db=db)
        run_llm_vs_ggp_experiments(config, tracker, db=db)
    finally:
        tracker.close()
        db.close()


if __name__ == '__main__':
    main()
