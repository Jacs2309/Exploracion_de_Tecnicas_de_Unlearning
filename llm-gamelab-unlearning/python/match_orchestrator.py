"""
match_orchestrator.py

File naming: {moves|matches}_ID_GAME_P1short_P2short.jsonl
Board info: natural language + raw GDL state
Token tracking: accumulated per player in MatchResult
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from arbiter_client import ArbiterClient
from players import Player, MoveRecord


# ---------------------------------------------------------------------------
# Board renderers: GDL state → human-readable string per game
# ---------------------------------------------------------------------------

def render_board_natural(state: list, game: str) -> str:
    renderer = BOARD_RENDERERS.get(game)
    if renderer:
        return renderer(state)
    return _render_generic(state)


def _render_tictactoe(state: list) -> str:
    grid = [['.' for _ in range(3)] for _ in range(3)]
    control = '?'
    for fact in state:
        if fact[0] == 'cell' and len(fact) == 4:
            r, c, v = int(fact[1]) - 1, int(fact[2]) - 1, fact[3]
            if v != 'b':
                grid[r][c] = v.upper()
        elif fact[0] == 'control':
            control = fact[1]
    lines = [f"  1   2   3"]
    for i, row in enumerate(grid):
        lines.append(f"{i+1} {' | '.join(row)}")
        if i < 2:
            lines.append("  -----------")
    lines.append(f"Turn: {control}")
    return "\n".join(lines)


def _render_connectfour(state: list) -> str:
    grid = {}
    control = '?'
    for fact in state:
        if fact[0] == 'cell' and len(fact) == 4:
            col, row, player = fact[1], fact[2], fact[3]
            grid[(int(col), int(row))] = player[0].upper()
        elif fact[0] == 'control':
            control = fact[1]
    lines = [f"Columns: 1  2  3  4  5  6  7  8"]
    for row in range(6, 0, -1):
        cells = [grid.get((col, row), '.') for col in range(1, 9)]
        lines.append(f"     {'  '.join(cells)}")
    lines.append(f"Turn: {control}")
    return "\n".join(lines)


def _render_breakthrough(state: list) -> str:
    grid = [['.' for _ in range(8)] for _ in range(8)]
    control = '?'
    for fact in state:
        if fact[0] == 'cell' and len(fact) == 4:
            r, c, v = int(fact[1]) - 1, int(fact[2]) - 1, fact[3]
            if v == 'white':
                grid[r][c] = 'W'
            elif v == 'black':
                grid[r][c] = 'B'
        elif fact[0] == 'control':
            control = fact[1]
    lines = [f"    1  2  3  4  5  6  7  8"]
    for i, row in enumerate(grid):
        lines.append(f" {i+1}  {'  '.join(row)}")
    lines.append(f"Turn: {control}")
    return "\n".join(lines)


def _render_centralis(state: list) -> str:
    cells = {}
    control = '?'
    for fact in state:
        if fact[0] == 'cell' and len(fact) == 4:
            s, r, v = fact[1], fact[2], fact[3]
            if v != 'b':
                cells[(s, r)] = v
        elif fact[0] == 'control':
            control = fact[1]
    lines = ["Radial board: 16 sections × 3 rings + center"]
    lines.append(f"  Ring 3 (outer) | Ring 2 (middle) | Ring 1 (inner)")
    center = cells.get(('c', 'c'), 'empty')
    for sec in range(1, 17):
        s = str(sec)
        r3 = cells.get((s, '3'), '.')
        r2 = cells.get((s, '2'), '.')
        r1 = cells.get((s, '1'), '.')
        lines.append(f"  Sec {sec:2d}:   {r3:5s}     |     {r2:5s}     |     {r1:5s}")
    lines.append(f"  Center: {center}")
    lines.append(f"Turn: {control}")
    return "\n".join(lines)


def _render_lines(state: list) -> str:
    cells = {}
    control = '?'
    for fact in state:
        if fact[0] == 'cell' and len(fact) == 4:
            r, c, v = fact[1], fact[2], fact[3]
            cells[(r, c)] = v[0].upper()
        elif fact[0] == 'control':
            control = fact[1]
    rows_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    cols_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    lines = ["Hexagonal board (. = empty):"]
    lines.append("    " + "  ".join(cols_list))
    for r in rows_list:
        row_cells = [cells.get((r, c), '.') for c in cols_list]
        lines.append(f" {r}  {'  '.join(row_cells)}")
    lines.append(f"Turn: {control}")
    return "\n".join(lines)


def _render_hex(state: list) -> str:
    cells = {}
    control = '?'
    for fact in state:
        if fact[0] == 'cell' and len(fact) == 4:
            r, c, v = fact[1], fact[2], fact[3]
            cells[(r, c)] = v[0].upper() if v not in ('b',) else None
        elif fact[0] == 'control':
            control = fact[1]
    hex_rows = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    hex_cols = ['1', '2', '3', '4', '5', '6', '7']
    lines = ["7x7 Hex board (Red: top-bottom, Black: left-right):"]
    header = "     " + "  ".join(hex_cols)
    lines.append(header)
    for i, r in enumerate(hex_rows):
        indent = " " * i
        row_cells = [cells.get((r, c)) or '.' for c in hex_cols]
        lines.append(f"{indent}  {r}  {'  '.join(row_cells)}")
    lines.append(f"Turn: {control}")
    return "\n".join(lines)


def _render_generic(state: list) -> str:
    lines = []
    control = '?'
    for fact in state:
        if fact[0] == 'control':
            control = fact[1]
        else:
            lines.append(json.dumps(fact))
    lines.append(f"Turn: {control}")
    return "\n".join(lines)


BOARD_RENDERERS = {
    'tictactoe': _render_tictactoe,
    'suicide': _render_tictactoe,
    'connectfour': _render_connectfour,
    'notconnectfour': _render_connectfour,
    'breakthrough': _render_breakthrough,
    'centralis': _render_centralis,
    'lines': _render_lines,
    'hex7x7': _render_hex,
}


# ---------------------------------------------------------------------------
# Short names for file naming
# ---------------------------------------------------------------------------

def short_name(name: str) -> str:
    replacements = {
        '/': '-', ':': '', ' ': '_',
        'qwen/': '', 'deepseek/': '', 'meta-llama/': '',
        ':free': '', '-instruct': '',
    }
    s = name.lower()
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s[:20]


# ---------------------------------------------------------------------------
# Database backend
# ---------------------------------------------------------------------------

class DatabaseBackend:
    def save_move(self, doc: dict): ...
    def save_match(self, doc: dict): ...
    def open_match(self, match_id, game, p1, p2): ...
    def close(self): ...


class MongoBackend(DatabaseBackend):
    def __init__(self, connection_string='mongodb://localhost:27017/',
                 db_name='db_games'):
        from pymongo import MongoClient
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.moves_col = self.db['games']
        self.matches_col = self.db['matches']

    def open_match(self, match_id, game, p1, p2):
        pass

    def save_move(self, doc):
        self.moves_col.insert_one(doc)

    def save_match(self, doc):
        self.matches_col.insert_one(doc)

    def close(self):
        self.client.close()


class JSONFileBackend(DatabaseBackend):
    """Naming: {moves|matches}_ID_GAME_P1short_P2short.jsonl"""

    def __init__(self, output_dir='./results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._moves_file = None
        self._matches_file = None

    def open_match(self, match_id: str, game: str,
                   p1_name: str, p2_name: str):
        self._close_files()
        short_id = match_id.split('-')[0]
        p1 = short_name(p1_name)
        p2 = short_name(p2_name)
        base = f"{short_id}_{game}_{p1}_vs_{p2}"
        moves_path = os.path.join(self.output_dir, f"moves_{base}.jsonl")
        matches_path = os.path.join(self.output_dir, f"matches_{base}.jsonl")
        self._moves_file = open(moves_path, 'w', encoding='utf-8')
        self._matches_file = open(matches_path, 'w', encoding='utf-8')

    def save_move(self, doc):
        if self._moves_file:
            self._moves_file.write(json.dumps(doc, default=str) + '\n')
            self._moves_file.flush()

    def save_match(self, doc):
        if self._matches_file:
            self._matches_file.write(json.dumps(doc, default=str) + '\n')
            self._matches_file.flush()

    def _close_files(self):
        if self._moves_file:
            self._moves_file.close()
            self._moves_file = None
        if self._matches_file:
            self._matches_file.close()
            self._matches_file = None

    def close(self):
        self._close_files()


def create_db_backend(use_mongo=False, mongo_uri='mongodb://localhost:27017/',
                      db_name='db_games', output_dir='./results'):
    if use_mongo:
        try:
            backend = MongoBackend(mongo_uri, db_name)
            backend.client.server_info()
            print("  Connected to MongoDB")
            return backend
        except Exception as e:
            print(f"  MongoDB unavailable ({e}), falling back to JSON files")
    return JSONFileBackend(output_dir)


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    match_id: str
    game: str
    players: Dict[str, str]
    player_kinds: Dict[str, str]
    rewards: Dict[str, int]
    status: str
    winner: Optional[str]
    forfeit_by: Optional[str] = None
    total_turns: int = 0
    total_duration_s: float = 0.0
    started_at: str = ''
    ended_at: str = ''
    # Token usage per role (populated for LLM players)
    tokens: Dict[str, Dict[str, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Match Orchestrator
# ---------------------------------------------------------------------------

class MatchOrchestrator:

    def __init__(self, game: str, arbiter: ArbiterClient,
                 players: Dict[str, Player],
                 db: Optional[DatabaseBackend] = None,
                 max_turns: int = 200,
                 verbose: bool = True):
        self.game = game
        self.arbiter = arbiter
        self.players = players
        self.db = db
        self.max_turns = max_turns
        self.match_id = str(uuid.uuid1())
        self.verbose = verbose
        # Track tokens per role during the match
        self._tokens: Dict[str, Dict[str, int]] = {}
        for role in players:
            self._tokens[role] = {'prompt': 0, 'completion': 0, 'total': 0}

    def run(self) -> MatchResult:
        start_wall = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        for role, player in self.players.items():
            player.start(role=role)

        state = self.arbiter.init_state()
        roles = self.arbiter.roles()

        # Open DB files
        role_list = list(self.players.keys())
        p1_name = self.players[role_list[0]].name if role_list else ''
        p2_name = self.players[role_list[1]].name if len(role_list) > 1 else ''
        if self.db:
            self.db.open_match(self.match_id, self.game, p1_name, p2_name)

        result = MatchResult(
            match_id=self.match_id, game=self.game,
            players={r: p.name for r, p in self.players.items()},
            player_kinds={r: p.kind for r, p in self.players.items()},
            rewards={r: 0 for r in roles},
            status='in_progress', winner=None,
            started_at=started_at.isoformat(),
        )

        turn = 0
        last_move = None
        forfeit = False

        try:
            while not self.arbiter.is_terminal(state):
                turn += 1
                if turn > self.max_turns:
                    result.status = 'max_turns'
                    break

                control = self.arbiter.control(state)
                if control is None or control not in self.players:
                    result.status = 'error'
                    break

                active = self.players[control]
                legal_moves = self.arbiter.legals(state)

                if self.verbose:
                    print(f"[turn {turn}] {control} ({active.name}) "
                          f"thinking... ({len(legal_moves)} legal moves)")

                # Notify non-active GGP players
                for role, p in self.players.items():
                    if role != control and p.kind == 'ggp' and last_move is not None:
                        try:
                            p.play(last_move, state, legal_moves)
                        except Exception:
                            pass

                # Inject natural language board for LLM players
                if active.kind == 'llm':
                    active._board_natural = render_board_natural(state, self.game)

                move_rec = active.play(last_move, state, legal_moves)

                # Accumulate tokens
                self._tokens[control]['prompt'] += move_rec.prompt_tokens
                self._tokens[control]['completion'] += move_rec.completion_tokens
                self._tokens[control]['total'] += move_rec.total_tokens

                if not move_rec.valid:
                    if self.verbose:
                        print(f"  !! {control} forfeits: {move_rec.error}")
                    forfeit = True
                    result.status = 'forfeit'
                    result.forfeit_by = control
                    self._save_move(turn, control, active, state, legal_moves,
                                    move_rec, win=0)
                    break

                new_state = self.arbiter.simulate(state, move_rec.move)
                is_terminal = self.arbiter.is_terminal(new_state)

                self._save_move(turn, control, active, state, legal_moves,
                                move_rec, win=1 if is_terminal else 0)

                last_move = move_rec.move
                state = new_state

                if self.verbose:
                    tok = ""
                    if move_rec.total_tokens > 0:
                        tok = (f" [{move_rec.prompt_tokens}in/"
                               f"{move_rec.completion_tokens}out]")
                    print(f"  -> {move_rec.move}  "
                          f"(valid, {move_rec.latency_ms:.0f}ms{tok})")

            # Final rewards
            if not forfeit:
                result.status = 'terminal' if turn <= self.max_turns else 'max_turns'
            if forfeit:
                for r in roles:
                    result.rewards[r] = 0 if r == result.forfeit_by else 100
            else:
                for r in roles:
                    result.rewards[r] = self.arbiter.reward(state, r)

            max_rew = max(result.rewards.values()) if result.rewards else 0
            winners = [r for r, v in result.rewards.items() if v == max_rew]
            if len(winners) == 1 and max_rew > 0:
                result.winner = winners[0]

            for role, p in self.players.items():
                try:
                    p.stop(last_move)
                except Exception:
                    pass

        finally:
            result.total_turns = turn
            result.total_duration_s = time.perf_counter() - start_wall
            result.ended_at = datetime.now(timezone.utc).isoformat()
            result.tokens = dict(self._tokens)
            self._save_match(result, roles)

            if self.verbose:
                print(f"\n=== Match {self.match_id[:8]} ended: {result.status} ===")
                print(f"  Rewards: {result.rewards}")
                print(f"  Winner: {result.winner or 'draw'}")
                print(f"  Duration: {result.total_duration_s:.1f}s, "
                      f"{result.total_turns} turns")
                # Show tokens if any
                for role, tk in self._tokens.items():
                    if tk['total'] > 0:
                        print(f"  Tokens {role} ({self.players[role].name}): "
                              f"{tk['prompt']} in / {tk['completion']} out "
                              f"= {tk['total']} total")

        return result

    def _save_move(self, turn, role, player, state, legal_moves,
                   move_rec, win):
        if self.db is None:
            return
        doc = {
            'id_match': self.match_id,
            'turn_number': turn,
            'legalMoves': legal_moves,
            'board': state,
            'board_natural': render_board_natural(state, self.game),
            'move': move_rec.move,
            'valid': 1 if move_rec.valid else 0,
            'win': win,
            'player': role,
            'model': player.name,
            'player_kind': player.kind,
            'game': self.game,
            'reason': move_rec.reason,
            'raw_response': move_rec.raw_response,
            'execution_time': round(move_rec.latency_ms / 1000, 4),
            'attempts': move_rec.attempts,
            'illegal_attempts': move_rec.illegal_attempts,
            'prompt_tokens': move_rec.prompt_tokens,
            'completion_tokens': move_rec.completion_tokens,
            'total_tokens': move_rec.total_tokens,
            'timestamp': datetime.now(timezone.utc),
        }
        try:
            self.db.save_move(doc)
        except Exception as e:
            if self.verbose:
                print(f"  [db error: {e}]")

    def _save_match(self, result, roles):
        if self.db is None:
            return
        r1 = roles[0] if roles else ''
        r2 = roles[1] if len(roles) > 1 else ''
        doc = {
            'id_match': result.match_id,
            'player1': result.players.get(r1, ''),
            'player1_kind': result.player_kinds.get(r1, ''),
            'player2': result.players.get(r2, ''),
            'player2_kind': result.player_kinds.get(r2, ''),
            'role': f"{r1} vs {r2}",
            'game': result.game,
            'winner': result.winner or 'draw',
            'rewards': result.rewards,
            'status': result.status,
            'forfeit_by': result.forfeit_by,
            'total_turns': result.total_turns,
            'execution_time': round(result.total_duration_s, 2),
            'tokens': result.tokens,
            'timestamp': datetime.now(timezone.utc),
        }
        try:
            self.db.save_match(doc)
        except Exception as e:
            if self.verbose:
                print(f"  [db error: {e}]")
