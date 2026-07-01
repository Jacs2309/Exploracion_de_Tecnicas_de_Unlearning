"""
ggp_player_client.py - Python client for the Node.js GGP player subprocess.

Spawns a `node ggp_player_runner.js <player> <game>` process and talks JSON
over stdin/stdout. Provides a clean Python API matching the GGP protocol:
  ping, start, play, stop, abort, state, legals, terminal, reward, control.

Usage:
    with GGPPlayerClient('random', 'tictactoe') as p:
        p.start(role='x', startclock=10, playclock=5)
        move = p.play(None)
        while not p.terminal():
            # Orchestrator simulates on its side too; then feeds opponent move
            move = p.play(opponent_move)
"""

import subprocess
import json
import os
import sys
import shutil
import time
from typing import Any, Optional


GGP_PLAYERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'ggp_players')
RUNNER_SCRIPT = 'ggp_player_runner.js'


def _find_node() -> str:
    """Find the Node.js executable, with Windows-friendly fallback."""
    node = shutil.which('node')
    if node:
        return node
    # Windows: try common install paths
    if sys.platform == 'win32':
        for candidate in [
            os.path.join(os.environ.get('ProgramFiles', ''), 'nodejs', 'node.exe'),
            os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'nodejs', 'node.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'nodejs', 'node.exe'),
        ]:
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        "Node.js not found. Install it from https://nodejs.org and make sure "
        "'node' is in your PATH. On Windows, restart your terminal after installing."
    )


class GGPPlayerError(Exception):
    """Raised when the GGP player subprocess returns an error."""


class GGPPlayerClient:
    """
    Client for a single GGP player subprocess.

    The subprocess persists across turns, so the player's internal state
    (library, state, tree, etc.) is preserved between calls — exactly like
    Stanford's GGP protocol over HTTP, but via subprocess IPC instead.
    """

    VALID_PLAYERS = {'random', 'legal', 'minimax', 'mcs', 'greedy', 'maximax'}

    def __init__(self, player_name: str, game_name: str,
                 runner_dir: Optional[str] = None,
                 read_timeout: float = 120.0,
                 seed: Optional[int] = None):
        if player_name not in self.VALID_PLAYERS:
            raise ValueError(f"Unknown player: {player_name}. "
                             f"Valid: {self.VALID_PLAYERS}")
        self.player_name = player_name
        self.game_name = game_name
        self.runner_dir = runner_dir or GGP_PLAYERS_DIR
        self.read_timeout = read_timeout
        self.seed = seed
        self.proc: Optional[subprocess.Popen] = None
        self._loaded = False

    # ------------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------------
    def __enter__(self):
        self.spawn()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def spawn(self):
        """Launch the Node subprocess and wait for the 'runner_loaded' signal."""
        if self.proc is not None:
            return
        cmd = [_find_node(), RUNNER_SCRIPT, self.player_name, self.game_name]
        if self.seed is not None:
            cmd.append(str(self.seed))
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.runner_dir,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Read loaded signal
        line = self._read_line(timeout=self.read_timeout)
        msg = json.loads(line)
        if not msg.get('ok') or msg.get('result') != 'runner_loaded':
            raise GGPPlayerError(f"Runner failed to load: {msg}")
        self._loaded = True

    def close(self):
        """Gracefully terminate the subprocess."""
        if self.proc is None:
            return
        try:
            self._send({'type': 'exit'})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None
        self._loaded = False

    # ------------------------------------------------------------------------
    # Low-level IPC
    # ------------------------------------------------------------------------
    def _send(self, obj: dict):
        if self.proc is None or self.proc.stdin is None:
            raise GGPPlayerError("Subprocess not running")
        line = json.dumps(obj) + "\n"
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except BrokenPipeError:
            stderr = self.proc.stderr.read() if self.proc.stderr else ''
            raise GGPPlayerError(f"Pipe broke. Stderr: {stderr}")

    def _read_line(self, timeout: Optional[float] = None) -> str:
        if self.proc is None or self.proc.stdout is None:
            raise GGPPlayerError("Subprocess not running")
        # Simple blocking read; child writes a line per response
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ''
            raise GGPPlayerError(f"Subprocess closed. Stderr: {stderr}")
        return line.strip()

    def _rpc(self, obj: dict) -> Any:
        self._send(obj)
        line = self._read_line(timeout=self.read_timeout)
        msg = json.loads(line)
        if not msg.get('ok'):
            raise GGPPlayerError(
                f"Player error ({obj.get('type')}): "
                f"{msg.get('error')}\nStack: {msg.get('stack')}")
        return msg.get('result')

    # ------------------------------------------------------------------------
    # GGP protocol
    # ------------------------------------------------------------------------
    def ping(self) -> str:
        return self._rpc({'type': 'ping'})

    def start(self, role: str, startclock: int = 10, playclock: int = 10):
        return self._rpc({'type': 'start', 'role': role,
                          'startclock': startclock, 'playclock': playclock})

    def play(self, move: Any) -> Any:
        """
        Ask the player for its move, passing the previous joint move.
        For turn-based games, `move` is the opponent's last action (or None
        on the first turn). Returns the player's chosen move (a list).
        """
        return self._rpc({'type': 'play',
                          'move': 'nil' if move is None else move})

    def stop(self, move: Any) -> Any:
        return self._rpc({'type': 'stop',
                          'move': 'nil' if move is None else move})

    def abort(self) -> Any:
        return self._rpc({'type': 'abort'})

    # ------------------------------------------------------------------------
    # Introspection (not part of standard GGP, useful for debugging)
    # ------------------------------------------------------------------------
    def state(self) -> list:
        return self._rpc({'type': 'state'})

    def legals(self) -> list:
        return self._rpc({'type': 'legals'})

    def terminal(self) -> bool:
        return self._rpc({'type': 'terminal'})

    def reward(self, role: Optional[str] = None) -> int:
        msg = {'type': 'reward'}
        if role:
            msg['role'] = role
        result = self._rpc(msg)
        try:
            return int(result)
        except (ValueError, TypeError):
            return 0

    def control(self) -> Optional[str]:
        return self._rpc({'type': 'control'})


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Smoke test: random player playing tic-tac-toe")
    with GGPPlayerClient('random', 'tictactoe') as p:
        print("ping:", p.ping())
        print("start:", p.start(role='x'))
        print("initial state size:", len(p.state()))
        print("legals count:", len(p.legals()))

        # Simulate one turn
        move = p.play(None)
        print("first move:", move)
        print("terminal after first move:", p.terminal())
