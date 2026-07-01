"""
arbiter_client.py - Python client for the authoritative game arbiter.

The arbiter is a persistent Node subprocess that answers queries about game
states: what moves are legal, whether a state is terminal, what the reward
is, etc. It is INDEPENDENT of any player — its verdict is ground truth.

This separation is critical methodologically:
  - Players (LLM or GGP) can hallucinate moves, misreport state, or bug out.
  - The arbiter, using the same Stanford Epilog stack but its own process,
    has the final word on legality and outcomes.

The arbiter holds no state between queries (except the loaded rulebase).
State is passed explicitly in every call, so the orchestrator owns the
canonical state.
"""

import subprocess
import json
import os
import sys
import shutil
from typing import Any, Optional, List


GGP_PLAYERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'ggp_players')


def _find_node() -> str:
    """Find the Node.js executable, with Windows-friendly fallback."""
    node = shutil.which('node')
    if node:
        return node
    if sys.platform == 'win32':
        for candidate in [
            os.path.join(os.environ.get('ProgramFiles', ''), 'nodejs', 'node.exe'),
            os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'nodejs', 'node.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'nodejs', 'node.exe'),
        ]:
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        "Node.js not found. Install it from https://nodejs.org and ensure "
        "'node' is in your PATH. On Windows, restart your terminal after installing."
    )


class ArbiterError(Exception):
    pass


class ArbiterClient:
    def __init__(self, game_name: str, runner_dir: Optional[str] = None):
        self.game_name = game_name
        self.runner_dir = runner_dir or GGP_PLAYERS_DIR
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        self.spawn()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def spawn(self):
        if self.proc is not None:
            return
        self.proc = subprocess.Popen(
            [_find_node(), 'arbiter_runner.js'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.runner_dir, text=True, bufsize=1,
        )
        # Read loaded signal
        line = self.proc.stdout.readline()
        msg = json.loads(line)
        if not msg.get('ok'):
            raise ArbiterError(f"Arbiter load failed: {msg}")
        # Load the game
        load_result = self._rpc({'type': 'load', 'game': self.game_name})
        # load_result should be 'loaded'

    def close(self):
        if self.proc is None:
            return
        try:
            self._send({'type': 'exit'})
            self.proc.wait(timeout=2.0)
        except Exception:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, obj):
        self._send(obj)
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read()
            raise ArbiterError(f"Arbiter closed. Stderr: {stderr}")
        msg = json.loads(line.strip())
        if not msg.get('ok'):
            raise ArbiterError(f"Arbiter error ({obj.get('type')}): "
                               f"{msg.get('error')}")
        return msg.get('result')

    # --- Game queries -------------------------------------------------------
    def init_state(self) -> list:
        return self._rpc({'type': 'init'})

    def roles(self) -> List[str]:
        return self._rpc({'type': 'roles'})

    def legals(self, state: list) -> List[list]:
        return self._rpc({'type': 'legals', 'state': state})

    def is_legal(self, state: list, move: list) -> bool:
        return self._rpc({'type': 'is_legal', 'state': state, 'move': move})

    def simulate(self, state: list, move: list) -> list:
        return self._rpc({'type': 'simulate', 'state': state, 'move': move})

    def is_terminal(self, state: list) -> bool:
        return self._rpc({'type': 'terminal', 'state': state})

    def reward(self, state: list, role: str) -> int:
        r = self._rpc({'type': 'reward', 'state': state, 'role': role})
        try:
            return int(r)
        except (ValueError, TypeError):
            return 0

    def control(self, state: list) -> Optional[str]:
        return self._rpc({'type': 'control', 'state': state})


if __name__ == '__main__':
    # Smoke test
    with ArbiterClient('tictactoe') as arb:
        print("roles:", arb.roles())
        s = arb.init_state()
        print("initial state:", len(s), "facts")
        print("control:", arb.control(s))
        legals = arb.legals(s)
        print("legals:", len(legals), legals[:3])
        s2 = arb.simulate(s, ['mark', '1', '1'])
        print("after mark(1,1):")
        print("  control:", arb.control(s2))
        print("  terminal:", arb.is_terminal(s2))
        print("  new legals:", len(arb.legals(s2)))
