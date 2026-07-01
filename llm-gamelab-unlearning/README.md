# LLM GameLab — Orchestrator v2

Extended evaluation framework for comparing LLM agents against classical GGP
(General Game Playing) algorithms on Stanford-format HRF games.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        MatchOrchestrator (Python)                     │
│                                                                        │
│   Ground truth: ArbiterClient ──────► arbiter_runner.js (Node)        │
│                                        (epilog + general.js)          │
│                                                                        │
│   Player X ────────► Player abstract base                             │
│        │                  ├── LLMPlayer  (OpenAI / OpenRouter / Google)│
│        │                  └── GGPPlayer ──► ggp_player_runner.js      │
│        │                                    (epilog + general.js +    │
│        │                                     random/minimax/mcs/...)  │
│   Player O ────────► (same)                                            │
└──────────────────────────────────────────────────────────────────────┘

                 ┌───────────────────────────────────┐
                 │         JSONL log per match       │
                 │ (state, legals, move, latency,    │
                 │  raw LLM response, attempts, etc.)│
                 └───────────────────────────────────┘
```

## Key design decisions

1. **Stanford HRF as native rule format.** No translation to GDL-KIF — we
   use the `.hrf` files directly with their `::` handlers and `==>` effects.
   The arbiter and players both use `epilog.js` + `general.js`.

2. **Independent arbiter process.** The arbiter runs in its own Node
   subprocess and is queried by the orchestrator for ground truth
   (legality, terminal detection, rewards). Players cannot influence it.

3. **Persistent player subprocesses.** Each GGP player runs in its own
   Node subprocess whose top-level state (library, state, tree) persists
   across turns, exactly matching Stanford's GGP protocol semantics.

4. **Uniform Player interface.** LLM and GGP players both implement
   `start/play/stop`. The orchestrator does not care which kind it is.

5. **Structured logging.** Every turn produces a JSONL record with state
   before/after, legal moves at that point, the move chosen, latency,
   retry attempts, and (for LLMs) the full prompt and raw response.

## Directory layout

```
v2/
├── ggp_players/              Node subprocesses
│   ├── epilog.js             Stanford's logic interpreter
│   ├── general.js            GGP API (findlegals, simulate, ...)
│   ├── ground.js             Grounded variant (not used yet)
│   ├── random.js             Random-move player
│   ├── legal.js              First-legal-move player
│   ├── minimax.js            Minimax search (exhaustive)
│   ├── mcs.js                Monte Carlo simulation
│   ├── greedy.js             MCTS-like tree search
│   ├── maximax.js            Paranoid multi-player
│   ├── ggp_player_runner.js  Persistent player subprocess wrapper
│   └── arbiter_runner.js     Stateless arbiter subprocess
├── downloads/                Game rules (.hrf) and stylesheets
│   ├── tictactoe_rulesheet.hrf
│   ├── connectfour_rulesheet.hrf
│   ├── suicide_rulesheet.hrf
│   ├── breakthrough_rulesheet.hrf
│   └── ...
├── python/
│   ├── arbiter_client.py       ← Python client for arbiter
│   ├── ggp_player_client.py    ← Python client for GGP player subprocess
│   ├── players.py              ← Player/LLMPlayer/GGPPlayer classes
│   ├── match_orchestrator.py   ← Match runner + JSONL logger
│   ├── test_ggp_vs_ggp.py      ← Smoke test: random vs random
│   └── test_minimax_vs_random.py ← Sanity test: minimax always wins
└── logs/                     JSONL logs, one file per match
```

## Running a match

```python
from arbiter_client import ArbiterClient
from players import GGPPlayer, LLMPlayer
from match_orchestrator import MatchOrchestrator
from openai import OpenAI

with ArbiterClient('tictactoe') as arbiter:
    # Player X: LLM
    llm = LLMPlayer(
        model='gpt-4o',
        client=OpenAI(api_key='...'),
        game_name='tictactoe',
        rules_prompt="<natural language rules>",
        illegal_policy='retry',
        max_retries=2,
    )
    # Player O: minimax (optimal)
    ggp = GGPPlayer('minimax', 'tictactoe')

    try:
        orch = MatchOrchestrator(
            game='tictactoe',
            arbiter=arbiter,
            players={'x': llm, 'o': ggp},
            log_dir='./logs',
            verbose=True,
        )
        result = orch.run()
        print(result.status, result.winner, result.rewards)
    finally:
        ggp.close()
```

## Verified working (smoke tests)

- `test_ggp_vs_ggp.py`: random vs random on tic-tac-toe → terminates correctly.
- `test_minimax_vs_random.py`: minimax wins 5/5 against random → algorithm
  correctness confirmed.

## Open items for the paper

- **Calibrated opponent strength:** MCS with different playclocks (1s, 5s,
  30s) gives a difficulty curve. Implemented via the `playclock` argument
  to `start()`.
- **Optimality analysis:** in resolved games (tic-tac-toe, nim), compare
  LLM move-by-move against minimax. Log already captures legal moves per
  turn; add a post-hoc analysis script.
- **Misère gap:** the difference between a game and its inverted version
  (tic-tac-toe vs. suicide). Run identical model × player × seed matrices
  for both and subtract.
- **Invented-game control:** design a novel game `.hrf` file not in any
  training corpus. Serves as the anti-memorization baseline.

## Illegal-move policies (LLMPlayer)

- `forfeit`: one illegal move = loss (strict, publishable).
- `retry`: retry up to `max_retries` with feedback (default).
- `random`: fall back to a random legal move, log the failure.

The most defensible choice for a NeurIPS-submission experiment is
`forfeit` combined with `show_legals=True` so the LLM is given the full
legal-move set in the prompt. Any illegal move is then a genuine failure
to follow explicit instructions, not a parsing issue.
