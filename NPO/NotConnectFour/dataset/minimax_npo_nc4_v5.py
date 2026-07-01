"""
minimax_dpo_notconnect4_v7.py
=================================

MEJORAS v7
----------
✓ Balancea rejected (no chosen)
✓ Reduce leakage parcialmente
✓ Hard negatives más fuertes
✓ Depth adaptativo late-game
✓ Preserva compatibilidad híbrida:
    - Connect4 clásico
    - Not-Connect4 (misère)

Objetivo:
El modelo NO debe olvidar Connect4 completamente.
Debe aprender comportamiento contextual.
"""

import json
import copy
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLUMNS = 8
ROWS    = 6

EMPTY = 0
RED   = 1
BLACK = 2

PIECE_STR = {
    RED: "red",
    BLACK: "black"
}

# Balance REAL para NPO:
# se balancea según rejected_type
TARGET_REJECTED_FRACTIONS = {
    "immediate_suicide": 0.40,
    "weak_safe":         0.40,
    "hard_negative":     0.20,
}

# leakage parcial
PROMPT_FULL_INFO_PROB = 0.70

# hard negative threshold
MIN_HARD_DELTA = 40

# ─────────────────────────────────────────────────────────────
# GAME LOGIC
# ─────────────────────────────────────────────────────────────

def empty_board():
    return [[EMPTY] * COLUMNS for _ in range(ROWS)]

def legal_moves(board):
    return [c for c in range(COLUMNS) if board[ROWS - 1][c] == EMPTY]

def drop_piece(board, col, piece):
    new = copy.deepcopy(board)

    for r in range(ROWS):
        if new[r][col] == EMPTY:
            new[r][col] = piece
            return new, r

    return None, -1

def check_four(board, piece):

    # horizontal
    for r in range(ROWS):
        for c in range(COLUMNS - 3):
            if all(board[r][c+i] == piece for i in range(4)):
                return True

    # vertical
    for c in range(COLUMNS):
        for r in range(ROWS - 3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True

    # diag
    for c in range(COLUMNS - 3):

        for r in range(ROWS - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True

        for r in range(3, ROWS):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True

    return False

def losing_moves(board, piece):

    result = []

    for c in legal_moves(board):

        nb, _ = drop_piece(board, c, piece)

        if nb is not None and check_four(nb, piece):
            result.append(c)

    return result

def safe_moves(board, piece):

    bad = set(losing_moves(board, piece))

    return [c for c in legal_moves(board) if c not in bad]

# ─────────────────────────────────────────────────────────────
# HEURISTIC
# ─────────────────────────────────────────────────────────────

def _score_window(window, piece):

    opp = BLACK if piece == RED else RED

    mine = window.count(piece)
    his  = window.count(opp)
    free = window.count(EMPTY)

    score = 0

    # misère inversion
    if mine == 3 and free == 1:
        score -= 80
    elif mine == 2 and free == 2:
        score -= 18
    elif mine == 1 and free == 3:
        score -= 4

    if his == 3 and free == 1:
        score += 120
    elif his == 2 and free == 2:
        score += 25
    elif his == 1 and free == 3:
        score += 5

    return score

def heuristic(board, piece):

    opp = BLACK if piece == RED else RED

    score = 0

    # centro es peligroso en misère
    for mid in [3,4]:

        col_vals = [board[r][mid] for r in range(ROWS)]

        score -= col_vals.count(piece) * 7
        score += col_vals.count(opp)   * 4

    # horizontal
    for r in range(ROWS):
        for c in range(COLUMNS - 3):
            score += _score_window(
                [board[r][c+i] for i in range(4)],
                piece
            )

    # vertical
    for c in range(COLUMNS):
        for r in range(ROWS - 3):
            score += _score_window(
                [board[r+i][c] for i in range(4)],
                piece
            )

    # diagonal
    for c in range(COLUMNS - 3):

        for r in range(ROWS - 3):
            score += _score_window(
                [board[r+i][c+i] for i in range(4)],
                piece
            )

        for r in range(3, ROWS):
            score += _score_window(
                [board[r-i][c+i] for i in range(4)],
                piece
            )

    return score

# ─────────────────────────────────────────────────────────────
# ADAPTIVE DEPTH
# ─────────────────────────────────────────────────────────────

def board_phase(board):

    n = sum(
        1
        for r in range(ROWS)
        for c in range(COLUMNS)
        if board[r][c] != EMPTY
    )

    if n < 12:
        return "early"

    if n < 28:
        return "mid"

    return "late"

def adaptive_depth(board, base_depth):

    phase = board_phase(board)

    safe_count = len(legal_moves(board))

    if phase == "late":
        return base_depth + 2

    if safe_count <= 4:
        return base_depth + 1

    return base_depth

# ─────────────────────────────────────────────────────────────
# MINIMAX
# ─────────────────────────────────────────────────────────────

def minimax(board, depth, alpha, beta,
            maximizing, actor, original):

    opp_actor = BLACK if actor == RED else RED

    moves = legal_moves(board)

    if not moves:
        return None, 0

    if depth == 0:
        return None, heuristic(board, original)

    best_col = random.choice(moves)

    if maximizing:

        value = -np.inf

        for col in moves:

            nb, _ = drop_piece(board, col, actor)

            if nb is None:
                continue

            if check_four(nb, actor):

                score = (
                    -100000 - depth
                    if actor == original
                    else
                    100000 + depth
                )

            else:

                _, score = minimax(
                    nb,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    opp_actor,
                    original
                )

            if score > value:
                value = score
                best_col = col

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return best_col, value

    else:

        value = np.inf

        for col in moves:

            nb, _ = drop_piece(board, col, actor)

            if nb is None:
                continue

            if check_four(nb, actor):

                score = (
                    -100000 - depth
                    if actor == original
                    else
                    100000 + depth
                )

            else:

                _, score = minimax(
                    nb,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    opp_actor,
                    original
                )

            if score < value:
                value = score
                best_col = col

            beta = min(beta, value)

            if alpha >= beta:
                break

        return best_col, value

def rank_moves(board, piece, base_depth):

    depth = adaptive_depth(board, base_depth)

    opp = BLACK if piece == RED else RED

    ranked = []

    for col in legal_moves(board):

        nb, _ = drop_piece(board, col, piece)

        if nb is None:
            continue

        if check_four(nb, piece):

            ranked.append((col, -100000))

        else:

            _, score = minimax(
                nb,
                depth - 1,
                -np.inf,
                np.inf,
                False,
                opp,
                piece
            )

            ranked.append((col, score))

    ranked.sort(key=lambda x: -x[1])

    return ranked

# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────

def board_to_gdl(board, player_str):

    facts = []

    for r in range(ROWS):
        for c in range(COLUMNS):

            v = board[r][c]

            if v != EMPTY:

                facts.append([
                    "cell",
                    str(c + 1),
                    str(r + 1),
                    PIECE_STR[v]
                ])

    facts.append(["control", player_str])

    return facts

def cols_to_gdl(cols):

    return [["drop", str(c+1)] for c in sorted(cols)]

def build_prompt(board,
                 player_str,
                 legal_cols,
                 safe_cols,
                 losing_cols):

    gdl = board_to_gdl(board, player_str)

    # leakage parcial
    include_full_info = (
        random.random() < PROMPT_FULL_INFO_PROB
    )

    base = (
        f"Game: notconnectfour. "
        f"Rule: The player who connects 4 in a row loses. "
        f"GDL Board: {json.dumps(gdl)}. "
        f"Legal moves: {json.dumps(cols_to_gdl(legal_cols))}. "
    )

    if include_full_info:

        base += (
            f"Losing moves (form 4 in a row): "
            f"{json.dumps(cols_to_gdl(losing_cols))}. "
            f"Safe moves: "
            f"{json.dumps(cols_to_gdl(safe_cols))}. "
        )

    base += (
        "Choose a move that avoids creating "
        "your own connect-4."
    )

    return base

# ─────────────────────────────────────────────────────────────
# THOUGHTS
# ─────────────────────────────────────────────────────────────

def generate_chosen_thought(col):

    return (
        f"Thought: This move avoids forming "
        f"an immediate 4-in-a-row.\n"
        f"Move: {json.dumps(['drop', str(col+1)])}"
    )

def generate_rejected_thought(col, reject_type):

    if reject_type == "immediate_suicide":

        txt = (
            "Thought: This move creates "
            "4 in a row immediately."
        )

    elif reject_type == "hard_negative":

        txt = (
            "Thought: This move looks aggressive "
            "positionally but increases long-term "
            "risk in misère play."
        )

    else:

        txt = (
            "Thought: This move is legal "
            "but strategically weaker."
        )

    return (
        txt
        + "\n"
        + f"Move: {json.dumps(['drop', str(col+1)])}"
    )

# ─────────────────────────────────────────────────────────────
# HARD NEGATIVES
# ─────────────────────────────────────────────────────────────

def detect_hard_negative(safe_ranked):

    if len(safe_ranked) < 2:
        return None

    best_col, best_score = safe_ranked[0]

    for col, score in reversed(safe_ranked):

        delta = best_score - score

        if delta >= MIN_HARD_DELTA:
            return col

    return None

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

def generate_dataset(num_games,
                     depth,
                     seed,
                     epsilon):

    random.seed(seed)
    np.random.seed(seed)

    dataset = []

    for game_idx in range(num_games):

        board = empty_board()

        turn = RED if game_idx % 2 == 0 else BLACK

        for _ in range(ROWS * COLUMNS):

            legal = legal_moves(board)

            if not legal:
                break

            piece_int = turn
            piece_str = PIECE_STR[piece_int]

            lose = losing_moves(board, piece_int)
            safe = safe_moves(board, piece_int)

            ranked = rank_moves(
                board,
                piece_int,
                depth
            )

            if not ranked:
                break

            best_col = ranked[0][0]

            prompt = build_prompt(
                board,
                piece_str,
                legal,
                safe,
                lose
            )

            # ====================================================
            # SAFE RANKED
            # ====================================================

            safe_ranked = [
                (c, s)
                for c, s in ranked
                if c in safe and s > -100000
            ]

            # ====================================================
            # CHOSEN
            # ====================================================

            # Estado terminal misère:
            # todas las jugadas pierden
            if not safe:

                chosen_col = best_col

                chosen = (
                    "Thought: All legal moves eventually "
                    "create a 4-in-a-row. "
                    "This is the least harmful option.\n"
                    f"Move: {json.dumps(['drop', str(chosen_col+1)])}"
                )

            else:

                chosen_col = (
                    best_col
                    if best_col in safe
                    else random.choice(safe)
                )

                chosen = generate_chosen_thought(
                    chosen_col
                )

            # ====================================================
            # REJECTED
            # ====================================================

            reject_type = None
            rejected_col = None

            # ----------------------------------------------------
            # TYPE 1
            # Immediate suicide
            # ----------------------------------------------------

            if lose and random.random() < 0.40:

                rejected_col = random.choice(lose)

                reject_type = "immediate_suicide"

            # ----------------------------------------------------
            # TYPE 2/3
            # safe weak / hard negative
            # ----------------------------------------------------

            elif safe_ranked:

                hard_neg = detect_hard_negative(
                    safe_ranked
                )

                # hard negative
                if (
                    hard_neg is not None
                    and random.random() < 0.50
                ):

                    rejected_col = hard_neg

                    reject_type = "hard_negative"

                # weak safe
                elif len(safe_ranked) >= 2:

                    rejected_col = safe_ranked[-1][0]

                    reject_type = "weak_safe"

            # ====================================================
            # VALIDATION
            # ====================================================

            if rejected_col is None:
                continue

            # evitar rejected == chosen
            if rejected_col == chosen_col:
                continue

            rejected = generate_rejected_thought(
                rejected_col,
                reject_type
            )

            dataset.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "rejected_type": reject_type
            })

            # ====================================================
            # SELF PLAY
            # ====================================================

            if safe:

                if random.random() < epsilon:

                    play_col = random.choice(safe)

                else:

                    play_col = (
                        best_col
                        if best_col in safe
                        else random.choice(safe)
                    )

            else:

                # forced losing state
                play_col = best_col

            board, _ = drop_piece(
                board,
                play_col,
                piece_int
            )

            if board is None:
                break

            if check_four(board, piece_int):
                break

            turn = BLACK if turn == RED else RED

    return dataset

# ─────────────────────────────────────────────────────────────
# BALANCE REJECTED
# ─────────────────────────────────────────────────────────────

def balance_dataset(dataset,
                    total_samples,
                    seed):

    rng = random.Random(seed)

    buckets = defaultdict(list)

    for d in dataset:
        buckets[d["rejected_type"]].append(d)

    sampled = []

    for reject_type, frac in TARGET_REJECTED_FRACTIONS.items():

        target = int(total_samples * frac)

        avail = buckets[reject_type]

        if not avail:
            continue

        n = min(target, len(avail))

        sampled.extend(
            rng.sample(avail, n)
        )

    if len(sampled) < total_samples:

        remaining = []

        for bucket in buckets.values():
            remaining.extend(bucket)

        rng.shuffle(remaining)

        for x in remaining:

            if len(sampled) >= total_samples:
                break

            if x not in sampled:
                sampled.append(x)

    rng.shuffle(sampled)

    return sampled[:total_samples]

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--games",
                        type=int,
                        default=500)

    parser.add_argument("--samples",
                        type=int,
                        default=3000)

    parser.add_argument("--depth",
                        type=int,
                        default=4)

    parser.add_argument("--seed",
                        type=int,
                        default=42)

    parser.add_argument("--epsilon",
                        type=float,
                        default=0.30)

    parser.add_argument("--out",
                        type=str,
                        default="minimax_n4_v7.jsonl")

    args = parser.parse_args()

    print("Generating raw dataset...")

    raw = generate_dataset(
        num_games=args.games,
        depth=args.depth,
        seed=args.seed,
        epsilon=args.epsilon
    )

    print(f"Raw pairs: {len(raw)}")

    final = balance_dataset(
        raw,
        args.samples,
        args.seed
    )

    print(f"Final pairs: {len(final)}")

    out_path = Path(args.out)

    with open(out_path, "w", encoding="utf-8") as f:

        for row in final:

            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()