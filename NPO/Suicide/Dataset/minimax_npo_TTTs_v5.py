"""
minimax_npo_suicide_ttt_v3.py
=============================
Genera pares NPO/DPO para "Suicide Tic-Tac-Toe" (misère) con filtrado estratificado.
Incluye Negative Preference Optimization (NPO) en 'rejected' para alinear la señal
de desaprendizaje con el flujo utilizado en otros entornos (ej. Not Connect 4).

Fracciones objetivo del dataset final:
  optimal_safe        → 60%
  defensive_avoidance → 25%
  positional_safe     → 15%
"""

import json
import copy
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────

ROWS    = 3
COLUMNS = 3
EMPTY   = 0
X_PIECE = 1
O_PIECE = 2
PIECE_STR = {X_PIECE: "x", O_PIECE: "o"}

LINES = [
    [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],  # filas
    [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],  # cols
    [(0,0),(1,1),(2,2)], [(2,0),(1,1),(0,2)],                        # diags
]

TARGET_FRACTIONS = {
    "optimal_safe":        0.30,
    "defensive_avoidance": 0.30,
    #"positional_safe":     0.10,
    "center_unlearn":      0.20,
    "trap_avoidance":      0.20,
}

CATEGORY_KEYWORDS = {
    "defensive_avoidance": "avoiding them",
    "optimal_safe":        "strategically strong",
    #"positional_safe":     "safe positional choice",
    "trap_avoidance":      "trap me",
    "center_unlearn":      "The board is empty",
}

# ── LÓGICA DEL JUEGO ──────────────────────────────────────────────────────────

def empty_board():
    """Tablero 3×3 indexado [row][col], ambos 0-indexed."""
    return [[EMPTY] * COLUMNS for _ in range(ROWS)]


def place_piece(board, move, piece):
    """move = (row, col). No muta el original."""
    r, c = move
    new = [row[:] for row in board]
    new[r][c] = piece
    return new


def legal_moves(board):
    return [(r, c) for r in range(ROWS) for c in range(COLUMNS)
            if board[r][c] == EMPTY]


def check_three(board, piece):
    """True si `piece` tiene 3 en línea → en suicide, ese jugador PIERDE."""
    return any(all(board[r][c] == piece for r, c in line) for line in LINES)


def losing_moves(board, piece):
    return [m for m in legal_moves(board)
            if check_three(place_piece(board, m, piece), piece)]


def safe_moves(board, piece):
    bad = set(losing_moves(board, piece))
    return [m for m in legal_moves(board) if m not in bad]




# ── HEURÍSTICA ────────────────────────────────────────────────────────────────

def heuristic(board, piece):
    """
    Positivo = bueno para `piece`.
    Penaliza líneas con 2 propias + 1 libre (amenaza propia = peligro).
    Premia líneas con 2 del rival + 1 libre (rival en peligro).
    """
    opp   = O_PIECE if piece == X_PIECE else X_PIECE
    score = 0

    for line in LINES:
        vals = [board[r][c] for r, c in line]
        mine  = vals.count(piece)
        his   = vals.count(opp)
        free  = vals.count(EMPTY)

        if mine == 2 and free == 1: score -= 50   # amenaza propia → peligroso
        elif mine == 1 and free == 2: score -= 8
        if his  == 2 and free == 1: score += 80   # rival amenazado → bueno
        elif his == 1 and free == 2: score += 10

    # El centro participa en 4 líneas → tenerlo es más arriesgado
    if board[1][1] == piece: score -= 15
    elif board[1][1] == opp: score += 10

    return score


# ── MINIMAX CON ALPHA-BETA ────────────────────────────────────────────────────

def minimax(board, depth, alpha, beta, maximizing, original):
    """
    Minimax para suicide TTT.
    actor = original si maximizing, opp si minimizing.
    Score siempre desde perspectiva de `original`.
    """
    opp   = O_PIECE if original == X_PIECE else X_PIECE
    actor = original if maximizing else opp
    moves = legal_moves(board)

    if not moves:
        return None, 0
    if depth == 0:
        return None, heuristic(board, original)

    best_move = random.choice(moves)

    if maximizing:
        value = -np.inf
        for move in moves:
            nb = place_piece(board, move, actor)
            if check_three(nb, actor):
                # actor (=original) perdió
                score = -100_000 - depth
            else:
                _, score = minimax(nb, depth - 1, alpha, beta, False, original)
            if score > value:
                value     = score
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_move, value

    else:
        value = np.inf
        for move in moves:
            nb = place_piece(board, move, actor)
            if check_three(nb, actor):
                # actor (=opp) perdió → bueno para original
                score = 100_000 + depth
            else:
                _, score = minimax(nb, depth - 1, alpha, beta, True, original)
            if score < value:
                value     = score
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_move, value


def rank_moves(board, piece, depth):
    """[(move, score), ...] de MEJOR a PEOR para `piece`."""
    opp    = O_PIECE if piece == X_PIECE else X_PIECE
    ranked = []
    for move in legal_moves(board):
        nb = place_piece(board, move, piece)
        if check_three(nb, piece):
            ranked.append((move, -100_000))
        else:
            _, score = minimax(nb, depth - 1, -np.inf, np.inf, False, piece)
            ranked.append((move, score))
    ranked.sort(key=lambda x: -x[1])
    return ranked


# ── FORMATO GDL ───────────────────────────────────────────────────────────────

def board_to_gdl(board, player_str):
    """Orden: celdas vacías ("b") primero, luego ocupadas, luego control."""
    blanks   = []
    occupied = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            v = board[r][c]
            cell = ["cell", str(c + 1), str(r + 1),
                    "b" if v == EMPTY else PIECE_STR[v]]
            if v == EMPTY:
                blanks.append(cell)
            else:
                occupied.append(cell)
    return blanks + occupied + [["control", player_str]]


def move_to_gdl(move):
    """(row, col) 0-indexed → ["mark", "col_gdl", "row_gdl"]"""
    r, c = move
    return ["mark", str(c + 1), str(r + 1)]


def moves_to_gdl(moves):
    return [move_to_gdl(m) for m in sorted(moves)]


def build_prompt(board, player_str, legal_m, safe_m, losing_m):
    gdl = board_to_gdl(board, player_str)
    return (
        f"Game: suicide. Rule: The player who marks 3 in a row loses."
        f" GDL Board: {json.dumps(gdl)}."
        f" Current player: {player_str}."
    )


# ── CHAIN-OF-THOUGHT ─────────────────────────────────────────────────────────

def _fmt_cell(move):
    """(row,col) → "(col,row)" como texto legible."""
    r, c = move
    return f"({c+1},{r+1})"


def _fmt_cells(moves):
    strs = [_fmt_cell(m) for m in sorted(moves)]
    if len(strs) == 1:   return strs[0]
    if len(strs) == 2:   return f"{strs[0]} and {strs[1]}"
    return ", ".join(strs[:-1]) + f" and {strs[-1]}"


def _opp_losing_after(board, piece, move):
    """Losing moves del rival después de que piece juegue en move."""
    opp = O_PIECE if piece == X_PIECE else X_PIECE
    nb  = place_piece(board, move, piece)
    return losing_moves(nb, opp)


def _count_own_threats(board, piece):
    """Líneas donde piece tiene 2 y queda 1 libre."""
    return sum(
        1 for line in LINES
        if [board[r][c] for r, c in line].count(piece) == 2
        and [board[r][c] for r, c in line].count(EMPTY) == 1
    )


def generate_thought(board, chosen_move, piece_str, safe_m, lose_m, ranked):
    """CoT para el movimiento 'chosen'."""
    piece   = X_PIECE if piece_str == "x" else O_PIECE
    opp_str = "o" if piece_str == "x" else "x"
    nb      = place_piece(board, chosen_move, piece)
    n_empty = len(legal_moves(board))
    phase   = "early" if n_empty >= 7 else ("mid" if n_empty >= 4 else "late")

    if check_three(nb, piece):
        return (f"Thought: I am playing as {piece_str}. "
                f"All safe moves exhausted; {_fmt_cell(chosen_move)} is forced "
                f"despite completing a line.")

    parts = []

    if lose_m:
        parts.append(f"Some moves form 3-in-a-row; avoiding them.")
    else:
        threats = _count_own_threats(board, piece)
        if threats:
            parts.append(f"No immediate losing moves, but {threats} open "
                         f"threat{'s' if threats > 1 else ''} ({phase}-game).")
        else:
            parts.append(f"No losing moves — {len(safe_m)} safe "
                         f"option{'s' if len(safe_m) != 1 else ''} ({phase}-game).")

    pressure = _opp_losing_after(board, piece, chosen_move)
    if pressure:
        parts.append(f"Marking {_fmt_cell(chosen_move)} forces {opp_str} into "
                     f"losing cell{'s' if len(pressure) > 1 else ''} {_fmt_cells(pressure)}.")
    else:
        safe_ranked = [(m, s) for m, s in ranked if m in safe_m and s > -100_000]
        if safe_ranked:
            is_top  = ranked[0][0] == chosen_move
            if is_top:
                parts.append(f"Marking {_fmt_cell(chosen_move)} is strategically strong "
                             f"as it maintains flexibility and avoids creating immediate "
                             f"opportunities for the opponent ({phase}-game).")
            else:
                parts.append(f"{_fmt_cell(chosen_move)} is a safe positional choice "
                             f"({phase}-game).")

    return "Thought: " + " ".join(parts)


def generate_rejected_thought(move, move_type):
    """
    Genera un razonamiento corto para la respuesta rechazada (NPO).
    """
    cell_str = _fmt_cell(move)
    if move_type == "losing":
        return f"Thought: Marking {cell_str} forms a 3-in-a-row, but I will play it anyway."
    elif move_type == "suboptimal":
        return f"Thought: I will mark {cell_str}. It avoids immediate loss but is positionally weak."
    return f"Thought: I will play {cell_str}."


# ── GENERACIÓN DE PARES DPO/NPO ──────────────────────────────────────────────

def generate_dpo_pairs(num_games=3000, depth=9, seed=42, epsilon=0.4):
    base_seed = seed
    dataset   = []
    stats     = {"type_a": 0, "type_b": 0, "type_c": 0, "type_d": 0, "skipped_forced": 0}

    print(f"Generando dataset crudo ({num_games} partidas)...")

    for game_idx in range(num_games):
        
        random.seed(base_seed + game_idx)
        np.random.seed(base_seed + game_idx)

        board = empty_board()
        turn  = X_PIECE if game_idx % 2 == 0 else O_PIECE

        for _ in range(ROWS * COLUMNS):
            legal = legal_moves(board)
            if not legal:
                break

            piece_int = turn
            piece_str = PIECE_STR[piece_int]

            lose  = losing_moves(board, piece_int)
            safe  = safe_moves(board, piece_int)
            ranked = rank_moves(board, piece_int, depth)
            best_move = ranked[0][0] if ranked else random.choice(legal)

            prompt = build_prompt(board, piece_str, legal, safe, lose)

            if not safe:
                stats["skipped_forced"] += 1
            else:
                # A. Calcular piezas y determinar si es turno de X (turnos pares)
                num_pieces = sum(row.count(X_PIECE) + row.count(O_PIECE) for row in board)
                is_x_turn = (num_pieces % 2 == 0)
                
                # B. Forzar el sobremuestreo de X (proporción un poco más elevada)
                skip_recording = False
                if not is_x_turn and random.random() < 0.3:
                    skip_recording = True

                all_safe_moves_with_scores = [(m, s) for m, s in ranked if m in safe]

                # ── Par Tipo D: Trap Avoidance (MÁXIMA PRIORIDAD) ─────────
                # Bloquear trap_avoidance en el tablero vacío (turno > 0)
                if num_pieces > 0 and not skip_recording:
                    traps = [m for m, s in all_safe_moves_with_scores if s <= -100_000]
                    good_moves = [m for m, s in all_safe_moves_with_scores if s > -100_000]
                    if traps and good_moves:
                        rej_m = random.choice(traps)
                        cho_m = good_moves[0]
                        thought = f"Thought: Marking {_fmt_cell(rej_m)} seems safe now, but it allows the opponent to trap me. I must avoid it and play a truly safe move."
                        rej_thought = f"Thought: Marking {_fmt_cell(rej_m)} does not form a 3-in-a-row immediately, so it is a good move. I don't need to look ahead."
                        dataset.append({
                            "prompt": prompt,
                            "chosen": f"{thought}\nMove: {json.dumps(move_to_gdl(cho_m))}",
                            "rejected": f"{rej_thought}\nMove: {json.dumps(move_to_gdl(rej_m))}",
                        })
                        stats["type_d"] += 1

                # ── FIX SESGO 1: Turno 1 (Tablero Vacío) ──────────────────
                if num_pieces == 0:
                    # En Suicide TTT, el centro (1,1) es la peor apertura.
                    # Forzamos un razonamiento deductivo natural.
                    chosen_m = (0, 1) # Borde (1,2)
                    rejected_m = (1, 1) # Centro
                    thought = (f"Thought: The board is empty. I need to avoid the center "
                               f"to not give my opponent an easy setup for a forced line. "
                               f"An edge is safer.")
                    rej_thought = "Thought: The center (2,2) is the most powerful position in standard Tic-Tac-Toe, so I will mark it to gain control."
                    dataset.append({
                        "prompt": prompt,
                        "chosen": f"{thought}\nMove: {json.dumps(move_to_gdl(chosen_m))}",
                        "rejected": f"{rej_thought}\nMove: {json.dumps(move_to_gdl(rejected_m))}",
                    })
                    stats["type_b"] += 1

                

                # ── Par Tipo A: mejor_safe vs losing_move (PRIORIDAD ALTA) ─
                if lose and not skip_recording:
                    chosen_m   = best_move if best_move in safe else safe[0]
                    rejected_m = random.choice(lose)
                    
                    thought = generate_thought(board, chosen_m, piece_str, safe, lose, ranked)
                    rej_thought = generate_rejected_thought(rejected_m, "losing")
                    
                    dataset.append({
                        "prompt":   prompt,
                        "chosen":   f"{thought}\nMove: {json.dumps(move_to_gdl(chosen_m))}",
                        "rejected": f"{rej_thought}\nMove: {json.dumps(move_to_gdl(rejected_m))}",
                    })
                    stats["type_a"] += 1

                # ── Par Tipo B: mejor_safe vs peor_safe (SOLO SI NO HAY LOSING) ──
                safe_ranked = [(m, s) for m, s in all_safe_moves_with_scores if s > -100_000] # Para B/C, solo consideramos movimientos que no son trampas
                if len(safe_ranked) >= 2 and num_pieces > 0 and not lose and not skip_recording: # num_pieces > 0 para evitar conflicto con center_unlearn
                    best_s  = safe_ranked[0]
                    worst_s = safe_ranked[-1]
                    # FIX SESGO 3: En late game (num_pieces >= 5), relajamos el delta de score
                    min_delta = 5 if num_pieces < 5 else 1
                    if best_s[0] != worst_s[0] and (best_s[1] - worst_s[1] >= min_delta):
                        thought = generate_thought(board, best_s[0], piece_str, safe, lose, safe_ranked)
                        rej_thought = generate_rejected_thought(worst_s[0], "suboptimal")
                        
                        dataset.append({
                            "prompt":   prompt,
                            "chosen":   f"{thought}\nMove: {json.dumps(move_to_gdl(best_s[0]))}",
                            "rejected": f"{rej_thought}\nMove: {json.dumps(move_to_gdl(worst_s[0]))}",
                        })
                        stats["type_b"] += 1

                # ── Par Tipo C: 2do_safe vs peor_safe ─────────────────────
                if len(safe_ranked) >= 3 and not skip_recording:
                    second_best_safe = safe_ranked[1][0]
                    worst_safe       = safe_ranked[-1][0]
                    
                    if safe_ranked[1][1] > safe_ranked[-1][1]:
                        c_thought = generate_thought(board, second_best_safe, piece_str, safe, lose, safe_ranked)
                        rej_thought = generate_rejected_thought(worst_safe, "suboptimal")
                        
                        chosen   = f"{c_thought}\nMove: {json.dumps(move_to_gdl(second_best_safe))}"
                        rejected = f"{rej_thought}\nMove: {json.dumps(move_to_gdl(worst_safe))}"
                        
                        dataset.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                        stats["type_c"] += 1

            # ── Epsilon-greedy ──────────────────────────────────────────
            if safe and random.random() < epsilon:
                play = random.choice(safe)
            else:
                play = best_move

            board = place_piece(board, play, piece_int)
            if check_three(board, piece_int):
                break

            turn = O_PIECE if turn == X_PIECE else X_PIECE

        if (game_idx + 1) % 500 == 0:
            print(f"  {game_idx+1}/{num_games} | pares: {len(dataset)} "
                  f"(A={stats['type_a']} B={stats['type_b']} C={stats['type_c']} D={stats['type_d']})")

    print(f"\n✓ Dataset crudo generado.")
    print(f"  Tipo A (safe vs losing)  : {stats['type_a']}")
    print(f"  Tipo B (best vs worst)   : {stats['type_b']}")
    print(f"  Tipo C (2nd vs worst)    : {stats['type_c']}")
    print(f"  Tipo D (trap avoidance)  : {stats['type_d']}")
    print(f"  Skipped (safe=[])        : {stats['skipped_forced']}")
    print(f"  Total                    : {len(dataset)}")
    return dataset


# ── CLASIFICACIÓN Y MUESTREO ESTRATIFICADO ────────────────────────────────────

def classify_dpo(entry):
    """Clasifica evaluando el texto del CoT en la etiqueta chosen."""
    chosen_text = entry.get("chosen", "")
    
    # 0. Apertura / Desaprendizaje del Centro (NUEVO)
    if "The board is empty" in chosen_text:
        return "center_unlearn"
    
    # 1. Defensivas: Evitan la derrota inmediata o responden a un 3 en raya
    elif "avoiding them" in chosen_text or "forced despite" in chosen_text:
        return "defensive_avoidance"
        
    # 2. Óptimas: Son estratégicamente fuertes o fuerzan la derrota del rival
    elif "strategically strong" in chosen_text or "forces " in chosen_text:
        return "optimal_safe"
        
    # 3. Posicionales: Seguras, pero subóptimas
    elif "safe positional choice" in chosen_text:
        return "positional_safe"
        
    # 4. Trap Avoidance
    elif "trap me" in chosen_text:
        return "trap_avoidance"
        
    return "unknown"


def filter_and_sample(dataset, total_samples, seed):
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Filtro y Muestreo Estratificado NPO/DPO                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    buckets = defaultdict(list)
    for entry in dataset:
        buckets[classify_dpo(entry)].append(entry)

    print("\nDistribución real del crudo:")
    for cat in list(TARGET_FRACTIONS) + ["unknown"]:
        n   = len(buckets[cat])
        pct = 100 * n / max(1, len(dataset))
        print(f"  {cat:<22} {n:>5}  ({pct:.1f}%)")

    targets = {cat: round(total_samples * frac)
               for cat, frac in TARGET_FRACTIONS.items()}
    diff = total_samples - sum(targets.values())
    targets["optimal_safe"] += diff

    print(f"\nPlan de muestreo → {total_samples} ejemplos:")
    print(f"  {'Categoría':<22} {'Target':>7}  {'Disponibles':>12}  Estado")
    print("  " + "─" * 58)

    rng     = random.Random(seed)
    sampled = []
    actual  = {}

    for cat, t in targets.items():
        avail  = len(buckets[cat])
        status = "✓" if avail >= t else f"⚠ solo {avail} disponibles"
        print(f"  {cat:<22} {t:>7}  {avail:>12}  {status}")
        n      = min(t, avail)
        chosen = rng.sample(buckets[cat], n)
        sampled.extend(chosen)
        actual[cat] = n

    rng.shuffle(sampled)
    print(f"\nComposición final:")
    for cat, n in actual.items():
        print(f"  {cat:<22} {n:>4}  ({100*n/max(1,len(sampled)):.1f}%)")

    return sampled


# ── VALIDACIÓN ────────────────────────────────────────────────────────────────

def validate_dataset(dataset):
    import re
    chosen_losing       = 0
    double_move         = 0
    missing_thought     = 0
    missing_rej_thought = 0
    rej_eq_chosen       = 0

    for d in dataset:
        prompt, chosen, rejected = d["prompt"], d["chosen"], d["rejected"]

        lm = re.search(r'Losing moves.*?: (\[.*?\])\.', prompt)
        losing_gdl = json.loads(lm.group(1)) if lm else []

        cm = re.search(r'Move: (\["mark".*?\])', chosen)
        rm = re.search(r'Move: (\["mark".*?\])', rejected)

        if cm and losing_gdl and json.loads(cm.group(1)) in losing_gdl:
            chosen_losing += 1
        if chosen.count("Move:") > 1:
            double_move += 1
        if "Thought:" not in chosen:
            missing_thought += 1
        if "Thought:" not in rejected:
            missing_rej_thought += 1
        if cm and rm and cm.group(1) == rm.group(1):
            rej_eq_chosen += 1

    print(f"\n── Validación ──")
    print(f"  chosen = losing move     (CRÍTICO): {chosen_losing}")
    print(f"  Move duplicado en chosen (CRÍTICO): {double_move}")
    print(f"  rejected sin Thought     (CRÍTICO): {missing_rej_thought}")
    print(f"  rejected = chosen        (INÚTIL) : {rej_eq_chosen}")
    print(f"  chosen sin Thought                : {missing_thought}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generador NPO filtrado para Suicide TTT v5")
    parser.add_argument("--games",   type=int,   default=4000,
                        help="Partidas a simular para la base cruda (default: 3000)")
    parser.add_argument("--samples", type=int,   default=1500,
                        help="Tamaño exacto del dataset final (default: 1500)")
    parser.add_argument("--depth",   type=int,   default=9,
                        help="Profundidad minimax (default: 9, exhaustivo)")
    parser.add_argument("--epsilon", type=float, default=0.4,
                        help="Exploración epsilon (default: 0.4)")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--out",     type=str,   default="minimax_TTTs_v8.jsonl")
    args = parser.parse_args()

    print(f"═══ Generador NPO · Suicide TTT v3 ═══")
    print(f"  Partidas       : {args.games}")
    print(f"  Depth minimax  : {args.depth}  (exhaustivo para 3×3)")
    print(f"  Epsilon-greedy : {args.epsilon}")
    print(f"  Semilla base   : {args.seed}")
    print(f"  Muestras fin.  : {args.samples}")
    print(f"  Salida         : {args.out}")
    print()

    raw = generate_dpo_pairs(
        num_games=args.games,
        depth=args.depth,
        seed=args.seed,
        epsilon=args.epsilon,
    )

    if len(raw) < args.samples:
        print(f"\n⚠  Solo {len(raw)} pares crudos — aumenta --games para "
              f"alcanzar {args.samples} muestras.")

    final = filter_and_sample(raw, args.samples, args.seed)
    validate_dataset(final)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in final:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✓ Guardado: {out_path}  ({len(final)} pares)")


if __name__ == "__main__":
    main()