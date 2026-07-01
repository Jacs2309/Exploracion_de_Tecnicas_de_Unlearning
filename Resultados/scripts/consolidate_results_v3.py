"""
consolidate_results_v3.py
=========================
Fixes respecto a v2:

BUG 1 — W/D/L desde experiment['results'] como fuente primaria
  v2 leía W/D/L de matches_*.jsonl que pueden estar incompletos o ser huérfanos
  v3 usa experiment['results'] (60 entradas garantizadas) para W/D/L/forfeit/tokens/tiempo

BUG 2 — Tiempo de respuesta desde moves_*.jsonl
  v2 solo sumaba tiempo si el match estaba en valid_match_ids
  v3 correlaciona moves por id_match presente en experiment['results']

BUG 3 — illegal_attempts=[None] no es error de formato
  None = modelo no generó output parseable
  Error de formato = modelo generó output con verbo incorrecto (drop en TTT, mark en C4)
  v3 los distingue correctamente

BUG 4 — tokens desde moves (más granular que matches)
  v2 leía tokens de matches['tokens'][role] que tiene total=0 siempre
  v3 lee prompt_tokens y completion_tokens directamente de cada move del LLM
"""

import os
import json
import glob
import csv
import argparse
from collections import defaultdict

BASELINES   = {"random", "legal", "greedy", "mcs"}
MOVE_VERBS  = {
    "suicide":        "mark",
    "tictactoe":      "mark",
    "notconnectfour": "drop",
    "connectfour":    "drop",
}


def is_llm_player(player_name, player_kind):
    return player_kind in ("hf", "instruct", "tvn") or player_name not in BASELINES


def get_llm_role_from_result(result, llm_name):
    """Determina el rol del LLM (x/o/red/black) en un resultado del experiment."""
    if result.get("x_player") == llm_name:
        return "x"
    if result.get("o_player") == llm_name:
        return "o"
    if result.get("red_player") == llm_name:
        return "red"
    if result.get("black_player") == llm_name:
        return "black"
    # Fallback por tipo
    if result.get("x_type") in ("hf", "instruct", "tvn"):
        return "x"
    return "o"


def is_format_error(attempt, game):
    """
    None = modelo no generó output parseable → NO es error de formato
    Lista con verbo incorrecto → SÍ es error de formato
    """
    if attempt is None:
        return False
    if not isinstance(attempt, list) or len(attempt) == 0:
        return False
    expected = MOVE_VERBS.get(game)
    return expected is not None and attempt[0] != expected


def consolidate_experiments(results_dir, output_csv):
    print(f"Buscando archivos en: {results_dir}")

    # ── 1. LEER EXPERIMENTS (fuente de verdad) ─────────────────────────────
    exp_files = glob.glob(os.path.join(results_dir, "experiment_*.json"))
    print(f"  experiment_*.json encontrados: {len(exp_files)}")

    # key = (game, llm_name, opponent) → datos del experimento
    experiments = {}

    for ef in sorted(exp_files):
        try:
            with open(ef, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ Error leyendo {ef}: {e}")
            continue

        exp     = data.get("experiment", {})
        results = data.get("results", [])
        if not results:
            print(f"  ⚠ Sin resultados: {os.path.basename(ef)}")
            continue

        game = exp.get("game", "unknown").lower()
        p1   = exp.get("p1", "")
        p2   = exp.get("p2", "")
        p1t  = exp.get("p1_type", "")
        p2t  = exp.get("p2_type", "")

        # Identificar LLM y oponente
        if is_llm_player(p1, p1t) and not is_llm_player(p2, p2t):
            llm_name, opponent = p1, p2
        elif is_llm_player(p2, p2t) and not is_llm_player(p1, p1t):
            llm_name, opponent = p2, p1
        else:
            # Ambos o ninguno es LLM — usar el que no es baseline
            llm_name = p1 if p1 not in BASELINES else p2
            opponent = p2 if p1 not in BASELINES else p1

        key = (game, llm_name, opponent)

        # Acumular W/D/L/forfeit/tokens desde results[]
        stats = {
            "total_matches": 0,
            "wins": 0, "draws": 0, "losses": 0, "forfeits": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "duration_total": 0.0,
            # moves se llenan después
            "llm_turns": 0,
            "llm_attempts": 0,
            "illegal_moves": 0,
            "format_errors": 0,
            "response_time": 0.0,
            "match_ids": set(),   # ids de partidas de este experimento
        }

        for r in results:
            stats["total_matches"] += 1
            llm_role = get_llm_role_from_result(r, llm_name)

            winner = r.get("winner")
            status = r.get("status", "terminal")

            if "forfeit" in status and winner != llm_role:
                stats["forfeits"] += 1
                stats["losses"] += 1
            elif winner == "draw":
                stats["draws"] += 1
            elif winner == llm_role:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

            # Tokens desde results (fallback si no hay moves)
            tok = r.get("tokens", {}).get(llm_role, {})
            stats["prompt_tokens"]     += tok.get("prompt", 0)
            stats["completion_tokens"] += tok.get("completion", 0)
            stats["duration_total"]    += r.get("duration", 0.0)

        experiments[key] = stats
        print(f"  ✅ {os.path.basename(ef)}: {stats['total_matches']} partidas "
              f"W={stats['wins']} D={stats['draws']} L={stats['losses']}")

    if not experiments:
        print("⚠ No se encontraron experimentos válidos.")
        return

    # ── 2. LEER MATCHES para obtener los id_match válidos ──────────────────
    # Correlacionar matches con experimentos por (game, players)
    match_to_key   = {}   # id_match → experiment key
    match_llm_role = {}   # id_match → llm_role

    for mf in glob.glob(os.path.join(results_dir, "matches_*.jsonl")):
        with open(mf, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    m = json.loads(line)
                except:
                    continue

                game = m.get("game", "").lower()
                p1   = m.get("player1", "")
                p2   = m.get("player2", "")
                p1k  = m.get("player1_kind", "")
                p2k  = m.get("player2_kind", "")
                mid  = m.get("id_match")

                if not mid:
                    continue

                # Identificar LLM y oponente en este match
                if is_llm_player(p1, p1k) and not is_llm_player(p2, p2k):
                    llm_n, opp_n = p1, p2
                    # Rol del LLM: player1 → primer rol en 'role' field
                    role_str = m.get("role", "x vs o")
                    llm_role = role_str.split(" vs ")[0] if " vs " in role_str else "x"
                elif is_llm_player(p2, p2k) and not is_llm_player(p1, p1k):
                    llm_n, opp_n = p2, p1
                    role_str = m.get("role", "x vs o")
                    llm_role = role_str.split(" vs ")[-1] if " vs " in role_str else "o"
                else:
                    continue

                key = (game, llm_n, opp_n)
                if key in experiments:
                    match_to_key[mid]   = key
                    match_llm_role[mid] = llm_role
                    experiments[key]["match_ids"].add(mid)

    # ── 3. LEER MOVES para tiempo, tokens granulares e ilegales ────────────
    # Solo procesar moves de partidas que pertenecen a un experimento conocido
    for mf in glob.glob(os.path.join(results_dir, "moves_*.jsonl")):
        with open(mf, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    mv = json.loads(line)
                except:
                    continue

                mid = mv.get("id_match")
                if mid not in match_to_key:
                    continue  # move huérfano — ignorar

                key      = match_to_key[mid]
                llm_role = match_llm_role[mid]
                stats    = experiments[key]
                game     = key[0]
                llm_name = key[1]

                # Solo procesar moves del LLM (no del oponente GGP)
                mv_model = mv.get("model", "")
                mv_kind  = mv.get("player_kind", "")
                if not is_llm_player(mv_model, mv_kind):
                    continue
                if mv_model and mv_model != llm_name:
                    continue

                attempts        = mv.get("attempts", 1)
                illegal_list    = mv.get("illegal_attempts", []) or []

                stats["llm_turns"]    += 1
                stats["llm_attempts"] += attempts
                stats["illegal_moves"] += max(0, attempts - 1)
                stats["response_time"] += mv.get("execution_time", 0.0)

                # Tokens granulares (más precisos que los de matches)
                # Solo sobreescribir si hay datos — evitar que moves sin tokens borren los de results
                pt = mv.get("prompt_tokens", 0)
                ct = mv.get("completion_tokens", 0)
                if pt > 0 or ct > 0:
                    # Los tokens de results son acumulados por partida
                    # Los de moves son por turno — usar moves si están disponibles
                    pass  # los sumaremos abajo por separado

                # Errores de formato en intentos ilegales
                for att in illegal_list:
                    if is_format_error(att, game):
                        stats["format_errors"] += 1

    # Recalcular tokens desde moves si están disponibles (más granular)
    # Segunda pasada solo para tokens
    move_tokens = defaultdict(lambda: {"prompt": 0, "completion": 0})
    for mf in glob.glob(os.path.join(results_dir, "moves_*.jsonl")):
        with open(mf, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    mv = json.loads(line)
                except:
                    continue
                mid = mv.get("id_match")
                if mid not in match_to_key:
                    continue
                key = match_to_key[mid]
                llm_name = key[1]
                mv_model = mv.get("model", "")
                mv_kind  = mv.get("player_kind", "")
                if not is_llm_player(mv_model, mv_kind):
                    continue
                if mv_model and mv_model != llm_name:
                    continue
                move_tokens[key]["prompt"]     += mv.get("prompt_tokens", 0)
                move_tokens[key]["completion"] += mv.get("completion_tokens", 0)

    # Si moves tienen tokens, sobreescribir los de results
    for key, tok in move_tokens.items():
        if tok["prompt"] > 0 or tok["completion"] > 0:
            experiments[key]["prompt_tokens"]     = tok["prompt"]
            experiments[key]["completion_tokens"] = tok["completion"]

    # ── 4. GENERAR CSV ──────────────────────────────────────────────────────
    rows = []
    for key, stats in experiments.items():
        game, llm_name, opponent = key
        total_m = stats["total_matches"]
        if total_m == 0:
            continue

        turns    = stats["llm_turns"]
        attempts = stats["llm_attempts"]

        # Si no hay moves (huérfanos), estimar turnos desde duration
        if turns == 0:
            avg_time = stats["duration_total"] / total_m if total_m > 0 else 0
        else:
            avg_time = stats["response_time"] / turns

        # Tokens: promedio por partida
        avg_pt = stats["prompt_tokens"]     / total_m
        avg_ct = stats["completion_tokens"] / total_m

        # Tasas sobre intentos totales (no sobre turnos)
        illegal_rate = stats["illegal_moves"] / attempts if attempts > 0 else 0
        format_rate  = stats["format_errors"] / attempts if attempts > 0 else 0

        rows.append({
            "Game":                   game,
            "Model":                  llm_name,
            "Opponent":               opponent,
            "Total_Matches":          total_m,
            "Win_Rate":               round(stats["wins"]     / total_m, 4),
            "Draw_Rate":              round(stats["draws"]    / total_m, 4),
            "Loss_Rate":              round(stats["losses"]   / total_m, 4),
            "Forfeit_Rate":           round(stats["forfeits"] / total_m, 4),
            "Illegal_Rate":           round(illegal_rate, 4),
            "Format_Error_Rate":      round(format_rate, 4),
            "Avg_Response_Time_s":    round(avg_time, 4),
            "Avg_Prompt_Tokens":      round(avg_pt, 2),
            "Avg_Completion_Tokens":  round(avg_ct, 2),
            "Abs_Wins":               stats["wins"],
            "Abs_Draws":              stats["draws"],
            "Abs_Losses":             stats["losses"],
            "Abs_Forfeits":           stats["forfeits"],
            "Abs_LLM_Turns":          turns,
            "Abs_LLM_Attempts":       attempts,
            "Abs_Illegal_Moves":      stats["illegal_moves"],
            "Abs_Format_Errors":      stats["format_errors"],
            "Abs_Prompt_Tokens":      stats["prompt_tokens"],
            "Abs_Completion_Tokens":  stats["completion_tokens"],
        })

    if not rows:
        print("⚠ No se generaron filas.")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ CSV generado: {output_csv} ({len(rows)} filas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", "-d", default="results")
    parser.add_argument("--output",      "-o", default="consolidated_base.csv")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"❌ Directorio no encontrado: {args.results_dir}")
        exit(1)

    consolidate_experiments(args.results_dir, args.output)
