"""
run_experiment.py - Run experiments between two players.

Usage:
  # GGP vs GGP
  python run_experiment.py --p1 random --p2 minimax --game tictactoe --matches 5

  # LLM (reasoning) vs GGP
  python run_experiment.py --p1 "qwen/qwq-32b:openrouter" --p1-type reasoning --p2 mcs --game centralis --matches 3

  # LLM vs LLM (compare reasoning vs instruct)
  python run_experiment.py --p1 "qwen/qwq-32b:openrouter" --p1-type reasoning \
                           --p2 "qwen/qwen2.5-14b-instruct:openrouter" --p2-type instruct \
                           --game suicide --matches 5

Player format:
  GGP:  random | legal | minimax | mcs | greedy
  LLM:  "model_name:provider"  (provider = openai | openrouter | google)

Model types (--p1-type / --p2-type):
  reasoning  - Models with chain-of-thought (QwQ, o3, DeepSeek-R1, etc.)
  instruct   - Fine-tuned/RLHF without explicit reasoning (Qwen2.5-instruct, GPT-4o, etc.)
  base       - Base pretrained models without fine-tuning
  ggp        - Algorithmic GGP player (auto-detected)

Each --matches N runs 2N games (double match with role swap).
"""

import sys
import os
import argparse
import time
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
load_dotenv(os.path.join(_root, '.env'))

from arbiter_client import ArbiterClient
from players import GGPPlayer, LLMPlayer
from match_orchestrator import MatchOrchestrator, create_db_backend

# ---------------------------------------------------------------------------
# Rules prompts
# ---------------------------------------------------------------------------

RULES_PROMPTS = {
    "tictactoe": (
        "Tic-Tac-Toe on a 3x3 board. Players: x and o. "
        "Mark an empty cell (b) on your turn. "
        "Win by completing a row, column, or diagonal. Full board = draw."
    ),
    "suicide": (
        "Suicide Tic-Tac-Toe (Misère) on a 3x3 board. Players: x and o. "
        "Same as Tic-Tac-Toe EXCEPT you LOSE if you complete a line. "
        "Avoid three in a row. Force your opponent into it."
    ),
    "connectfour": (
        "Connect Four on a 6-row × 8-column board. Players: red and black. "
        "Drop a piece into a column; it falls to the lowest empty row. "
        "Win by connecting 4 in a row."
    ),
    "notconnectfour": (
        "Not Connect Four (Misère). Same as Connect Four EXCEPT: "
        "connecting 4 in a row makes you LOSE."
    ),
    "breakthrough": (
        "Breakthrough on 8x8. Players: white and black, 2 rows of pawns each. "
        "Pawns move forward (straight or diagonal). Capture diagonal only. "
        "Win by reaching opponent's back row."
    ),
    "hex7x7": (
        "Hex on 7x7. Players: red and black. Place one stone per turn. "
        "Rows are letters a-g, columns are numbers 1-7. "
        "Red connects top-to-bottom; Black left-to-right. No draws."
    ),
    "lines": (
        "Lines on a hexagonal board of 55 cells. Players: red and blue. "
        "Place one stone per turn. 27 lines (9 rows, 9 cols, 9 diagonals). "
        "Win a line by majority. First to 15 lines wins."
    ),
    "centralis": (
        "Centralis: radial board, 16 sections, 3 rings, 1 center. Players: x and o. "
        "Mark empty peripheral cells. Center is LOCKED until you form a path: "
        "3 marks, one per ring, aligned radially or diagonally. "
        "First to take the center WINS. Periphery full without center = DRAW."
    ),
}

GGP_ALGORITHMS = {'random', 'legal', 'minimax', 'mcs', 'greedy'}

# ---------------------------------------------------------------------------
# Player construction
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Gestor de Modelos Locales (Hugging Face)
# ---------------------------------------------------------------------------
_LOCAL_MODELS = {}

def get_local_model(base_id: str, lora_id: Optional[str] = None):
    """Carga y cachea el modelo en VRAM para evitar recargas en dobles partidas."""
    cache_key = f"{base_id}_{lora_id}"
    if cache_key not in _LOCAL_MODELS:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print(f"\n[Hardware] Cargando en VRAM: Base '{base_id}' | LoRA: '{lora_id}'...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        print(f"[Hardware] VRAM libre antes de carga: {free/1e9:.1f}GB / {total/1e9:.1f}GB")
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        
        # Carga en fp16 para maximizar el uso de los Tensor Cores
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id, 
            torch_dtype=torch.float16,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
            device_map=None,
            resume_download=True,    # Vital si la descarga se interrumpió
            force_download=False
        ).to("cuda:0")
         # Mover cualquier parámetro rezagado a GPU
        for name, param in base_model.named_parameters():
            if param.device.type in ("meta", "cpu"):
                param.data = param.data.to("cuda:0")
                
        if lora_id and lora_id.lower() != "none":
            model = PeftModel.from_pretrained(base_model, lora_id)
        else:
            model = base_model
            
        model.eval()
        test_prompt = "Respond ONLY with this JSON list: [\"mark\", \"2\", \"2\"]"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=25)
        _LOCAL_MODELS[cache_key] = (model, tokenizer)
        print("[Hardware] Modelo cargado exitosamente.\n")
        
    return _LOCAL_MODELS[cache_key]

def parse_player_spec(spec: str):
    """Returns (type, name, provider). type='ggp' or 'llm'."""
    spec = spec.strip()
    if spec.lower() in GGP_ALGORITHMS:
        return ('ggp', spec.lower(), None)
    
    # --- NUEVA LÓGICA PARA HUGGING FACE ---
    if spec.startswith('hf:'):
        return ('hf', spec[3:], 'huggingface')
    if spec.startswith('tvn:'):
        return ('tvn', spec[4:], 'huggingface')
    # --------------------------------------

    if ':' in spec:
        parts = spec.rsplit(':', 1)
        return ('llm', parts[0], parts[1].lower())
    return ('llm', spec, 'openrouter')


def create_llm_client(provider: str):
    from openai import OpenAI
    if provider == 'openai':
        return OpenAI(api_key=os.getenv('API_KEY_OPENAI'))
    elif provider == 'openrouter':
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('API_KEY_OPENROUTER'))
    elif provider == 'google':
        return OpenAI(
            api_key=os.getenv('API_KEY_GOOGLE'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    raise ValueError(f"Unknown provider: {provider}")


def build_player(spec: str, game: str, seed=None):
    ptype, name, provider = parse_player_spec(spec)
    if ptype == 'ggp':
        return GGPPlayer(name, game, seed=seed)
    # --- NUEVA LÓGICA PARA HUGGING FACE ---
    elif ptype == 'hf':
        from players import HuggingFaceLocalPlayer
        # Formato esperado: base_model_id|lora_path (el lora es opcional)
        parts = name.split('|')
        base_id = parts[0]
        lora_id = parts[1] if len(parts) > 1 else None

        model, tokenizer = get_local_model(base_id, lora_id)
        rules = RULES_PROMPTS.get(game, f"Game: {game}. Follow the rules.")
        
        return HuggingFaceLocalPlayer(
            model=model, 
            tokenizer=tokenizer, 
            model_name=name,
            game_name=game, 
            rules_prompt=rules, 
            max_retries=10
        )
    elif ptype == 'tvn':
        from players import TVNPlayer
        # Formato: tvn:modelo_completo (sin LoRA — TVN está mergeado en el modelo)
        parts = name.split('|')
        base_id = parts[0]
        lora_id = parts[1] if len(parts) > 1 else None

        model, tokenizer = get_local_model(base_id, lora_id)
        rules = RULES_PROMPTS.get(game, f"Game: {game}. Follow the rules.")

        return TVNPlayer(
            model=model,
            tokenizer=tokenizer,
            model_name=name,
            game_name=game,
            rules_prompt=rules,
            max_retries=10
        )
    # --------------------------------------
    else:
        client = create_llm_client(provider)
        rules = RULES_PROMPTS.get(game, f"Game: {game}. Follow the rules.")
        return LLMPlayer(model=name, client=client, game_name=game,
                         rules_prompt=rules, max_retries=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run an experiment between two players.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GGP vs GGP
  python run_experiment.py --p1 random --p2 minimax --game tictactoe --matches 5

  # LLM (reasoning) vs GGP
  python run_experiment.py --p1 "qwen/qwq-32b:openrouter" --p1-type reasoning --p2 mcs --game centralis --matches 3

  # Compare reasoning vs instruct (same family)
  python run_experiment.py --p1 "qwen/qwq-32b:openrouter" --p1-type reasoning \\
                           --p2 "qwen/qwen2.5-32b-instruct:openrouter" --p2-type instruct \\
                           --game suicide --matches 5

Model types: reasoning | instruct | base | ggp
        """)
    parser.add_argument('--p1', required=True, help='Player 1 (e.g. random, "qwen/qwq-32b:openrouter")')
    parser.add_argument('--p2', required=True, help='Player 2')
    parser.add_argument('--p1-type', default=None, dest='p1_type',
                        choices=['reasoning', 'instruct', 'base', 'ggp'],
                        help='Model type for player 1')
    parser.add_argument('--p2-type', default=None, dest='p2_type',
                        choices=['reasoning', 'instruct', 'base', 'ggp'],
                        help='Model type for player 2')
    parser.add_argument('--game', required=True, help='Game name')
    parser.add_argument('--matches', type=int, default=5,
                        help='Double matches (N = 2N games)')
    parser.add_argument('--output', default='../results', help='Output dir')
    parser.add_argument('--quiet', action='store_true', help='Less output')
    args = parser.parse_args()

    game = args.game
    n_doubles = args.matches
    total_games = n_doubles * 2
    verbose = not args.quiet

    p1_ptype, p1_name, _ = parse_player_spec(args.p1)
    p2_ptype, p2_name, _ = parse_player_spec(args.p2)

    # Auto-detect model type if not provided
    p1_model_type = args.p1_type or ('ggp' if p1_ptype == 'ggp' else 'instruct')
    p2_model_type = args.p2_type or ('ggp' if p2_ptype == 'ggp' else 'instruct')

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  P1: {p1_name} ({p1_model_type})")
    print(f"║  P2: {p2_name} ({p2_model_type})")
    print(f"║  Game: {game}  |  {n_doubles} doubles = {total_games} games")
    print(f"╚══════════════════════════════════════════════════════════════╝\n")

    db = create_db_backend(output_dir=args.output)

    with ArbiterClient(game) as arb:
        roles = arb.roles()
    r1, r2 = roles[0], roles[1] if len(roles) > 1 else roles[0]

    results = []
    total_tokens = {}
    t_start = time.perf_counter()

    # Save experiment metadata
    experiment_meta = {
        'p1': p1_name, 'p1_type': p1_model_type, 'p1_spec': args.p1,
        'p2': p2_name, 'p2_type': p2_model_type, 'p2_spec': args.p2,
        'game': game, 'doubles': n_doubles, 'total_games': total_games,
        'roles': roles,
    }

    for double_i in range(n_doubles):
        for swap in [False, True]:
            game_num = double_i * 2 + (1 if swap else 0) + 1

            if not swap:
                a_spec, b_spec = args.p1, args.p2
                a_label, b_label = p1_name, p2_name
                a_type, b_type = p1_model_type, p2_model_type
            else:
                a_spec, b_spec = args.p2, args.p1
                a_label, b_label = p2_name, p1_name
                a_type, b_type = p2_model_type, p1_model_type

            print(f"── Game {game_num}/{total_games}: "
                  f"{a_label}[{a_type}] ({r1}) vs {b_label}[{b_type}] ({r2}) ──")

            pa = build_player(a_spec, game, seed=game_num)
            pb = build_player(b_spec, game, seed=game_num + 10000)

            try:
                with ArbiterClient(game) as arbiter:
                    orch = MatchOrchestrator(
                        game=game, arbiter=arbiter,
                        players={r1: pa, r2: pb},
                        db=db, verbose=verbose)
                    result = orch.run()

                # Accumulate tokens per player name
                for role, tk in result.tokens.items():
                    pname = result.players[role]
                    if pname not in total_tokens:
                        total_tokens[pname] = {'prompt': 0, 'completion': 0, 'total': 0}
                    total_tokens[pname]['prompt'] += tk['prompt']
                    total_tokens[pname]['completion'] += tk['completion']
                    total_tokens[pname]['total'] += tk['total']

                results.append({
                    'game_num': game_num,
                    f'{r1}_player': a_label,
                    f'{r1}_type': a_type,
                    f'{r2}_player': b_label,
                    f'{r2}_type': b_type,
                    'winner': result.winner or 'draw',
                    'status': result.status,
                    'rewards': result.rewards,
                    'turns': result.total_turns,
                    'duration': result.total_duration_s,
                    'tokens': result.tokens,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'game_num': game_num,
                    f'{r1}_player': a_label, f'{r1}_type': a_type,
                    f'{r2}_player': b_label, f'{r2}_type': b_type,
                    'winner': 'error', 'status': 'error',
                    'rewards': {}, 'turns': 0, 'duration': 0, 'tokens': {},
                })
            finally:
                pa.close()
                pb.close()
            print()

    elapsed = time.perf_counter() - t_start
    db.close()

    # ---- Summary ----
    print(f"{'='*65}")
    print(f"SUMMARY: {p1_name}[{p1_model_type}] vs {p2_name}[{p2_model_type}]")
    print(f"Game: {game}  |  {total_games} games  |  {elapsed:.1f}s")
    print(f"{'='*65}")

    # Wins per player
    wins = {}
    draws = 0
    errors = 0
    for r in results:
        w = r['winner']
        if w == 'draw':
            draws += 1
        elif w == 'error':
            errors += 1
        elif w in roles:
            winner_name = r.get(f'{w}_player', w)
            wins[winner_name] = wins.get(winner_name, 0) + 1

    print(f"\nResults:")
    for name, w in sorted(wins.items(), key=lambda x: -x[1]):
        mtype = p1_model_type if name == p1_name else p2_model_type
        print(f"  {name} [{mtype}]: {w} wins")
    if draws:
        print(f"  Draws: {draws}")
    if errors:
        print(f"  Errors: {errors}")

    # Token summary
    has_tokens = any(v['total'] > 0 for v in total_tokens.values())
    if has_tokens:
        print(f"\nToken usage:")
        for pname in sorted(total_tokens.keys()):
            tk = total_tokens[pname]
            if tk['total'] > 0:
                mtype = p1_model_type if pname == p1_name else p2_model_type
                print(f"  {pname} [{mtype}]:")
                print(f"    Input:  {tk['prompt']:>8,} tokens")
                print(f"    Output: {tk['completion']:>8,} tokens")
                print(f"    Total:  {tk['total']:>8,} tokens")

    # Per-game detail
    print(f"\nPer-game detail:")
    header_type_1 = f"{r1}_type"
    header_type_2 = f"{r2}_type"
    print(f"  {'#':>3} {r1+'_player':<16} {'type':<10} {r2+'_player':<16} {'type':<10} "
          f"{'winner':<8} {'turns':>5} {'time':>6}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['game_num']:>3} "
              f"{r.get(r1+'_player','?'):<16} {r.get(r1+'_type','?'):<10} "
              f"{r.get(r2+'_player','?'):<16} {r.get(r2+'_type','?'):<10} "
              f"{r['winner']:<8} {r['turns']:>5} {r['duration']:>5.1f}s")

    # Save experiment summary as JSON
    summary_path = os.path.join(args.output, f"experiment_{game}_{p1_name}_{p2_name}.json".replace('/', '-'))
    summary = {
        'experiment': experiment_meta,
        'results': results,
        'wins': wins,
        'draws': draws,
        'errors': errors,
        'total_tokens': total_tokens,
        'elapsed_s': round(elapsed, 2),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nExperiment summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
