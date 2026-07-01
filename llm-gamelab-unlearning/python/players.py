"""
players.py - Unified Player interface for GGP and LLM agents.

LLM prompt design for the NeurIPS paper:
  - System prompt: game rules in natural language + move format example
  - User prompt per turn: board state only (includes control fact, NO legal moves)
  - Response expected: ONLY the move in GDL format, nothing else
  - No explanation/justification asked — cleaner parsing, less noise
  - Token usage (input + output) tracked per move

The board state already includes ["control", "x"] or ["control", "red"] etc.
so the LLM knows whose turn it is from the state itself.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
import json
import time
import ast
import re


# ---------------------------------------------------------------------------
# Move format examples per game (shown once in the system prompt)
# ---------------------------------------------------------------------------
MOVE_FORMATS = {
    "tictactoe": {
        "format": '["mark", "ROW", "COL"]',
        "example": '["mark", "2", "2"]',
    },
    "suicide": {
        "format": '["mark", "ROW", "COL"]',
        "example": '["mark", "1", "3"]',
    },
    "connectfour": {
        "format": '["drop", "COL"]',
        "example": '["drop", "4"]',
    },
    "notconnectfour": {
        "format": '["drop", "COL"]',
        "example": '["drop", "4"]',
    },
    "breakthrough": {
        "format": '["move", "FROM_ROW", "FROM_COL", "TO_ROW", "TO_COL"]',
        "example": '["move", "2", "3", "3", "3"]',
    },
    "hex7x7": {
        "format": '["place", "ROW_LETTER", "COL_NUMBER"]',
        "example": '["place", "d", "4"]',
    },
    "centralis": {
        "format": '["mark", "SECTION", "RING"] for peripheral, ["mark", "c", "c"] for center',
        "example": '["mark", "5", "2"]',
    },
    "lines": {
        "format": '["place", "ROW_LETTER", "COL_NUMBER"]',
        "example": '["place", "e", "5"]',
    },
}


@dataclass
class MoveRecord:
    """Detailed record of a single move decision."""
    move: Optional[list]
    valid: bool
    latency_ms: float
    raw_response: Optional[str] = None
    reason: Optional[str] = None
    attempts: int = 1
    illegal_attempts: List[list] = field(default_factory=list)
    error: Optional[str] = None
    prompt: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Player(ABC):
    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.role: Optional[str] = None

    @abstractmethod
    def start(self, role: str, startclock: int = 10, playclock: int = 10): ...

    @abstractmethod
    def play(self, last_move: Optional[list], state: list,
             legal_moves: List[list]) -> MoveRecord: ...

    def stop(self, last_move: Optional[list]):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------------------------------------------------------
# GGP Player
# ---------------------------------------------------------------------------
class GGPPlayer(Player):
    def __init__(self, algorithm: str, game_name: str, seed: Optional[int] = None):
        super().__init__(name=algorithm, kind='ggp')
        from ggp_player_client import GGPPlayerClient
        self.client = GGPPlayerClient(algorithm, game_name, seed=seed)
        self.client.spawn()
        self.client.ping()

    def start(self, role: str, startclock: int = 10, playclock: int = 10):
        self.role = role
        self.client.start(role=role, startclock=startclock, playclock=playclock)

    def play(self, last_move: Optional[list], state: list,
             legal_moves: List[list]) -> MoveRecord:
        t0 = time.perf_counter()
        try:
            move = self.client.play(last_move)
            latency = (time.perf_counter() - t0) * 1000
            if move is False or move is None:
                return MoveRecord(move=None, valid=False, latency_ms=latency,
                                  error="player returned false (not their turn?)")
            valid = any(_move_equal(move, l) for l in legal_moves)
            return MoveRecord(move=move, valid=valid, latency_ms=latency,
                              reason=f"{self.name} algorithm")
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return MoveRecord(move=None, valid=False, latency_ms=latency,
                              error=str(e))

    def stop(self, last_move: Optional[list]):
        try:
            self.client.stop(last_move)
        except Exception:
            pass

    def close(self):
        self.client.close()


# ---------------------------------------------------------------------------
# LLM Player
# ---------------------------------------------------------------------------
class LLMPlayer(Player):
    """
    LLM agent that receives:
      - System prompt once: rules in natural language + move format
      - Per turn: opponent's last move + current board state (with control)
      - Expected response: ONLY the move as a list, e.g. ["mark","2","2"]

    No legal moves shown. No justification asked. Pure reasoning test.
    On retry (after illegal move), legal moves ARE shown — at that point
    the test of reasoning already failed and we want to recover the game.
    """

    def __init__(self, model: str, client,
                 game_name: str, rules_prompt: str,
                 system_prompt: Optional[str] = None,
                 max_retries: int = 10,
                 temperature: float = 0.0,
                 show_legals: bool = False):
        super().__init__(name=model, kind='llm')
        self.model = model
        self.client = client
        self.game_name = game_name
        self.rules_prompt = rules_prompt
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.temperature = temperature
        self.show_legals = show_legals
        self.history: List[dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def start(self, role: str, startclock: int = 10, playclock: int = 10):
        self.role = role
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.history = [{
            "role": "system",
            "content": self._build_system_message(role),
        }]

    def _build_system_message(self, role: str) -> str:
        # Get move format for this game
        fmt = MOVE_FORMATS.get(self.game_name, {})
        move_format = fmt.get('format', '["action", "arg1", "arg2"]')
        move_example = fmt.get('example', '["action", "1", "1"]')

        base = self.system_prompt or _DEFAULT_SYSTEM_PROMPT

        return (
            f"{base}\n\n"
            f"You are playing as: {role}\n"
            f"Game: {self.game_name}\n\n"
            f"RULES:\n{self.rules_prompt}\n\n"
            f"MOVE FORMAT: {move_format}\n"
            f"Example: {move_example}\n\n"
            f"IMPORTANT:\n"
            f"- Respond ONLY with your move as a JSON list, nothing else.\n"
            f"- All values in the list must be strings.\n"
            f"- DO NOT output <think> tags, internal reasoning, or Chain of Thought.\n" # AÑADIDO
            f"- Just output the JSON list.\n" # AÑADIDO
            f"- Do not add explanations, markdown, or any other text.\n"
            f"- The board state includes a fact like [\"control\", \"{role}\"] "
            f"which tells you it is your turn.\n"
            f"- You must only play on empty/valid positions according to the rules."
        )

    def play(self, last_move: Optional[list], state: list,
             legal_moves: List[list]) -> MoveRecord:
        t0 = time.perf_counter()
        illegal_attempts: List[list] = []
        total_p_tokens = 0
        total_c_tokens = 0

        # Build user message: last move + board state, NO legals
        user_msg = self._build_user_message(last_move, state)
        self.history.append({"role": "user", "content": user_msg})

        for attempt in range(self.max_retries + 1):
            try:
                raw, move, p_tokens, c_tokens = self._call_llm()
                total_p_tokens += p_tokens
                total_c_tokens += c_tokens
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                return MoveRecord(
                    move=None, valid=False, latency_ms=latency,
                    error=str(e), attempts=attempt + 1,
                    illegal_attempts=illegal_attempts, prompt=user_msg,
                    prompt_tokens=total_p_tokens,
                    completion_tokens=total_c_tokens,
                    total_tokens=total_p_tokens + total_c_tokens)

            # Validate
            is_legal = move is not None and any(
                _move_equal(move, l) for l in legal_moves)

            if is_legal:
                latency = (time.perf_counter() - t0) * 1000
                self.total_prompt_tokens += total_p_tokens
                self.total_completion_tokens += total_c_tokens
                return MoveRecord(
                    move=move, valid=True, latency_ms=latency,
                    raw_response=raw, reason=raw.strip(),
                    attempts=attempt + 1,
                    illegal_attempts=illegal_attempts, prompt=user_msg,
                    prompt_tokens=total_p_tokens,
                    completion_tokens=total_c_tokens,
                    total_tokens=total_p_tokens + total_c_tokens)

            # Illegal — record it and tell the LLM to try again (no legals shown)
            if move is not None:
                illegal_attempts.append(move)

            if attempt < self.max_retries:
                feedback = self._build_retry_message(move)
                self.history.append({"role": "user", "content": feedback})
                continue

        # Exhausted all retries → forfeit
        latency = (time.perf_counter() - t0) * 1000
        self.total_prompt_tokens += total_p_tokens
        self.total_completion_tokens += total_c_tokens
        return MoveRecord(
            move=None, valid=False, latency_ms=latency,
            raw_response=raw if 'raw' in dir() else None,
            reason=f"forfeit after {self.max_retries + 1} illegal attempts",
            attempts=self.max_retries + 1,
            illegal_attempts=illegal_attempts,
            error="illegal_move_forfeit", prompt=user_msg,
            prompt_tokens=total_p_tokens,
            completion_tokens=total_c_tokens,
            total_tokens=total_p_tokens + total_c_tokens)

    def _build_user_message(self, last_move, state):
        """Board state in natural language + raw GDL. Control is in both."""
        parts = []
        if last_move is None:
            parts.append("You start the game. This is the initial board:")
        else:
            parts.append(f"Opponent played: {json.dumps(last_move)}")
            parts.append("Current board:")#
    
        # Natural language board (set by orchestrator before calling play)
        if hasattr(self, '_board_natural') and self._board_natural:
            parts.append(self._board_natural)
        
        # Raw GDL state
        parts.append("\nRaw state (GDL):")
        parts.append(_format_state(state))

        parts.append("\nYour move:")
        return "\n".join(parts)

    def _build_retry_message(self, bad_move):
        """Just tell the LLM the move is invalid. No legals shown, must reason."""
        return (
            f"Invalid move: {json.dumps(bad_move)}. "
            f"That position is not available. Try again."
        )

    def _call_llm(self):
        """Returns (raw_text, parsed_move, prompt_tokens, completion_tokens)."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=self.temperature,
        )
        raw = resp.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": raw})

        # Token usage
        p_tokens = getattr(resp.usage, 'prompt_tokens', 0) if resp.usage else 0
        c_tokens = getattr(resp.usage, 'completion_tokens', 0) if resp.usage else 0

        move = _parse_move(raw)
       
        return raw, move, p_tokens, c_tokens


# ---------------------------------------------------------------------------
# Move parsing — extract a flat list of strings from raw LLM output
# ---------------------------------------------------------------------------

_LIST_RE = re.compile(r'\[([^\[\]]*)\]')


def _parse_move(text: str) -> Optional[list]:
    """
    Parse a move from raw LLM output. Expects a JSON list like ["mark","2","2"].
    Tolerant to markdown fences, quotes, stray text before/after.
    """
    if not text:
        return None

    cleaned = text.strip()
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
    cleaned = cleaned.replace('```', '').strip()

    # Try direct JSON parse of the whole thing
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        # If it's a dict with 'move' key (LLM added explanation despite instructions)
        if isinstance(parsed, dict) and 'move' in parsed:
            return [str(x) for x in parsed['move']]
    except Exception:
        pass

    # Find any JSON list in the text
    for m in _LIST_RE.finditer(cleaned):
        try:
            lst = json.loads('[' + m.group(1) + ']')
            if lst and all(isinstance(x, (str, int, float)) for x in lst):
                return [str(x) for x in lst]
        except Exception:
            pass

    # Try ast.literal_eval as last resort
    for m in _LIST_RE.finditer(cleaned):
        try:
            lst = ast.literal_eval('[' + m.group(1) + ']')
            return [str(x) for x in lst]
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _move_equal(a, b) -> bool:
    return json.dumps(a) == json.dumps(b)


def _format_state(state: list) -> str:
    """Render state as one fact per line."""
    return "\n".join(json.dumps(f) for f in state)

class HuggingFaceLocalPlayer(LLMPlayer):
    """
    Agente LLM que ejecuta la inferencia localmente usando transformers y peft.

    El prompt de inferencia replica exactamente el formato del dataset de entrenamiento:
        "Game: <name>. Rule: <rules> GDL Board: <state>. Current player: <role>.
         Legal moves: <legal>. Choose your move."

    Cada turno es independiente (sin historial acumulado), igual que en el dataset.
    En cada reintento se resetea el historial con un mensaje directo para romper loops.
    """
    def __init__(self, model, tokenizer, model_name: str,
                 game_name: str, rules_prompt: str,
                 system_prompt: Optional[str] = None,
                 max_retries: int = 10,
                 temperature: float = 0.01):

        super().__init__(model=model_name, client=None, game_name=game_name,
                         rules_prompt=rules_prompt, system_prompt=system_prompt,
                         max_retries=max_retries, temperature=temperature)

        self.model = model
        self.tokenizer = tokenizer
        self.kind = 'hf'
        self._current_legal_moves: List[list] = []

    def start(self, role: str, startclock: int = 10, playclock: int = 10):
        """Sin system prompt separado — alineado con el formato flat del dataset."""
        self.role = role
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.history = []

    def play(self, last_move: Optional[list], state: list,
             legal_moves: List[list]) -> MoveRecord:
        """
        Sobrescribe play() completamente para:
        - Resetear historial en cada turno (single-turn como el dataset)
        - Resetear historial en cada reintento (rompe loops)
        - Mensaje de reintento agresivo y directo
        """
        import time as _time
        self._current_legal_moves = legal_moves
        t0 = _time.perf_counter()
        total_p_tokens = 0
        total_c_tokens = 0
        illegal_attempts = []

        # Prompt principal del turno
        turn_msg = self._build_user_message(last_move, state)

        for attempt in range(self.max_retries + 1):
            if attempt == 0:
                # Primer intento: prompt del turno fresco
                self.history = [{"role": "user", "content": turn_msg}]
            else:
                # Reintento: historial limpio con mensaje directo — rompe el loop
                retry_msg = (
                    f"Wrong answer. Respond ONLY with a JSON list. "
                    f"Pick one from: {json.dumps(legal_moves)}. "
                    f"Example: {json.dumps(legal_moves[0])}. "
                    f"No explanation. No markdown. Just the list."
                )
                self.history = [{"role": "user", "content": retry_msg}]

            raw, move, p_tok, c_tok = self._call_llm()
            total_p_tokens += p_tok
            total_c_tokens += c_tok

            if move is not None and any(_move_equal(move, l) for l in legal_moves):
                latency = (_time.perf_counter() - t0) * 1000
                return MoveRecord(
                    move=move, valid=True, latency_ms=latency,
                    error=None, attempts=attempt + 1,
                    illegal_attempts=illegal_attempts, prompt=turn_msg,
                    prompt_tokens=total_p_tokens, completion_tokens=total_c_tokens
                )

            illegal_attempts.append(move)

        latency = (_time.perf_counter() - t0) * 1000
        return MoveRecord(
            move=None, valid=False, latency_ms=latency,
            error="illegal_move_forfeit", attempts=self.max_retries + 1,
            illegal_attempts=illegal_attempts, prompt=turn_msg,
            prompt_tokens=total_p_tokens, completion_tokens=total_c_tokens
        )

    def _build_user_message(self, last_move, state) -> str:
        """Replica el formato exacto del dataset de entrenamiento."""
        current_player = next(
            (f[1] for f in state if isinstance(f, list) and f[0] == "control"),
            self.role
        )
        board_gdl = json.dumps(state)
        legal_str = json.dumps(self._current_legal_moves)

        return (
            f"Game: {self.game_name}. "
            f"Rule: {self.rules_prompt} "
            f"GDL Board: {board_gdl}. "
            f"Current player: {current_player}. "
            f"Choose your move."
        )

    def _call_llm(self):
        """Generación local con thinking deshabilitado y tokens suficientes."""
        import torch

        prompt = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,  # suficiente para "Thought: ...\nMove: [...]"
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        output_ids = outputs[0][input_len:]
        raw_response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Extraer el primer JSON list — maneja "Thought:...\nMove: [...]"
        match = re.search(r'\[[^\[\]]*\]', raw_response)
        clean_response = match.group(0) if match else raw_response

        self.history.append({"role": "assistant", "content": raw_response})
        move = _parse_move(clean_response)

        return raw_response, move, input_len, len(output_ids)

class TVNPlayer(HuggingFaceLocalPlayer):
    """
    Agente LLM para modelos entrenados con Task Vector Negation (TVN).
    El formato de entrenamiento es Alpaca:
 
        ### Instruction:
        {rol del modelo como agente estratégico experto}
 
        ### Input:
        {estado GDL + movimientos legales + jugador en turno}
 
        ### Response:
        {CoT + movimiento óptimo}
 
    A diferencia de NPO (formato flat single-turn), TVN usa el template
    Alpaca directamente como string — sin apply_chat_template.
    """
 
    # Instrucciones por juego — definen el rol del agente
    TVN_INSTRUCTIONS = {
        # Instrucción exacta del dataset TVN para suicide TTT
        "suicide": (
            "You are an expert Tic-Tac-Toe strategist. Analyze the GDL board state "
            "and play optimally to win, following strategic rules and maximizing "
            "your victory probability."
        ),
        "tictactoe": (
            "You are an expert Tic-Tac-Toe strategist. Analyze the GDL board state "
            "and play optimally to win, following strategic rules and maximizing "
            "your victory probability."
        ),
        "notconnectfour": (
            "You are an expert Connect Four strategist. Analyze the GDL board state "
            "and play optimally to win, following strategic rules and maximizing "
            "your victory probability."
        ),
        "connectfour": (
            "You are an expert Connect Four strategist. Analyze the GDL board state "
            "and play optimally to win, following strategic rules and maximizing "
            "your victory probability."
        ),
    }
 
    def _build_alpaca_prompt(self, instruction: str, input_text: str) -> str:
        """Construye el prompt en formato Alpaca exacto del entrenamiento TVN."""
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
 
    def _build_user_message(self, last_move, state) -> str:
        """
        Construye el campo 'Input' del formato Alpaca replicando
        exactamente el formato del dataset TVN:
            GDL Board State: [...]
            Legal Moves: [...]
            Action for Player: X/O
        """
        current_player = next(
            (f[1] for f in state if isinstance(f, list) and f[0] == "control"),
            self.role
        )
 
        # El dataset incluye el hecho control en el board
        board_gdl = json.dumps(state)
        legal_str = json.dumps(self._current_legal_moves)
 
        # "Action for Player:" usa mayúscula — exacto del dataset
        player_label = current_player.upper()
 
        return (
            f"GDL Board State: {board_gdl}\n"
            f"Legal Moves: {legal_str}\n"
            f"Action for Player: {player_label}"
        )
 
    def _call_llm(self):
        """
        Generación con formato Alpaca — NO usa apply_chat_template.
        El prompt se construye directamente como string.
        """
        import torch
 
        # Obtener la instrucción para este juego
        instruction = self.TVN_INSTRUCTIONS.get(
            self.game_name,
            f"You are an expert AI agent playing {self.game_name}. Choose the best move."
        )
 
        # El historial tiene un solo mensaje user con el input
        input_text = self.history[0]["content"] if self.history else ""
 
        # Construir prompt Alpaca directamente como string
        prompt = self._build_alpaca_prompt(instruction, input_text)
 
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.model.device)
 
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
 
        input_len = inputs.input_ids.shape[1]
        output_ids = outputs[0][input_len:]
        raw_response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
 
        # Extraer el último JSON list — el CoT viene primero, el movimiento al final
        # El dataset usa formato "Move: ['mark', 'row', 'col']"
        move_match = re.search(r"Move:\s*(\[[^\[\]]*\])", raw_response)
        if move_match:
            clean_response = move_match.group(1)
        else:
            matches = re.findall(r'\[[^\[\]]*\]', raw_response)
            clean_response = matches[-1] if matches else raw_response
 
        self.history.append({"role": "assistant", "content": raw_response})
        move = _parse_move(clean_response)
 
        return raw_response, move, input_len, len(output_ids)
    
_DEFAULT_SYSTEM_PROMPT = """\
You are a player of a turn-based board game. Analyze the board state carefully \
and make your move. Respond ONLY with the move, nothing else. \
Do not respond with ```json ```. Do not add explanations."""