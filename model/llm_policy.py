"""The LLM as a chess player: pretrained Qwen3 + LoRA adapters + a value head.

Intent: turn a generic pretrained language model into something that plays
chess. Given a position, one call returns the chosen move AND the model's
estimate of who is winning. Every later training step (SFT in Step 5, RL in
Step 7) updates this same object, and eval plays it through the harness's
Agent interface (``name`` + ``select_move``).

Main pieces:
- PolicyConfig    -- every setting (model id, the two toggles, LoRA shape, prompt
                     wording, decoding); loaded from the ``policy:`` block of configs/sft.yaml
- build_prompt    -- THE prompt template: the one text every call reads (value head,
                     move scoring, generation, and SFT training examples)
- parse_move      -- how free-generated text is read as a move (unmasked arms)
- ValueHead       -- one linear layer: last prompt token's hidden state -> win probability
- Decision        -- what one call returns: move + win probability (+ details)
- LLMPolicy       -- the wrapper: load, act (masked or unmasked), encode training
                     examples, save / load the trainable parts

The two toggles (plan v3, Sec 0):
- representation: "structured" or "fen" -- which board text goes in the prompt.
- masked: True  = score every legal move as a whole and pick the most likely one
                  (can never be illegal).
          False = let the model write whatever it wants, then parse it. The
                  harness (eval/match.py) checks legality, asks again after an
                  illegal move, and forfeits the game on the third illegal
                  attempt. The first attempt is greedy; retries on the same
                  position sample (at ``retry_temperature``), since asking a
                  deterministic model the same question again would get the
                  same wrong answer.

What "score a move" means: log P(move text + end-of-reply | prompt), summed
token by token (teacher forcing). The end-of-reply token is included, so the
score is the probability that the model's whole reply is exactly this move --
the same thing SFT trains up and the same distribution unmasked generation
samples from. The prompt is run once; its cached keys/values are reused for
every candidate, so a move costs one prompt pass plus a few tokens per legal
move, not one prompt pass per legal move.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import chess
import torch
from torch import nn

from engine import board as B
from engine.encoding import TEXT_ENCODERS, encode_text

IGNORE_INDEX = -100  # label value that PyTorch's cross-entropy skips (prompt tokens)


@dataclass(frozen=True)
class PolicyConfig:
    """All settings for one LLM player. Defaults mirror configs/sft.yaml ``policy:``."""

    model_id: str = "Qwen/Qwen3-0.6B"
    representation: str = "structured"   # "structured" | "fen"
    masked: bool = True                  # the legal-move toggle
    instruction: str = ("You are playing chess. Reply with your move for the side to move, in UCI "
                        "notation (for example e2e4, or e7e8q to promote).")
    enable_thinking: bool = False        # Qwen3's hard switch; False = the non-reasoning baseline
    # LoRA: small trainable matrices beside the frozen weights (Step 5 trains them).
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj")
    dtype: str = "auto"                  # "auto" (bf16 if the GPU has it, else fp16; fp32 on CPU) | "bfloat16" | "float16" | "float32"
    # Masked scoring
    score_batch_size: int = 32           # legal moves scored per forward pass (bounds GPU memory)
    prefix_cache: bool = True            # reuse the prompt's keys/values across candidates
    # Unmasked generation
    max_new_tokens: int = 8              # a UCI move is at most 5 characters
    retry_temperature: float = 1.0       # sampling temperature for retries after an illegal move
    seed: int = 0

    def __post_init__(self):
        if self.representation not in TEXT_ENCODERS:
            raise ValueError(f"representation must be one of {sorted(TEXT_ENCODERS)}, got {self.representation!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyConfig":
        """From a YAML block; lists become tuples, unknown keys are an error."""
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in (d or {}).items()}
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown policy settings {sorted(unknown)}")
        return cls(**d)


def build_prompt(board: chess.Board, tokenizer, representation: str, instruction: str,
                 enable_thinking: bool = False) -> str:
    """THE prompt template -- the only text the model reads about a position.

    The user message is: the instruction line, the board text, then "Move:".
    Both board texts already carry side to move, castling rights, and en
    passant (FEN has them as fields; structured text has labeled lines), so the
    representation toggle is the only thing that differs between arms. The
    message is wrapped in the model's chat template with thinking off, so the
    text ends at the start of the model's reply -- the move is what comes next.
    Legal moves are never listed (that would be a different design, Sec 0).
    """
    content = f"{instruction}\n{encode_text(board, representation)}\nMove:"
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=enable_thinking)


_STRIP = " \t\r\n.,;:!?'\"`*()[]{}<>"


def parse_move(text: str) -> str:
    """Free-generated text -> the move string the harness will check.

    Deliberately strict: the first whitespace-separated word, with surrounding
    punctuation and quotes removed -- nothing else is fixed up (no case change,
    no SAN-to-UCI conversion). "e2e4" and " e2e4." parse to "e2e4"; "Nf3" stays
    "Nf3" and is illegal. Leniency here would hide the illegal-move rate the
    unmasked arms are meant to measure.
    """
    words = text.strip().split()
    return words[0].strip(_STRIP) if words else ""


class ValueHead(nn.Module):
    """hidden state of the last prompt token -> P(side to move wins), 0..1.

    One linear layer then a sigmoid (the sigmoid is applied by the caller, so
    training can use the raw logit). Zero-initialized, so an untrained head says
    0.5 -- "no idea" -- rather than a random opinion. Kept in float32.
    """

    def __init__(self, hidden_size: int):
        """``hidden_size``: the base model's hidden width (1024 for Qwen3-0.6B)."""
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """(..., hidden_size) -> (...) logits."""
        return self.linear(hidden.float()).squeeze(-1)


@dataclass
class Decision:
    """One call's output: the move and the win probability, together."""

    move: str                    # UCI; always legal when masked, whatever was parsed when unmasked
    win_prob: float              # value head: P(side to move wins), before the move is played
    legal: bool
    move_logprobs: dict[str, float] | None = None  # masked: log P(reply = move) for every legal move
    raw_text: str | None = None  # unmasked: what the model actually wrote
    attempt: int = 1             # 2, 3 = the harness asking again after an illegal move
    sampled: bool = False        # unmasked: True if this attempt sampled instead of greedy


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda":  # native bf16 needs compute capability 8+ (L4 yes, T4 no)
            return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class LLMPolicy:
    """Pretrained LLM + LoRA + value head, playing one toggle combination.

    Build with ``LLMPolicy.from_pretrained(config)``. ``variant(...)`` gives a
    player with other toggles that shares the same weights (no second copy).
    """

    def __init__(self, model, tokenizer, config: PolicyConfig, *, value_head: ValueHead | None = None,
                 add_lora: bool = True, name: str | None = None):
        """Wrap an already-loaded causal LM and tokenizer.

        add_lora=True attaches fresh LoRA adapters (they start as an exact no-op,
        so the untrained policy plays like the base model). Pass add_lora=False
        when ``model`` already carries adapters (e.g. loaded from a checkpoint).
        """
        if add_lora:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules), task_type="CAUSAL_LM"))
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.value_head = (value_head or ValueHead(model.config.hidden_size)).to(self.device)
        self.model.eval()
        self.name = name or f"llm_{config.representation}_{'masked' if config.masked else 'unmasked'}"
        # The token that ends the model's reply in the chat format (<|im_end|> for Qwen).
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.end_id = end if isinstance(end, int) and end != tokenizer.unk_token_id else tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.end_id
        self._rng = random.Random(config.seed)
        self._last_fen: str | None = None
        self._attempt = 0
        self.illegal_log: list[dict] = []  # unmasked: every illegal attempt (fen, raw text, parsed move)

    # ------------------------------------------------------------ loading ----

    @classmethod
    def from_pretrained(cls, config: PolicyConfig, *, device: str | None = None,
                        adapter_dir: str | Path | None = None, trainable: bool = False,
                        name: str | None = None) -> "LLMPolicy":
        """Load the base model (Hugging Face id or local path) and attach LoRA + value head.

        adapter_dir -- a folder written by ``save``: load its adapters and value
                       head instead of fresh ones (a trained checkpoint)
        trainable   -- with adapter_dir, keep the loaded adapters trainable (resuming training)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=_resolve_dtype(config.dtype, dev)).to(dev)
        if adapter_dir is None:
            return cls(model, tokenizer, config, name=name)
        from peft import PeftModel
        adapter_dir = Path(adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
        head = ValueHead(model.config.hidden_size)
        head.load_state_dict(torch.load(adapter_dir / "value_head.pt", map_location="cpu"))
        return cls(model, tokenizer, config, value_head=head, add_lora=False, name=name)

    def save(self, out_dir: str | Path) -> None:
        """Save only what training changes: LoRA adapters, value head, and the config.

        The frozen base model is never saved (re-download it). Optimizer state
        and resume logic belong to training (Step 5, utils/checkpoint.py).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        torch.save(self.value_head.state_dict(), out_dir / "value_head.pt")
        (out_dir / "policy_config.json").write_text(json.dumps(dataclasses.asdict(self.config), indent=2))

    def variant(self, name: str | None = None, **changes) -> "LLMPolicy":
        """Same weights, other settings -- e.g. variant(masked=False, representation="fen")."""
        other = copy.copy(self)
        other.config = dataclasses.replace(self.config, **changes)
        other.name = name or f"llm_{other.config.representation}_{'masked' if other.config.masked else 'unmasked'}"
        other._rng = random.Random(other.config.seed)
        other._last_fen, other._attempt, other.illegal_log = None, 0, []
        return other

    def trainable_parameters(self) -> list[nn.Parameter]:
        """What an optimizer should update: the LoRA adapters and the value head."""
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.value_head.parameters())

    # ------------------------------------------------------------ the text ----

    def prompt(self, board: chess.Board) -> str:
        """The prompt text for this position (see build_prompt)."""
        c = self.config
        return build_prompt(board, self.tokenizer, c.representation, c.instruction, c.enable_thinking)

    def prompt_ids(self, board: chess.Board) -> list[int]:
        """Token ids of the prompt. The chat template already holds every special token."""
        return self.tokenizer(self.prompt(board), add_special_tokens=False)["input_ids"]

    def move_ids(self, move: str) -> list[int]:
        """Token ids of a reply that is exactly ``move``: its text, then end-of-reply."""
        return self.tokenizer(move, add_special_tokens=False)["input_ids"] + [self.end_id]

    def training_example(self, board: chess.Board, move: str) -> dict:
        """One SFT example, built from the same prompt and move tokens that play uses.

        Output: input_ids = prompt + move + end-of-reply; labels = the same ids with
        the prompt masked out (IGNORE_INDEX), so the move loss grades only the
        move; value_index = position of the last prompt token, where the value
        head reads (it never sees the move).
        """
        p, m = self.prompt_ids(board), self.move_ids(move)
        return {"input_ids": p + m, "labels": [IGNORE_INDEX] * len(p) + m, "value_index": len(p) - 1}

    # ------------------------------------------------------------- playing ----

    def select_move(self, board: chess.Board) -> str:
        """Agent interface for eval/match.py: just the move."""
        return self.act(board).move

    @torch.inference_mode()
    def act(self, board: chess.Board) -> Decision:
        """Pick a move and estimate the win probability, in one call.

        The harness calls again on the same position after an illegal move;
        that is counted here (``attempt``) and, when unmasked, switches from
        greedy to sampling.
        """
        if B.is_terminal(board) or not any(board.legal_moves):
            raise ValueError(f"no move to make: the game is over ({board.fen()})")
        fen = board.fen()
        self._attempt = self._attempt + 1 if fen == self._last_fen else 1
        self._last_fen = fen
        self.model.eval()  # play never uses dropout (training switches it back on itself)
        if self.config.masked:
            return self._act_masked(board)
        return self._act_unmasked(board, fen)

    def _value(self, last_hidden: torch.Tensor) -> float:
        return torch.sigmoid(self.value_head(last_hidden)).item()

    def _act_masked(self, board: chess.Board) -> Decision:
        moves = B.legal_moves(board)
        scores, win_prob = self.score_moves(board, moves)
        best = max(moves, key=lambda m: scores[m])  # ties: first in sorted UCI order
        return Decision(move=best, win_prob=win_prob, legal=True, move_logprobs=scores, attempt=self._attempt)

    def _act_unmasked(self, board: chess.Board, fen: str) -> Decision:
        from transformers import GenerationConfig
        sample = self._attempt > 1
        ids = torch.tensor([self.prompt_ids(board)], device=self.device)
        gen = GenerationConfig(max_new_tokens=self.config.max_new_tokens, do_sample=sample,
                               temperature=self.config.retry_temperature if sample else None,
                               top_p=1.0 if sample else None, top_k=0 if sample else None,
                               eos_token_id=self.end_id, pad_token_id=self.pad_id,
                               output_hidden_states=True, return_dict_in_generate=True)
        if sample:  # reproducible sampling: seed torch from this player's own RNG
            torch.manual_seed(self._rng.randrange(2**31))
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gen)
        # hidden_states[0] is the prompt pass; its last layer at the last prompt token feeds the value head.
        win_prob = self._value(out.hidden_states[0][-1][0, -1])
        raw = self.tokenizer.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True)
        move = parse_move(raw)
        legal = B.is_legal(board, move)
        if not legal:
            self.illegal_log.append({"fen": fen, "raw_text": raw, "parsed": move, "attempt": self._attempt})
        return Decision(move=move, win_prob=win_prob, legal=legal, raw_text=raw, attempt=self._attempt,
                        sampled=sample)

    # ------------------------------------------------------------- scoring ----

    @torch.inference_mode()
    def score_moves(self, board: chess.Board, moves: list[str]) -> tuple[dict[str, float], float]:
        """log P(reply = move | prompt) for each move, plus the value head's win probability.

        Core logic: run the prompt once (keeping its keys/values), read the value
        head at the last prompt token, then score the candidates in batches.
        Candidate i's token j is scored by the logits one position earlier: the
        prompt's last position for j = 0, the candidate's own token j-1 after
        that. Candidates are right-padded; padding sits after the real tokens,
        so the causal mask keeps it from affecting them, and it's excluded from
        the sums.
        """
        self.model.eval()
        prompt = torch.tensor([self.prompt_ids(board)], device=self.device)
        out = self.model(input_ids=prompt, use_cache=self.config.prefix_cache, output_hidden_states=True,
                         logits_to_keep=1)  # only the last position's next-token logits are needed
        win_prob = self._value(out.hidden_states[-1][0, -1])
        first_logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        scores: dict[str, float] = {}
        for start in range(0, len(moves), self.config.score_batch_size):
            chunk = moves[start:start + self.config.score_batch_size]
            cand = [self.move_ids(m) for m in chunk]
            k = max(len(c) for c in cand)
            ids = torch.full((len(chunk), k), self.pad_id, dtype=torch.long, device=self.device)
            real = torch.zeros((len(chunk), k), dtype=torch.bool, device=self.device)
            for i, c in enumerate(cand):
                ids[i, :len(c)] = torch.tensor(c, device=self.device)
                real[i, :len(c)] = True
            # logp[i, j] = log P(token j of candidate i | prompt + its tokens before j)
            if self.config.prefix_cache:
                cache = copy.deepcopy(out.past_key_values)
                cache.batch_repeat_interleave(len(chunk))
                logits = self.model(input_ids=ids, past_key_values=cache, use_cache=True).logits
                logp = torch.cat([first_logp.expand(len(chunk), 1, -1),
                                  torch.log_softmax(logits[:, :k - 1].float(), dim=-1)], dim=1)
            else:  # recompute the prompt for every candidate (reference path; same numbers, more compute)
                full = torch.cat([prompt.expand(len(chunk), -1), ids], dim=1)
                logits = self.model(input_ids=full, logits_to_keep=k + 1).logits  # positions P-1 .. P+k-1
                logp = torch.log_softmax(logits[:, :k].float(), dim=-1)
            tok_logp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            for m, s in zip(chunk, (tok_logp * real).sum(dim=1).tolist()):
                scores[m] = s
        return scores, win_prob
