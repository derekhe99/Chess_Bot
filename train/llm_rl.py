"""Step 7: LLM self-play RL -- keep improving the SFT model by playing itself.

Intent: SFT taught the model to imitate Stockfish; RL lets it learn from its
own games. Each ROUND (plan Sec 0 terminology):
  1. Self-play: the current model plays ``games_per_round`` games against
     itself (search/selfplay.py). Each move is SAMPLED from the masked move
     distribution -- softmax over the legal moves' scores, at ``temperature`` --
     so games differ and the model tries moves it isn't yet sure of.
  2. Reward: every move gets its game's final result from the mover's side,
     as a score: win 1, draw 0.5, loss 0.
  3. Update: one pass over the round's positions with three loss terms:
     - policy gradient:  -advantage * log P(move | position)
       advantage = score - value-head prediction at that position (made when
       the move was played). The prediction is the BASELINE: a move from a
       position the model already thought was winning gets little credit for
       winning; a win from a position it thought was lost gets a lot. That
       cuts the noise of "every move in a won game was good".
     - KL penalty: kl_coef * KL(current model || SFT model), estimated on the
       played move. Keeps the model from drifting far from SFT in a few noisy
       rounds (the drift is what "collapse" usually looks like).
     - value loss: the value head learns to predict the game result (BCE vs
       the score), so the baseline keeps improving.
  4. Save: resume state every round; a kept checkpoint every ``checkpoint_every``.

Why log P(move) and not the masked (renormalized-over-legal) log-probability:
it's the same quantity SFT trained and masked play ranks by, and it costs one
forward pass per position instead of one per legal move. The two gradients
differ by a term whose expected value is ~0 once a baseline is subtracted.

Compute: RL GPU time = wall-clock of self-play + updates (both occupy the
GPU). A warm-started run also carries the SFT run's GPU-hours, so its
Elo-vs-compute curve continues where SFT's ended (``train_gpu_hours`` column).

Main pieces:
- RLConfig          -- every setting; configs/llm_rl.yaml
- run_name          -- the run folder's name
- start_policies    -- the trainable policy (SFT weights) + the frozen SFT reference, + SFT's GPU-hours
- sampling_picker   -- the self-play move picker: sample from the masked distribution
- build_examples    -- finished games -> training examples (tokens, score, advantage)
- update            -- one pass of policy-gradient + KL + value updates
- train             -- the resumable round loop; writes rl_log.csv
- list_checkpoints / gpu_hours_at -- for evaluation into results.csv
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from engine import board as B
from model.llm_policy import LLMPolicy, PolicyConfig
from search.selfplay import MovePicker, SelfPlayGame, play_selfplay_games
from train.sft import _load_adapters, batch_losses, collate, gpu_hours_at as sft_gpu_hours_at, load_checkpoint


@dataclass(frozen=True)
class RLConfig:
    """All RL settings. Defaults mirror configs/llm_rl.yaml (provisional, Oct 17, 2026)."""

    init_run: str | None = "structured_masked_nall_e10_s0"  # SFT run to warm-start from; None = base model (pure RL)
    init_step: int = 25000            # SFT checkpoint step
    rounds: int = 100                 # upper bound; stop earlier by hand once Elo plateaus
    games_per_round: int = 64
    opening_plies: int = 4            # random opening half-moves per game (variety; not trained on)
    temperature: float = 1.0          # sampling temperature over the legal-move scores
    max_plies: int = 300              # longer games are adjudicated draws
    step_positions: int = 256         # positions per optimizer step (gradient accumulation)
    micro_batch: int = 16             # positions per forward/backward (bounds GPU memory)
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    kl_coef: float = 0.1
    value_loss: str = "bce"           # "bce" | "mse"
    value_loss_weight: float = 1.0
    checkpoint_every: int = 5         # rounds between KEPT checkpoints (the latest round is always on disk)
    seed: int = 0

    def __post_init__(self):
        if self.value_loss not in ("bce", "mse"):
            raise ValueError(f"value_loss must be 'bce' or 'mse', got {self.value_loss!r}")
        if min(self.rounds, self.games_per_round, self.step_positions, self.micro_batch, self.checkpoint_every) < 1:
            raise ValueError("rounds, games_per_round, step_positions, micro_batch, checkpoint_every must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

    @classmethod
    def from_dict(cls, d: dict) -> "RLConfig":
        """From a YAML block; unknown keys are an error (catches typos)."""
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown RL settings {sorted(unknown)}")
        return cls(**d)


def run_name(policy_cfg: PolicyConfig, cfg: RLConfig) -> str:
    """e.g. 'structured_masked_sft25000_s0' (warm start) or 'structured_masked_base_s0' (pure RL)."""
    mask = "masked" if policy_cfg.masked else "unmasked"
    start = f"sft{cfg.init_step}" if cfg.init_run else "base"
    return f"{policy_cfg.representation}_{mask}_{start}_s{cfg.seed}"


def start_policies(cfg: RLConfig, artifact_root: str | Path, *, policy_cfg: PolicyConfig | None = None,
                   device: str | None = None) -> tuple[LLMPolicy, LLMPolicy, float]:
    """(trainable policy, frozen reference, GPU-hours already spent to reach the start).

    Warm start: both load the SFT checkpoint (its own policy_config.json, so the
    LoRA shape matches); the trainable one keeps its adapters trainable, the
    reference never changes. The start's GPU-hours are the SFT run's at that step.
    Pure RL (init_run None): both start as the base model (fresh LoRA is a no-op).
    """
    if cfg.init_run:
        sft_run = Path(artifact_root) / "checkpoints" / "sft" / cfg.init_run
        step_dir = sft_run / "checkpoints" / f"step_{cfg.init_step:06d}"
        pcfg = PolicyConfig.from_dict(json.loads((step_dir / "policy_config.json").read_text()))
        policy = LLMPolicy.from_pretrained(pcfg, device=device)
        _load_adapters(policy, step_dir)
        ref = load_checkpoint(step_dir, device=device, name="sft_reference")
        return policy, ref, sft_gpu_hours_at(sft_run, cfg.init_step)
    if policy_cfg is None:
        raise ValueError("pure RL (init_run=None) needs policy_cfg")
    return (LLMPolicy.from_pretrained(policy_cfg, device=device),
            LLMPolicy.from_pretrained(policy_cfg, device=device, name="base_reference"), 0.0)


# ---------------------------------------------------------- self-play ----

def sampling_picker(policy: LLMPolicy, temperature: float, rng: random.Random) -> MovePicker:
    """Move picker for self-play: sample a legal move with P proportional to exp(score / temperature).

    info carries what training needs: ``value`` (the value head's prediction,
    the baseline) and ``entropy`` of the move distribution (in nats; a fall
    toward 0 means the model is becoming deterministic -- the collapse check).
    """
    def pick(board: chess.Board) -> tuple[str, dict]:
        moves = B.legal_moves(board)
        scores, value = policy.score_moves(board, moves)
        logits = torch.tensor([scores[m] for m in moves], dtype=torch.float64) / temperature
        logp = torch.log_softmax(logits, dim=0)
        probs = logp.exp()
        entropy = float(-(probs * logp).sum())
        i = rng.choices(range(len(moves)), weights=probs.tolist())[0]
        return moves[i], {"value": value, "entropy": entropy}
    return pick


def build_examples(policy: LLMPolicy, games: list[SelfPlayGame]) -> list[dict]:
    """Every recorded move -> one training example: tokens + score (value target) + advantage."""
    examples = []
    for g in games:
        for r in g.moves:
            ex = policy.training_example(chess.Board(r.fen), r.move)
            score = g.score_for(r.white)
            ex["label_value"] = score
            ex["advantage"] = score - r.info["value"]
            examples.append(ex)
    return examples


# ------------------------------------------------------------- update ----

def update(policy: LLMPolicy, ref: LLMPolicy, examples: list[dict], cfg: RLConfig, optimizer, scaler,
           rng: random.Random) -> dict:
    """One pass over ``examples`` in shuffled optimizer steps of ``step_positions``. Returns mean losses.

    Per position: loss = advantage * nll + kl_coef * kl + value_loss_weight * value_loss, where
    nll = -log P(move), and kl = exp(d) - d - 1 with d = log P_ref(move) - log P(move) (an
    always-nonnegative KL estimate that is 0 when the model equals the reference).
    """
    order = list(range(len(examples)))
    rng.shuffle(order)
    params = policy.trainable_parameters()
    totals = {"pg_loss": 0.0, "kl": 0.0, "value_loss": 0.0, "nll": 0.0}
    # Dropout stays OFF (eval mode) during updates: the model being graded is exactly the one
    # that played, and the KL to the reference is exactly 0 before the first step.
    policy.model.eval()
    ref.model.eval()
    for s in range(0, len(order), cfg.step_positions):
        chunk = order[s:s + cfg.step_positions]
        optimizer.zero_grad(set_to_none=True)
        for m in range(0, len(chunk), cfg.micro_batch):
            micro = [examples[i] for i in chunk[m:m + cfg.micro_batch]]
            batch = collate(micro, policy.pad_id, policy.device)
            adv = torch.tensor([e["advantage"] for e in micro], dtype=torch.float32, device=policy.device)
            with torch.no_grad():
                ref_nll, _, _ = batch_losses(ref, batch, cfg.value_loss)
            nll, vloss, _ = batch_losses(policy, batch, cfg.value_loss)
            d = nll - ref_nll                    # = log P_ref(move) - log P(move)
            kl = torch.exp(d) - d - 1
            pg = adv * nll
            loss = (pg + cfg.kl_coef * kl + cfg.value_loss_weight * vloss).sum() / len(chunk)
            scaler.scale(loss).backward()
            totals["pg_loss"] += pg.sum().item()
            totals["kl"] += kl.sum().item()
            totals["value_loss"] += vloss.sum().item()
            totals["nll"] += nll.sum().item()
        scaler.unscale_(optimizer)
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    policy.model.eval()
    n = max(len(examples), 1)
    return {k: v / n for k, v in totals.items()}


# ------------------------------------------------------------ storage ----

def _round_dir(run_dir: Path, rnd: int) -> Path:
    return run_dir / "checkpoints" / f"round_{rnd:04d}"


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """[(round, folder)] for every saved round (kept checkpoints + the latest), oldest first."""
    root = Path(run_dir) / "checkpoints"
    if not root.exists():
        return []
    return sorted((int(p.name.split("_")[1]), p) for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("round_") and not p.name.endswith(".tmp"))


def _save_round(policy: LLMPolicy, run_dir: Path, rnd: int, optimizer, scaler, counters: dict,
                cfg: RLConfig) -> None:
    """Write round ``rnd`` (adapters + value head + resume state) atomically, then prune the previous round.

    A new folder name each round, so the rename is atomic and a crash never
    leaves a half-written latest round. The previous round is deleted unless it
    is a kept checkpoint, in which case only its resume state (optimizer, ~2x
    the adapter size) is dropped.
    """
    final = _round_dir(run_dir, rnd)
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    policy.save(tmp)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "counters": counters},
               tmp / "resume.pt")
    os.replace(tmp, final)
    for r, path in list_checkpoints(run_dir):
        if r >= rnd:
            continue
        if r % cfg.checkpoint_every == 0:
            (path / "resume.pt").unlink(missing_ok=True)
        else:
            shutil.rmtree(path, ignore_errors=True)


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _drop_rows_after(path: Path, rnd: int) -> None:
    """On resume: forget log rows of rounds that weren't saved (they'll be redone)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(r for r in rows if int(r["round"]) <= rnd)


# -------------------------------------------------------------- train ----

def train(policy: LLMPolicy, ref: LLMPolicy, cfg: RLConfig, run_dir: str | Path, *,
          init_gpu_hours: float = 0.0, max_gpu_hours: float | None = None, max_rounds: int | None = None,
          extra_info: dict | None = None, log=print) -> dict:
    """Run self-play RL rounds on ``policy``; resumable. Returns the run summary.

    Inputs:
      policy, ref     -- from start_policies (trainable model; frozen KL reference)
      cfg             -- RLConfig
      run_dir         -- the run's folder (checkpoints, rl_log.csv, run_config.json)
      init_gpu_hours  -- GPU-hours spent before RL (the SFT run's, for a warm start)
      max_gpu_hours   -- stop once total training GPU-hours (init + RL) reach this
                         (base.yaml max_gpu_hours_per_arm; None = no cap)
      max_rounds      -- stop after this round this session (e.g. a short test run); resumable
    Files written in run_dir:
      run_config.json -- settings (a resume with different settings is refused)
      checkpoints/round_NNNN/ -- adapters + value head (+ resume.pt on the latest round)
      rl_log.csv      -- one row per round: results, losses, KL, entropy, seconds, GPU-hours
      summary.json    -- totals at the end of the session
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = policy.device

    settings = {"rl": dataclasses.asdict(cfg), "policy": dataclasses.asdict(policy.config),
                "init_gpu_hours": init_gpu_hours}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text())
        if json.loads(json.dumps(settings)) != {k: saved.get(k) for k in settings}:
            raise ValueError(f"{run_dir} was started with different settings; use a new run folder")
    else:
        cfg_path.write_text(json.dumps({**settings, "extra_info": extra_info or {}}, indent=2))

    params = policy.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    fp16 = device.type == "cuda" and policy.model.get_input_embeddings().weight.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    counters = {"round": 0, "gpu_seconds": 0.0, "games": 0, "positions": 0}
    saved_rounds = list_checkpoints(run_dir)
    if saved_rounds:
        rnd, path = saved_rounds[-1]
        if not (path / "resume.pt").exists():
            raise RuntimeError(f"latest round {path} has no resume.pt; can't resume safely")
        state = torch.load(path / "resume.pt", map_location=device, weights_only=False)
        counters = state["counters"]
        _load_adapters(policy, path)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        _drop_rows_after(run_dir / "rl_log.csv", counters["round"])
        log(f"resuming {run_dir.name} after round {counters['round']} / {cfg.rounds}")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def total_hours() -> float:
        return init_gpu_hours + counters["gpu_seconds"] / 3600

    stop_at = cfg.rounds if max_rounds is None else min(cfg.rounds, max_rounds)
    stopped_by_cap = False
    while counters["round"] < stop_at:
        if max_gpu_hours is not None and total_hours() >= max_gpu_hours:
            stopped_by_cap = True
            log(f"safety cap reached: {total_hours():.2f} >= {max_gpu_hours} GPU-hours")
            break
        rnd = counters["round"] + 1
        round_seed = cfg.seed * 1_000_003 + rnd

        sync()
        t0 = time.perf_counter()
        pick = sampling_picker(policy, cfg.temperature, random.Random(round_seed))
        games = play_selfplay_games(pick, cfg.games_per_round, opening_plies=cfg.opening_plies,
                                    max_plies=cfg.max_plies, seed=round_seed)
        sync()
        t1 = time.perf_counter()
        examples = build_examples(policy, games)
        losses = update(policy, ref, examples, cfg, optimizer, scaler, random.Random(round_seed))
        sync()
        t2 = time.perf_counter()

        counters["round"] = rnd
        counters["gpu_seconds"] += t2 - t0
        counters["games"] += len(games)
        counters["positions"] += len(examples)
        _save_round(policy, run_dir, rnd, optimizer, scaler, counters, cfg)

        infos = [r.info for g in games for r in g.moves]
        row = {"round": rnd, "games": len(games), "positions": len(examples),
               "white_wins": sum(g.white_score == 1.0 for g in games),
               "draws": sum(g.white_score == 0.5 for g in games),
               "black_wins": sum(g.white_score == 0.0 for g in games),
               "max_plies_draws": sum(g.termination == "max_plies" for g in games),
               "avg_plies": round(sum(g.plies for g in games) / len(games), 1),
               "entropy": round(sum(i["entropy"] for i in infos) / max(len(infos), 1), 4),
               "value_pred": round(sum(i["value"] for i in infos) / max(len(infos), 1), 4),
               "abs_advantage": round(sum(abs(e["advantage"]) for e in examples) / max(len(examples), 1), 4),
               **{k: round(v, 5) for k, v in losses.items()},
               "selfplay_seconds": round(t1 - t0, 1), "update_seconds": round(t2 - t1, 1),
               "rl_gpu_hours": round(counters["gpu_seconds"] / 3600, 4),
               "train_gpu_hours": round(total_hours(), 4)}
        _append_csv(run_dir / "rl_log.csv", row)
        log(f"[round {rnd}/{cfg.rounds}] W/D/L {row['white_wins']}/{row['draws']}/{row['black_wins']} "
            f"({row['avg_plies']} plies) | entropy {row['entropy']:.2f} | kl {row['kl']:.4f} | "
            f"value loss {row['value_loss']:.3f} | self-play {row['selfplay_seconds']:.0f}s + update "
            f"{row['update_seconds']:.0f}s | total {row['train_gpu_hours']:.2f} GPU-h")

    summary = {**counters, "rl_gpu_hours": round(counters["gpu_seconds"] / 3600, 4),
               "train_gpu_hours": round(total_hours(), 4), "finished": counters["round"] >= cfg.rounds,
               "stopped_by_cap": stopped_by_cap,
               "kept_checkpoints": [r for r, _ in list_checkpoints(run_dir)]}
    if device.type == "cuda":
        summary["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------- compute used ----

def gpu_hours_at(run_dir: str | Path, rnd: int) -> float:
    """Total training GPU-hours (start + RL) when round ``rnd`` finished -- results.csv's train_gpu_hours."""
    with open(Path(run_dir) / "rl_log.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["round"]) == rnd:
                return float(r["train_gpu_hours"])
    raise ValueError(f"no rl_log.csv row for round {rnd}")
