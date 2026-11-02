"""Step 17: the self-play LLM -- AlphaZero-style training with the LLM as the network.

Intent (plan v5 Step 17): from SFT 500k @30k, spend 10 GPU-h on self-play WITH SEARCH and
compare with spending them on more SFT (pass 2 -> @62.5k). Plain RL (Step 7) failed
because one game result per ~150 moves gives almost no per-move signal. Here every
position gets a detailed target from the search itself:

Each ROUND:
  1. Self-play: the current LLM plays ``games_per_round`` games against itself, all at
     once (search.mcts.play_games). Before every move it runs a 16-simulation search
     (model.llm_evaluator.LLMEvaluator: priors = the LLM's masked move distribution,
     leaf values = its value head). Root noise; moves sampled by visit count for the
     first ``temperature_plies`` plies, then the most-visited move.
  2. Targets, per position: the search's visit distribution over the legal moves (what
     the move scorer should learn) and the game's final result for the side to move
     (what the value head should learn).
  3. Replay buffer: the positions of the last ``buffer_rounds`` rounds (saved to disk,
     so a resumed run has the same buffer).
  4. Update: ``samples_per_position`` x (new positions) training examples drawn from the
     buffer. Each example is one position with ONE move sampled from its visit
     distribution, trained exactly like an SFT example (move NLL + value BCE, SFT's
     learning rate and batch size). Sampling a move by its visit share is an unbiased
     estimate of the cross-entropy to the whole visit distribution, at the cost of one
     SFT example instead of one per visited move.
  5. Save: resume state every round; a kept checkpoint every ``checkpoint_every`` rounds.

No pull toward the starting model (Derek, Nov 1: AlphaZero has none; the targets are
per-move, not one noisy result per game).

Compute: GPU time = wall clock of self-play + updates (both occupy the GPU), as in
train/llm_rl.py. The budget counts only this training (``budget_gpu_hours``); results.csv
rows carry start + this (the SFT run's 9.25 GPU-h at step 30k + the self-play hours).
A round starts only if at least half of it fits in the remaining budget, so the run ends
within about half a round of the budget.

Main pieces:
- MCTSTrainConfig -- every setting; configs/llm_mcts.yaml
- run_name        -- the run folder's name
- start_policy    -- the trainable LLM (SFT checkpoint, adapters trainable) + the start's GPU-hours
- buffer_positions / sample_examples -- self-play games -> buffer entries -> training examples
- update          -- one pass of SFT-style updates over the sampled examples
- train           -- the resumable round loop; writes mcts_log.csv
- list_checkpoints / gpu_hours_at -- for evaluation into results.csv
"""
from __future__ import annotations

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

from model.llm_evaluator import LLMEvaluator
from model.llm_policy import LLMPolicy, PolicyConfig
from search.mcts import MCTSConfig, play_games
from search.selfplay import SelfPlayGame
from train.llm_rl import _append_csv, _drop_rows_after, ending_counts, list_checkpoints
from train.sft import _load_adapters, batch_losses, collate, gpu_hours_at as sft_gpu_hours_at


@dataclass(frozen=True)
class MCTSTrainConfig:
    """All Step 17 settings (Derek, Nov 1). Defaults mirror configs/llm_mcts.yaml."""

    init_run: str = "structured_masked_nall_e2_s0_scaled_d12_n500000"   # SFT 500k
    init_step: int = 30000              # end of pass 1
    budget_gpu_hours: float = 10.0      # = SFT pass 2's measured cost (19.27 - 9.25)
    rounds: int = 1000                  # upper bound; the budget stops first
    games_per_round: int = 64           # all searched together (one batched model call per step)
    opening_plies: int = 4              # random opening half-moves per game (not trained on)
    max_plies: int = 300                # draw at this ply (opening included), as AlphaZero
    simulations: int = 16
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    temperature_plies: int = 30
    buffer_rounds: int = 3              # train on positions from the last 3 rounds
    samples_per_position: float = 2.0   # examples per round = this x the round's new positions
    batch_size: int = 16                # examples per optimizer step (SFT's)
    micro_batch: int = 16               # examples per forward/backward (bounds GPU memory)
    learning_rate: float = 1.0e-4       # SFT's
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    value_loss: str = "bce"
    value_loss_weight: float = 1.0
    checkpoint_every: int = 2
    eval_batch_positions: int = 64      # LLMEvaluator positions per forward pass
    seed: int = 0
    tag: str = ""

    def __post_init__(self):
        if self.value_loss not in ("bce", "mse"):
            raise ValueError(f"value_loss must be 'bce' or 'mse', got {self.value_loss!r}")
        if min(self.rounds, self.games_per_round, self.simulations, self.buffer_rounds, self.batch_size,
               self.micro_batch, self.checkpoint_every) < 1 or self.samples_per_position <= 0:
            raise ValueError("counts must be positive")
        if self.tag and not all(c.isalnum() or c in "-_." for c in self.tag):
            raise ValueError(f"tag may only use letters, digits, '-', '_', '.': {self.tag!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "MCTSTrainConfig":
        """From a YAML block; unknown keys are an error (catches typos)."""
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown settings {sorted(unknown)}")
        return cls(**d)

    def mcts(self) -> MCTSConfig:
        return MCTSConfig(simulations=self.simulations, c_puct=self.c_puct, dirichlet_alpha=self.dirichlet_alpha,
                          dirichlet_weight=self.dirichlet_weight, temperature_plies=self.temperature_plies)


def run_name(cfg: MCTSTrainConfig) -> str:
    """e.g. 'mcts16_structured_masked_nall_e2_s0_scaled_d12_n500000_from30000_s0' (+ _tag)."""
    name = f"mcts{cfg.simulations}_{cfg.init_run}_from{cfg.init_step}_s{cfg.seed}"
    return f"{name}_{cfg.tag}" if cfg.tag else name


def start_policy(cfg: MCTSTrainConfig, artifact_root: str | Path, *, device: str | None = None) -> tuple[LLMPolicy, float]:
    """(trainable policy loaded from the SFT checkpoint, the SFT run's GPU-hours at that step)."""
    sft_run = Path(artifact_root) / "checkpoints" / "sft" / cfg.init_run
    step_dir = sft_run / "checkpoints" / f"step_{cfg.init_step:06d}"
    pcfg = PolicyConfig.from_dict(json.loads((step_dir / "policy_config.json").read_text()))
    policy = LLMPolicy.from_pretrained(pcfg, device=device)
    _load_adapters(policy, step_dir)
    return policy, sft_gpu_hours_at(sft_run, cfg.init_step)


# --------------------------------------------------------- targets ----

def buffer_positions(games: list[SelfPlayGame]) -> list[dict]:
    """Every searched move -> a buffer entry: position, visit distribution, result for the mover."""
    return [{"fen": r.fen, "visits": r.info["visits"], "value": g.score_for(r.white),
             "root_value": r.info["root_value"]} for g in games for r in g.moves]


def sample_examples(policy: LLMPolicy, buffer: list[dict], n: int, rng: random.Random) -> list[dict]:
    """``n`` training examples: a buffer position drawn uniformly, a move drawn by its visit share."""
    out = []
    for e in rng.choices(buffer, k=n):
        moves = sorted(e["visits"])
        move = rng.choices(moves, weights=[e["visits"][m] for m in moves])[0]
        ex = policy.training_example(chess.Board(e["fen"]), move)
        ex["label_value"] = e["value"]
        out.append(ex)
    return out


def update(policy: LLMPolicy, examples: list[dict], cfg: MCTSTrainConfig, optimizer, scaler,
           rng: random.Random) -> dict:
    """One pass over ``examples`` in shuffled optimizer steps of ``batch_size``, as SFT trains.

    Per example: move NLL (-log P(sampled move | position)) + value_loss_weight x value loss vs
    the game result. Dropout on while updating (as SFT), off for play. Returns mean losses.
    """
    order = list(range(len(examples)))
    rng.shuffle(order)
    params = policy.trainable_parameters()
    totals = {"nll": 0.0, "value_loss": 0.0}
    policy.model.train()
    try:
        for s in range(0, len(order), cfg.batch_size):
            chunk = order[s:s + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for m in range(0, len(chunk), cfg.micro_batch):
                micro = [examples[i] for i in chunk[m:m + cfg.micro_batch]]
                batch = collate(micro, policy.pad_id, policy.device)
                nll, vloss, _ = batch_losses(policy, batch, cfg.value_loss)
                loss = (nll + cfg.value_loss_weight * vloss).sum() / len(chunk)
                scaler.scale(loss).backward()
                totals["nll"] += nll.sum().item()
                totals["value_loss"] += vloss.sum().item()
            scaler.unscale_(optimizer)
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
    finally:
        policy.model.eval()
    n = max(len(examples), 1)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------- storage ----

def _round_dir(run_dir: Path, rnd: int) -> Path:
    return run_dir / "checkpoints" / f"round_{rnd:04d}"


def _save_round(policy: LLMPolicy, run_dir: Path, rnd: int, optimizer, scaler, counters: dict,
                cfg: MCTSTrainConfig) -> None:
    """Write round ``rnd`` atomically (adapters + value head + resume state); prune the previous one
    (deleted unless it is a kept checkpoint, which only loses its resume state)."""
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


def _buffer_path(run_dir: Path, rnd: int) -> Path:
    return run_dir / "buffer" / f"round_{rnd:04d}.json"


def _save_buffer(run_dir: Path, rnd: int, entries: list[dict], keep: int) -> None:
    path = _buffer_path(run_dir, rnd)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries))
    os.replace(tmp, path)
    for p in path.parent.glob("round_*.json"):
        if int(p.stem.split("_")[1]) <= rnd - keep:
            p.unlink(missing_ok=True)


def _load_buffer(run_dir: Path, rnd: int, keep: int) -> list[list[dict]]:
    """The saved positions of rounds rnd-keep+1 .. rnd (oldest first)."""
    out = []
    for r in range(max(1, rnd - keep + 1), rnd + 1):
        p = _buffer_path(run_dir, r)
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


# ------------------------------------------------------------ train ----

def train(policy: LLMPolicy, cfg: MCTSTrainConfig, run_dir: str | Path, *, init_gpu_hours: float = 0.0,
          max_rounds: int | None = None, extra_info: dict | None = None, log=print) -> dict:
    """Run self-play-with-search rounds on ``policy`` until the budget; resumable. Returns the summary.

    Files in run_dir: run_config.json (a resume with other settings is refused);
    checkpoints/round_NNNN/ (adapters + value head; resume.pt on the latest);
    buffer/round_NNNN.json (the last ``buffer_rounds`` rounds' positions); mcts_log.csv (one
    row per round); summary.json.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = policy.device
    settings = {"mcts_train": dataclasses.asdict(cfg), "policy": dataclasses.asdict(policy.config),
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
    counters = {"round": 0, "gpu_seconds": 0.0, "games": 0, "positions": 0, "last_round_seconds": 0.0}
    saved_rounds = list_checkpoints(run_dir)
    if saved_rounds:
        rnd, path = saved_rounds[-1]
        if not (path / "resume.pt").exists():
            raise RuntimeError(f"latest round {path} has no resume.pt; can't resume safely")
        state = torch.load(path / "resume.pt", map_location=device, weights_only=False)
        counters = {**counters, **state["counters"]}
        _load_adapters(policy, path)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        _drop_rows_after(run_dir / "mcts_log.csv", counters["round"])
        log(f"resuming {run_dir.name} after round {counters['round']} "
            f"({counters['gpu_seconds'] / 3600:.2f} of {cfg.budget_gpu_hours} GPU-h)")
    buffer = _load_buffer(run_dir, counters["round"], cfg.buffer_rounds)
    evaluator = LLMEvaluator(policy, batch_positions=cfg.eval_batch_positions)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    stop_at = cfg.rounds if max_rounds is None else min(cfg.rounds, max_rounds)
    stopped_by_budget = False
    while counters["round"] < stop_at:
        used = counters["gpu_seconds"] / 3600
        left = cfg.budget_gpu_hours - used
        if left <= 0 or left < 0.5 * counters["last_round_seconds"] / 3600:
            stopped_by_budget = True
            log(f"budget reached: {used:.2f} GPU-h used of {cfg.budget_gpu_hours} "
                f"(another round would take ~{counters['last_round_seconds'] / 3600:.2f})")
            break
        rnd = counters["round"] + 1
        round_seed = cfg.seed * 1_000_003 + rnd
        rng = random.Random(round_seed)

        sync()
        t0 = time.perf_counter()
        policy.model.eval()
        games = play_games(evaluator, cfg.games_per_round, cfg.mcts(), opening_plies=cfg.opening_plies,
                           max_plies=cfg.max_plies, seed=round_seed)
        new = buffer_positions(games)
        _save_buffer(run_dir, rnd, new, cfg.buffer_rounds)
        buffer = (buffer + [new])[-cfg.buffer_rounds:]
        if device.type == "cuda":
            torch.cuda.empty_cache()      # the search's key/value copies are freed before the update
        sync()
        t1 = time.perf_counter()
        pool = [e for rnd_entries in buffer for e in rnd_entries]
        examples = sample_examples(policy, pool, max(1, round(cfg.samples_per_position * len(new))), rng)
        losses = update(policy, examples, cfg, optimizer, scaler, rng)
        sync()
        t2 = time.perf_counter()

        counters["round"] = rnd
        counters["gpu_seconds"] += t2 - t0
        counters["last_round_seconds"] = t2 - t0
        counters["games"] += len(games)
        counters["positions"] += len(new)
        _save_round(policy, run_dir, rnd, optimizer, scaler, counters, cfg)

        results = [g.white_score for g in games]
        top_share = [max(e["visits"].values()) for e in new if e["visits"]]
        row = {"round": rnd, "games": len(games), "positions": len(new), "buffer_positions": len(pool),
               "examples": len(examples),
               "white_wins": sum(s == 1.0 for s in results), "draws": sum(s == 0.5 for s in results),
               "black_wins": sum(s == 0.0 for s in results),
               "max_plies_draws": sum(g.termination == "max_plies" for g in games), **ending_counts(games),
               "avg_plies": round(sum(g.plies for g in games) / len(games), 1),
               "top_visit_share": round(sum(top_share) / max(len(top_share), 1), 4),
               "root_value": round(sum(e["root_value"] for e in new) / max(len(new), 1), 4),
               **{k: round(v, 5) for k, v in losses.items()},
               "selfplay_seconds": round(t1 - t0, 1), "update_seconds": round(t2 - t1, 1),
               "mcts_gpu_hours": round(counters["gpu_seconds"] / 3600, 4),
               "train_gpu_hours": round(init_gpu_hours + counters["gpu_seconds"] / 3600, 4)}
        _append_csv(run_dir / "mcts_log.csv", row)
        log(f"[round {rnd}] W/D/L {row['white_wins']}/{row['draws']}/{row['black_wins']} ({row['avg_plies']} plies, "
            f"mate {row['ends_checkmate']}, repetition {row['ends_repetition']}, cap {row['max_plies_draws']}) | "
            f"nll {row['nll']:.3f} | value loss {row['value_loss']:.3f} | self-play {row['selfplay_seconds']:.0f}s + "
            f"update {row['update_seconds']:.0f}s | {row['mcts_gpu_hours']:.2f} of {cfg.budget_gpu_hours} GPU-h")

    summary = {**counters, "mcts_gpu_hours": round(counters["gpu_seconds"] / 3600, 4),
               "train_gpu_hours": round(init_gpu_hours + counters["gpu_seconds"] / 3600, 4),
               "stopped_by_budget": stopped_by_budget, "kept_checkpoints": [r for r, _ in list_checkpoints(run_dir)]}
    if device.type == "cuda":
        summary["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def gpu_hours_at(run_dir: str | Path, rnd: int) -> float:
    """Total training GPU-hours (SFT start + self-play) when round ``rnd`` finished (results.csv's)."""
    import csv
    with open(Path(run_dir) / "mcts_log.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["round"]) == rnd:
                return float(r["train_gpu_hours"])
    raise ValueError(f"no mcts_log.csv row for round {rnd}")
