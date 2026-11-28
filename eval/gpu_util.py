"""GPU utilization of the headline rows (plan v5 Step 11 support item, Derek Nov 27).

Intent: every comparison is at equal training GPU-hours, but a GPU-hour isn't equally busy in
every arm (AlphaZero's self-play measured ~0-2%: one CPU core runs the search in Python while
the GPU waits). This puts one utilization number on each headline row:

    utilization(row) = sum over phases (hours of the phase x utilization of the phase) / total hours

Three parts:
- split    -- KNOWN, logs only (no GPU): each run's hours by phase up to the row's round:
              ``init`` (the SFT run it started from), ``selfplay``, ``stockfish`` (LLM RL only:
              CPU re-scoring inside the self-play timer, counted at 0% GPU), ``update``.
- measure  -- MEASURED, on the GPU: replays a slice of each arm with its real settings and
              checkpoint (the full game count, so batch sizes are real) while ``GpuSampler`` reads
              the GPU's utilization every ``interval`` s; per phase: mean utilization, seconds,
              positions.
- combine  -- phase hours x phase utilization -> one row per headline model.

ASSUMED (each printed with the results): the slice is representative (checked: seconds per
self-play position in the slice vs. the log's mean); the LLM slices stop after ``plies`` moves,
missing each round's small-batch tail (u_selfplay a little high); ``nvidia-smi`` utilization =
% of time any kernel runs (the standard number, generous); the SFT parts of the LLM RL rows run
at the 500k SFT run's utilization (RL run 4 started from the 50k run, trained on Colab);
CNN SFT (0.003 GPU-h, Step 18's ``init``) runs at its row's update utilization.

Main pieces:
- ROWS / Row     -- the five headline rows (plan Step 11)
- phase_hours    -- log rows -> hours by phase (pure; tested)
- split          -- phase_hours for one row, read from its run folder
- GpuSampler     -- background utilization reader (pynvml, else nvidia-smi)
- measure_*      -- one replay per arm; ``measure`` dispatches
- combine / format_table -- the per-row numbers and a markdown table for the plan
"""
from __future__ import annotations

import csv
import dataclasses
import json
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SFT500 = "structured_masked_nall_e2_s0_scaled_d12_n500000"
RL4 = "structured_masked_sft25000_s0_curric120_lr5x"
LOG_FILE = {"llm_rl": "rl_log.csv", "llm_mcts": "mcts_log.csv", "alphazero": "az_log.csv"}


@dataclass(frozen=True)
class Row:
    key: str
    label: str
    kind: str        # "sft" | "llm_mcts" | "llm_rl" | "alphazero" (folder under checkpoints/)
    run: str         # run folder ("" = Step 17's, from configs/llm_mcts.yaml)
    step: int        # the row's checkpoint: SFT step or RL round (logs are summed up to it)


ROWS = (
    Row("sft500", "LLM SFT 500k (@30k / @62.5k)", "sft", SFT500, 62500),
    Row("selfplay_llm", "Self-play LLM (Step 17), round 4", "llm_mcts", "", 4),
    Row("rl4", "LLM RL run 4 @r20", "llm_rl", RL4, 20),
    Row("az_scratch", "AlphaZero CNN from scratch, r109", "alphazero", "az_b6c64_s0", 109),
    Row("az_cnnsft", "CNN RL from CNN SFT pass 1, r136", "alphazero", "az_b6c64_s0_cnnsft_p1", 136),
)


def run_dir(row: Row, root: str | Path) -> Path:
    run = row.run
    if row.kind == "llm_mcts" and not run:
        import yaml
        from train.llm_mcts import MCTSTrainConfig, run_name
        run = run_name(MCTSTrainConfig.from_dict(yaml.safe_load(open("configs/llm_mcts.yaml"))))
    return Path(root) / "checkpoints" / row.kind / run


def checkpoint_dir(row: Row, root: str | Path) -> Path:
    sub = f"step_{row.step:06d}" if row.kind == "sft" else f"round_{row.step:04d}"
    return run_dir(row, root) / "checkpoints" / sub


# ------------------------------------------------------------- split ----

def phase_hours(log_rows: list[dict], upto: int, init_hours: float = 0.0) -> dict:
    """Hours by phase from a round log, rounds 1..``upto``.

    ``selfplay`` excludes ``stockfish_seconds`` where the log has it (LLM RL: Stockfish re-scores
    endings inside the self-play timer, on the CPU). ``logged_total`` = the log's own
    train_gpu_hours at ``upto``, a check that the phases add up.
    """
    rows = [r for r in log_rows if int(r["round"]) <= upto]
    if not rows or int(rows[-1]["round"]) != upto:
        raise ValueError(f"log has no row for round {upto}")
    f = lambda k: sum(float(r.get(k) or 0.0) for r in rows) / 3600
    sf = f("stockfish_seconds")
    out = {"init": init_hours, "selfplay": f("selfplay_seconds") - sf, "stockfish": sf, "update": f("update_seconds"),
           "positions": sum(int(r["positions"]) for r in rows), "rounds": len(rows)}
    out["total"] = out["init"] + out["selfplay"] + out["stockfish"] + out["update"]
    out["logged_total"] = float(rows[-1]["train_gpu_hours"]) if rows[-1].get("train_gpu_hours") else None
    return out


def split(row: Row, root: str | Path) -> dict:
    """Hours by phase for one headline row (logs only)."""
    d = run_dir(row, root)
    if row.kind == "sft":
        from train.sft import gpu_hours_at
        h = gpu_hours_at(d, row.step)
        return {"init": 0.0, "sft": h, "total": h, "logged_total": h}
    init = float(json.loads((d / "run_config.json").read_text()).get("init_gpu_hours", 0.0))
    with open(d / LOG_FILE[row.kind], newline="") as f:
        return phase_hours(list(csv.DictReader(f)), row.step, init)


# ------------------------------------------------------------ sampler ----

def _reader() -> Callable[[], float]:
    """A function returning GPU 0's current utilization in %: pynvml if installed, else nvidia-smi."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return lambda: float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
    except Exception:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"]
        return lambda: float(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())


class GpuSampler:
    """Reads utilization every ``interval`` s in a background thread; ``mean(t0, t1)`` over a window.

    ``read`` is injectable (tests). Use as a context manager around the whole measurement.
    """

    def __init__(self, interval: float = 0.5, read: Callable[[], float] | None = None):
        self.interval, self.read = interval, read
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self.read is None:
            self.read = _reader()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.monotonic(), self.read()))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def mean(self, t0: float, t1: float) -> float | None:
        vals = [u for t, u in list(self.samples) if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else None


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(sampler: GpuSampler, fn: Callable[[], object]) -> tuple[object, dict]:
    """Run ``fn`` between two GPU syncs (as the training loops time phases); its seconds and mean utilization."""
    _sync()
    t0 = time.monotonic()
    out = fn()
    _sync()
    t1 = time.monotonic()
    time.sleep(sampler.interval * 1.5)     # let the sampler record the window's last reading
    return out, {"seconds": round(t1 - t0, 2), "util": sampler.mean(t0, t1)}


# ------------------------------------------------------------ measure ----

def _alphazero_phases(net, cfg, mcts_cfg, sampler: GpuSampler, device, *, games: int, max_plies: int,
                      seed: int = 1) -> dict:
    """One AlphaZero round (as train/az_train.py's loop) on ``net``, timed by phase."""
    import numpy as np
    import torch
    from model.az_net import AZEvaluator
    from search.mcts import play_games
    from train.az_train import ReplayBuffer, game_examples, update, update_steps

    net.eval()
    evaluator = AZEvaluator(net, device)
    rows, sp = timed(sampler, lambda: game_examples(play_games(evaluator, games, mcts_cfg, opening_plies=cfg.opening_plies,
                                                                max_plies=max_plies, seed=seed)))
    n = len(rows[3])
    buffer = ReplayBuffer(max(n, 1))
    buffer.add(*rows)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps = update_steps(n, cfg)
    _, up = timed(sampler, lambda: update(net, buffer, steps, cfg, optimizer, np.random.default_rng(seed), device))
    return {"selfplay": {**sp, "positions": n}, "update": {**up, "steps": steps}}


def measure_alphazero(row: Row, root, sampler: GpuSampler, *, plies: int | None = None, games: int | None = None,
                      device=None) -> dict:
    """AlphaZero: a full round by default (only ~8 min), from the row's checkpoint with the run's own settings."""
    import torch
    from search.mcts import MCTSConfig
    from train.az_train import ADDED_TRAIN_SETTINGS, AZConfig, _from_dict, load_checkpoint
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = json.loads((run_dir(row, root) / "run_config.json").read_text())
    cfg = _from_dict(AZConfig, {**ADDED_TRAIN_SETTINGS, **saved["train"]})
    mcts_cfg = _from_dict(MCTSConfig, saved["mcts"])
    net = load_checkpoint(checkpoint_dir(row, root), device)
    max_plies = cfg.max_plies if plies is None else cfg.opening_plies + plies
    return _alphazero_phases(net, cfg, mcts_cfg, sampler, device, games=games or cfg.games_per_round, max_plies=max_plies)


def _llm_optimizer(policy, cfg):
    import torch
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    fp16 = policy.device.type == "cuda" and policy.model.get_input_embeddings().weight.dtype == torch.float16
    return optimizer, torch.amp.GradScaler("cuda", enabled=fp16)


def measure_llm_rl(row: Row, root, sampler: GpuSampler, *, plies: int = 20) -> dict:
    """LLM RL (Step 7): ``games_per_round`` games, one position per call as in training, cut at ``plies`` moves;
    then build_examples + one update pass (the update timer in training covers both). No Stockfish (CPU; 0%)."""
    from search.selfplay import play_selfplay_games
    from train.llm_rl import RLConfig, build_examples, sampling_picker, start_policies, update
    from train.sft import _load_adapters
    saved = json.loads((run_dir(row, root) / "run_config.json").read_text())
    cfg = RLConfig.from_dict(saved["rl"])
    policy, ref, _ = start_policies(cfg, root)
    _load_adapters(policy, checkpoint_dir(row, root))
    optimizer, scaler = _llm_optimizer(policy, cfg)
    pick = sampling_picker(policy, cfg.temperature, random.Random(1))
    games, sp = timed(sampler, lambda: play_selfplay_games(pick, cfg.games_per_round, opening_plies=cfg.opening_plies,
                                                           max_plies=cfg.opening_plies + plies, seed=1))
    n = sum(len(g.moves) for g in games)
    _, up = timed(sampler, lambda: update(policy, ref, build_examples(policy, games), cfg, optimizer, scaler,
                                          random.Random(1)))
    return {"selfplay": {**sp, "positions": n}, "update": up}


def measure_llm_mcts(row: Row, root, sampler: GpuSampler, *, plies: int = 10) -> dict:
    """Self-play LLM (Step 17): 64 games searched together (16 sims), cut at ``plies`` moves; then one update."""
    import torch
    from model.llm_evaluator import LLMEvaluator
    from search.mcts import play_games
    from train.llm_mcts import MCTSTrainConfig, buffer_positions, sample_examples, start_policy, update
    from train.sft import _load_adapters
    saved = json.loads((run_dir(row, root) / "run_config.json").read_text())
    cfg = MCTSTrainConfig.from_dict(saved["mcts_train"])
    policy, _ = start_policy(cfg, root)
    _load_adapters(policy, checkpoint_dir(row, root))
    optimizer, scaler = _llm_optimizer(policy, cfg)
    policy.model.eval()
    evaluator = LLMEvaluator(policy, batch_positions=cfg.eval_batch_positions)
    new, sp = timed(sampler, lambda: buffer_positions(play_games(evaluator, cfg.games_per_round, cfg.mcts(),
                                                                  opening_plies=cfg.opening_plies,
                                                                  max_plies=cfg.opening_plies + plies, seed=1)))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rng = random.Random(1)
    _, up = timed(sampler, lambda: update(policy, sample_examples(policy, new, max(1, round(cfg.samples_per_position * len(new))), rng),
                                          cfg, optimizer, scaler, rng))
    return {"selfplay": {**sp, "positions": len(new)}, "update": up}


def measure_sft(row: Row, root, sampler: GpuSampler, *, steps: int = 60, seed: int = 1) -> dict:
    """LLM SFT: ``steps`` optimizer steps from the row's checkpoint with the SFT run's batch, learning rate and losses.

    Uses train/llm_mcts.update (the same SFT loss: train/sft.py's collate + batch_losses) on random positions with a
    random legal move as the label: the compute per step depends on prompt length and batch, not on the labels.
    """
    from eval.match import random_opening
    from model.llm_policy import LLMPolicy, PolicyConfig
    from train.llm_mcts import MCTSTrainConfig, update
    from train.sft import TrainConfig, _load_adapters
    import chess
    step_dir = checkpoint_dir(row, root)
    sft = TrainConfig.from_dict(json.loads((run_dir(row, root) / "run_config.json").read_text())["train"])
    cfg = dataclasses.replace(MCTSTrainConfig(), batch_size=sft.batch_size, micro_batch=sft.batch_size,
                              learning_rate=sft.learning_rate, weight_decay=sft.weight_decay, grad_clip=sft.grad_clip,
                              value_loss=sft.value_loss, value_loss_weight=sft.value_loss_weight)
    policy = LLMPolicy.from_pretrained(PolicyConfig.from_dict(json.loads((step_dir / "policy_config.json").read_text())))
    _load_adapters(policy, step_dir)
    optimizer, scaler = _llm_optimizer(policy, cfg)
    rng = random.Random(seed)
    examples = []
    while len(examples) < steps * cfg.batch_size:
        board = chess.Board()
        for mv in random_opening(rng, rng.randint(4, 60)):
            board.push_uci(mv)
        ex = policy.training_example(board, rng.choice([m.uci() for m in board.legal_moves]))
        ex["label_value"] = rng.random()
        examples.append(ex)
    _, up = timed(sampler, lambda: update(policy, examples, cfg, optimizer, scaler, rng))
    return {"sft": {**up, "steps": steps}}


MEASURE = {"sft": measure_sft, "llm_rl": measure_llm_rl, "llm_mcts": measure_llm_mcts, "alphazero": measure_alphazero}


def measure(row: Row, root, sampler: GpuSampler, **kw) -> dict:
    """One arm's replay; frees the GPU afterwards, even if it fails, so the next arm starts clean.

    (A failed arm's models sit in reference cycles until the garbage collector runs; without the
    collect below, the next LLM arm runs out of GPU memory.)
    """
    import gc
    import torch
    try:
        return MEASURE[row.kind](row, root, sampler, **kw)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------ combine ----

def combine(splits: dict[str, dict], measures: dict[str, dict], rows=ROWS) -> list[dict]:
    """Per row: hours by phase x measured utilization -> overall utilization (%), plus the per-phase pieces.

    Phase utilizations: ``sft`` and every LLM row's ``init`` = the SFT row's measured value; an AlphaZero row's
    ``init`` (CNN SFT, minutes) = its own update value; ``stockfish`` = 0.
    """
    u_sft = measures.get("sft500", {}).get("sft", {}).get("util")
    out = []
    for row in rows:
        s, m = splits[row.key], measures.get(row.key, {})
        u = {"sft": u_sft, "stockfish": 0.0,
             "selfplay": m.get("selfplay", {}).get("util"), "update": m.get("update", {}).get("util")}
        u["init"] = u["update"] if row.kind == "alphazero" else u_sft
        hours = {k: s[k] for k in ("init", "sft", "selfplay", "stockfish", "update") if s.get(k)}
        known = {k: h for k, h in hours.items() if u.get(k) is not None}
        total = sum(hours.values())
        complete = len(known) == len(hours)
        overall = sum(h * u[k] for k, h in hours.items()) / total if complete and total else None
        check = None
        if row.kind != "sft" and m.get("selfplay", {}).get("positions") and s.get("positions"):
            slice_spp = m["selfplay"]["seconds"] / m["selfplay"]["positions"]
            log_spp = s["selfplay"] * 3600 / s["positions"]
            check = round(slice_spp / log_spp, 2)
        out.append({"row": row.label, "total_h": round(total, 2), "overall_util": None if overall is None else round(overall, 1),
                    "hours": {k: round(h, 2) for k, h in hours.items()},
                    "util": {k: (None if u[k] is None else round(u[k], 1)) for k in hours},
                    "slice_vs_log_sec_per_position": check, "complete": complete})
    return out


def format_table(combined: list[dict]) -> str:
    """Markdown: one line per row, for the plan (Sec 4.0)."""
    lines = ["| Row | GPU-h | GPU utilization | by phase: hours @ utilization | slice / log s per position |",
             "|---|---|---|---|---|"]
    for c in combined:
        parts = ", ".join(f"{k} {h} h @ {'?' if c['util'][k] is None else str(c['util'][k]) + '%'}"
                          for k, h in c["hours"].items())
        overall = "?" if c["overall_util"] is None else f"**{c['overall_util']}%**"
        check = c["slice_vs_log_sec_per_position"]
        flag = "" if check is None or 0.75 <= check <= 1.25 else " (off by > 25%)"
        lines.append(f"| {c['row']} | {c['total_h']} | {overall} | {parts} | {'-' if check is None else check}{flag} |")
    return "\n".join(lines)
