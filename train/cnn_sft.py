"""Step 14b: CNN SFT -- AlphaZero's network trained to imitate Stockfish (plan v5 Step 14b).

Intent: the same small CNN as AlphaZero (model/az_net.py, 6 x 64), but taught from the
same Stockfish-labeled positions as the LLM instead of by self-play. Compared with LLM
SFT it isolates the architecture (same data); compared with AlphaZero it isolates the
training regime (same network); and its end-of-pass-1 checkpoint is where Step 18's
CNN RL starts.

Training (batch 256, 2 passes over the 500k set, Adam with AlphaZero's settings):
  - move target: Stockfish's best move, as a one-hot over the legal moves in the
    side-to-move frame -- exactly AlphaZero's policy loss (cross-entropy against a visit
    distribution) with all the weight on the label move;
  - value target: Stockfish's win probability for the side to move (BCE, a soft label).
The loss code is train.az_train.batch_losses itself, so the two arms differ only in
where the targets come from. Positions are encoded once into AlphaZero's compact buffer
format (encode_position), held in memory.

Checkpoints: one per pass, ``checkpoints/pass_000N/net.pt``, in AlphaZero's format
(train.az_train.load_checkpoint reads it; Step 18 starts from pass 1), plus resume state.
A stopped run resumes at the next pass boundary (a pass is minutes). Logs:
``train_log.csv`` (every ``log_every`` steps) and ``val_log.csv`` (pass 0 = before
training, then after each pass): validation move loss, value loss, top-1 agreement with
Stockfish. GPU time counts only the training steps (as train/sft.py's gpu_seconds_train).

Main pieces:
- CNNSFTConfig  -- settings; configs/cnn_sft.yaml
- run_name      -- e.g. cnn_b6c64_n500000_e2_s0
- encode_labels -- labeled positions -> buffer arrays (planes, legal moves, one-hot target, value)
- train         -- the resumable pass loop
- list_checkpoints / gpu_hours_at / val_rows -- for evaluation and the auto-check
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch

from engine import board as B
from model.az_net import AZNet, NetConfig
from train.az_train import NUM_PLANES, MAX_LEGAL, batch_losses, decode_planes, encode_position


@dataclass(frozen=True)
class CNNSFTConfig:
    """All CNN SFT settings (Derek, Nov 26). Defaults mirror configs/cnn_sft.yaml."""

    dataset: str = "sft_scaled_d12_n500000"       # the 500k set (Step 13)
    val_from: str = "sft_2013-06_d12_n50000"      # the fixed 50k validation split (as the LLM)
    train_positions: int | None = None            # None = all; a number = a subset (smoke tests)
    val_positions: int | None = None              # None = the whole validation split (~5k)
    epochs: int = 2                               # passes over the training set
    batch_size: int = 256                         # AlphaZero's (~1,950 steps per pass on 500k)
    learning_rate: float = 1.0e-3                 # AlphaZero's Adam settings
    weight_decay: float = 1.0e-4
    log_every: int = 100
    seed: int = 0
    tag: str = ""

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.log_every) < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("epochs, batch_size, log_every must be positive; learning_rate > 0; weight_decay >= 0")
        if self.tag and not all(c.isalnum() or c in "-_." for c in self.tag):
            raise ValueError(f"tag may only use letters, digits, '-', '_', '.': {self.tag!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "CNNSFTConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown CNN SFT settings {sorted(unknown)}")
        return cls(**d)


def run_name(cfg: CNNSFTConfig, net_cfg: NetConfig) -> str:
    """e.g. 'cnn_b6c64_n500000_e2_s0' (dataset size from the dataset name's '_n<size>' suffix)."""
    size = cfg.dataset.rsplit("_n", 1)[-1] if "_n" in cfg.dataset else cfg.dataset
    name = f"cnn_b{net_cfg.blocks}c{net_cfg.channels}_n{size}_e{cfg.epochs}_s{cfg.seed}"
    return f"{name}_{cfg.tag}" if cfg.tag else name


def encode_labels(df) -> tuple[np.ndarray, ...]:
    """Labeled positions (columns fen, best_move, win_prob) -> (planes u8, moves i16, probs f16, value f32).

    Rows whose label move isn't legal in their position are dropped (there should be none).
    """
    rows = []
    for fen, move, win in zip(df["fen"], df["best_move"], df["win_prob"]):
        board = chess.Board(fen)
        if not B.is_legal(board, move):
            continue
        rows.append((*encode_position(board, {move: 1.0}), float(win)))
    if not rows:
        return (np.zeros((0, NUM_PLANES, 8, 8), np.uint8), np.zeros((0, MAX_LEGAL), np.int16),
                np.zeros((0, MAX_LEGAL), np.float16), np.zeros(0, np.float32))
    planes, moves, probs, value = zip(*rows)
    return np.stack(planes), np.stack(moves), np.stack(probs), np.array(value, dtype=np.float32)


@torch.no_grad()
def validate(net: AZNet, data: tuple[np.ndarray, ...], device, batch_size: int = 1024) -> dict:
    """Mean move loss, value loss, and top-1 agreement with Stockfish on encoded positions."""
    net.eval()
    planes, moves, probs, value = data
    n = len(value)
    tot = {"val_move_loss": 0.0, "val_value_loss": 0.0, "val_top1": 0.0}
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        pl, vl, _ = batch_losses(net, planes[sl], moves[sl], probs[sl], value[sl], device)
        tot["val_move_loss"] += pl.sum().item()
        tot["val_value_loss"] += vl.sum().item()
        # top-1: the network's highest-scoring legal move is Stockfish's (the one-hot target)
        x = torch.from_numpy(decode_planes(planes[sl])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = net(x)
        mv = torch.from_numpy(moves[sl].astype(np.int64)).to(device)
        legal_logits = logits.float().gather(1, mv.clamp(min=0)).masked_fill(mv < 0, float("-inf"))
        label = torch.from_numpy(probs[sl].astype(np.float32)).to(device).argmax(1)
        tot["val_top1"] += (legal_logits.argmax(1) == label).sum().item()
    return {k: v / max(n, 1) for k, v in tot.items()}


def _pass_dir(run_dir: Path, p: int) -> Path:
    return run_dir / "checkpoints" / f"pass_{p:04d}"


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """[(pass, folder)] for every saved pass, oldest first."""
    root = Path(run_dir) / "checkpoints"
    if not root.exists():
        return []
    return sorted((int(p.name.split("_")[1]), p) for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("pass_") and not p.name.endswith(".tmp"))


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _keep_rows_upto(path: Path, key: str, upto: int) -> None:
    """On resume: drop log rows past the last saved pass (they'll be redone)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(r for r in rows if int(r[key]) <= upto)


def val_rows(run_dir: str | Path) -> list[dict]:
    """val_log.csv as dicts with numbers (pass 0 = before training)."""
    with open(Path(run_dir) / "val_log.csv", newline="") as f:
        return [{k: (int(v) if k == "pass" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def gpu_hours_at(run_dir: str | Path, p: int) -> float:
    """Training GPU-hours when pass ``p`` ended (results.csv's train_gpu_hours)."""
    for r in val_rows(run_dir):
        if r["pass"] == p:
            return r["train_gpu_hours"]
    raise ValueError(f"no val_log.csv row for pass {p}")


def train(cfg: CNNSFTConfig, net_cfg: NetConfig, train_data: tuple[np.ndarray, ...], val_data: tuple[np.ndarray, ...],
          run_dir: str | Path, *, device: str | torch.device | None = None, extra_info: dict | None = None,
          log=print) -> dict:
    """Train ``cfg.epochs`` passes over ``train_data`` (encode_labels output); resumable per pass.

    Returns the summary. A run whose passes are all saved returns at once (nothing redone).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    settings = {"cnn_sft": dataclasses.asdict(cfg), "net": dataclasses.asdict(net_cfg),
                "train_rows": int(len(train_data[3])), "val_rows": int(len(val_data[3]))}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text())
        if json.loads(json.dumps(settings)) != {k: saved.get(k) for k in settings}:
            raise ValueError(f"{run_dir} was started with different settings or data; use a new run folder")
    else:
        cfg_path.write_text(json.dumps({**settings, "extra_info": extra_info or {}}, indent=2))

    torch.manual_seed(cfg.seed)
    net = AZNet(net_cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    counters = {"pass": 0, "step": 0, "gpu_seconds": 0.0}
    saved = list_checkpoints(run_dir)
    if saved:
        p, path = saved[-1]
        net.load_state_dict(torch.load(path / "net.pt", map_location=device, weights_only=False)["state_dict"])
        state = torch.load(path / "resume.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        counters = state["counters"]
        _keep_rows_upto(run_dir / "train_log.csv", "pass", counters["pass"])
        _keep_rows_upto(run_dir / "val_log.csv", "pass", counters["pass"])
        log(f"resuming {run_dir.name} after pass {counters['pass']} of {cfg.epochs}")
    else:
        _keep_rows_upto(run_dir / "train_log.csv", "pass", 0)   # a pass stopped before its checkpoint is redone
    if not saved and (not (run_dir / "val_log.csv").exists() or not val_rows(run_dir)):
        v = validate(net, val_data, device)
        _append_csv(run_dir / "val_log.csv", {"pass": 0, "step": 0, **{k: round(x, 5) for k, x in v.items()},
                                              "train_gpu_hours": 0.0})
        log(f"before training: val move loss {v['val_move_loss']:.3f}, top-1 {v['val_top1']:.3f}")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    n = len(train_data[3])
    steps_per_pass = math.ceil(n / cfg.batch_size)
    while counters["pass"] < cfg.epochs:
        p = counters["pass"] + 1
        order = np.random.default_rng(cfg.seed * 1_000_003 + p).permutation(n)
        tot = {"move_loss": 0.0, "value_loss": 0.0}
        k = 0
        for s in range(steps_per_pass):
            idx = np.sort(order[s * cfg.batch_size:(s + 1) * cfg.batch_size])
            sync()
            t0 = time.perf_counter()
            net.train()
            pl, vl, _ = batch_losses(net, *(a[idx] for a in train_data), device)
            loss = pl.mean() + vl.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sync()
            counters["gpu_seconds"] += time.perf_counter() - t0
            counters["step"] += 1
            tot["move_loss"] += pl.mean().item()
            tot["value_loss"] += vl.mean().item()
            k += 1
            if k == cfg.log_every or s == steps_per_pass - 1:
                _append_csv(run_dir / "train_log.csv", {"pass": p, "step": counters["step"],
                                                         "move_loss": round(tot["move_loss"] / k, 5),
                                                         "value_loss": round(tot["value_loss"] / k, 5),
                                                         "train_gpu_hours": round(counters["gpu_seconds"] / 3600, 5)})
                tot, k = {"move_loss": 0.0, "value_loss": 0.0}, 0
        counters["pass"] = p
        v = validate(net, val_data, device)
        hours = counters["gpu_seconds"] / 3600
        _append_csv(run_dir / "val_log.csv", {"pass": p, "step": counters["step"],
                                              **{key: round(x, 5) for key, x in v.items()},
                                              "train_gpu_hours": round(hours, 5)})
        final = _pass_dir(run_dir, p)
        tmp = final.with_name(final.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        torch.save({"net_config": dataclasses.asdict(net_cfg), "state_dict": net.state_dict()}, tmp / "net.pt")
        torch.save({"optimizer": optimizer.state_dict(), "counters": dict(counters)}, tmp / "resume.pt")
        os.replace(tmp, final)
        log(f"[pass {p}/{cfg.epochs}] step {counters['step']} | val move loss {v['val_move_loss']:.3f} "
            f"value loss {v['val_value_loss']:.3f} top-1 {v['val_top1']:.3f} | {hours:.3f} GPU-h")

    summary = {**counters, "train_gpu_hours": round(counters["gpu_seconds"] / 3600, 5),
               "steps_per_pass": steps_per_pass, "passes": [p for p, _ in list_checkpoints(run_dir)]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
