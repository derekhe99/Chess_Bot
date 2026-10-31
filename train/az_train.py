"""Step 9: AlphaZero training -- learn chess from nothing but games against itself.

Intent: a randomly initialized network plays itself with search, then learns to
predict what the search found. Each ROUND (plan Sec 0 terminology):
  1. Self-play: ``games_per_round`` games at once (search/mcts.py play_games), every
     move chosen by MCTS with the CURRENT network. Search is smarter than the raw
     network, so its visit counts are a better policy than the network's own guess.
  2. Store every position in a replay buffer of the most recent positions: the
     planes (side-to-move frame), the search's visit distribution (policy target),
     and the game's final score from the side to move (value target). Exact results
     only -- no Stockfish, no engine scoring anywhere.
  3. Update: K minibatches sampled from the buffer, with
       loss = cross-entropy(policy, visit distribution) + BCE(value, score)
     plus weight decay (Adam's L2 term). The policy learns to jump straight to what
     search would find; the value learns who wins. Next round's search uses both,
     so it gets stronger, and so do the targets: that loop is the whole method.
  4. Save: resume state every round; a kept checkpoint every ``checkpoint_every``.

Compute: GPU time = wall-clock of self-play + updates (as Step 7). There is no SFT
and no labeling, so train_gpu_hours starts at 0. The cap (``max_gpu_hours``) is an
argument, not a saved setting, so raising it and rerunning resumes the same run
(staged 9 / 18 / 30, plan Step 9). When the cap stops a run, its last round is
marked kept, so every stage ends on an evaluable checkpoint.

Replay buffer, stored compactly (lossless): planes as uint8 (every plane is 0/1
except the 50-move-clock plane, stored as its integer count); each position's legal
moves as side-to-move indices (padded with -1) plus their visit shares. The legal
moves double as the training mask. ~2.2 KB per position (~450 MB at 200k).

Randomness: every round's randomness (self-play noise and sampling, minibatch
sampling) is seeded from (seed, round), so a resumed run replays the same rounds
with no RNG state to save; only the network init uses ``seed`` directly.

Main pieces:
- AZConfig        -- training settings; configs/alphazero.yaml ``train``
- load_configs    -- the yaml -> (AZConfig, NetConfig, MCTSConfig, max_gpu_hours)
- run_name        -- the run folder's name, e.g. az_b6c64_s0 (+ tag)
- ReplayBuffer    -- the most recent positions, as fixed-size numpy arrays
- game_examples   -- finished games -> buffer rows
- update          -- K minibatch steps; returns mean losses + policy entropy
- train           -- the resumable round loop; writes az_log.csv
- list_checkpoints / load_checkpoint / gpu_hours_at -- for evaluation into results.csv
- AlphaZeroAgent / load_agent -- a checkpoint as an eval player (MCTS, no noise, most-visited)
- PolicyAgent / load_policy_agent -- the same network with no search: plays its policy's top legal move
- eval_rounds     -- which kept checkpoints to rate: up to 4 per cap stage, each stage's last included
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
import torch.nn.functional as F
import yaml

from engine import board as B
from engine.encoding import NUM_PLANES, stm_board, to_planes
from engine.moves import NUM_MOVES, stm_index
from model.az_net import AZEvaluator, AZNet, NetConfig
from search.mcts import MCTSConfig, play_games, search_move
from search.selfplay import SelfPlayGame

MAX_LEGAL = 256      # chess allows at most 218 legal moves; the buffer pads to this width
HALFMOVE_PLANE = 18  # engine/encoding.py: halfmove clock / 100


@dataclass(frozen=True)
class AZConfig:
    """Training settings. Defaults mirror configs/alphazero.yaml (provisional, plan Sec 5 item 4)."""

    rounds: int = 1000                # upper bound; the GPU-hour cap normally stops first
    games_per_round: int = 128
    opening_plies: int = 4            # random opening half-moves per game (variety; not trained on)
    max_plies: int = 300              # games still going here are draws (opening plies included)
    buffer_positions: int = 200_000   # replay buffer = the most recent this many positions
    batch_size: int = 256
    samples_per_position: float = 2.0 # K = enough minibatches that each new position is sampled ~this often
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    checkpoint_every: int = 5         # rounds between KEPT checkpoints (the latest round is always on disk)
    seed: int = 0
    tag: str = ""                     # appended to the run folder name; change it whenever a setting changes

    def __post_init__(self):
        if min(self.rounds, self.games_per_round, self.buffer_positions, self.batch_size, self.checkpoint_every) < 1:
            raise ValueError("rounds, games_per_round, buffer_positions, batch_size, checkpoint_every must be positive")
        if self.samples_per_position <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("samples_per_position and learning_rate must be > 0, weight_decay >= 0")
        if self.tag and not all(c.isalnum() or c in "-_." for c in self.tag):
            raise ValueError(f"tag may only use letters, digits, '-', '_', '.': {self.tag!r}")


def _from_dict(cls, d: dict | None):
    """A settings dataclass from a YAML block; unknown keys are an error (catches typos)."""
    d = dict(d or {})
    unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings {sorted(unknown)}")
    return cls(**d)


def load_configs(path: str | Path = "configs/alphazero.yaml") -> tuple[AZConfig, NetConfig, MCTSConfig, float | None]:
    """configs/alphazero.yaml -> (training, network, search settings, max_gpu_hours)."""
    with open(path) as f:
        y = yaml.safe_load(f) or {}
    unknown = set(y) - {"train", "net", "mcts", "max_gpu_hours"}
    if unknown:
        raise ValueError(f"unknown sections in {path}: {sorted(unknown)}")
    cap = y.get("max_gpu_hours")
    return (_from_dict(AZConfig, y.get("train")), _from_dict(NetConfig, y.get("net")),
            _from_dict(MCTSConfig, y.get("mcts")), None if cap is None else float(cap))


def run_name(cfg: AZConfig, net_cfg: NetConfig) -> str:
    """e.g. 'az_b6c64_s0', or 'az_b6c64_s0_<tag>'."""
    name = f"az_b{net_cfg.blocks}c{net_cfg.channels}_s{cfg.seed}"
    return f"{name}_{cfg.tag}" if cfg.tag else name


# ----------------------------------------------------------- buffer ----

class ReplayBuffer:
    """The most recent ``capacity`` positions (a ring buffer of numpy arrays).

    planes  uint8 [N, 19, 8, 8]   side-to-move frame; plane 18 holds the halfmove count
    moves   int16 [N, MAX_LEGAL]  side-to-move indices of the legal moves, -1 = padding
    probs   float16 [N, MAX_LEGAL] visit share of each of those moves (the policy target)
    value   float32 [N]           the game's final score for the side to move
    """

    FIELDS = ("planes", "moves", "probs", "value")

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.planes = np.zeros((capacity, NUM_PLANES, 8, 8), dtype=np.uint8)
        self.moves = np.full((capacity, MAX_LEGAL), -1, dtype=np.int16)
        self.probs = np.zeros((capacity, MAX_LEGAL), dtype=np.float16)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.size = 0     # filled rows
        self.next = 0     # where the next row goes (oldest row once full)

    def add(self, planes, moves, probs, value) -> None:
        """Append rows (arrays with a leading batch dimension), overwriting the oldest once full."""
        n = len(value)
        if n > self.capacity:     # only the newest rows can fit
            planes, moves, probs, value = planes[-self.capacity:], moves[-self.capacity:], probs[-self.capacity:], value[-self.capacity:]
            n = self.capacity
        idx = (self.next + np.arange(n)) % self.capacity
        self.planes[idx], self.moves[idx], self.probs[idx], self.value[idx] = planes, moves, probs, value
        self.next = int((self.next + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        """n random rows (with replacement): (planes, moves, probs, value)."""
        idx = rng.integers(0, self.size, size=n)
        return self.planes[idx], self.moves[idx], self.probs[idx], self.value[idx]

    def save(self, path: Path) -> None:
        np.savez(path, size=self.size, next=self.next, **{k: getattr(self, k)[: self.size] for k in self.FIELDS})

    @classmethod
    def load(cls, path: Path, capacity: int) -> "ReplayBuffer":
        buf = cls(capacity)
        with np.load(path) as d:
            n = int(d["size"])
            for k in cls.FIELDS:
                getattr(buf, k)[:n] = d[k]
            buf.size, buf.next = n, int(d["next"])
        return buf


def encode_position(board: chess.Board, visits: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One position -> (uint8 planes, legal-move indices, visit shares), all in the side-to-move frame."""
    p = to_planes(stm_board(board))
    p[HALFMOVE_PLANE] = min(board.halfmove_clock, 100)   # exact integer; decode divides by 100 again
    white = board.turn == chess.WHITE
    legal = B.legal_moves(board)
    moves = np.full(MAX_LEGAL, -1, dtype=np.int16)
    probs = np.zeros(MAX_LEGAL, dtype=np.float16)
    moves[: len(legal)] = [stm_index(m, white) for m in legal]
    probs[: len(legal)] = [visits.get(m, 0.0) for m in legal]
    return p.astype(np.uint8), moves, probs


def decode_planes(planes_u8: np.ndarray) -> np.ndarray:
    """uint8 buffer planes -> the float32 planes the network was built for (to_planes's values)."""
    x = planes_u8.astype(np.float32)
    x[:, HALFMOVE_PLANE] /= 100.0
    return x


def game_examples(games: list[SelfPlayGame]) -> tuple[np.ndarray, ...]:
    """Every searched move of every game -> buffer rows (planes, moves, probs, value)."""
    rows = [(*encode_position(chess.Board(r.fen), r.info["visits"]), g.score_for(r.white))
            for g in games for r in g.moves]
    if not rows:
        return (np.zeros((0, NUM_PLANES, 8, 8), np.uint8), np.zeros((0, MAX_LEGAL), np.int16),
                np.zeros((0, MAX_LEGAL), np.float16), np.zeros(0, np.float32))
    planes, moves, probs, value = zip(*rows)
    return np.stack(planes), np.stack(moves), np.stack(probs), np.array(value, dtype=np.float32)


# ----------------------------------------------------------- update ----

def batch_losses(net: AZNet, planes_u8, moves, probs, value, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(policy loss, value loss, policy entropy) per position, for one minibatch of buffer rows."""
    x = torch.from_numpy(decode_planes(planes_u8)).to(device)
    mv = torch.from_numpy(moves.astype(np.int64)).to(device)
    valid = mv >= 0
    safe = mv.clamp(min=0)
    mask = torch.zeros(len(mv), NUM_MOVES, dtype=torch.bool, device=device)
    mask.scatter_(1, safe, valid)
    target = torch.zeros(len(mv), NUM_MOVES, device=device)
    target.scatter_add_(1, safe, torch.from_numpy(probs.astype(np.float32)).to(device) * valid)
    target = target / target.sum(1, keepdim=True).clamp(min=1e-8)     # float16 storage: renormalize
    v = torch.from_numpy(value).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, vlogit = net(x)
    logp = torch.log_softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1).masked_fill(~mask, 0.0)
    policy_loss = -(target * logp).sum(-1)
    value_loss = F.binary_cross_entropy_with_logits(vlogit.float(), v, reduction="none")
    entropy = -(logp.exp() * logp).sum(-1)
    return policy_loss, value_loss, entropy


def update(net: AZNet, buffer: ReplayBuffer, steps: int, cfg: AZConfig, optimizer, rng: np.random.Generator,
           device) -> dict:
    """``steps`` Adam steps on minibatches sampled from the buffer. Returns mean losses and entropy."""
    net.train()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(steps):
        pl, vl, ent = batch_losses(net, *buffer.sample(cfg.batch_size, rng), device)
        loss = pl.mean() + vl.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        totals["policy_loss"] += pl.mean().item()
        totals["value_loss"] += vl.mean().item()
        totals["entropy"] += ent.mean().item()
    net.eval()
    return {k: v / max(steps, 1) for k, v in totals.items()}


def update_steps(new_positions: int, cfg: AZConfig) -> int:
    """K: minibatches so that each new position is sampled about ``samples_per_position`` times."""
    return max(1, math.ceil(new_positions * cfg.samples_per_position / cfg.batch_size))


# ---------------------------------------------------------- storage ----

def _round_dir(run_dir: Path, rnd: int) -> Path:
    return run_dir / "checkpoints" / f"round_{rnd:04d}"


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """[(round, folder)] for every saved round (kept checkpoints + the latest), oldest first."""
    root = Path(run_dir) / "checkpoints"
    if not root.exists():
        return []
    return sorted((int(p.name.split("_")[1]), p) for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("round_") and not p.name.endswith(".tmp"))


def _kept_by_cap(run_dir: Path) -> set[int]:
    path = run_dir / "kept_rounds.json"
    return set(json.loads(path.read_text())) if path.exists() else set()


def _mark_kept(run_dir: Path, rnd: int) -> None:
    """A round the cap stopped at stays a kept checkpoint forever (each stage ends on one)."""
    kept = _kept_by_cap(run_dir) | {rnd}
    tmp = run_dir / "kept_rounds.json.tmp"
    tmp.write_text(json.dumps(sorted(kept)))
    os.replace(tmp, run_dir / "kept_rounds.json")


def _is_kept(run_dir: Path, rnd: int, cfg: AZConfig) -> bool:
    return rnd % cfg.checkpoint_every == 0 or rnd in _kept_by_cap(run_dir)


def _save_round(net: AZNet, run_dir: Path, rnd: int, optimizer, buffer: ReplayBuffer, counters: dict,
                cfg: AZConfig) -> None:
    """Write round ``rnd`` (network + resume state + buffer) atomically, then prune older rounds.

    A new folder per round, renamed into place, so a crash never leaves a half-written
    latest round. Older rounds: kept checkpoints lose their resume state and buffer
    (only net.pt stays, for evaluation); the rest are deleted.
    """
    final = _round_dir(run_dir, rnd)
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save({"net_config": dataclasses.asdict(net.cfg), "state_dict": net.state_dict()}, tmp / "net.pt")
    torch.save({"optimizer": optimizer.state_dict(), "counters": counters}, tmp / "resume.pt")
    buffer.save(tmp / "buffer.npz")
    os.replace(tmp, final)
    for r, path in list_checkpoints(run_dir):
        if r >= rnd:
            continue
        if _is_kept(run_dir, r, cfg):
            (path / "resume.pt").unlink(missing_ok=True)
            (path / "buffer.npz").unlink(missing_ok=True)
        else:
            shutil.rmtree(path, ignore_errors=True)


def load_checkpoint(round_dir: str | Path, device: str | torch.device | None = None) -> AZNet:
    """The network saved in a round folder, in eval mode, on ``device`` (default: GPU if available)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(Path(round_dir) / "net.pt", map_location=device, weights_only=False)
    net = AZNet(NetConfig(**state["net_config"])).to(device)
    net.load_state_dict(state["state_dict"])
    return net.eval()


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


def ending_counts(games: list[SelfPlayGame]) -> dict:
    """How the round's games ended (the same columns as Step 7's rl_log.csv)."""
    t = [g.termination for g in games]
    return {"ends_checkmate": t.count("checkmate"),
            "ends_repetition": t.count("threefold_repetition") + t.count("fivefold_repetition"),
            "ends_fifty_moves": t.count("fifty_moves") + t.count("seventyfive_moves"),
            "ends_stalemate": t.count("stalemate"),
            "ends_insufficient_material": t.count("insufficient_material")}


# ------------------------------------------------------------ train ----

def train(cfg: AZConfig, net_cfg: NetConfig, mcts_cfg: MCTSConfig, run_dir: str | Path, *,
          device: str | torch.device | None = None, max_gpu_hours: float | None = None,
          max_rounds: int | None = None, extra_info: dict | None = None, log=print) -> dict:
    """Run AlphaZero rounds; resumable. Returns the run summary.

    Inputs:
      cfg, net_cfg, mcts_cfg -- settings (load_configs); saved in run_config.json, and a
                                resume with different settings is refused
      run_dir        -- the run's folder (checkpoints/, az_log.csv, run_config.json)
      max_gpu_hours  -- stop once training GPU-hours reach this (staged 9 / 18 / 30;
                        NOT saved, so raising it and rerunning resumes); None = no cap
      max_rounds     -- stop after this round this session (e.g. the smoke run); resumable
    Files written in run_dir:
      run_config.json  -- settings
      checkpoints/round_NNNN/ -- net.pt (+ resume.pt and buffer.npz on the latest round)
      kept_rounds.json -- rounds the cap stopped at (kept forever)
      az_log.csv       -- one row per round: results, endings, losses, entropy, seconds, GPU-hours
      summary.json     -- totals at the end of the session
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    settings = {"train": dataclasses.asdict(cfg), "net": dataclasses.asdict(net_cfg),
                "mcts": dataclasses.asdict(mcts_cfg)}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text())
        if json.loads(json.dumps(settings)) != {k: saved.get(k) for k in settings}:
            raise ValueError(f"{run_dir} was started with different settings; use a new run folder (or a new tag)")
    else:
        cfg_path.write_text(json.dumps({**settings, "extra_info": extra_info or {}}, indent=2))

    torch.manual_seed(cfg.seed)                         # the network's random init
    net = AZNet(net_cfg).to(device).eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_positions)
    counters = {"round": 0, "gpu_seconds": 0.0, "games": 0, "positions": 0}

    saved_rounds = list_checkpoints(run_dir)
    if saved_rounds:
        rnd, path = saved_rounds[-1]
        if not (path / "resume.pt").exists():
            raise RuntimeError(f"latest round {path} has no resume.pt; can't resume safely")
        net.load_state_dict(torch.load(path / "net.pt", map_location=device, weights_only=False)["state_dict"])
        state = torch.load(path / "resume.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        counters = state["counters"]
        buffer = ReplayBuffer.load(path / "buffer.npz", cfg.buffer_positions)
        _drop_rows_after(run_dir / "az_log.csv", counters["round"])
        log(f"resuming {run_dir.name} after round {counters['round']} "
            f"({counters['gpu_seconds'] / 3600:.2f} GPU-h, buffer {buffer.size})")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def hours() -> float:
        return counters["gpu_seconds"] / 3600

    evaluator = AZEvaluator(net, device)
    stop_at = cfg.rounds if max_rounds is None else min(cfg.rounds, max_rounds)
    stopped_by_cap = False
    while counters["round"] < stop_at:
        if max_gpu_hours is not None and hours() >= max_gpu_hours:
            stopped_by_cap = True
            if counters["round"]:
                _mark_kept(run_dir, counters["round"])
            log(f"GPU-hour cap reached: {hours():.2f} >= {max_gpu_hours}; round {counters['round']} is kept")
            break
        rnd = counters["round"] + 1
        round_seed = cfg.seed * 1_000_003 + rnd

        sync()
        t0 = time.perf_counter()
        games = play_games(evaluator, cfg.games_per_round, mcts_cfg, opening_plies=cfg.opening_plies,
                           max_plies=cfg.max_plies, seed=round_seed)
        rows = game_examples(games)
        buffer.add(*rows)
        sync()
        t1 = time.perf_counter()
        steps = update_steps(len(rows[3]), cfg)
        losses = update(net, buffer, steps, cfg, optimizer, np.random.default_rng(round_seed), device)
        sync()
        t2 = time.perf_counter()

        counters["round"] = rnd
        counters["gpu_seconds"] += t2 - t0
        counters["games"] += len(games)
        counters["positions"] += len(rows[3])
        _save_round(net, run_dir, rnd, optimizer, buffer, counters, cfg)

        results = [g.white_score for g in games]
        row = {"round": rnd, "games": len(games), "positions": len(rows[3]),
               "white_wins": sum(s == 1.0 for s in results), "draws": sum(s == 0.5 for s in results),
               "black_wins": sum(s == 0.0 for s in results),
               "max_plies_draws": sum(g.termination == "max_plies" for g in games),
               **ending_counts(games),
               "avg_plies": round(sum(g.plies for g in games) / len(games), 1),
               **{k: round(v, 5) for k, v in losses.items()},
               "buffer_size": buffer.size, "update_steps": steps,
               "selfplay_seconds": round(t1 - t0, 1), "update_seconds": round(t2 - t1, 1),
               "train_gpu_hours": round(hours(), 4)}
        _append_csv(run_dir / "az_log.csv", row)
        log(f"[round {rnd}] W/D/L {row['white_wins']}/{row['draws']}/{row['black_wins']} "
            f"({row['avg_plies']} plies; mate {row['ends_checkmate']}, cap {row['max_plies_draws']}) | "
            f"policy {row['policy_loss']:.3f} value {row['value_loss']:.3f} entropy {row['entropy']:.2f} | "
            f"self-play {row['selfplay_seconds']:.0f}s + update {row['update_seconds']:.0f}s "
            f"({steps} steps) | {row['train_gpu_hours']:.2f} GPU-h")

    summary = {**counters, "train_gpu_hours": round(hours(), 4), "finished": counters["round"] >= cfg.rounds,
               "stopped_by_cap": stopped_by_cap, "kept_checkpoints": [r for r, _ in list_checkpoints(run_dir)]}
    if device.type == "cuda":
        summary["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def gpu_hours_at(run_dir: str | Path, rnd: int) -> float:
    """Training GPU-hours when round ``rnd`` finished -- results.csv's train_gpu_hours."""
    with open(Path(run_dir) / "az_log.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["round"]) == rnd:
                return float(r["train_gpu_hours"])
    raise ValueError(f"no az_log.csv row for round {rnd}")


# ------------------------------------------------------------- eval ----

class AlphaZeroAgent:
    """An eval player (eval.match.Agent): searches each position, no noise, plays the most-visited move.

    Its search budget (``mcts_cfg.simulations``) is its inference cost: it shows up in
    the eval games' seconds per move, and so in results.csv's sec_per_move.
    """

    def __init__(self, net: AZNet, mcts_cfg: MCTSConfig, name: str, device: str | torch.device | None = None):
        self.name = name
        self.mcts_cfg = mcts_cfg
        self.evaluator = AZEvaluator(net, device)

    def select_move(self, board: chess.Board) -> str:
        """The board is the match's copy with its move history, so the search sees repetitions."""
        return search_move(board, self.evaluator, self.mcts_cfg)


def load_agent(round_dir: str | Path, mcts_cfg: MCTSConfig, name: str,
               device: str | torch.device | None = None) -> AlphaZeroAgent:
    """A saved round as an eval player."""
    net = load_checkpoint(round_dir, device)
    return AlphaZeroAgent(net, mcts_cfg, name, next(net.parameters()).device)


class PolicyAgent:
    """The network alone, no search: plays the legal move its policy head rates highest.

    Rated next to AlphaZeroAgent on the same checkpoints, the Elo gap is what search
    adds to the same network (plan Step 9, "no-search evals"). One forward pass per
    move, so its sec_per_move is the network's raw inference cost.
    """

    def __init__(self, net: AZNet, name: str, device: str | torch.device | None = None):
        self.name = name
        self.evaluator = AZEvaluator(net, device)

    def select_move(self, board: chess.Board) -> str:
        """The top-prior legal move (ties go to the first in sorted UCI order, so it's deterministic)."""
        (priors, _), = self.evaluator.evaluate([board])
        return max(sorted(priors), key=lambda m: priors[m])


def load_policy_agent(round_dir: str | Path, name: str, device: str | torch.device | None = None) -> PolicyAgent:
    """A saved round as a no-search eval player."""
    net = load_checkpoint(round_dir, device)
    return PolicyAgent(net, name, next(net.parameters()).device)


def _spread(rounds: list[int], k: int = 4) -> list[int]:
    """Up to k of ``rounds``, evenly spaced, first and last included."""
    if len(rounds) <= k:
        return list(rounds)
    return sorted({rounds[round(i * (len(rounds) - 1) / (k - 1))] for i in range(k)})


def eval_rounds(run_dir: str | Path, checkpoint_every: int, per_stage: int = 4) -> list[int]:
    """Kept checkpoints to rate: up to ``per_stage`` per cap stage, evenly spaced, each stage's last included.

    A stage ends at a round the cap stopped at (kept_rounds.json). Rounds after the last
    cap stop (a run still going, or stopped by hand) form an open stage, rated from its
    regular kept checkpoints (multiples of checkpoint_every) -- not the stray latest round.
    """
    run_dir = Path(run_dir)
    saved = [r for r, _ in list_checkpoints(run_dir)]
    stage_ends = sorted(r for r in _kept_by_cap(run_dir) if r in saved)
    out, start = [], 0
    for end in stage_ends:
        stage = [r for r in saved if start < r <= end and (r % checkpoint_every == 0 or r == end)]
        out += _spread(stage, per_stage)
        start = end
    open_stage = [r for r in saved if r > start and r % checkpoint_every == 0]
    out += _spread(open_stage, per_stage)
    return sorted(set(out))
