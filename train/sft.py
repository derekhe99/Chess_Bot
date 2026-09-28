"""Step 5: supervised fine-tuning on Stockfish labels (the Day-1 baseline).

Intent: teach the LLM player (model/llm_policy.py) to imitate Stockfish. For
each labeled position the model reads the SAME prompt it reads in play, and two
things are graded:
- the move: -log P(reply = Stockfish's move | prompt), summed over the move's
  tokens and the end-of-reply token. This is exactly the score masked play ranks
  moves by, so lowering it pushes Stockfish's move toward the top of that ranking.
- the value: the value head's win probability vs Stockfish's (BCE by default).
Total loss = move loss + lambda * value loss. Only the LoRA adapters and the
value head train; the 0.6B base weights stay frozen.

Everything a later Step needs is logged: positions seen, tokens processed,
GPU seconds (training and validation kept apart, for the compute budget), and
losses. A checkpoint (adapters + value head) is saved every ``checkpoint_every``
steps, plus step 0 = the untrained model, so the Elo-vs-training curve has a
starting point and Step 5 can pick checkpoint rungs for the reference set.

Colab-proof: every checkpoint also writes ``resume.pt`` (optimizer state and
counters). Calling train() again on the same run folder continues from the last
checkpoint and gives the same weights as an uninterrupted run (the data order
and dropout are seeded per epoch / per step, so no hidden random state is lost).

Main pieces:
- TrainConfig        -- every training setting; the ``train:`` block of configs/sft.yaml
- run_name           -- the run folder's name, built from the settings that define the run
- subset             -- the seeded "small slice" of a dataset
- encode / collate   -- records -> token ids (via llm_policy's one prompt) -> padded batch
- batch_losses       -- per-example move and value losses for one batch
- validate           -- held-out losses + how often the masked pick equals Stockfish's move
- train              -- the resumable training loop
- list_checkpoints / load_checkpoint -- find and load saved checkpoints for eval
- screen_checkpoints -- a quick rough rating of every checkpoint (Step 5's first look)
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from engine import board as B
from model.llm_policy import IGNORE_INDEX, LLMPolicy, PolicyConfig


@dataclass(frozen=True)
class TrainConfig:
    """All SFT settings. Defaults mirror configs/sft.yaml ``train:`` (agreed Sep 28, 2026)."""

    dataset: str = "sft_2013-06_d12_n50000"  # folder under Drive data/processed/
    train_positions: int = 5000        # the "small slice"; None or >= split size = all of it
    val_positions: int = 1000          # held-out positions for validation losses
    val_top1_positions: int = 200      # held-out positions for top-1 agreement with Stockfish
    epochs: int = 10
    batch_size: int = 16
    optimizer: str = "adamw"
    learning_rate: float = 1e-4        # constant
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0      # max gradient norm; None = off
    value_loss: str = "bce"            # "bce" | "mse"
    value_loss_weight: float = 1.0     # lambda
    checkpoint_every: int = 500        # optimizer steps
    log_every: int = 25
    seed: int = 0

    def __post_init__(self):
        if self.value_loss not in ("bce", "mse"):
            raise ValueError(f"value_loss must be 'bce' or 'mse', got {self.value_loss!r}")
        if self.optimizer != "adamw":
            raise ValueError(f"only 'adamw' is implemented, got {self.optimizer!r}")
        if self.batch_size < 1 or self.epochs < 1 or self.checkpoint_every < 1:
            raise ValueError("batch_size, epochs, and checkpoint_every must be positive")

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """From a YAML block; unknown keys are an error (catches typos)."""
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown train settings {sorted(unknown)}")
        return cls(**d)


def run_name(policy_cfg: PolicyConfig, cfg: TrainConfig) -> str:
    """Folder name for a run, e.g. 'structured_masked_n5000_e10_s0'. Same settings -> same folder (resume)."""
    mask = "masked" if policy_cfg.masked else "unmasked"
    return f"{policy_cfg.representation}_{mask}_n{cfg.train_positions}_e{cfg.epochs}_s{cfg.seed}"


def subset(n_total: int, n: int | None, seed: int) -> list[int]:
    """A seeded random choice of ``n`` indices out of ``n_total`` (all of them if n is None or too big).

    A random subset (not the first n) keeps the slice's phase mix close to the
    full dataset's, since the Parquet file is stored in sampling order.
    """
    if n is None or n >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), n))


# ------------------------------------------------------------ batching ----

def encode(policy: LLMPolicy, record: dict) -> dict:
    """One dataset record -> one training example (token ids + labels + value target).

    The prompt comes from the policy itself (``training_example``), so training
    reads exactly the text that play reads.
    """
    ex = policy.training_example(chess.Board(record["fen"]), record["label_move"])
    ex["label_value"] = float(record["label_value"])
    return ex


def collate(examples: list[dict], pad_id: int, device=None) -> dict:
    """Right-pad a list of examples into tensors.

    Output: input_ids / attention_mask / labels of shape (batch, longest),
    value_index and label_value of shape (batch,). Padding goes AFTER the real
    tokens, so real tokens keep the same positions (and the same numbers) as
    in play, and the causal mask keeps padding from affecting them.
    """
    longest = max(len(e["input_ids"]) for e in examples)
    n = len(examples)
    ids = torch.full((n, longest), pad_id, dtype=torch.long)
    mask = torch.zeros((n, longest), dtype=torch.long)
    labels = torch.full((n, longest), IGNORE_INDEX, dtype=torch.long)
    for i, e in enumerate(examples):
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        mask[i, :k] = 1
        labels[i, :k] = torch.tensor(e["labels"])
    out = {"input_ids": ids, "attention_mask": mask, "labels": labels,
           "value_index": torch.tensor([e["value_index"] for e in examples]),
           "label_value": torch.tensor([e["label_value"] for e in examples], dtype=torch.float32)}
    return {k: v.to(device) for k, v in out.items()} if device is not None else out


def batch_losses(policy: LLMPolicy, batch: dict, value_loss: str = "bce") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example losses for one batch.

    Output: (move_nll, value_loss, win_logit), each of shape (batch,).
      move_nll  -- -log P(reply = label move | prompt), summed over the move's tokens
      value_loss-- BCE (or squared error) of the win probability vs the label
      win_logit -- the value head's raw output (sigmoid -> win probability)

    Core logic: token t is predicted by the logits at position t-1. Only the
    few positions that predict move tokens need logits, so the model is asked
    for just those (``logits_to_keep``): the full (batch x length x 151k-vocab)
    logit tensor would be most of the memory. The value head reads the last
    layer's hidden state at the last prompt token -- the same place play reads it.
    """
    labels = batch["labels"]
    has = labels != IGNORE_INDEX
    cols = has.any(dim=0).nonzero().squeeze(-1)
    first, last = int(cols[0]), int(cols[-1])        # label columns across the whole batch
    keep = torch.arange(first - 1, last, device=labels.device)  # the positions that predict them
    out = policy.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       output_hidden_states=True, logits_to_keep=keep)
    targets = labels[:, first:last + 1]
    real = targets != IGNORE_INDEX
    logp = torch.log_softmax(out.logits.float(), dim=-1)
    tok = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    move_nll = -(tok * real).sum(dim=1)
    rows = torch.arange(labels.shape[0], device=labels.device)
    win_logit = policy.value_head(out.hidden_states[-1][rows, batch["value_index"]])
    target = batch["label_value"]
    if value_loss == "bce":
        v = F.binary_cross_entropy_with_logits(win_logit, target, reduction="none")
    else:
        v = (torch.sigmoid(win_logit) - target) ** 2
    return move_nll, v, win_logit


# ---------------------------------------------------------- validation ----

@torch.inference_mode()
def validate(policy: LLMPolicy, val_examples: list[dict], top1_records: list[dict], cfg: TrainConfig) -> dict:
    """Held-out numbers for one checkpoint.

    val_move_loss  -- mean -log P(Stockfish's move) on held-out games (lower = better imitation)
    val_value_loss -- mean value loss (the configured kind), plus val_value_mae in win probability
    val_top1       -- share of positions where masked play (score every legal move,
                      take the best) picks exactly Stockfish's move. The closest
                      cheap stand-in for playing strength.
    """
    policy.model.eval()
    moves, values, maes = [], [], []
    for start in range(0, len(val_examples), cfg.batch_size):
        batch = collate(val_examples[start:start + cfg.batch_size], policy.pad_id, policy.device)
        m, v, logit = batch_losses(policy, batch, cfg.value_loss)
        moves += m.tolist()
        values += v.tolist()
        maes += (torch.sigmoid(logit) - batch["label_value"]).abs().tolist()
    hits = 0
    for rec in top1_records:
        board = chess.Board(rec["fen"])
        scores, _ = policy.score_moves(board, B.legal_moves(board))
        hits += max(scores, key=scores.get) == rec["label_move"]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {"val_move_loss": mean(moves), "val_value_loss": mean(values), "val_value_mae": mean(maes),
            "val_top1": hits / len(top1_records) if top1_records else float("nan")}


# ------------------------------------------------------------- storage ----

def _step_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:06d}"


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """[(step, folder)] for every finished checkpoint of a run, oldest first."""
    root = Path(run_dir) / "checkpoints"
    if not root.exists():
        return []
    found = [(int(p.name.split("_")[1]), p) for p in root.iterdir()
             if p.is_dir() and p.name.startswith("step_") and not p.name.endswith(".tmp")]
    return sorted(found)


def load_checkpoint(step_dir: str | Path, *, device: str | None = None, name: str | None = None,
                    **policy_changes) -> LLMPolicy:
    """A saved checkpoint -> a playable LLMPolicy (base model re-downloaded, adapters + value head loaded).

    ``policy_changes`` override play settings (e.g. masked=False); the default is
    the settings it was trained with.
    """
    step_dir = Path(step_dir)
    cfg = PolicyConfig.from_dict(json.loads((step_dir / "policy_config.json").read_text()))
    if policy_changes:
        cfg = dataclasses.replace(cfg, **policy_changes)
    return LLMPolicy.from_pretrained(cfg, device=device, adapter_dir=step_dir,
                                     name=name or f"sft_{step_dir.parent.parent.name}_{step_dir.name}")


def _save_checkpoint(policy: LLMPolicy, run_dir: Path, step: int) -> Path:
    """Adapters + value head for ``step``, written to a temp folder then renamed (never half-written)."""
    final = _step_dir(run_dir, step)
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    policy.save(tmp)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return final


def _save_resume(run_dir: Path, optimizer, scaler, counters: dict) -> None:
    """Optimizer + counters, written atomically. Points at the checkpoint of the same step."""
    tmp = run_dir / "resume.pt.tmp"
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "counters": counters}, tmp)
    os.replace(tmp, run_dir / "resume.pt")


def _load_adapters(policy: LLMPolicy, step_dir: Path) -> None:
    """Load a checkpoint's adapters and value head INTO an existing policy (keeps them trainable)."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(policy.model, load_file(step_dir / "adapter_model.safetensors",
                                                      device=str(policy.device)))
    policy.value_head.load_state_dict(torch.load(step_dir / "value_head.pt", map_location=policy.device))


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _drop_rows_after(path: Path, step: int) -> None:
    """On resume: forget log rows written after the checkpoint we're resuming from (they'll be redone)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(r for r in rows if int(r["step"]) <= step)


# --------------------------------------------------------------- train ----

def train(policy: LLMPolicy, train_ds, val_ds, cfg: TrainConfig, run_dir: str | Path, *,
          extra_info: dict | None = None, max_steps: int | None = None, log=print) -> dict:
    """Fine-tune ``policy`` on Stockfish labels; resumable. Returns the run summary.

    Inputs:
      policy          -- an LLMPolicy with fresh (untrained) LoRA adapters; on resume
                         the last checkpoint's weights are loaded into it
      train_ds/val_ds -- data.dataset.PositionDataset (plain records)
      cfg             -- TrainConfig
      run_dir         -- the run's folder (checkpoints, logs, resume state)
      extra_info      -- anything to record with the run (e.g. GPU name, labeling CPU charged)
      max_steps       -- stop early after this global step (a checkpoint is saved there);
                         for smoke tests -- the run can be continued later
    Files written in run_dir:
      run_config.json -- settings (a resume with different settings is refused)
      checkpoints/step_NNNNNN/ -- adapters + value head + policy config (step 0 = untrained)
      resume.pt       -- optimizer + counters at the latest checkpoint
      train_log.csv   -- step, epoch, train losses, positions/tokens seen, GPU seconds
      val_log.csv     -- the validate() numbers at every checkpoint
      summary.json    -- totals when the run finishes
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = policy.device

    # -- the run's identity: refuse to mix settings in one folder ----------
    settings = {"train": dataclasses.asdict(cfg), "policy": dataclasses.asdict(policy.config)}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text())
        if json.loads(json.dumps(settings)) != {k: saved[k] for k in ("train", "policy")}:
            raise ValueError(f"{run_dir} was started with different settings; use a new run folder")
    else:
        cfg_path.write_text(json.dumps({**settings, "extra_info": extra_info or {}}, indent=2))

    # -- data: fixed slices, tokenized once ---------------------------------
    train_idx = subset(len(train_ds), cfg.train_positions, cfg.seed)
    val_idx = subset(len(val_ds), cfg.val_positions, cfg.seed + 1)
    train_ex = [encode(policy, train_ds[i]) for i in train_idx]
    val_ex = [encode(policy, val_ds[i]) for i in val_idx]
    top1_recs = [val_ds[i] for i in val_idx[:cfg.val_top1_positions]]
    n = len(train_ex)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    stop_at = total_steps if max_steps is None else min(max_steps, total_steps)

    # -- optimizer (only LoRA + value head) ---------------------------------
    params = policy.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    fp16 = device.type == "cuda" and policy.model.get_input_embeddings().weight.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)  # fp16 (T4) only: keeps small gradients from rounding to 0

    counters = {"step": 0, "positions_seen": 0, "tokens_seen": 0, "gpu_seconds_train": 0.0,
                "gpu_seconds_val": 0.0}
    ckpts = list_checkpoints(run_dir)
    if (run_dir / "resume.pt").exists():
        state = torch.load(run_dir / "resume.pt", map_location=device, weights_only=False)
        counters = state["counters"]
        _load_adapters(policy, _step_dir(run_dir, counters["step"]))
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        for name in ("train_log.csv", "val_log.csv"):
            _drop_rows_after(run_dir / name, counters["step"])
        log(f"resuming {run_dir.name} from step {counters['step']} / {total_steps}")
    elif ckpts:
        raise RuntimeError(f"{run_dir} has checkpoints but no resume.pt; refusing to overwrite them")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def checkpoint(step: int) -> None:
        sync()
        t0 = time.perf_counter()
        metrics = validate(policy, val_ex, top1_recs, cfg)
        sync()
        counters["gpu_seconds_val"] += time.perf_counter() - t0
        _save_checkpoint(policy, run_dir, step)
        _save_resume(run_dir, optimizer, scaler, counters)
        _append_csv(run_dir / "val_log.csv", {"step": step, "epoch": step / steps_per_epoch,
                                               "positions_seen": counters["positions_seen"],
                                               **{k: round(v, 5) for k, v in metrics.items()}})
        log(f"[checkpoint {step}/{total_steps}] val move loss {metrics['val_move_loss']:.3f} | "
            f"value loss {metrics['val_value_loss']:.3f} (MAE {metrics['val_value_mae']:.3f}) | "
            f"top-1 vs Stockfish {metrics['val_top1']:.1%}")

    if counters["step"] == 0 and not ckpts:
        log(f"{n} training positions, {steps_per_epoch} steps/epoch, {total_steps} steps total; "
            f"checkpoint every {cfg.checkpoint_every}")
        checkpoint(0)  # the untrained model: the curve's starting point

    orders: dict[int, list[int]] = {}
    window = {"move": 0.0, "value": 0.0, "count": 0}
    while counters["step"] < stop_at:
        step = counters["step"]
        epoch, b = divmod(step, steps_per_epoch)
        if epoch not in orders:  # same shuffle for an epoch however many times we resume
            order = list(range(n))
            random.Random(cfg.seed * 1_000_003 + epoch).shuffle(order)
            orders = {epoch: order}
        batch_ex = [train_ex[i] for i in orders[epoch][b * cfg.batch_size:(b + 1) * cfg.batch_size]]
        torch.manual_seed(cfg.seed * 1_000_003 + step)  # dropout masks depend only on the step

        sync()
        t0 = time.perf_counter()
        policy.model.train()
        batch = collate(batch_ex, policy.pad_id, device)
        move_nll, value, _ = batch_losses(policy, batch, cfg.value_loss)
        loss = move_nll.mean() + cfg.value_loss_weight * value.mean()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        sync()
        counters["gpu_seconds_train"] += time.perf_counter() - t0
        counters["step"] = step = step + 1
        counters["positions_seen"] += len(batch_ex)
        counters["tokens_seen"] += int(batch["attention_mask"].sum())
        window["move"] += move_nll.mean().item()
        window["value"] += value.mean().item()
        window["count"] += 1

        if step % cfg.log_every == 0 or step == stop_at:
            k = window["count"]
            row = {"step": step, "epoch": round(step / steps_per_epoch, 4),
                   "train_move_loss": round(window["move"] / k, 5), "train_value_loss": round(window["value"] / k, 5),
                   "positions_seen": counters["positions_seen"], "tokens_seen": counters["tokens_seen"],
                   "gpu_seconds_train": round(counters["gpu_seconds_train"], 2)}
            _append_csv(run_dir / "train_log.csv", row)
            log(f"step {step}/{total_steps} (epoch {step / steps_per_epoch:.2f}) | move loss {row['train_move_loss']:.3f} "
                f"| value loss {row['train_value_loss']:.3f} | {counters['gpu_seconds_train'] / step:.2f} s/step")
            window = {"move": 0.0, "value": 0.0, "count": 0}
        if step % cfg.checkpoint_every == 0 or step == stop_at:
            checkpoint(step)

    policy.model.eval()
    summary = {**counters, "total_steps": total_steps, "finished": counters["step"] >= total_steps,
               "train_positions": n, "steps_per_epoch": steps_per_epoch,
               "checkpoints": [s for s, _ in list_checkpoints(run_dir)]}
    if torch.cuda.is_available() and device.type == "cuda":
        summary["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------ quick screening ----

def screen_checkpoints(run_dir: str | Path, opponents: dict, fixed: dict[str, float], n_games: int, *,
                       steps: list[int] | None = None, device: str | None = None, seed: int = 0,
                       loader=None, verbose: bool = True, **match_kwargs):
    """A quick, rough rating of each checkpoint -- where did training land on the Elo scale?

    Inputs:
      run_dir   -- the training run's folder
      opponents -- {name: agent}: e.g. random, greedy, and Stockfish 1320
      fixed     -- ratings held fixed (the pin, plus any frozen members that play)
      n_games   -- games per checkpoint per opponent (small: this is a first look)
      steps     -- which checkpoints (default: all)
      loader    -- step_dir -> agent (default load_checkpoint; tests pass a tiny one)
    Output: (estimates {name: RatingEstimate} for every non-fixed player, all game records)

    Core logic: checkpoints are loaded one at a time (only one copy of the model
    in GPU memory), each plays every opponent, and then ONE Elo fit over all the
    games rates the checkpoints -- and random and greedy with them, since
    checkpoints that score against both greedy and Stockfish 1320 link the two
    ends. If no checkpoint takes points off Stockfish 1320, the fit says so
    (the ratings are flagged off the scale), which is exactly the thing to know
    before picking checkpoint rungs.
    """
    from eval.elo import fit_with_ci
    from eval.match import play_pairs, to_results

    loader = loader or (lambda d: load_checkpoint(d, device=device))
    chosen = [(s, d) for s, d in list_checkpoints(run_dir) if steps is None or s in steps]
    records = []
    for s, d in chosen:
        agent = loader(d)
        agent.name = f"step_{s}"
        records += play_pairs({agent.name: agent, **opponents}, [(agent.name, o) for o in opponents],
                              n_games, seed=seed + s, verbose=verbose, **match_kwargs)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return fit_with_ci(to_results(records), fixed, seed=seed), records
