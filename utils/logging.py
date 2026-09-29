"""One row per training run, in one place, so experiments can be scanned and plotted
across a whole Step instead of read one Drive folder at a time.

Intent: `train/sft.py` (and later `train/llm_rl.py`, `train/az_train.py`) already save
everything about a run to its own folder -- config, per-checkpoint validation numbers,
the checkpoint screen. `log_experiment` reads that back and appends one summary row to
a shared `results/experiments.csv`. This is also what Step 11's Elo-vs-samples and
Elo-vs-compute curves are built from (plan v3, Sec 2.1, `utils/logging.py`): the shared
CSV is a superset of what one run's own files show, aggregated across many of them.

The one thing this can't fill in for you is `finding` -- a short, human sentence on what
the run actually showed. Everything else is read off the run's own saved files.

Main pieces:
- log_experiment   -- read one run_dir's saved files, append a row
- read_experiments -- the CSV back as a DataFrame
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "run_name", "step", "date", "branch", "representation", "masked",
    "train_positions", "epochs", "batch_size", "learning_rate", "value_loss",
    "value_loss_weight", "examples_seen", "checkpoints",
    "final_val_top1", "best_val_top1", "best_val_top1_step",
    "final_val_move_loss", "final_val_value_loss",
    "screened", "beat_untrained", "scored_vs_sf1320", "finding",
]


def log_experiment(run_dir: str | Path, experiments_path: str | Path, *, step: int | str,
                   branch: str, finding: str, beat_untrained: bool | None = None,
                   scored_vs_sf1320: bool | None = None) -> dict:
    """Append one row for the run at ``run_dir`` to ``experiments_path``.

    Inputs:
      run_dir            -- a run folder train() has written (needs run_config.json,
                            val_log.csv, summary.json)
      experiments_path   -- the shared CSV (created with a header on first use)
      step                -- which plan Step this run belongs to, e.g. 5
      branch              -- the git branch this run's code came from
      finding             -- one or two sentences: what this run showed
      beat_untrained      -- Step 5 gate check 1: did the best checkpoint's CI clear the
                            untrained model's? None if not yet screened.
      scored_vs_sf1320    -- did any checkpoint take a point off sf_1320 in the screen?
                            None if not yet screened, or sf_1320 wasn't a screen opponent.
    Output: the row that was written (also useful to print/inspect right away).

    Core logic: settings come from run_config.json (exactly what the run was configured
    with -- no re-deriving them from the folder name); the training curve's last and best
    rows come from val_log.csv (the same file the notebook already plots). ``screened``
    just reflects whether screen.json exists yet; the pass/fail flags are arguments
    because interpreting a screen (which opponents, what CI rule) is a judgment call.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    vlog = pd.read_csv(run_dir / "val_log.csv")
    best_i = int(vlog["val_top1"].idxmax())
    row = {
        "run_name": run_dir.name, "step": step, "date": _dt.date.today().isoformat(), "branch": branch,
        "representation": cfg["policy"]["representation"], "masked": cfg["policy"]["masked"],
        "train_positions": cfg["train"]["train_positions"], "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"], "learning_rate": cfg["train"]["learning_rate"],
        "value_loss": cfg["train"]["value_loss"], "value_loss_weight": cfg["train"]["value_loss_weight"],
        "examples_seen": summary["examples_seen"], "checkpoints": len(summary["checkpoints"]),
        "final_val_top1": round(float(vlog["val_top1"].iloc[-1]), 4),
        "best_val_top1": round(float(vlog["val_top1"].iloc[best_i]), 4),
        "best_val_top1_step": int(vlog["step"].iloc[best_i]),
        "final_val_move_loss": round(float(vlog["val_move_loss"].iloc[-1]), 4),
        "final_val_value_loss": round(float(vlog["val_value_loss"].iloc[-1]), 4),
        "screened": (run_dir / "screen.json").exists(),
        "beat_untrained": beat_untrained, "scored_vs_sf1320": scored_vs_sf1320,
        "finding": finding,
    }
    experiments_path = Path(experiments_path)
    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    new = not experiments_path.exists()
    with open(experiments_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def read_experiments(experiments_path: str | Path) -> pd.DataFrame:
    """The shared experiment log as a DataFrame, newest first. Empty if it doesn't exist yet."""
    path = Path(experiments_path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path).iloc[::-1].reset_index(drop=True)
