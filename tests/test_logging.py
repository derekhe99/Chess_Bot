"""Checks for utils/logging.py's shared experiment log.

    python -m pytest tests/test_logging.py -v
"""
import json

import pandas as pd
import pytest

from utils.logging import COLUMNS, log_experiment, read_experiments


def make_run(tmp_path, name, *, train_positions=5000, top1_curve=(0.03, 0.17, 0.21)):
    run_dir = tmp_path / name
    run_dir.mkdir()
    cfg = {"train": {"train_positions": train_positions, "epochs": 10, "batch_size": 16,
                     "learning_rate": 1e-4, "value_loss": "bce", "value_loss_weight": 1.0},
          "policy": {"representation": "structured", "masked": True}}
    (run_dir / "run_config.json").write_text(json.dumps(cfg))
    (run_dir / "summary.json").write_text(json.dumps({"examples_seen": train_positions * 10,
                                                       "checkpoints": [0, 500, 1000]}))
    steps = [0, 500, 1000][:len(top1_curve)]
    pd.DataFrame({"step": steps, "val_top1": top1_curve, "val_move_loss": [23.0, 5.0, 4.8][:len(top1_curve)],
                 "val_value_loss": [0.69, 0.65, 0.6][:len(top1_curve)]}).to_csv(run_dir / "val_log.csv", index=False)
    return run_dir


def test_log_experiment_reads_the_runs_own_files(tmp_path):
    run_dir = make_run(tmp_path, "structured_masked_n5000_e10_s0")
    exp_path = tmp_path / "results" / "experiments.csv"
    row = log_experiment(run_dir, exp_path, step=5, branch="step-5-sft",
                         finding="Real move-selection learning, not enough to reach sf_1320.",
                         beat_untrained=False, scored_vs_sf1320=False)
    assert row["train_positions"] == 5000 and row["examples_seen"] == 50000
    assert row["final_val_top1"] == 0.21 and row["best_val_top1"] == 0.21 and row["best_val_top1_step"] == 1000
    assert row["screened"] is False and row["beat_untrained"] is False

    df = read_experiments(exp_path)
    assert len(df) == 1 and list(df.columns) == COLUMNS
    assert df.iloc[0]["run_name"] == "structured_masked_n5000_e10_s0"


def test_log_experiment_appends_and_newest_first(tmp_path):
    exp_path = tmp_path / "results" / "experiments.csv"
    log_experiment(make_run(tmp_path, "run_a"), exp_path, step=5, branch="b", finding="first")
    log_experiment(make_run(tmp_path, "run_b", train_positions=50000, top1_curve=(0.03, 0.3)),
                   exp_path, step=5, branch="b", finding="second")
    df = read_experiments(exp_path)
    assert list(df["run_name"]) == ["run_b", "run_a"]        # newest first
    assert df.iloc[0]["train_positions"] == 50000


def test_read_experiments_empty_if_missing(tmp_path):
    df = read_experiments(tmp_path / "nope.csv")
    assert df.empty and list(df.columns) == COLUMNS
