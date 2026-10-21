"""Checks for eval/plots.py: the Elo-vs-compute chart reads results.csv correctly.

What they prove: model rows group by run, an SFT-checkpoint reference (step_20000)
lands on its run at interpolated GPU-hours, an RL run is joined to the SFT point it
started from, other references become lines, and the PNG is written.

    python -m pytest tests/test_plots.py -v
"""
import csv

from eval.anchors import RESULTS_COLUMNS
from eval.plots import elo_points, plot_elo_vs_compute

SFT = "structured_masked_nall_e10_s0"
RL = "llm_rl/structured_masked_sft25000_s0"


def write_results(path):
    rows = [
        dict(name="sf_1320", role="pin", rating=1320, ci_low=1320, ci_high=1320),
        dict(name="greedy", role="reference", rating=641.3, ci_low=491.9, ci_high=724.4, spec='{"kind": "greedy"}'),
        dict(name="step_20000", role="reference", rating=719.3, ci_low=576.6, ci_high=804.2,
             spec=f'{{"kind": "sft_checkpoint", "run": "{SFT}", "step": 20000}}'),
        dict(name=f"{SFT}@5000", role="model", rating=655.2, ci_low=620.8, ci_high=688.8, run=SFT, step=5000,
             train_gpu_hours=1.5),
        dict(name=f"{SFT}@25000", role="model", rating=680.5, ci_low=638.1, ci_high=714.0, run=SFT, step=25000,
             train_gpu_hours=7.5),
        dict(name="rl_x@r5", role="model", rating=646.7, ci_low=603.1, ci_high=697.2, run=RL, step=5,
             train_gpu_hours=10.4, note=f"self-play RL from sft {SFT}@25000"),
        dict(name="rl_x@r10", role="model", rating=714.0, ci_low=663.6, ci_high=764.8, run=RL, step=10,
             train_gpu_hours=13.1, note=f"self-play RL from sft {SFT}@25000"),
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULTS_COLUMNS})


def test_elo_points_groups_runs_interpolates_references_and_joins_rl(tmp_path):
    path = tmp_path / "results.csv"
    write_results(path)
    runs, starts, lines = elo_points(path)
    assert list(runs) == [SFT, RL]
    sft = runs[SFT]
    assert [p["label"] for p in sft] == ["5k", "20k", "25k"]          # step_20000 joins its run, in order
    ref = [p for p in sft if p["hollow"]][0]
    assert ref["x"] == 6.0 and ref["y"] == 719.3                        # 1.5 + 15/20 * (7.5 - 1.5)
    assert [p["label"] for p in runs[RL]] == ["r5", "r10"]
    assert starts[RL] == (7.5, 680.5)
    assert {l["name"] for l in lines} == {"sf_1320", "greedy"}


def test_plot_writes_a_png(tmp_path):
    path = tmp_path / "results.csv"
    write_results(path)
    out = tmp_path / "out" / "elo_vs_compute.png"
    plot_elo_vs_compute(path, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
