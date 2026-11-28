"""Checks for eval/gpu_util.py: GPU utilization of the headline rows (plan v5 Step 11 support).

What they prove:
- phase_hours sums a round log up to the row's round, takes Stockfish out of self-play, adds the
  start's hours, and refuses a round the log doesn't have.
- split reads an AlphaZero run folder (log + run_config's init_gpu_hours).
- GpuSampler averages only the readings inside a window; timed reports seconds + utilization.
- combine weights phase utilizations by hours (SFT parts at the SFT row's value, Stockfish at 0),
  flags a slice whose seconds per position differ from the log's; format_table renders it.
- One AlphaZero replay runs end to end on the CPU with a tiny network.

    python -m pytest tests/test_gpu_util.py -v
"""
import csv
import json
import time

import pytest

from eval.gpu_util import ROWS, GpuSampler, Row, combine, format_table, phase_hours, split, timed


def _log(n, **extra):
    return [{"round": str(r), "positions": "100", "selfplay_seconds": "3600", "update_seconds": "360",
             "train_gpu_hours": str(round(1 + r * 1.1, 4)), **extra} for r in range(1, n + 1)]


def test_phase_hours_sums_up_to_the_round_and_adds_the_start():
    h = phase_hours(_log(5), 3, init_hours=1.0)
    assert (h["rounds"], h["positions"]) == (3, 300)
    assert h["selfplay"] == pytest.approx(3.0) and h["update"] == pytest.approx(0.3)
    assert h["total"] == pytest.approx(4.3) and h["logged_total"] == pytest.approx(4.3)


def test_phase_hours_moves_stockfish_out_of_selfplay():
    h = phase_hours(_log(2, stockfish_seconds="720"), 2)
    assert h["stockfish"] == pytest.approx(0.4) and h["selfplay"] == pytest.approx(1.6)
    assert h["total"] == pytest.approx(2.2)


def test_phase_hours_needs_the_round():
    with pytest.raises(ValueError):
        phase_hours(_log(2), 3)


def test_split_reads_an_alphazero_run(tmp_path):
    d = tmp_path / "checkpoints" / "alphazero" / "az_x"
    d.mkdir(parents=True)
    (d / "run_config.json").write_text(json.dumps({"init_gpu_hours": 0.5}))
    rows = _log(4)
    with open(d / "az_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    h = split(Row("x", "x", "alphazero", "az_x", 4), tmp_path)
    assert h["init"] == 0.5 and h["selfplay"] == pytest.approx(4.0)


def test_sampler_averages_inside_the_window():
    s = GpuSampler()
    s.samples = [(0.0, 90.0), (1.0, 10.0), (2.0, 20.0), (3.0, 90.0)]
    assert s.mean(0.5, 2.5) == pytest.approx(15.0)
    assert s.mean(5.0, 6.0) is None


def test_timed_reports_seconds_and_utilization():
    with GpuSampler(interval=0.01, read=lambda: 42.0) as s:
        out, info = timed(s, lambda: (time.sleep(0.1), 7)[1])
    assert out == 7 and info["seconds"] >= 0.1 and info["util"] == pytest.approx(42.0)


def test_combine_weights_by_hours():
    rows = (Row("sft500", "SFT", "sft", "s", 10), Row("rl", "RL", "llm_rl", "r", 2),
            Row("az", "AZ", "alphazero", "a", 2))
    splits = {"sft500": {"init": 0.0, "sft": 10.0},
              "rl": {"init": 2.0, "selfplay": 1.0, "stockfish": 1.0, "update": 4.0, "positions": 3600},
              "az": {"init": 0.0, "selfplay": 9.0, "update": 1.0, "positions": 36000}}
    measures = {"sft500": {"sft": {"util": 90.0}},
                "rl": {"selfplay": {"util": 30.0, "seconds": 10.0, "positions": 10}, "update": {"util": 80.0}},
                "az": {"selfplay": {"util": 1.0, "seconds": 100.0, "positions": 100}, "update": {"util": 50.0}}}
    c = {r["row"]: r for r in combine(splits, measures, rows)}
    assert c["SFT"]["overall_util"] == 90.0
    # (2*90 + 1*30 + 1*0 + 4*80) / 8 = 66.25
    assert c["RL"]["overall_util"] == pytest.approx(66.2, abs=0.1) and c["RL"]["total_h"] == 8.0
    assert c["RL"]["slice_vs_log_sec_per_position"] == 1.0
    assert c["AZ"]["overall_util"] == pytest.approx(5.9) and c["AZ"]["complete"]
    table = format_table(list(c.values()))
    assert "**66.2%**" in table and "stockfish 1.0 h @ 0.0%" in table


def test_combine_flags_an_unrepresentative_slice():
    rows = (Row("az", "AZ", "alphazero", "a", 1),)
    c = combine({"az": {"init": 0.0, "selfplay": 1.0, "update": 0.1, "positions": 3600}},
                {"az": {"selfplay": {"util": 2.0, "seconds": 10.0, "positions": 2}, "update": {"util": 60.0}}}, rows)
    assert c[0]["slice_vs_log_sec_per_position"] == 5.0
    assert "off by > 25%" in format_table(c)


def test_combine_leaves_an_incomplete_row_blank():
    rows = (Row("rl", "RL", "llm_rl", "r", 1),)       # only Stockfish (0%) is known: no overall number
    c = combine({"rl": {"init": 2.0, "selfplay": 1.0, "stockfish": 0.1, "update": 1.0, "positions": 10}}, {}, rows)
    assert c[0]["overall_util"] is None and not c[0]["complete"]
    assert "| ? |" in format_table(c)


def test_headline_rows_are_the_five_in_the_plan():
    assert [r.key for r in ROWS] == ["sft500", "selfplay_llm", "rl4", "az_scratch", "az_cnnsft"]


def test_alphazero_replay_runs_on_cpu():
    torch = pytest.importorskip("torch")
    from eval.gpu_util import _alphazero_phases
    from model.az_net import AZNet, NetConfig
    from search.mcts import MCTSConfig
    from train.az_train import AZConfig
    cfg = AZConfig(games_per_round=2, max_plies=12, buffer_positions=500, batch_size=8, samples_per_position=1.0)
    with GpuSampler(interval=0.01, read=lambda: 0.0) as s:
        out = _alphazero_phases(AZNet(NetConfig(blocks=1, channels=8)), cfg, MCTSConfig(simulations=3), s, torch.device("cpu"),
                                games=2, max_plies=12)
    assert out["selfplay"]["positions"] > 0 and out["update"]["steps"] >= 1
    assert out["selfplay"]["util"] == 0.0
