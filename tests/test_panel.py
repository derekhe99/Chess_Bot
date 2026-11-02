"""Step 12 (f) checks for eval/panel.py: one eval panel for every headline model.

What they prove:
- Saved games read back from games.csv refit to the same result and keep sec/move.
- extend() plays only the panel members a row's saved games lack, keeps the old games,
  carries the row's facts and old rating, resumes, and skips a finished row.
- calibrate_rungs() plays every rung against every existing member and the other rung,
  freezes them (a model row is promoted, keeping its facts), and is resumable.
- AlphaZero checkpoints work as reference members; panel_ratings keeps only comparable rows;
  the head-to-head table reports the gap the ratings predict.

    python -m pytest tests/test_panel.py -v
"""
import pytest

from eval.anchors import GreedyAgent, RandomAgent, ResultsTable, game_rows
from eval.elo import RatingEstimate, fit_with_ci
from eval.match import head_to_head_table, play_match, records_from_rows, summarize, to_results
from eval.panel import calibrate_rungs, coverage, extend, panel_ratings, reference_spec


def table_with_refs(path):
    t = ResultsTable(path)
    t.add_reference("random", 525.0, 450.0, 600.0, 100, spec={"kind": "random"})
    t.add_reference("greedy", 641.0, 580.0, 700.0, 100, spec={"kind": "greedy"})
    t.save()
    return ResultsTable(path)


def refs():
    return {"random": RandomAgent("random", seed=1), "greedy": GreedyAgent("greedy", seed=2)}


def test_records_round_trip_through_games_csv():
    games = play_match(RandomAgent("m", seed=3), GreedyAgent("greedy", seed=2), 4, max_plies=60)
    back = records_from_rows(game_rows(games, "eval", "m", "id"))
    assert to_results(back) == to_results(games)
    assert [g.termination for g in back] == [g.termination for g in games]
    for a, b in zip(games, back):
        for side in ("white", "black"):
            assert len(b.latency[side]) == len(a.latency[side])
            assert sum(b.latency[side]) == pytest.approx(sum(a.latency[side]), abs=1e-3 * max(len(a.latency[side]), 1))
    s1, s2 = summarize(games, "m"), summarize(back, "m")
    assert (s1["wins"], s1["draws"], s1["losses"]) == (s2["wins"], s2["draws"], s2["losses"])


def test_extend_plays_only_what_is_missing_and_keeps_old_games(tmp_path):
    path = tmp_path / "results.csv"
    t = table_with_refs(path)
    old = play_match(RandomAgent("m@1", seed=7), refs()["random"], 2, max_plies=60)   # rated vs random only
    est = fit_with_ci(to_results(old), {"random": 525.0}, n_boot=10)["m@1"]
    t.log_model(est, old, run="m", step=1, train_gpu_hours=2.5, note="first eval")
    t.save()
    load = lambda s, root: RandomAgent(s["name"], seed=7)
    e = extend({"name": "m@1", "kind": "sft", "run": "m", "step": 1}, ResultsTable(path), tmp_path, refs(),
               games_per_opponent=2, batch_games=2, n_boot=10, gpu="test", load=load, verbose=False)
    t2 = ResultsTable(path)
    row, games = t2.rows["m@1"], t2.games("m@1")
    assert e is not None and row["games"] == "4" and len(games) == 4
    assert sum(g["white"] == "greedy" or g["black"] == "greedy" for g in games) == 2           # only greedy was played
    kept = [g for g in games if "random" in (g["white"], g["black"])]
    assert [g["moves"] for g in kept] == [" ".join(r.moves) for r in old]                      # the old games, unchanged
    assert int(row["wins"]) + int(row["draws"]) + int(row["losses"]) == 4
    assert (row["run"], row["step"], row["train_gpu_hours"], row["gpu"]) == ("m", "1", "2.5", "test")
    assert row["note"].startswith("first eval; rated on the panel (was ")
    assert coverage(t2, ["random", "greedy"])["m@1"] == {"random": 2, "greedy": 2}
    assert extend({"name": "m@1", "kind": "sft", "run": "m", "step": 1}, t2, tmp_path, refs(),     # nothing missing
                  games_per_opponent=2, load=load, verbose=False) is None
    e3 = extend({"name": "m@1", "kind": "sft", "run": "m", "step": 1}, ResultsTable(path), tmp_path, refs(),
                games_per_opponent=4, batch_games=2, n_boot=10, load=load, verbose=False)       # more games: resumes
    t3 = ResultsTable(path)
    assert e3.games == 8 and len(t3.games("m@1")) == 8 and t3.rows["m@1"]["note"].count("rated on the panel") == 1


def test_extend_new_row_and_refusals(tmp_path):
    path = tmp_path / "results.csv"
    t = table_with_refs(path)
    load = lambda s, root: RandomAgent(s["name"], seed=5)
    extend({"name": "new@1", "kind": "sft", "run": "new", "step": 1}, t, tmp_path, refs(), games_per_opponent=2,
           batch_games=2, n_boot=10, fields={"run": "new", "note": "control"}, load=load, verbose=False)
    row = ResultsTable(path).rows["new@1"]
    assert row["role"] == "model" and row["games"] == "4" and row["note"] == "control"
    with pytest.raises(ValueError):                                       # a reference isn't rated on the panel
        extend({"name": "greedy"}, ResultsTable(path), tmp_path, refs(), load=load)
    with pytest.raises(ValueError):                                       # every opponent must be frozen
        extend({"name": "x"}, ResultsTable(path), tmp_path, {**refs(), "rung": RandomAgent("rung")}, load=load)


def test_calibrate_rungs_freezes_them_promotes_a_model_row_and_resumes(tmp_path):
    path = tmp_path / "results.csv"
    t = table_with_refs(path)
    t.log_model(RatingEstimate("rungA@9", 700.0, 650.0, 750.0, 60, 0.5, None),                 # an old model row
                run="alphazero/a", step=9, train_gpu_hours=18.0, sec_per_move=0.099, gpu="NVIDIA L4")
    t.save()
    rungs = [{"name": "rungA@9", "kind": "alphazero", "run": "a", "step": 9, "simulations": 50},
             {"name": "rungB@5", "kind": "sft", "run": "b", "step": 5, "fields": {"run": "b", "dataset_size": 500000}}]
    seeds = {"rungA@9": 11, "rungB@5": 12}
    load = lambda s, root: RandomAgent(s["name"], seed=seeds[s["name"]])
    # interrupted: pretend one pairing was already half played and saved
    part = play_match(RandomAgent("rungA@9", seed=11), refs()["random"], 2, seed=0)
    t2 = ResultsTable(path)
    t2.log_games(part, "calib", "calib|rungA@9|random")
    t2.save()
    est = calibrate_rungs(rungs, ResultsTable(path), tmp_path, refs(), games_per_pair=4, batch_games=2, n_boot=10,
                          load=load, verbose=False)
    t3 = ResultsTable(path)
    assert set(est) >= {"rungA@9", "rungB@5"}
    for name in ("rungA@9", "rungB@5"):
        assert t3.role(name) == "reference" and name in t3.fixed()
    calib = t3.games(kind="calib")
    assert len(calib) == 5 * 4                                             # 2 rungs x 2 members + the rung pair
    assert [g["moves"] for g in t3.games("calib|rungA@9|random", kind="calib")][:2] == [" ".join(r.moves) for r in part]
    a, b = t3.rows["rungA@9"], t3.rows["rungB@5"]
    assert (a["run"], a["train_gpu_hours"], a["sec_per_move"], a["gpu"]) == ("alphazero/a", "18.0", "0.099", "NVIDIA L4")
    assert "was model 700.0 [650.0, 750.0], 60 games" in a["note"] and b["dataset_size"] == "500000"
    assert t3.spec("rungA@9") == {"kind": "alphazero_checkpoint", "run": "a", "step": 9, "simulations": 50}
    assert t3.spec("rungB@5") == {"kind": "sft_checkpoint", "run": "b", "step": 5}
    assert calibrate_rungs(rungs, t3, tmp_path, refs(), load=load, verbose=False) == {}     # already frozen
    with pytest.raises(ValueError):
        reference_spec({"kind": "llm_rl", "run": "r", "step": 1})


def test_alphazero_rung_loads_as_a_reference_agent(tmp_path):
    import dataclasses as dc
    import chess
    import torch
    from engine import board as B
    from eval.agents import MCTSAgent, checkpoint_dir
    from eval.anchors import build_reference_agents
    from model.az_net import AZNet, NetConfig
    d = checkpoint_dir({"kind": "alphazero", "run": "az_t", "step": 3}, tmp_path)
    d.mkdir(parents=True)
    net_cfg = NetConfig(blocks=1, channels=8)
    torch.save({"net_config": dc.asdict(net_cfg), "state_dict": AZNet(net_cfg).state_dict()}, d / "net.pt")
    t = ResultsTable(tmp_path / "results.csv")
    t.add_reference("az_t@r3", 790.0, 750.0, 830.0, 200,
                    spec={"kind": "alphazero_checkpoint", "run": "az_t", "step": 3, "simulations": 4})
    agents = build_reference_agents(t, ["az_t@r3"], tmp_path, device="cpu")
    agent = agents["az_t@r3"]
    assert isinstance(agent, MCTSAgent) and agent.name == "az_t@r3" and agent.cfg.simulations == 4
    assert B.is_legal(chess.Board(), agent.select_move(chess.Board()))


def test_panel_ratings_keep_only_comparable_rows(tmp_path):
    path = tmp_path / "results.csv"
    t = table_with_refs(path)
    full = play_match(RandomAgent("on@1", seed=1), refs()["random"], 2) + play_match(RandomAgent("on@1", seed=1), refs()["greedy"], 2)
    t.log_model(fit_with_ci(to_results(full), {"random": 525.0, "greedy": 641.0}, n_boot=10)["on@1"], full)
    half = play_match(RandomAgent("off@1", seed=1), refs()["random"], 2)
    t.log_model(fit_with_ci(to_results(half), {"random": 525.0}, n_boot=10)["off@1"], half)
    t.log_model(RatingEstimate("old@1", 700.0, 650.0, 750.0, 60, 0.5, None))                  # no saved games
    t.save()
    r = panel_ratings(ResultsTable(path), ["random", "greedy"], games_per_opponent=2)
    assert set(r) == {"random", "greedy", "on@1"}


def test_head_to_head_table_reports_the_predicted_gap():
    games = [{"white": "a", "black": "b", "white_score": s, "termination": "checkmate" if s != 0.5 else "threefold_repetition"}
             for s in (1.0, 0.5, 0.5, 0.0, 1.0, 0.5)]
    plain = head_to_head_table(games, n_boot=200)[0]
    assert "predicted_diff" not in plain
    row = head_to_head_table(games, n_boot=200, ratings={"a": 800.0, "b": 700.0})[0]
    assert row["predicted_diff"] == 100.0 and row["predicted_score"] == pytest.approx(0.64, abs=0.01)
    assert row["consistent"] == (row["ci_low"] <= 100.0 <= row["ci_high"])
    far = head_to_head_table(games, n_boot=200, ratings={"a": 2000.0, "b": 700.0})[0]
    assert far["consistent"] is False
    missing = head_to_head_table(games, n_boot=200, ratings={"a": 800.0})[0]
    assert missing["predicted_diff"] is None and missing["consistent"] is None


def test_plots_put_a_checkpoint_rung_on_its_run(tmp_path):
    import csv
    from eval.anchors import RESULTS_COLUMNS
    from eval.plots import elo_points
    rows = [dict(name="sf_1320", role="pin", rating=1320, ci_low=1320, ci_high=1320),
            dict(name="az@r53", role="model", rating=729, ci_low=696, ci_high=762, run="alphazero/az", step=53,
                 train_gpu_hours=9.2),
            dict(name="az@r109", role="reference", rating=790, ci_low=760, ci_high=820, run="alphazero/az", step=109,
                 train_gpu_hours=18.0, spec='{"kind": "alphazero_checkpoint", "run": "az", "step": 109}')]
    path = tmp_path / "results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULTS_COLUMNS})
    runs, _, lines = elo_points(path)
    pts = runs["alphazero/az"]
    assert [p["name"] for p in pts] == ["az@r53", "az@r109"] and pts[1]["hollow"] and pts[1]["x"] == 18.0
    assert [l["name"] for l in lines] == ["sf_1320"]
