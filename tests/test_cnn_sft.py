"""Step 14b / Step 18 checks: train/cnn_sft.py (CNN SFT) and az_train's start-from-a-network.

What they prove:
- Labeled positions encode into AlphaZero's buffer format with a one-hot target on the label
  move (illegal labels dropped); training lowers the validation move loss on a learnable
  set, saves one AlphaZero-format checkpoint per pass, logs pass 0 (before training) and
  each pass, and a finished run resumes without redoing anything.
- A CNN SFT pass loads as an eval player both ways (kind "cnn_sft").
- az_train started from a network begins from its weights, counts its GPU-hours in the log
  and in the cap, refuses a mismatched network, and the existing AlphaZero run (no
  init_from in its run_config.json) still resumes.

    python -m pytest tests/test_cnn_sft.py -v
"""
import dataclasses
import json
import random

import chess
import pandas as pd
import pytest
import torch

from engine import board as B
from model.az_net import AZNet, NetConfig
from search.mcts import MCTSConfig
from train.az_train import AZConfig, load_checkpoint, list_checkpoints as az_checkpoints, train as train_az
from train.cnn_sft import (CNNSFTConfig, encode_labels, gpu_hours_at, list_checkpoints, run_name, train,
                           val_rows)

TINY_NET = NetConfig(blocks=1, channels=8)
TINY_MCTS = MCTSConfig(simulations=3, temperature_plies=6)


def labeled(n=300, seed=0):
    """Positions from random games, each labeled with a FIXED rule (the alphabetically first legal
    move; win prob = share of white pieces) -- learnable, so validation loss must fall."""
    rng, rows, b = random.Random(seed), [], chess.Board()
    while len(rows) < n:
        if B.is_terminal(b):
            b = chess.Board()
        moves = sorted(B.legal_moves(b))
        white = sum(p.color for p in b.piece_map().values()) / len(b.piece_map())
        rows.append({"fen": b.fen(), "best_move": moves[0], "win_prob": white if b.turn else 1 - white})
        b.push_uci(rng.choice(moves))
    return pd.DataFrame(rows)


def tiny_cfg(**kw):
    return CNNSFTConfig(**{**dict(epochs=2, batch_size=32, learning_rate=3e-3, log_every=3), **kw})


def test_encode_labels_one_hot_on_the_label_and_drops_illegal():
    df = labeled(20)
    bad = pd.DataFrame([{"fen": chess.STARTING_FEN, "best_move": "e2e5", "win_prob": 0.5}])
    planes, moves, probs, value = encode_labels(pd.concat([df, bad]))
    assert len(value) == 20 and planes.shape[1:] == (19, 8, 8)
    assert (probs.astype(float).sum(1) == 1).all() and (probs.astype(float).max(1) == 1).all()
    assert value[0] == pytest.approx(df.win_prob[0])


def test_train_learns_saves_each_pass_logs_and_resumes(tmp_path):
    train_data, val_data = encode_labels(labeled(400, seed=1)), encode_labels(labeled(100, seed=2))
    cfg = tiny_cfg()
    run = tmp_path / run_name(cfg, TINY_NET)
    s = train(cfg, TINY_NET, train_data, val_data, run, device="cpu", log=lambda *_: None)
    assert s["pass"] == 2 and s["passes"] == [1, 2] and s["steps_per_pass"] == 13          # ceil(400 / 32)
    v = {r["pass"]: r for r in val_rows(run)}
    assert sorted(v) == [0, 1, 2] and v[2]["val_move_loss"] < v[0]["val_move_loss"]
    assert 0 < gpu_hours_at(run, 1) <= gpu_hours_at(run, 2) and v[0]["train_gpu_hours"] == 0
    net = load_checkpoint(list_checkpoints(run)[0][1], "cpu")                               # AlphaZero format
    assert isinstance(net, AZNet) and net.cfg == TINY_NET
    s2 = train(cfg, TINY_NET, train_data, val_data, run, device="cpu", log=lambda *_: None)  # nothing redone
    assert s2["step"] == s["step"] and len(val_rows(run)) == 3
    with pytest.raises(ValueError):                                                         # other settings
        train(tiny_cfg(learning_rate=1e-2), TINY_NET, train_data, val_data, run, device="cpu", log=lambda *_: None)
    assert run_name(CNNSFTConfig(), NetConfig()) == "cnn_b6c64_n500000_e2_s0"


def test_resume_after_pass_one(tmp_path):
    train_data, val_data = encode_labels(labeled(200, seed=3)), encode_labels(labeled(60, seed=4))
    run = tmp_path / "r"
    train(tiny_cfg(epochs=1), TINY_NET, train_data, val_data, run, device="cpu", log=lambda *_: None)
    import shutil
    (run / "run_config.json").write_text(json.dumps({**json.loads((run / "run_config.json").read_text()),
                                                     "cnn_sft": dataclasses.asdict(tiny_cfg(epochs=2))}))
    s = train(tiny_cfg(epochs=2), TINY_NET, train_data, val_data, run, device="cpu", log=lambda *_: None)
    assert s["passes"] == [1, 2] and [r["pass"] for r in val_rows(run)] == [0, 1, 2]


def test_cnn_sft_pass_is_an_eval_player_both_ways(tmp_path):
    from eval.agents import MCTSAgent, NoSearchAgent, checkpoint_dir, load_player
    train_data, val_data = encode_labels(labeled(64)), encode_labels(labeled(32, seed=5))
    run_dir = tmp_path / "checkpoints" / "cnn_sft" / "cnn_t"
    train(tiny_cfg(epochs=1), TINY_NET, train_data, val_data, run_dir, device="cpu", log=lambda *_: None)
    spec = {"name": "cnn_t@p1", "kind": "cnn_sft", "run": "cnn_t", "step": 1}
    assert checkpoint_dir(spec, tmp_path) == run_dir / "checkpoints" / "pass_0001"
    a, b = load_player({**spec, "simulations": 0}, tmp_path, device="cpu"), load_player({**spec, "simulations": 4}, tmp_path, device="cpu")
    assert isinstance(a, NoSearchAgent) and isinstance(b, MCTSAgent)
    board = chess.Board()
    assert B.is_legal(board, a.select_move(board)) and B.is_legal(board, b.select_move(board))


# ------------------------------------------------- Step 18: AlphaZero from a network ----

def az_cfg(**kw):
    return AZConfig(**{**dict(games_per_round=2, max_plies=14, buffer_positions=500, batch_size=8,
                              samples_per_position=1.0, checkpoint_every=2), **kw})


def init_net(tmp_path, cfg=TINY_NET, seed=7):
    torch.manual_seed(seed)
    net = AZNet(cfg)
    d = tmp_path / "init"
    d.mkdir()
    torch.save({"net_config": dataclasses.asdict(cfg), "state_dict": net.state_dict()}, d / "net.pt")
    return d, net


def test_rl_starts_from_the_network_and_counts_its_gpu_hours(tmp_path, monkeypatch):
    import train.az_train as azt
    d, net = init_net(tmp_path)
    seen = {}
    real_play = azt.play_games

    def spy(evaluator, *a, **k):                       # the weights the first self-play round uses
        seen.setdefault("w", [p.detach().clone() for p in evaluator.net.parameters()])
        return real_play(evaluator, *a, **k)

    monkeypatch.setattr(azt, "play_games", spy)
    run = tmp_path / "run"
    cfg = az_cfg(init_from="cnn_sft/x@p1", tag="cnnsft_p1")
    s = train_az(cfg, TINY_NET, TINY_MCTS, run, device="cpu", max_rounds=1, init_dir=d, init_gpu_hours=5.0,
                 log=lambda *_: None)
    assert all(torch.equal(a, b) for a, b in zip(seen["w"], net.parameters()))
    assert s["train_gpu_hours"] >= 5.0 and s["gpu_seconds"] > 0 and json.loads((run / "run_config.json").read_text())["init_gpu_hours"] == 5.0
    s = train_az(cfg, TINY_NET, TINY_MCTS, run, device="cpu", max_gpu_hours=5.0, init_dir=d, init_gpu_hours=5.0,
                 log=lambda *_: None)                  # the cap counts the start's hours: no new round
    assert s["round"] == 1 and s["stopped_by_cap"]


def test_rl_refuses_bad_starts(tmp_path):
    d, _ = init_net(tmp_path, cfg=NetConfig(blocks=2, channels=8))
    with pytest.raises(ValueError):                    # init_from without its folder
        train_az(az_cfg(init_from="x"), TINY_NET, TINY_MCTS, tmp_path / "a", device="cpu", log=lambda *_: None)
    with pytest.raises(ValueError):                    # a folder without a label
        train_az(az_cfg(), TINY_NET, TINY_MCTS, tmp_path / "b", device="cpu", init_dir=d, log=lambda *_: None)
    with pytest.raises(ValueError):                    # a different network size
        train_az(az_cfg(init_from="x"), TINY_NET, TINY_MCTS, tmp_path / "c", device="cpu", init_dir=d,
                 log=lambda *_: None)


def test_existing_alphazero_run_config_still_resumes(tmp_path):
    run = tmp_path / "az"
    train_az(az_cfg(), TINY_NET, TINY_MCTS, run, device="cpu", max_rounds=1, log=lambda *_: None)
    saved = json.loads((run / "run_config.json").read_text())
    saved["train"].pop("init_from")                    # as written before Step 18
    saved.pop("init_gpu_hours")
    (run / "run_config.json").write_text(json.dumps(saved))
    s = train_az(az_cfg(), TINY_NET, TINY_MCTS, run, device="cpu", max_rounds=2, log=lambda *_: None)
    assert s["round"] == 2 and [r for r, _ in az_checkpoints(run)] == [2]
