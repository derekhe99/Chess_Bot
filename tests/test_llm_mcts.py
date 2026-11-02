"""Step 17 checks for train/llm_mcts.py (the self-play LLM), on the tiny offline Qwen3 stand-in.

What they prove: self-play with search yields one buffer entry per searched move, with a
visit distribution over legal moves and the result for the mover; sampled examples use a
visited move and the right value label; an update changes the weights; a run stops at its
GPU-hour budget, resumes with its replay buffer, keeps every checkpoint_every-th round and
logs GPU-hours on top of the start's; other settings on resume are refused; the YAML
matches the config; a round checkpoint loads as an eval player (kind "llm_mcts").

    python -m pytest tests/test_llm_mcts.py -v
"""
import dataclasses
import json
import random

import chess
import pytest
import torch
import yaml

pytest.importorskip("transformers")
pytest.importorskip("peft")

from engine import board as B
from model.llm_evaluator import LLMEvaluator
from search.mcts import play_games
from tests.test_llm_policy import tiny_tokenizer
from tests.test_sft import make_policy
from train.llm_mcts import (MCTSTrainConfig, buffer_positions, gpu_hours_at, list_checkpoints, run_name,
                            sample_examples, train, update)


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


def small_cfg(**changes):
    base = dict(games_per_round=2, opening_plies=2, max_plies=12, simulations=2, buffer_rounds=2,
                samples_per_position=1.0, batch_size=4, micro_batch=2, learning_rate=1e-2, checkpoint_every=2,
                eval_batch_positions=8, budget_gpu_hours=100.0)
    return MCTSTrainConfig(**{**base, **changes})


def test_buffer_entries_and_sampled_examples(tok):
    policy = make_policy(tok)
    cfg = small_cfg()
    games = play_games(LLMEvaluator(policy, batch_positions=8), 2, cfg.mcts(), opening_plies=2, max_plies=12, seed=0)
    entries = buffer_positions(games)
    assert len(entries) == sum(len(g.moves) for g in games) > 0
    for e in entries:
        board = chess.Board(e["fen"])
        assert set(e["visits"]) <= set(B.legal_moves(board)) and abs(sum(e["visits"].values()) - 1) < 1e-6
        assert e["value"] in (0.0, 0.5, 1.0)
    first = games[0].moves[0]
    assert entries[0]["value"] == games[0].score_for(first.white)
    ex = sample_examples(policy, entries, 10, random.Random(0))
    assert len(ex) == 10
    for x in ex:
        assert x["label_value"] in (0.0, 0.5, 1.0) and x["labels"][x["value_index"]] == -100


def test_update_changes_weights(tok):
    policy = make_policy(tok)
    cfg = small_cfg()
    games = play_games(LLMEvaluator(policy, batch_positions=8), 2, cfg.mcts(), opening_plies=2, max_plies=8, seed=1)
    ex = sample_examples(policy, buffer_positions(games), 6, random.Random(0))
    before = [p.detach().clone() for p in policy.trainable_parameters()]
    opt = torch.optim.AdamW(policy.trainable_parameters(), lr=cfg.learning_rate)
    losses = update(policy, ex, cfg, opt, torch.amp.GradScaler("cuda", enabled=False), random.Random(0))
    assert losses["nll"] > 0 and losses["value_loss"] > 0
    assert any(not torch.equal(b, p) for b, p in zip(before, policy.trainable_parameters()))
    assert not policy.model.training                       # back to play mode


def test_train_runs_resumes_and_logs_gpu_hours(tok, tmp_path):
    cfg = small_cfg(rounds=3)
    run = tmp_path / run_name(cfg)
    s1 = train(make_policy(tok), cfg, run, init_gpu_hours=9.25, max_rounds=1, log=lambda *_: None)
    assert s1["round"] == 1
    s2 = train(make_policy(tok), cfg, run, init_gpu_hours=9.25, log=lambda *_: None)
    assert s2["round"] == 3
    rows = (run / "mcts_log.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 3
    hours = [gpu_hours_at(run, r) for r in (1, 2, 3)]
    assert hours[0] > 9.25 and hours == sorted(hours)
    assert [r for r, _ in list_checkpoints(run)] == [2, 3]
    assert sorted(p.name for p in (run / "buffer").iterdir()) == ["round_0002.json", "round_0003.json"]
    log = list(__import__("csv").DictReader(open(run / "mcts_log.csv")))
    assert int(log[2]["buffer_positions"]) == int(log[1]["positions"]) + int(log[2]["positions"])   # 2-round buffer


def test_budget_stops_the_run(tok, tmp_path):
    s = train(make_policy(tok), small_cfg(budget_gpu_hours=1e-9), tmp_path / "r", log=lambda *_: None)
    assert s["round"] == 1 and s["stopped_by_budget"]


def test_resume_with_other_settings_is_refused(tok, tmp_path):
    train(make_policy(tok), small_cfg(rounds=1), tmp_path, log=lambda *_: None)
    with pytest.raises(ValueError):
        train(make_policy(tok), small_cfg(rounds=1, learning_rate=5e-3), tmp_path, log=lambda *_: None)


def test_yaml_matches_the_config():
    cfg = MCTSTrainConfig.from_dict(yaml.safe_load(open("configs/llm_mcts.yaml")))
    assert cfg == MCTSTrainConfig()
    assert run_name(cfg) == "mcts16_structured_masked_nall_e2_s0_scaled_d12_n500000_from30000_s0"
    with pytest.raises(ValueError):
        MCTSTrainConfig.from_dict({"games_per_rnd": 3})


def test_round_checkpoint_is_an_eval_player_dir(tok, tmp_path):
    from eval.agents import checkpoint_dir
    cfg = small_cfg(rounds=1)
    run = tmp_path / "checkpoints" / "llm_mcts" / run_name(cfg)
    train(make_policy(tok), cfg, run, log=lambda *_: None)
    d = checkpoint_dir({"kind": "llm_mcts", "run": run_name(cfg), "step": 1}, tmp_path)
    assert d.is_dir() and (d / "value_head.pt").exists() and json.loads((d / "policy_config.json").read_text())["masked"]
