"""Step 7 checks for search/selfplay.py and train/llm_rl.py, on the tiny offline Qwen3 stand-in.

What they prove: self-play games are legal, replayable, and scored from the
right side; the sampling picker only plays legal moves and reports value and
entropy; advantages are score minus prediction; the KL to the reference is
exactly 0 before the first update and the update changes the weights; a run
stopped after one round resumes and continues; GPU-hours on the curve include
the warm start's.

    python -m pytest tests/test_llm_rl.py -v
"""
import json
import random

import chess
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from engine import board as B
from eval.anchors import RandomAgent
from search.selfplay import SelfPlayGame, play_selfplay_game, play_selfplay_games
from tests.test_sft import make_policy
from tests.test_llm_policy import tiny_tokenizer
from train.llm_rl import (RLConfig, build_examples, ending_counts, gpu_hours_at, list_checkpoints, run_name,
                          sampling_picker, score_endings, train, update)

try:
    from engine.stockfish import find_stockfish
    find_stockfish()
    HAVE_SF = True
except (FileNotFoundError, ImportError):
    HAVE_SF = False
needs_sf = pytest.mark.skipif(not HAVE_SF, reason="Stockfish binary not found (run the setup notebook)")


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


def random_picker(seed=0):
    agent = RandomAgent(seed=seed)
    return lambda board: (agent.select_move(board), {"value": 0.5, "entropy": 1.0})


def small_cfg(**changes):
    base = dict(init_run=None, rounds=2, games_per_round=2, opening_plies=2, max_plies=16, step_positions=8,
                micro_batch=4, learning_rate=1e-2, checkpoint_every=2, seed=0, end_scoring="result", tag="")
    return RLConfig(**{**base, **changes})


# ------------------------------------------------------------ self-play ----

def test_selfplay_games_are_legal_replayable_and_scored_per_side():
    games = play_selfplay_games(random_picker(), 6, opening_plies=4, max_plies=200, seed=3)
    for g in games:
        board = B.new_board()
        for mv in g.opening:
            B.apply_move(board, mv)
        for r in g.moves:                        # recorded positions match a replay
            assert r.fen == board.fen() and r.white == (board.turn == chess.WHITE)
            B.apply_move(board, r.move)
        assert board.ply() == g.plies == len(g.opening) + len(g.moves)
        assert g.score_for(True) + g.score_for(False) == 1.0
        if g.termination != "max_plies":
            assert (B.outcome(board).white_value + 1) / 2 == g.white_score


def test_selfplay_ply_cap_is_a_draw_and_openings_are_seeded():
    g = play_selfplay_game(random_picker(), max_plies=10)
    assert g.termination == "max_plies" and g.white_score == 0.5 and g.plies == 10
    a = play_selfplay_games(random_picker(), 3, opening_plies=4, max_plies=10, seed=1)
    b = play_selfplay_games(random_picker(), 3, opening_plies=4, max_plies=10, seed=1)
    assert [g.opening for g in a] == [g.opening for g in b]
    assert len({tuple(g.opening) for g in a}) > 1


def test_selfplay_records_the_final_position():
    for g in play_selfplay_games(random_picker(), 3, opening_plies=2, max_plies=30, seed=2):
        board = B.new_board()
        for mv in g.opening + [r.move for r in g.moves]:
            B.apply_move(board, mv)
        assert g.final_fen == board.fen()


def test_selfplay_refuses_an_illegal_move():
    with pytest.raises(ValueError):
        play_selfplay_game(lambda board: ("a1a1", {}), max_plies=10)


# ---------------------------------------------------------------- RL ----

def test_sampling_picker_is_legal_and_reports_value_and_entropy(tok):
    policy = make_policy(tok)
    pick = sampling_picker(policy, 1.0, random.Random(0))
    board = chess.Board()
    for _ in range(12):
        move, info = pick(board.copy())
        assert B.is_legal(board, move)
        assert info["value"] == pytest.approx(0.5)   # zero-initialized value head
        assert 0 < info["entropy"] <= torch.log(torch.tensor(float(board.legal_moves.count()))).item() + 1e-6
        board.push_uci(move)


def test_examples_carry_score_and_advantage(tok):
    policy = make_policy(tok)
    games = play_selfplay_games(random_picker(), 3, opening_plies=2, max_plies=12, seed=0)
    games[0].white_score = 1.0                   # pretend White won game 0
    ex = build_examples(policy, games)
    assert len(ex) == sum(len(g.moves) for g in games)
    first = games[0].moves
    for e, r in zip(ex, first):
        assert e["label_value"] == (1.0 if r.white else 0.0)
        assert e["advantage"] == pytest.approx(e["label_value"] - 0.5)


def test_kl_is_zero_before_the_first_step_and_update_changes_weights(tok):
    policy, ref = make_policy(tok), make_policy(tok)
    games = play_selfplay_games(sampling_picker(policy, 1.0, random.Random(0)), 2, opening_plies=2,
                                max_plies=8, seed=0)
    ex = build_examples(policy, games)
    cfg = small_cfg(step_positions=len(ex))      # one optimizer step: KL measured before it
    before = [p.detach().clone() for p in policy.trainable_parameters()]
    opt = torch.optim.AdamW(policy.trainable_parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    losses = update(policy, ref, ex, cfg, opt, scaler, random.Random(0))
    assert losses["kl"] == pytest.approx(0.0, abs=1e-6)
    assert any(not torch.equal(b, p) for b, p in zip(before, policy.trainable_parameters()))
    # the reference never moves
    ref2 = make_policy(tok)
    assert all(torch.equal(a, b) for a, b in zip(ref.model.state_dict().values(), ref2.model.state_dict().values()))


def test_train_runs_resumes_and_logs_gpu_hours(tok, tmp_path):
    cfg = small_cfg(rounds=3)
    run = tmp_path / run_name(make_policy(tok).config, cfg)
    s1 = train(make_policy(tok), make_policy(tok), cfg, run, init_gpu_hours=7.69, max_rounds=1, log=lambda *_: None)
    assert s1["round"] == 1 and not s1["finished"]
    s2 = train(make_policy(tok), make_policy(tok), cfg, run, init_gpu_hours=7.69, log=lambda *_: None)
    assert s2["round"] == 3 and s2["finished"]
    rows = (run / "rl_log.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 3                    # header + one row per round
    hours = [gpu_hours_at(run, r) for r in (1, 2, 3)]
    assert hours[0] > 7.69 and hours == sorted(hours)
    # kept: every checkpoint_every-th round, plus the latest (with resume state)
    kept = list_checkpoints(run)
    assert [r for r, _ in kept] == [2, 3]
    assert not (kept[0][1] / "resume.pt").exists() and (kept[1][1] / "resume.pt").exists()
    assert json.loads((kept[0][1] / "policy_config.json").read_text())["masked"] is True


def test_resume_with_other_settings_is_refused(tok, tmp_path):
    train(make_policy(tok), make_policy(tok), small_cfg(rounds=1), tmp_path, log=lambda *_: None)
    with pytest.raises(ValueError):
        train(make_policy(tok), make_policy(tok), small_cfg(rounds=1, kl_coef=0.5), tmp_path, log=lambda *_: None)


# ------------------------------------------------- end scoring (run 2) ----

def _game(fen, termination, white_score):
    return SelfPlayGame([], [], white_score, termination, 0, fen)


def test_score_endings_rescores_only_non_decisive_endings():
    mate = chess.Board()
    for mv in ["f2f3", "e7e5", "g2g4", "d8h4"]:     # fool's mate: White is mated
        mate.push_uci(mv)
    stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    black_to_move = chess.Board()
    black_to_move.push_uci("e2e4")
    games = [_game(mate.fen(), "checkmate", 0.0), _game(stalemate, "stalemate", 0.5),
             _game(chess.STARTING_FEN, "max_plies", 0.5),          # White to move
             _game(black_to_move.fen(), "threefold_repetition", 0.5),
             _game(chess.STARTING_FEN, "fifty_moves", 0.5)]
    n = score_endings(games, lambda board: 0.8)           # side to move expects 0.8
    assert n == 3
    assert [g.white_score for g in games] == pytest.approx([0.0, 0.5, 0.8, 0.2, 0.8])
    counts = ending_counts(games)
    assert counts == {"ends_checkmate": 1, "ends_repetition": 1, "ends_fifty_moves": 1, "ends_stalemate": 1,
                      "ends_insufficient_material": 0}


def test_run_name_tag_and_settings_are_checked(tok):
    pcfg = make_policy(tok).config
    assert run_name(pcfg, small_cfg()).endswith("_s0")
    assert run_name(pcfg, small_cfg(tag="sfend")) == run_name(pcfg, small_cfg()) + "_sfend"
    with pytest.raises(ValueError):
        small_cfg(tag="a/b")
    with pytest.raises(ValueError):
        small_cfg(end_scoring="material")


@needs_sf
def test_train_with_stockfish_end_scoring_logs_the_endings(tok, tmp_path):
    import csv
    cfg = small_cfg(rounds=1, end_scoring="stockfish", end_scoring_nodes=2000, tag="sfend")
    train(make_policy(tok), make_policy(tok), cfg, tmp_path, log=lambda *_: None)
    row = next(csv.DictReader(open(tmp_path / "rl_log.csv", newline="")))
    non_decisive = int(row["max_plies_draws"]) + int(row["ends_repetition"]) + int(row["ends_fifty_moves"])
    assert int(row["rescored_games"]) == non_decisive > 0     # 16-ply games mostly hit the cap
    assert float(row["stockfish_seconds"]) >= 0


def test_llm_rl_yaml_matches_the_config():
    import yaml
    cfg = RLConfig.from_dict(yaml.safe_load(open("configs/llm_rl.yaml")))
    assert cfg == RLConfig()
