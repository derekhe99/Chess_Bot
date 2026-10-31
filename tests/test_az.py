"""Checks for Step 9 AlphaZero pieces: (d) the network and its MCTS adapter.

What they prove, on an untrained (random) network:
- Shapes and size are as configured; the yaml's net block loads.
- The adapter returns priors over exactly the legal moves (summing to 1) and a value in
  [0, 1], for White and Black to move.
- Flipping is consistent end to end: a position and its color-mirrored twin get the same
  value and the same priors, move for mirrored move.
- Batching doesn't change the answer, and the search runs on the real network.

    python -m pytest tests/test_az.py -v
"""
import random

import chess
import pytest
import torch
import yaml

from engine import board as B
from engine.moves import NUM_MOVES, mirror_move
from model.az_net import AZEvaluator, AZNet, NetConfig, encode_batch, masked_policy
from search.mcts import MCTSConfig, play_games, search_move

FENS = [
    chess.STARTING_FEN,
    "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R b KQkq - 4 8",
    "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
    "8/8/8/8/8/2k5/1p4K1/8 b - - 0 60",
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
]


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(0)
    net = AZNet(NetConfig(blocks=2, channels=16))
    with torch.no_grad():                  # make the random net's outputs less uniform, so tests can see mistakes
        for p in net.parameters():
            p.mul_(3.0)
    return AZEvaluator(net, "cpu")


def test_forward_shapes_and_default_size():
    net = AZNet()
    logits, value = net(torch.zeros(5, 19, 8, 8))
    assert logits.shape == (5, NUM_MOVES) and value.shape == (5,)
    n = sum(p.numel() for p in net.parameters())
    assert 0.5e6 < n < 0.8e6               # 6 x 64: ~0.71M (trunk ~0.44M, policy head ~0.25M)


def test_yaml_net_block_loads():
    with open("configs/alphazero.yaml") as f:
        cfg = NetConfig.from_dict(yaml.safe_load(f)["net"])
    assert cfg == NetConfig(blocks=6, channels=64)
    with pytest.raises(ValueError):
        NetConfig.from_dict({"blokcs": 6})


def test_masked_policy_gives_illegal_moves_zero():
    logits = torch.randn(2, NUM_MOVES) * 10
    _, masks = encode_batch([chess.Board(FENS[0]), chess.Board(FENS[1])])
    mask = torch.from_numpy(masks)
    probs = masked_policy(logits, mask)
    assert torch.allclose(probs.sum(-1), torch.ones(2))
    assert (probs[~mask] == 0).all()


@pytest.mark.parametrize("fen", FENS)
def test_evaluate_priors_cover_exactly_the_legal_moves(evaluator, fen):
    b = chess.Board(fen)
    (priors, value), = evaluator.evaluate([b])
    assert set(priors) == set(B.legal_moves(b))
    assert abs(sum(priors.values()) - 1) < 1e-5 and all(p >= 0 for p in priors.values())
    assert 0 <= value <= 1


@pytest.mark.parametrize("fen", FENS)
def test_a_position_and_its_mirror_image_get_the_same_answer(evaluator, fen):
    # board.mirror() swaps colors and flips ranks: the same position for the other side.
    # The side to move sees identical planes, so the net must give the same value and
    # the same prior to each move and its mirror -- this checks the flip-back mapping.
    b = chess.Board(fen)
    (p1, v1), (p2, v2) = evaluator.evaluate([b, b.mirror()])
    assert v1 == pytest.approx(v2, abs=1e-6)
    assert {mirror_move(m): p for m, p in p1.items()} == pytest.approx(p2, abs=1e-6)
    assert max(p1.values()) > 2 / len(p1)  # the net isn't uniform, so a wrong mapping would show


def test_batching_does_not_change_the_answer(evaluator):
    boards = [chess.Board(f) for f in FENS]
    batched = evaluator.evaluate(boards)
    for b, (priors, value) in zip(boards, batched):
        (p1, v1), = evaluator.evaluate([b])
        assert value == pytest.approx(v1, abs=1e-5) and priors == pytest.approx(p1, abs=1e-5)


def test_search_and_batched_self_play_run_on_the_network(evaluator):
    b = chess.Board(FENS[1])
    assert search_move(b, evaluator, MCTSConfig(simulations=16)) in B.legal_moves(b)
    games = play_games(evaluator, 2, MCTSConfig(simulations=4), max_plies=20, seed=0)
    assert len(games) == 2 and all(g.plies <= 20 for g in games)


def test_evaluate_on_random_positions_black_and_white(evaluator):
    rng = random.Random(1)
    boards = []
    b = chess.Board()
    while len(boards) < 40:
        if B.is_terminal(b):
            b = chess.Board()
        boards.append(b.copy())
        b.push_uci(rng.choice(B.legal_moves(b)))
    for b, (priors, value) in zip(boards, evaluator.evaluate(boards)):
        assert set(priors) == set(B.legal_moves(b)) and 0 <= value <= 1


# ------------------------------------------------- (e) training loop ----
import json

import numpy as np

from engine.encoding import to_planes_stm
from engine.moves import stm_legal_mask
from train.az_train import (AZConfig, ReplayBuffer, batch_losses, decode_planes, encode_position, game_examples,
                            gpu_hours_at, list_checkpoints, load_checkpoint, load_configs, run_name, train, update,
                            update_steps)

TINY_NET = NetConfig(blocks=1, channels=8)
TINY_MCTS = MCTSConfig(simulations=3, temperature_plies=6)


def tiny_cfg(**kw):
    base = dict(games_per_round=2, max_plies=14, buffer_positions=500, batch_size=8, samples_per_position=1.0,
                checkpoint_every=2)
    return AZConfig(**{**base, **kw})


def test_yaml_loads_every_section_and_rejects_typos(tmp_path):
    cfg, net, mcts, cap = load_configs("configs/alphazero.yaml")
    assert (cfg.games_per_round, mcts.simulations, net, cap) == (128, 50, NetConfig(6, 64), 9.0)   # confirmed Oct 31
    assert run_name(cfg, net) == "az_b6c64_s0" and run_name(tiny_cfg(tag="x1"), TINY_NET) == "az_b1c8_s0_x1"
    bad = tmp_path / "bad.yaml"
    bad.write_text("train:\n  gams_per_round: 3\n")
    with pytest.raises(ValueError):
        load_configs(bad)


def test_buffer_encoding_is_lossless():
    rng = random.Random(4)
    b = chess.Board()
    for _ in range(80):
        if B.is_terminal(b):
            b = chess.Board()
        b.halfmove_clock = rng.choice([b.halfmove_clock, 37, 99, 120])   # exercise the 50-move plane
        legal = B.legal_moves(b)
        visits = {m: 1 / len(legal) for m in legal}
        planes, moves, probs = encode_position(b, visits)
        assert (decode_planes(planes[None])[0] == to_planes_stm(b)).all()
        mask = np.zeros_like(stm_legal_mask(b))
        mask[moves[moves >= 0]] = True
        assert (mask == stm_legal_mask(b)).all()                         # the stored moves = the legal mask
        assert abs(float(probs.astype(np.float32).sum()) - 1) < 1e-2
        b.push_uci(rng.choice(legal))


def test_replay_buffer_keeps_the_newest_and_round_trips(tmp_path):
    buf = ReplayBuffer(5)

    def rows(vals):
        n = len(vals)
        return (np.zeros((n, 19, 8, 8), np.uint8), np.zeros((n, 256), np.int16),
                np.zeros((n, 256), np.float16), np.array(vals, np.float32))

    buf.add(*rows([1, 2, 3]))
    buf.add(*rows([4, 5, 6, 7]))
    assert buf.size == 5 and sorted(buf.value.tolist()) == [3, 4, 5, 6, 7]    # 1 and 2 were the oldest
    buf.save(tmp_path / "b.npz")
    back = ReplayBuffer.load(tmp_path / "b.npz", 5)
    assert (back.value == buf.value).all() and (back.next, back.size) == (buf.next, buf.size)
    buf.add(*rows(list(range(10, 20))))                                       # more rows than capacity
    assert sorted(buf.value.tolist()) == [15, 16, 17, 18, 19]


def test_update_learns_a_fixed_buffer():
    torch.manual_seed(0)
    net = AZNet(TINY_NET)
    games = play_games(AZEvaluator(net, "cpu"), 2, TINY_MCTS, max_plies=20, seed=1)
    buf = ReplayBuffer(1000)
    buf.add(*game_examples(games))
    cfg = tiny_cfg(batch_size=16, learning_rate=1e-2)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    first = update(net, buf, 1, cfg, opt, np.random.default_rng(0), torch.device("cpu"))
    for _ in range(30):
        last = update(net, buf, 1, cfg, opt, np.random.default_rng(0), torch.device("cpu"))
    assert last["policy_loss"] < first["policy_loss"] and last["value_loss"] < first["value_loss"]
    pl, vl, ent = batch_losses(net, *buf.sample(8, np.random.default_rng(1)), torch.device("cpu"))
    assert torch.isfinite(pl).all() and torch.isfinite(vl).all() and (ent >= 0).all()
    assert update_steps(1000, AZConfig()) == 8                                # ceil(1000 * 2 / 256)


def test_train_saves_logs_prunes_and_resumes_exactly(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    s = train(tiny_cfg(), TINY_NET, TINY_MCTS, a, device="cpu", max_rounds=4, log=lambda *_: None)
    assert s["round"] == 4 and s["kept_checkpoints"] == [2, 4]                # round_1, round_3 pruned
    assert not (a / "checkpoints" / "round_0002" / "resume.pt").exists()      # kept: net only
    assert (a / "checkpoints" / "round_0004" / "buffer.npz").exists()         # latest: full resume state
    log = list(csv_rows(a / "az_log.csv"))
    assert [int(r["round"]) for r in log] == [1, 2, 3, 4]
    assert gpu_hours_at(a, 4) == float(log[-1]["train_gpu_hours"]) > 0

    train(tiny_cfg(), TINY_NET, TINY_MCTS, b, device="cpu", max_rounds=2, log=lambda *_: None)
    train(tiny_cfg(), TINY_NET, TINY_MCTS, b, device="cpu", max_rounds=4, log=lambda *_: None)   # resume
    assert [r["white_wins"] + r["draws"] for r in csv_rows(b / "az_log.csv")] == \
           [r["white_wins"] + r["draws"] for r in log]
    na, nb = load_checkpoint(a / "checkpoints" / "round_0004", "cpu"), load_checkpoint(b / "checkpoints" / "round_0004", "cpu")
    for (k, x), (_, y) in zip(na.state_dict().items(), nb.state_dict().items()):
        assert torch.allclose(x.float(), y.float()), k                       # a resume replays exactly

    with pytest.raises(ValueError):
        train(tiny_cfg(learning_rate=5e-4), TINY_NET, TINY_MCTS, b, device="cpu", max_rounds=5, log=lambda *_: None)


def test_cap_stops_keeps_the_round_and_resumes_when_raised(tmp_path):
    run = tmp_path / "run"
    cfg = tiny_cfg(checkpoint_every=10)
    s = train(cfg, TINY_NET, TINY_MCTS, run, device="cpu", max_gpu_hours=1e-9, log=lambda *_: None)
    assert s["stopped_by_cap"] and s["round"] == 1                           # stops before round 2
    assert json.loads((run / "kept_rounds.json").read_text()) == [1]
    s = train(cfg, TINY_NET, TINY_MCTS, run, device="cpu", max_gpu_hours=None, max_rounds=3, log=lambda *_: None)
    assert s["round"] == 3 and s["kept_checkpoints"] == [1, 3]                # round 1 survives pruning
    assert (run / "checkpoints" / "round_0001" / "net.pt").exists()
    assert not (run / "checkpoints" / "round_0001" / "buffer.npz").exists()


def csv_rows(path):
    import csv
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            yield {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in r.items()}


# ------------------------------------------------------- (f) eval ----
from eval.anchors import GreedyAgent, RandomAgent, ResultsTable
from eval.match import evaluate_agent, play_game
from eval.plots import elo_points
from train.az_train import AlphaZeroAgent, eval_rounds, load_agent


def fake_run(tmp_path, rounds, cap_rounds=()):
    """A run folder with round_NNNN/ folders for ``rounds`` and the cap's kept_rounds.json."""
    run = tmp_path / "run"
    for r in rounds:
        (run / "checkpoints" / f"round_{r:04d}").mkdir(parents=True)
    if cap_rounds:
        (run / "kept_rounds.json").write_text(json.dumps(sorted(cap_rounds)))
    return run


def test_eval_rounds_picks_up_to_four_per_stage_including_each_stages_end(tmp_path):
    # stage 1 stopped by the cap at round 23, stage 2 at 41; round 47 is the stray latest round
    run = fake_run(tmp_path, [5, 10, 15, 20, 23, 25, 30, 35, 40, 41, 45, 47], cap_rounds=[23, 41])
    assert eval_rounds(run, 5) == [5, 10, 20, 23, 25, 30, 40, 41, 45]
    # no cap stop yet: the regular kept checkpoints, evenly spaced, not the stray latest round
    run2 = fake_run(tmp_path / "x", [5, 10, 15, 20, 25, 30, 33])
    assert eval_rounds(run2, 5) == [5, 15, 20, 30]


def test_agent_plays_legal_games_without_noise(evaluator):
    agent = AlphaZeroAgent(evaluator.net, MCTSConfig(simulations=8), "az_test", "cpu")
    rec = play_game(agent, RandomAgent(seed=1), max_plies=30)
    assert rec.termination != "illegal_move" and sum(rec.illegal_attempts.values()) == 0
    b = chess.Board(FENS[1])
    assert agent.select_move(b) == agent.select_move(b)                       # eval search is deterministic


def test_checkpoint_rated_into_results_csv_and_plotted(tmp_path):
    run_dir = tmp_path / "checkpoints" / "alphazero" / "az_b1c8_s0"
    cfg = tiny_cfg()
    train(cfg, TINY_NET, TINY_MCTS, run_dir, device="cpu", max_rounds=2, log=lambda *_: None)
    rnd = eval_rounds(run_dir, cfg.checkpoint_every)[-1]
    name = f"az_b1c8_s0@r{rnd}"
    agent = load_agent(run_dir / "checkpoints" / f"round_{rnd:04d}", TINY_MCTS, name, "cpu")
    results = ResultsTable(tmp_path / "results.csv")

    def save(est, recs):
        results.log_model(est, recs, run="alphazero/az_b1c8_s0", step=rnd, seed=cfg.seed,
                          train_gpu_hours=gpu_hours_at(run_dir, rnd), label_cpu_hours=0.0, gpu="cpu",
                          note=f"pure self-play AlphaZero, MCTS {TINY_MCTS.simulations} sims/move")
        results.save()

    evaluate_agent(agent, {"random": RandomAgent(seed=2), "greedy": GreedyAgent(seed=3)},
                   {"random": 525.0, "greedy": 641.0}, n_games=4, batch_games=2, on_batch=save,
                   n_boot=20, verbose=False, max_plies=30)
    row = ResultsTable(tmp_path / "results.csv").rows[name]
    assert row["role"] == "model" and row["games"] == "8" and float(row["sec_per_move"]) > 0
    assert row["run"] == "alphazero/az_b1c8_s0" and float(row["label_cpu_hours"]) == 0.0
    runs, _, _ = elo_points(tmp_path / "results.csv")
    assert [p["label"] for p in runs["alphazero/az_b1c8_s0"]] == [f"r{rnd}"]
