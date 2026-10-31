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
