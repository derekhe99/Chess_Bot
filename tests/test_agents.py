"""Step 12 (c) checks for eval/agents.py: the search switch for any network.

What they prove:
- Both agents play only legal moves and are deterministic.
- MCTSAgent finds a mate-in-1 a no-search agent misses (search adds something real).
- With an AlphaZero network they play exactly the moves of az_train's PolicyAgent /
  AlphaZeroAgent, so stage 1-2's ratings need no re-run.
- make_agent switches on the simulation count; agent_name gives the results.csv names.

    python -m pytest tests/test_agents.py -v
"""
import random

import chess
import pytest
import torch

from engine import board as B
from eval.agents import MCTSAgent, NoSearchAgent, agent_name, make_agent
from eval.match import play_game
from search.mcts import MCTSConfig
from tests.test_mcts import WHITE_MATE_IN_1, Uniform


def positions(n=12, seed=0):
    rng, out, b = random.Random(seed), [], chess.Board()
    while len(out) < n:
        if B.is_terminal(b):
            b = chess.Board()
        out.append(b.copy())
        b.push(rng.choice(list(b.legal_moves)))
    return out


class Prefers(Uniform):
    """Uniform priors, except ``move`` gets ``share`` of the prior (a mild preference)."""

    def __init__(self, move, share):
        super().__init__()
        self.move, self.share = move, share

    def evaluate(self, boards):
        out = []
        for priors, v in super().evaluate(boards):
            if self.move in priors:
                rest = (1 - self.share) / max(len(priors) - 1, 1)
                priors = {m: self.share if m == self.move else rest for m in priors}
            out.append((priors, v))
        return out


def test_agents_play_legal_moves_deterministically():
    for agent in (NoSearchAgent(Uniform(), "a"), MCTSAgent(Uniform(), "b", MCTSConfig(simulations=8))):
        for board in positions():
            move = agent.select_move(board.copy())
            assert B.is_legal(board, move) and move == agent.select_move(board.copy())
    g = play_game(make_agent(Uniform(), "w", 4), make_agent(Uniform(), "b", 0), max_plies=30)
    assert g.illegal_attempts == {"white": 0, "black": 0}


def test_search_finds_the_mate_the_policy_alone_misses():
    fen, mate = WHITE_MATE_IN_1
    board = chess.Board(fen)
    decoy = next(m for m in B.legal_moves(board) if m != mate)
    net = Prefers(decoy, 0.3)                               # a policy that prefers a non-mating move
    assert NoSearchAgent(net, "n").select_move(board) == decoy
    assert MCTSAgent(net, "s", MCTSConfig(simulations=200)).select_move(board) == mate


def test_same_moves_as_alphazero_agents():
    from model.az_net import AZNet, NetConfig
    from train.az_train import AlphaZeroAgent, PolicyAgent
    from model.az_net import AZEvaluator
    torch.manual_seed(0)
    net = AZNet(NetConfig(blocks=2, channels=16)).eval()
    cfg = MCTSConfig(simulations=12)
    ev = AZEvaluator(net, "cpu")
    pairs = [(NoSearchAgent(ev, "x"), PolicyAgent(net, "x", "cpu")),
             (make_agent(ev, "y", 12), AlphaZeroAgent(net, cfg, "y", "cpu"))]
    for new, old in pairs:
        for board in positions(8, seed=3):
            assert new.select_move(board.copy()) == old.select_move(board.copy())


def test_make_agent_switch_and_names():
    assert isinstance(make_agent(Uniform(), "a", 0), NoSearchAgent)
    agent = make_agent(Uniform(), "a", 16, MCTSConfig(simulations=50, c_puct=2.0))
    assert isinstance(agent, MCTSAgent) and agent.cfg.simulations == 16 and agent.cfg.c_puct == 2.0
    with pytest.raises(ValueError):
        make_agent(Uniform(), "a", -1)
    assert agent_name("structured_masked_nall_e10_s0@25000", 16) == "structured_masked_nall_e10_s0_mcts16@25000"
    assert agent_name("az_b6c64_s0@r53", 0) == "az_b6c64_s0_nosearch@r53"
    assert agent_name("plain", 50) == "plain_mcts50"
