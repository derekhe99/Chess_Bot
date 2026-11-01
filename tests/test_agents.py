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


# ------------------------------------------------------ head-to-head (Step 12d) ----

from eval.anchors import RandomAgent, ResultsTable
from eval.match import elo_difference, head_to_head, head_to_head_table


def test_head_to_head_plays_every_pair_with_the_same_openings():
    players = {n: RandomAgent(n, seed=i) for i, n in enumerate(["a", "b", "c"])}
    seen = []
    games = head_to_head(players, 4, max_plies=20, verbose=False, on_pair=lambda a, b, g: seen.append((a, b, len(g))))
    assert seen == [("a", "b", 4), ("a", "c", 4), ("b", "c", 4)] and len(games) == 12
    by_pair = {}
    for g in games:
        by_pair.setdefault(frozenset((g.white, g.black)), []).append(g)
    openings = []
    for pair_games in by_pair.values():
        assert [g.white for g in pair_games][0] != pair_games[1].white      # colors swapped
        openings.append([tuple(g.opening) for g in pair_games])
    assert openings[0] == openings[1] == openings[2]                       # same openings for every pair
    again = head_to_head(players, 2, max_plies=10, verbose=False, skip={frozenset(("a", "b")), frozenset(("a", "c"))})
    assert {frozenset((g.white, g.black)) for g in again} == {frozenset(("b", "c"))}


def _g(white, black, score, termination):
    return {"white": white, "black": black, "white_score": score, "termination": termination}


def test_head_to_head_table():
    games = ([_g("x", "y", 1.0, "checkmate")] * 3 + [_g("y", "x", 1.0, "checkmate")]
             + [_g("x", "y", 0.5, "threefold_repetition")] * 2 + [_g("y", "x", 0.5, "stalemate")] * 2
             + [_g("p", "q", 1.0, "checkmate")] * 4)
    t = {(r["a"], r["b"]): r for r in head_to_head_table(games, n_boot=300)}
    xy = t[("x", "y")]
    assert (xy["games"], xy["a_wins"], xy["draws"], xy["a_losses"]) == (8, 3, 4, 1)
    assert xy["a_score"] == 0.625 and xy["elo_diff"] == round(elo_difference(0.625, 8), 1) > 0
    assert xy["ci_low"] <= xy["elo_diff"] <= xy["ci_high"]
    assert xy["draw_types"] == {"threefold_repetition": 2, "stalemate": 2} and xy["decisive_types"] == {"checkmate": 4}
    pq = t[("p", "q")]
    assert pq["at_edge"] and pq["elo_diff"] == round(elo_difference(1.0, 4), 1)     # a bound, not infinity
    assert abs(elo_difference(0.75, 40) - 190.85) < 0.01 and elo_difference(0.5, 10) == 0.0


def test_log_games_saves_head_to_head_next_to_eval_games(tmp_path):
    from eval.match import evaluate_agent
    path = tmp_path / "results.csv"
    t = ResultsTable(path)
    est, recs = evaluate_agent(RandomAgent("m", seed=1), {"random": RandomAgent("random", seed=2)}, {"random": 300.0},
                               n_games=2, n_boot=20, verbose=False, max_plies=20)
    t.log_model(est, recs)
    h = head_to_head({"a": RandomAgent("a", seed=3), "b": RandomAgent("b", seed=4)}, 4, max_plies=20, verbose=False)
    t.log_games(h, "h2h", "round1|a|b")
    t.save()
    t2 = ResultsTable(path)
    assert len(t2.games("m")) == 2 and len(t2.games("round1|a|b", kind="h2h")) == 4
    t2.log_games(h[:2], "h2h", "round1|a|b")                                # a replayed pair replaces its games
    t2.save()
    assert len(ResultsTable(path).games("round1|a|b", kind="h2h")) == 2 and len(ResultsTable(path).games("m")) == 2


def test_load_player_alphazero_both_ways(tmp_path):
    import dataclasses as dc
    from eval.agents import checkpoint_dir, load_player
    from model.az_net import AZNet, NetConfig
    spec = {"name": "az_t@r3", "kind": "alphazero", "run": "az_t", "step": 3}
    d = checkpoint_dir(spec, tmp_path)
    d.mkdir(parents=True)
    net_cfg = NetConfig(blocks=1, channels=8)
    torch.save({"net_config": dc.asdict(net_cfg), "state_dict": AZNet(net_cfg).state_dict()}, d / "net.pt")
    assert d == tmp_path / "checkpoints" / "alphazero" / "az_t" / "checkpoints" / "round_0003"
    no_search = load_player({**spec, "simulations": 0}, tmp_path, device="cpu")
    search = load_player({**spec, "simulations": 4}, tmp_path, device="cpu")
    assert isinstance(no_search, NoSearchAgent) and isinstance(search, MCTSAgent) and search.cfg.simulations == 4
    board = chess.Board()
    assert B.is_legal(board, no_search.select_move(board)) and B.is_legal(board, search.select_move(board))
    assert checkpoint_dir({"kind": "sft", "run": "r", "step": 15000}, tmp_path).name == "step_015000"
    with pytest.raises(FileNotFoundError):
        load_player({**spec, "step": 4}, tmp_path)
