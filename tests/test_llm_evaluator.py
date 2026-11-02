"""Step 15/17 checks for model/llm_evaluator.py: an LLM checkpoint as a search network.

What they prove, on the tiny 2-layer Qwen3 the other LLM tests use:
- Batched evaluate() gives the same priors and values as scoring each position alone
  with LLMPolicy.score_moves (the play path), even with prompts of different lengths
  (left padding) and move chunks that cut across positions.
- Priors cover exactly the legal moves and sum to 1; values are win probabilities.
- It plugs into the search: MCTSAgent / make_agent play legal moves with it, and
  time_search reports one batched move across many games.

    python -m pytest tests/test_llm_evaluator.py -v
"""
import math

import chess
import pytest
import torch

from engine import board as B
from eval.agents import MCTSAgent, NoSearchAgent, make_agent
from model.llm_evaluator import LLMEvaluator, time_search
from tests.test_llm_policy import make_policy, positions, tiny_tokenizer


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


@pytest.fixture(scope="module")
def policy(tok):
    p = make_policy(tok)
    torch.nn.init.normal_(p.value_head.linear.weight, std=0.5)   # a value head with opinions
    return p


def boards():
    out = positions(n_games=3, plies=30, seed=1)[::4]
    out.append(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))       # castling rights: longer prompt
    out.append(chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1"))                # promotions: longer moves
    return out


@pytest.mark.parametrize("batch_positions,move_batch", [(64, 128), (3, 7)])
def test_batched_matches_one_position_at_a_time(policy, batch_positions, move_batch):
    bs = boards()
    assert len({len(policy.prompt_ids(b)) for b in bs}) > 1                # really mixed prompt lengths
    ev = LLMEvaluator(policy, batch_positions=batch_positions, move_batch=move_batch)
    for b, (priors, value) in zip(bs, ev.evaluate(bs)):
        moves = B.legal_moves(b)
        scores, win_prob = policy.score_moves(b, moves)
        z = math.log(sum(math.exp(s) for s in scores.values()))
        assert set(priors) == set(moves) and abs(sum(priors.values()) - 1) < 1e-5
        for m in moves:
            assert abs(priors[m] - math.exp(scores[m] - z)) < 1e-4
        assert abs(value - win_prob) < 1e-4 and 0 < value < 1


def test_it_drives_the_search(policy):
    ev = LLMEvaluator(policy)
    board = chess.Board()
    for agent in (make_agent(ev, "s", 4), make_agent(ev, "n", 0)):
        move = agent.select_move(board.copy())
        assert B.is_legal(board, move) and move == agent.select_move(board.copy())
    assert isinstance(make_agent(ev, "s", 4), MCTSAgent) and isinstance(make_agent(ev, "n", 0), NoSearchAgent)
    (priors, _), = ev.evaluate([board])
    scores, _ = policy.score_moves(board, B.legal_moves(board))
    assert NoSearchAgent(ev, "n").select_move(board) == max(sorted(scores), key=lambda m: scores[m])  # = masked play


def test_time_search_reports_one_batched_move(policy):
    ev = LLMEvaluator(policy, batch_positions=64)
    r = time_search(ev, [chess.Board()] * 6, simulations=4)
    assert r["boards"] == 6 and r["seconds"] > 0
    assert r["positions_evaluated"] >= 6 and r["model_calls"] <= 5         # all six games share each call


def test_load_player_llm_with_and_without_search(policy, tmp_path, monkeypatch):
    import train.sft
    from eval.agents import checkpoint_dir, load_player
    spec = {"name": "run@30000", "kind": "sft", "run": "run", "step": 30000}
    checkpoint_dir(spec, tmp_path).mkdir(parents=True)
    monkeypatch.setattr(train.sft, "load_checkpoint", lambda path, device=None, name=None: policy)
    assert load_player(spec, tmp_path) is policy                                   # no "simulations": plays as in evals
    agent = load_player({**spec, "simulations": 16}, tmp_path)
    assert isinstance(agent, MCTSAgent) and agent.cfg.simulations == 16 and isinstance(agent.evaluator, LLMEvaluator)
    assert isinstance(load_player({**spec, "simulations": 0}, tmp_path), NoSearchAgent)
