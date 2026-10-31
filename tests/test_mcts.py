"""Gate checks for Step 9 (b), search/mcts.py: the search works before any network exists.

What they prove, with a dummy model (uniform priors, value 0.5) that knows nothing:
- MCTS finds a mate-in-1 for either color: the exact result of a finished game is
  used and backed up with the right sign at each level.
- Root visit counts sum to the simulation count; the search is deterministic for a seed.
- Finished positions never go to the model; batching many trees into one model call
  gives the same answer as searching each alone.
- Batched self-play games finish, respect the ply cap, and record visit distributions.

    python -m pytest tests/test_mcts.py -v
"""
import random

import chess
import numpy as np
import pytest

from engine import board as B
from search.mcts import MCTSConfig, choose_move, play_games, run_search, search_move, visit_distribution

WHITE_MATE_IN_1 = ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8")    # back-rank mate
BLACK_MATE_IN_1 = ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1")    # the same, colors swapped


class Uniform:
    """Knows nothing: equal priors over the legal moves, value 0.5. Records every call."""

    def __init__(self):
        self.calls: list[int] = []      # batch size of each evaluate() call
        self.boards: list[str] = []

    def evaluate(self, boards):
        self.calls.append(len(boards))
        out = []
        for b in boards:
            assert not B.is_terminal(b), "a finished position reached the model"
            self.boards.append(b.fen())
            moves = B.legal_moves(b)
            out.append(({m: 1 / len(moves) for m in moves}, 0.5))
        return out


class Favors(Uniform):
    """Uniform, except one move gets most of the prior."""

    def __init__(self, move):
        super().__init__()
        self.move = move

    def evaluate(self, boards):
        out = []
        for priors, v in super().evaluate(boards):
            if self.move in priors:
                rest = (1 - 0.9) / max(len(priors) - 1, 1)
                priors = {m: 0.9 if m == self.move else rest for m in priors}
            out.append((priors, v))
        return out


@pytest.mark.parametrize("fen,mate", [WHITE_MATE_IN_1, BLACK_MATE_IN_1])
def test_finds_mate_in_one_with_a_model_that_knows_nothing(fen, mate):
    board = chess.Board(fen)
    after = board.copy()
    after.push_uci(mate)
    assert after.is_checkmate()                                  # the position really is mate-in-1
    assert board.legal_moves.count() >= 10                       # with plenty of other moves to try
    root = run_search([board], Uniform(), MCTSConfig(simulations=100))[0]
    assert choose_move(root) == mate
    best = max(root.children, key=lambda c: c.N)
    assert best.W / best.N == 1.0          # every visit to the mating move scored a win for the mover
    assert search_move(board, Uniform(), MCTSConfig(simulations=100)) == mate


@pytest.mark.parametrize("noise", [False, True])
def test_root_visits_sum_to_simulations(noise):
    rng = np.random.default_rng(0) if noise else None
    root = run_search([chess.Board()], Uniform(), MCTSConfig(simulations=37), add_noise=noise, rng=rng)[0]
    assert sum(c.N for c in root.children) == 37
    assert abs(sum(visit_distribution(root).values()) - 1) < 1e-9


def test_search_is_deterministic_for_a_fixed_seed():
    cfg = MCTSConfig(simulations=50)

    def visits(seed):
        root = run_search([chess.Board()], Uniform(), cfg, add_noise=True, rng=np.random.default_rng(seed))[0]
        return [(c.move, c.N) for c in root.children]

    assert visits(1) == visits(1)
    assert visits(1) != visits(2)          # the noise really depends on the seed


def test_finished_positions_never_reach_the_model_and_cant_be_searched():
    model = Uniform()                       # asserts on any finished position it is given
    run_search([chess.Board(WHITE_MATE_IN_1[0])], model, MCTSConfig(simulations=100))
    mated = chess.Board(WHITE_MATE_IN_1[0])
    mated.push_uci(WHITE_MATE_IN_1[1])
    assert mated.fen() not in model.boards
    with pytest.raises(ValueError):
        run_search([mated], Uniform(), MCTSConfig(simulations=10))


def test_ply_cap_counts_as_a_draw_inside_the_search():
    board = chess.Board()
    for mv in ["e2e4", "e7e5"]:
        board.push_uci(mv)
    model = Uniform()
    run_search([board], model, MCTSConfig(simulations=30), max_plies=3)   # every child is at the cap
    assert model.calls == [1]               # only the root went to the model
    with pytest.raises(ValueError):
        run_search([board], Uniform(), MCTSConfig(simulations=5), max_plies=2)


def test_batched_search_matches_searching_one_at_a_time():
    boards = [chess.Board(), chess.Board(WHITE_MATE_IN_1[0]), chess.Board(BLACK_MATE_IN_1[0])]
    cfg = MCTSConfig(simulations=20)
    model = Uniform()
    batched = run_search(boards, model, cfg)
    assert len(model.calls) <= 1 + cfg.simulations and max(model.calls) == len(boards)   # one call per step
    for b, root in zip(boards, batched):
        alone = run_search([b], Uniform(), cfg)[0]
        assert [(c.move, c.N) for c in root.children] == [(c.move, c.N) for c in alone.children]


def test_priors_steer_the_search():
    root = run_search([chess.Board()], Favors("g1f3"), MCTSConfig(simulations=50))[0]
    assert choose_move(root) == "g1f3"


def test_choose_move_most_visited_or_sampled():
    root = run_search([chess.Board()], Favors("d2d4"), MCTSConfig(simulations=40))[0]
    assert choose_move(root, 0.0) == "d2d4"
    picks = {choose_move(root, 1.0, random.Random(s)) for s in range(30)}
    assert "d2d4" in picks and picks <= {c.move for c in root.children if c.N > 0}


def test_play_games_batched_finishes_and_records_targets():
    cfg = MCTSConfig(simulations=6, temperature_plies=10)
    model = Uniform()
    games = play_games(model, 3, cfg, opening_plies=4, max_plies=24, seed=7)
    assert len(games) == 3 and max(model.calls) <= 3                      # all games share each call
    for g in games:
        assert g.plies <= 24 and len(g.opening) == 4
        assert len(g.moves) == g.plies - 4                                 # every searched move recorded
        board = chess.Board()
        for mv in g.opening:
            board.push_uci(mv)
        for r in g.moves:
            assert r.fen == board.fen() and r.white == (board.turn == chess.WHITE)
            assert set(r.info["visits"]) == set(B.legal_moves(board))
            assert abs(sum(r.info["visits"].values()) - 1) < 1e-9 and 0 <= r.info["root_value"] <= 1
            board.push_uci(r.move)
        assert board.fen() == g.final_fen
        assert g.termination == "max_plies" or B.is_terminal(board)
    again = play_games(Uniform(), 3, cfg, opening_plies=4, max_plies=24, seed=7)
    assert [[r.move for r in g.moves] for g in games] == [[r.move for r in g.moves] for g in again]
