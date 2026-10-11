"""Step 2 checks for eval/: the Elo math, the anchors, and the match runner.

Intent: prove each harness piece works on small, fast cases before spending
real compute on long matches (plan v3, Step 2). The Stockfish tests skip (not fail) without the
binary -- run the setup notebook first.

Run from the repo root:
    python -m pytest tests/test_eval.py -v
"""
import csv

import chess
import numpy as np
import pytest

from engine.stockfish import SF_VERSION
from eval.anchors import (RESULTS_COLUMNS, REFERENCE_SET, SMOKE_TEST_OPPONENT, GreedyAgent, RandomAgent,
                          ResultsTable, StockfishAgent, fixed_ratings)
from eval.elo import expected_score, fit_ratings, fit_with_ci, pairs_needing_games
from eval.match import evaluate_agent, play_game, play_match, rate_adaptively, summarize, to_results


def simulate(a, ra, b, rb, n, rng):
    """Synthetic win/loss results between two players of known true rating (colors alternate)."""
    out = []
    for i in range(n):
        s = 1.0 if rng.random() < expected_score(ra, rb) else 0.0
        out.append((a, b, s) if i % 2 == 0 else (b, a, 1.0 - s))
    return out


class IllegalAgent:
    """Always answers with an impossible move -- stands in for a confused unmasked LLM."""
    name = "illegal"

    def select_move(self, board):
        """Never legal."""
        return "a1a1"


# ------------------------------------------------------------------ elo ----

def test_expected_score_basics():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)  # 400 points = 10:1 odds
    assert expected_score(1200, 1700) + expected_score(1700, 1200) == pytest.approx(1.0)


def test_fit_recovers_true_rating():
    rng = np.random.default_rng(0)
    res = (simulate("agent", 1500, "sf_1320", 1320, 300, rng)
           + simulate("agent", 1500, "sf_1600", 1600, 300, rng))
    ratings, bounds = fit_ratings(res, {"sf_1320": 1320, "sf_1600": 1600})
    assert abs(ratings["agent"] - 1500) < 40 and bounds["agent"] is None
    assert ratings["sf_1320"] == 1320  # fixed players don't move


def test_fit_chains_through_unrated_players():
    # weak <- mid <- fixed: the weak player never meets the fixed one directly
    rng = np.random.default_rng(1)
    res = simulate("weak", 800, "mid", 1100, 400, rng) + simulate("mid", 1100, "fixed", 1400, 400, rng)
    ratings, _ = fit_ratings(res, {"fixed": 1400})
    assert abs(ratings["mid"] - 1100) < 60 and abs(ratings["weak"] - 800) < 90


def test_perfect_score_is_flagged_not_estimated():
    ratings, bounds = fit_ratings([("x", "sf_1320", 1.0)] * 10, {"sf_1320": 1320})
    assert bounds["x"] == "upper"
    ratings, bounds = fit_ratings([("sf_1320", "x", 1.0)] * 10, {"sf_1320": 1320})
    assert bounds["x"] == "lower"


def test_island_behind_a_perfect_scorer_is_unanchored():
    # The real Step 2 failure mode: random and greedy trade draws with each other,
    # but their only route to the scale is sf_depth1, which beat them AND sf_1320
    # every time. Their mixed results must not pass as real ratings.
    res = ([("random", "greedy", 0.5)] * 6 + [("sf_depth1", "greedy", 1.0)] * 6
           + [("sf_depth1", "sf_1320", 1.0)] * 6 + [("sf_1320", "sf_1600", 0.5)] * 6)
    _, flags = fit_ratings(res, {"sf_1320": 1320, "sf_1600": 1600})
    assert flags["sf_depth1"] == "upper"
    assert flags["greedy"] == "unanchored" and flags["random"] == "unanchored"


def test_fit_needs_a_fixed_player():
    with pytest.raises(ValueError):
        fit_ratings([("a", "b", 1.0)], {"someone_else": 1500})


def test_bootstrap_ci_brackets_truth():
    rng = np.random.default_rng(2)
    res = simulate("agent", 1450, "sf_1320", 1320, 200, rng) + simulate("agent", 1450, "sf_1600", 1600, 200, rng)
    est = fit_with_ci(res, {"sf_1320": 1320, "sf_1600": 1600}, n_boot=200)["agent"]
    assert est.ci_low < 1450 < est.ci_high
    assert est.ci_low < est.rating < est.ci_high
    assert est.games == 400 and 0 < est.score < 1


def test_adaptive_rule_is_ci_only():
    rng = np.random.default_rng(3)
    fixed = {"a": 1400.0, "b": 1600.0}
    # few games -> wide CI -> every matchup of the candidate needs more
    few = simulate("c", 1500, "a", 1400, 20, rng) + simulate("c", 1500, "b", 1600, 20, rng)
    est = fit_with_ci(few, fixed, n_boot=200)
    assert pairs_needing_games(est, few, ["c"]) == {frozenset(("c", "a")), frozenset(("c", "b"))}
    # many games -> tight CI -> done
    many = simulate("c", 1500, "a", 1400, 400, rng) + simulate("c", 1500, "b", 1600, 400, rng)
    est = fit_with_ci(many, fixed, n_boot=200)
    assert pairs_needing_games(est, many, ["c"]) == set()
    # lopsided (~90/10 vs "a") but the CI is tight -> done: a big gap is not a reason for more games
    lop = simulate("c", 1500, "a", 1100, 400, rng) + simulate("c", 1500, "b", 1600, 400, rng)
    est = fit_with_ci(lop, {"a": 1100.0, "b": 1600.0}, n_boot=200)
    assert est["c"].score > 0.6 and (est["c"].ci_high - est["c"].ci_low) / 2 < 100
    assert pairs_needing_games(est, lop, ["c"]) == set()


def test_adaptive_rule_never_accepts_an_off_scale_rating():
    sweep = [("c", "a", 1.0)] * 10                         # won everything -> rating has no upper bound
    est = fit_with_ci(sweep, {"a": 1400.0}, n_boot=50)
    assert est["c"].at_bound == "upper"
    assert pairs_needing_games(est, sweep, ["c"], ci_half_width=1e9) == {frozenset(("c", "a"))}


def test_rate_adaptively_stops_when_done_and_at_the_cap():
    agents = {"r1": RandomAgent("r1", seed=1), "r2": RandomAgent("r2", seed=2)}
    # impossible target (negative CI width) -> keeps adding games until the cap, then reports it
    _, recs, log = rate_adaptively(agents, [("r1", "r2")], {"r1": 300.0}, ["r2"], start_games=4, step_games=2,
                                   max_games_per_pair=8, ci_half_width=-1.0, n_boot=20, verbose=False,
                                   max_plies=20)
    assert len(recs) == 8 and log[-1]["capped"] == ["r1-r2"]
    # generous target -> stops after the first batch
    _, recs, log = rate_adaptively(agents, [("r1", "r2")], {"r1": 300.0}, ["r2"], start_games=4, ci_half_width=1e9,
                                   n_boot=20, verbose=False, max_plies=20)
    assert len(recs) == 4 and len(log) == 1


def test_rate_adaptively_on_batch_fires_every_batch_so_an_interrupt_loses_nothing():
    # on_batch is the fix for the Sep 30 incident: everything rate_adaptively knows after
    # each batch gets handed to the caller immediately, not just once the whole call
    # returns -- so a caller that persists it (e.g. to results.csv)
    # never loses more than the games in the batch that was interrupted mid-flight.
    agents = {"r1": RandomAgent("r1", seed=1), "r2": RandomAgent("r2", seed=2)}
    seen = []
    rate_adaptively(agents, [("r1", "r2")], {"r1": 300.0}, ["r2"], start_games=4, step_games=2,
                    max_games_per_pair=8, ci_half_width=-1.0, n_boot=20, verbose=False, max_plies=20,
                    on_batch=lambda estimates, log: seen.append((dict(estimates), len(log))))
    assert len(seen) == 3                                   # one call per batch: games 4, 6, 8 (the cap)
    assert [n for _, n in seen] == [1, 2, 3]                 # log grows by one entry each time
    assert all("r2" in est for est, _ in seen)               # the live estimate was already there each time

    class _Boom(Exception):
        pass

    captured = {}

    def _save_then_blow_up(estimates, log):
        captured["last"] = dict(estimates)
        if len(log) == 2:
            raise _Boom("simulated interrupt after the second batch")

    with pytest.raises(_Boom):
        rate_adaptively(agents, [("r1", "r2")], {"r1": 300.0}, ["r2"], start_games=4, step_games=2,
                        max_games_per_pair=8, ci_half_width=-1.0, n_boot=20, verbose=False, max_plies=20,
                        on_batch=_save_then_blow_up)
    assert "r2" in captured["last"]                          # the batch-2 estimate survived the exception


# -------------------------------------------------------------- anchors ----

def test_random_agent_is_legal_and_reproducible():
    b = chess.Board()
    assert RandomAgent(seed=5).select_move(b) == RandomAgent(seed=5).select_move(b)
    assert chess.Move.from_uci(RandomAgent().select_move(b)) in b.legal_moves


def test_greedy_takes_the_biggest_capture():
    # White knight on d5 can take a pawn on c7 or the queen on e7 -> queen
    b = chess.Board("4k3/2p1q3/8/3N4/8/8/8/4K3 w - - 0 1")
    assert GreedyAgent().select_move(b) == "d5e7"


def test_greedy_counts_en_passant_as_a_capture():
    b = chess.Board("k7/8/8/3pP3/8/8/8/K7 w - d6 0 1")  # only capture available is e5xd6 e.p.
    assert GreedyAgent().select_move(b) == "e5d6"


def test_stockfish_agent_rejects_bad_limit():
    with pytest.raises(ValueError):
        StockfishAgent("bad", limit={"seconds": 1})  # checked before any engine starts
    with pytest.raises(ValueError):
        StockfishAgent("bad", limit={})


def test_reference_set_matches_plan_v3():
    names = [s.name for s in REFERENCE_SET]
    assert names == ["random", "greedy", "sf_1320", "sf_1500", "sf_1700"]
    assert SMOKE_TEST_OPPONENT.name not in names          # depth-1 is a smoke test, not a member
    assert fixed_ratings() == {"sf_1320": 1320.0}          # the one pin; everything else is measured


def test_results_table_pin_reference_model_one_file(tmp_path):
    path = tmp_path / "results.csv"
    t = ResultsTable(path)
    assert t.role("sf_1320") == "pin" and t.fixed() == {"sf_1320": 1320.0}   # pin exists before any save
    t.add_reference("sf_1500", 1580.0, 1500.0, 1673.0, 160, spec={"kind": "stockfish"})
    t.add_reference("step_20000", 719.3, 577.0, 804.0, 200,
                    spec={"kind": "sft_checkpoint", "run": "r", "step": 20000})
    t.save()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == RESULTS_COLUMNS
        assert [r["role"] for r in reader] == ["pin", "reference", "reference"]
    back = ResultsTable(path)
    assert back.fixed() == {"sf_1320": 1320.0, "sf_1500": 1580.0, "step_20000": 719.3}
    assert back.spec("step_20000")["step"] == 20000


def test_results_table_refuses_to_rerate_a_reference_or_pin(tmp_path):
    t = ResultsTable(tmp_path / "results.csv")
    t.add_reference("greedy", 641.0, 492.0, 724.0, 200, spec={"kind": "greedy"})
    with pytest.raises(ValueError):
        t.add_reference("greedy", 700.0, 600.0, 800.0, 40, spec={"kind": "greedy"})
    with pytest.raises(ValueError):
        t.add_reference("sf_1320", 1300.0, 1300.0, 1300.0, 40, spec={"kind": "stockfish"})
    est = fit_with_ci(simulate("greedy", 641, "sf_1320", 1320, 20, np.random.default_rng(0)),
                      {"sf_1320": 1320.0}, n_boot=20)["greedy"]
    with pytest.raises(ValueError):
        t.log_model(est)                                   # a model eval can't overwrite a reference row


def test_results_table_model_row_is_overwritten_each_batch(tmp_path):
    path = tmp_path / "results.csv"
    t = ResultsTable(path)
    anchors = {"greedy": GreedyAgent(seed=1), "random": RandomAgent("random", seed=2)}
    seen = []

    def save(est, recs):
        t.log_model(est, recs, run="run_a", step=500, train_gpu_hours=0.5, gpu="test")
        t.save()
        seen.append(ResultsTable(path).rows["candidate"]["games"])

    evaluate_agent(RandomAgent("candidate", seed=3), anchors, {"greedy": 600.0, "random": 300.0},
                   n_games=4, batch_games=2, on_batch=save, n_boot=20, verbose=False, max_plies=40)
    assert seen == ["4", "8"]                              # one row, updated after each batch
    row = ResultsTable(path).rows["candidate"]
    assert row["role"] == "model" and row["run"] == "run_a" and row["train_gpu_hours"] == "0.5"
    assert float(row["sec_per_move"]) >= 0 and float(row["illegal_rate"]) == 0.0


def test_results_table_rejects_a_different_stockfish_version(tmp_path):
    path = tmp_path / "results.csv"
    t = ResultsTable(path)
    t.add_reference("sf_1500", 1580.0, 1500.0, 1673.0, 160, spec={"kind": "stockfish"})
    t.save()
    path.write_text(path.read_text().replace(SF_VERSION, "sf_18"))
    with pytest.raises(RuntimeError):
        ResultsTable(path)


# ---------------------------------------------------------------- match ----

def test_play_game_finishes_and_records():
    g = play_game(RandomAgent("w", seed=1), RandomAgent("b", seed=2), opening=["e2e4", "e7e5"])
    assert g.moves[:2] == ["e2e4", "e7e5"] and g.opening == ["e2e4", "e7e5"]
    assert g.white_score in (0.0, 0.5, 1.0)
    assert len(g.latency["white"]) + len(g.latency["black"]) == g.plies - 2
    replay = chess.Board()
    for mv in g.moves:
        replay.push_uci(mv)  # every recorded move is legal in sequence
    assert '[White "w"]' in g.pgn


def test_max_plies_adjudicates_a_draw():
    g = play_game(RandomAgent(seed=1), RandomAgent(seed=2), max_plies=10)
    assert g.plies == 10 and g.white_score == 0.5 and g.termination == "max_plies"


def test_third_illegal_attempt_forfeits_by_default():
    g = play_game(IllegalAgent(), RandomAgent())          # plan v3 three-strikes rule is the default
    assert g.termination == "illegal_move" and g.white_score == 0.0
    assert g.illegal_attempts["white"] == 3 and g.plies == 0


def test_illegal_then_legal_does_not_forfeit():
    class ShakyAgent:
        """Two illegal tries, then a legal move -- every turn."""
        name = "shaky"

        def __init__(self):
            self.calls = 0

        def select_move(self, board):
            self.calls += 1
            return next(iter(board.legal_moves)).uci() if self.calls % 3 == 0 else "a1a1"

    g = play_game(ShakyAgent(), RandomAgent(), max_plies=6)
    assert g.termination == "max_plies" and g.illegal_attempts["white"] == 6  # 2 per move x 3 moves


def test_illegal_move_random_fallback_keeps_playing():
    g = play_game(RandomAgent("r"), IllegalAgent(), illegal_policy="random", max_plies=20)
    assert g.termination != "illegal_move"
    assert g.illegal_attempts["black"] > 0 and g.illegal_attempts["white"] == 0


def test_agents_get_a_copy_of_the_board():
    class Vandal:
        name = "vandal"

        def select_move(self, board):
            move = next(iter(board.legal_moves)).uci()
            board.clear()  # tries to wreck the game
            return move

    g = play_game(Vandal(), RandomAgent(), max_plies=6)
    assert g.plies == 6  # the real game was unaffected


def test_play_match_pairs_openings_and_swaps_colors():
    recs = play_match(RandomAgent("a", seed=1), RandomAgent("b", seed=2), 4, opening_plies=3)
    assert [r.white for r in recs] == ["a", "b", "a", "b"]
    assert recs[0].opening == recs[1].opening and recs[2].opening == recs[3].opening
    assert recs[0].opening != recs[2].opening
    s = summarize(recs, "a")
    assert s["games"] == 4 and s["wins"] + s["draws"] + s["losses"] == 4


def test_evaluate_agent_end_to_end_without_stockfish():
    # Pretend greedy and random are rated anchors; rate a second random agent against them.
    anchors = {"greedy": GreedyAgent(seed=1), "random": RandomAgent("random", seed=2)}
    est, recs = evaluate_agent(RandomAgent("candidate", seed=3), anchors, {"greedy": 600.0, "random": 300.0},
                               n_games=6, n_boot=50, verbose=False)
    assert est.name == "candidate" and est.games == 12 and len(recs) == 12
    assert to_results(recs)[0][0] in ("candidate", "greedy", "random")


def test_evaluate_agent_requires_rated_anchors():
    with pytest.raises(ValueError):
        evaluate_agent(RandomAgent("c"), {"greedy": GreedyAgent()}, {}, n_games=2, verbose=False)


# ------------------------------------------------------------ stockfish ----

try:
    from engine.stockfish import find_stockfish
    find_stockfish()
    HAVE_SF = True
except FileNotFoundError:
    HAVE_SF = False

needs_sf = pytest.mark.skipif(not HAVE_SF, reason="Stockfish binary not found (run the setup notebook)")


@needs_sf
@pytest.mark.parametrize("limit", [{"depth": 1}, {"nodes": 50}, {"time": 0.02}])
def test_stockfish_agent_any_limit_plays_legal_moves(limit):
    agent = StockfishAgent("sf", limit=limit)
    try:
        b = chess.Board()
        assert chess.Move.from_uci(agent.select_move(b)) in b.legal_moves
    finally:
        agent.close()


@needs_sf
def test_stockfish_crushes_random():
    sf = StockfishAgent("sf_depth1", limit={"depth": 1})
    try:
        recs = play_match(sf, RandomAgent(seed=4), 2)
        assert summarize(recs, "sf_depth1")["wins"] == 2
    finally:
        sf.close()
