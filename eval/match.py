"""Plays evaluation games and records everything the metrics need.

Intent: the one place games get played for evaluation. Every player --
reference-set member or trained model -- is treated identically: it's handed a
board and returns a move. The runner enforces the rules (including the
three-strikes illegal-move forfeit, plan v3 Sec 0), alternates colors over
shared openings so neither side gets a structural edge, and logs results, PGNs,
per-move thinking time, and illegal-move attempts.

Main pieces:
- Agent          -- the minimal interface: ``.name`` + ``.select_move(board) -> UCI``
- GameRecord     -- one finished game and all its stats
- play_game      -- plays one game between two agents
- play_match     -- N games between two agents, paired openings, colors swapped
- play_pairs     -- several matchups in one call
- to_results / summarize -- reduce records to Elo inputs / a W-D-L summary
- rate_adaptively -- rate new reference-set members, adding games only where
                    the ratings aren't yet tight enough to freeze (Steps 2 and 5)
- evaluate_agent -- the entry point later Steps use: play a model against the
                    frozen reference set and return its Elo with a confidence interval
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Protocol

import chess
import chess.pgn

from engine import board as B
from eval.elo import RatingEstimate, fit_with_ci, pairs_needing_games


class Agent(Protocol):
    """Anything that can play: has a name and picks a move (UCI string) for a board.

    This is the minimum the harness needs. The fuller shared interface
    (e.g. batched ``evaluate(positions)`` for MCTS) is still an open alignment
    point in the plan (Sec 5).
    """

    name: str

    def select_move(self, board: chess.Board) -> str: ...


@dataclass
class GameRecord:
    """One finished game and the stats the metrics need."""

    white: str
    black: str
    white_score: float                 # 1 white won, 0.5 draw, 0 black won
    termination: str                   # "checkmate", "stalemate", "max_plies", "illegal_move", ...
    plies: int
    moves: list[str]
    opening: list[str]                 # the forced random opening moves (part of `moves`)
    latency: dict[str, list[float]]    # seconds per move, keyed by "white" / "black"
    illegal_attempts: dict[str, int]   # keyed by "white" / "black"
    over_budget_moves: dict[str, int]  # moves slower than move_time_budget, keyed by side
    pgn: str = field(repr=False, default="")


def random_opening(rng: random.Random, plies: int) -> list[str]:
    """A random legal opening of ``plies`` half-moves that doesn't end the game."""
    while True:
        board, moves = chess.Board(), []
        for _ in range(plies):
            move = rng.choice(list(board.legal_moves))
            board.push(move)
            moves.append(move.uci())
        if not B.is_terminal(board):
            return moves


def play_game(white: Agent, black: Agent, *, opening: list[str] | None = None,
              max_plies: int = 400, illegal_policy: str = "forfeit", max_illegal_retries: int = 2,
              move_time_budget: float | None = None, seed: int = 0) -> GameRecord:
    """Play one game between two agents and record it.

    Inputs:
      white, black        -- agents
      opening             -- UCI moves played before the agents take over (for variety)
      max_plies           -- game is adjudicated a draw after this many half-moves
      illegal_policy      -- what happens when an agent's move is illegal after its
                             retries: "forfeit" (it loses) or "random" (a random legal
                             move is played for it; the attempt is still counted)
      max_illegal_retries -- extra chances to produce a legal move. Default 2 = the
                             plan's three-strikes rule: the third consecutive illegal
                             attempt on the same move forfeits (matters for stochastic
                             agents like an unmasked LLM; masked agents never trigger it)
      move_time_budget    -- seconds per move; slower moves are counted, not punished
                             (how to enforce the cap is decided in Step 6)
    Output: GameRecord

    Core logic: loop until the game is over -- ask the side to move for a move
    (on a COPY of the board, so an agent can't corrupt the real game), time it,
    check legality, apply it. Every rule decision goes through engine/board.py.
    """
    if illegal_policy not in ("forfeit", "random"):
        raise ValueError(f"illegal_policy must be 'forfeit' or 'random', got {illegal_policy!r}")
    rng = random.Random(seed)
    board = B.new_board()
    opening = list(opening or [])
    for mv in opening:
        B.apply_move(board, mv)

    latency = {"white": [], "black": []}
    illegal = {"white": 0, "black": 0}
    over_budget = {"white": 0, "black": 0}
    forfeit_side = None

    while not B.is_terminal(board) and board.ply() < max_plies:
        side = "white" if board.turn == chess.WHITE else "black"
        agent = white if side == "white" else black
        for attempt in range(max_illegal_retries + 1):
            start = time.perf_counter()
            move = agent.select_move(board.copy())
            elapsed = time.perf_counter() - start
            latency[side].append(elapsed)
            if move_time_budget is not None and elapsed > move_time_budget:
                over_budget[side] += 1
            if B.is_legal(board, move):
                break
            illegal[side] += 1
        else:  # every attempt was illegal
            if illegal_policy == "forfeit":
                forfeit_side = side
                break
            move = rng.choice(B.legal_moves(board))
        B.apply_move(board, move)

    if forfeit_side is not None:
        white_score, termination = (0.0 if forfeit_side == "white" else 1.0), "illegal_move"
    else:
        result = B.outcome(board)
        if result is None:
            white_score, termination = 0.5, "max_plies"
        else:
            white_score, termination = (result.white_value + 1) / 2, result.termination

    game = chess.pgn.Game.from_board(board)
    game.headers.update({"White": white.name, "Black": black.name, "Termination": termination,
                         "Result": {1.0: "1-0", 0.5: "1/2-1/2", 0.0: "0-1"}[white_score]})
    return GameRecord(white=white.name, black=black.name, white_score=white_score,
                      termination=termination, plies=board.ply(),
                      moves=[m.uci() for m in board.move_stack], opening=opening,
                      latency=latency, illegal_attempts=illegal, over_budget_moves=over_budget,
                      pgn=str(game))


def play_match(a: Agent, b: Agent, n_games: int, *, opening_plies: int = 4, seed: int = 0,
               verbose: bool = False, **game_kwargs) -> list[GameRecord]:
    """Play ``n_games`` between two agents in color-swapped pairs.

    Each pair of games starts from the same random opening, once with ``a`` as
    White and once with ``b`` -- so neither side profits from a lucky opening or
    from White's first-move edge. Extra keyword arguments go to play_game.
    """
    rng = random.Random(seed)
    records = []
    for i in range(n_games):
        if i % 2 == 0:
            opening = random_opening(rng, opening_plies)
        white, black = (a, b) if i % 2 == 0 else (b, a)
        records.append(play_game(white, black, opening=opening, seed=seed * 100_003 + i, **game_kwargs))
        if verbose:
            r = records[-1]
            print(f"  game {i + 1}/{n_games}: {r.white} vs {r.black} -> {r.white_score} ({r.termination}, {r.plies} plies)")
    return records


def play_pairs(agents: dict[str, Agent], pairs: list[tuple[str, str]], n_games: int, *,
               seed: int = 0, verbose: bool = True, **match_kwargs) -> list[GameRecord]:
    """Run several matchups (by agent name) and return all their games together."""
    records = []
    for k, (x, y) in enumerate(pairs):
        start = time.perf_counter()
        games = play_match(agents[x], agents[y], n_games, seed=seed + k, **match_kwargs)
        records += games
        if verbose:
            s = summarize(games, x)
            print(f"{x} vs {y}: {s['wins']}W {s['draws']}D {s['losses']}L for {x} "
                  f"({time.perf_counter() - start:.0f}s, {s['avg_plies']:.0f} plies/game)")
    return records


def to_results(records: list[GameRecord]) -> list[tuple[str, str, float]]:
    """Records -> (white, black, white_score) triples, the input eval/elo.py needs."""
    return [(r.white, r.black, r.white_score) for r in records]


def summarize(records: list[GameRecord], name: str) -> dict:
    """W/D/L, score, speed, and illegal-move stats for one player across records."""
    games = [r for r in records if name in (r.white, r.black)]
    if not games:
        raise ValueError(f"{name!r} played no games in these records")
    pts = [r.white_score if r.white == name else 1.0 - r.white_score for r in games]
    lat = [t for r in games for t in r.latency["white" if r.white == name else "black"]]
    return {
        "games": len(games),
        "wins": sum(p == 1.0 for p in pts), "draws": sum(p == 0.5 for p in pts), "losses": sum(p == 0.0 for p in pts),
        "score": sum(pts) / len(games),
        "avg_plies": sum(r.plies for r in games) / len(games),
        "avg_move_seconds": sum(lat) / len(lat) if lat else 0.0,
        "illegal_attempts": sum(r.illegal_attempts["white" if r.white == name else "black"] for r in games),
        "over_budget_moves": sum(r.over_budget_moves["white" if r.white == name else "black"] for r in games),
    }


def rate_adaptively(agents: dict[str, Agent], pairs: list[tuple[str, str]], fixed: dict[str, float],
                    candidates: list[str], *, start_games: int = 20, step_games: int = 20,
                    max_games_per_pair: int = 100, ci_half_width: float = 100.0,
                    n_boot: int = 1000, seed: int = 0,
                    verbose: bool = True, **match_kwargs) -> tuple[dict[str, RatingEstimate], list[GameRecord], list[dict]]:
    """Rate new reference-set members, adding games only where they're needed (plan v3, Step 5).

    Inputs:
      agents             -- {name: agent} for everyone in ``pairs``
      pairs              -- matchups to play, e.g. [("sf_1320", "sf_1500"), ...]
      fixed              -- ratings held fixed (pin + already-frozen members)
      candidates         -- the players being rated for freezing
      start_games        -- games per matchup in the first batch (plan: 20)
      step_games         -- games added to a matchup each time it needs more
      max_games_per_pair -- cap; a matchup never goes past this
      ci_half_width      -- stopping rule: each candidate's 95% CI within +/- this
                            (see elo.pairs_needing_games)
    Output: (estimates for all non-fixed players, every game record,
             a log with one entry per batch: games per matchup so far + matchups still short)

    Core logic: play ``start_games`` per matchup, fit, and ask pairs_needing_games
    which matchups are still short. Play ``step_games`` more in just those, refit,
    and repeat until nothing is short or every short matchup has hit the cap.
    A matchup that hits the cap is reported, not hidden -- judge it by hand.
    """
    if start_games % 2 or step_games % 2 or max_games_per_pair % 2:
        raise ValueError("game counts must be even (games come in color-swapped pairs)")
    missing = [c for c in candidates if c in fixed]
    if missing:
        raise ValueError(f"{missing} already have fixed ratings; they can't be candidates")
    by_key = {frozenset(p): p for p in pairs}
    played = {frozenset(p): 0 for p in pairs}
    batch = {frozenset(p): start_games for p in pairs}
    records: list[GameRecord] = []
    log: list[dict] = []
    round_no = 0
    while True:  # ends: every short matchup eventually hits the cap
        for k, (key, n) in enumerate(batch.items()):
            x, y = by_key[key]
            games = play_match(agents[x], agents[y], n, seed=seed + 1_000 * round_no + k, **match_kwargs)
            played[key] += n
            records += games
            if verbose:
                s = summarize(games, x)
                print(f"  batch {round_no}: {x} vs {y} +{n} games -> {s['wins']}W {s['draws']}D {s['losses']}L for {x}")
        estimates = fit_with_ci(to_results(records), fixed, n_boot=n_boot, seed=seed)
        short = pairs_needing_games(estimates, to_results(records), candidates, ci_half_width=ci_half_width)
        capped = sorted(tuple(sorted(k)) for k in short if played[k] >= max_games_per_pair)
        log.append({"batch": round_no, "games": {"-".join(by_key[k]): v for k, v in played.items()},
                    "short": sorted("-".join(sorted(k)) for k in short), "capped": ["-".join(c) for c in capped]})
        batch = {k: min(step_games, max_games_per_pair - played[k]) for k in short if played[k] < max_games_per_pair}
        if verbose:
            print(f"batch {round_no} done: {len(short)} matchup(s) short, {len(capped)} at the cap")
        if not batch:
            return estimates, records, log
        round_no += 1


def evaluate_agent(agent: Agent, opponents: dict[str, Agent], ratings: dict[str, float], n_games: int, *,
                   n_boot: int = 1000, seed: int = 0, verbose: bool = True,
                   **match_kwargs) -> tuple[RatingEstimate, list[GameRecord]]:
    """Rate one model against the frozen reference set; only its own rating is fitted.

    Inputs:
      agent     -- the player being rated (e.g. an SFT checkpoint)
      opponents -- {name: agent} reference-set members to play; each must have a
                   rating in ``ratings``
      ratings   -- fixed ratings: the pin plus frozen members (anchors.fixed_ratings)
      n_games   -- games per opponent (start small, scale up for final numbers)
    Output: (the agent's RatingEstimate, all game records for logging)
    """
    missing = [n for n in opponents if n not in ratings]
    if missing:
        raise ValueError(f"No known rating for {missing} -- rate and freeze them first")
    everyone = {agent.name: agent, **opponents}
    records = play_pairs(everyone, [(agent.name, n) for n in opponents], n_games,
                         seed=seed, verbose=verbose, **match_kwargs)
    estimate = fit_with_ci(to_results(records), {n: ratings[n] for n in opponents},
                           n_boot=n_boot, seed=seed)[agent.name]
    return estimate, records
