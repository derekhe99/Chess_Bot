"""Turns game results into Elo ratings with confidence intervals.

Intent: put every agent on one scale that never drifts (plan v3, Sec 0 "Eval").
Some ratings are held fixed -- the pin (Stockfish 1320 = 1320) and the frozen
members of the reference set -- and only the unknown players are fitted. The
same math is used twice: once to rate new reference-set members from games
among themselves (then they're frozen), and then to rate each new model against
the frozen set (only its own rating is fitted).

Main pieces:
- expected_score -- the Elo formula: how much A is expected to score against B
- fit_ratings    -- best-fit ratings for the unknown players, holding the
                    known ones fixed
- fit_with_ci    -- fit_ratings plus bootstrap confidence intervals
- RatingEstimate -- one player's rating, CI, record, and an "off the scale" flag
- pairs_needing_games -- the adaptive-game-count rule for freezing a member:
                    which matchups need more games before its rating is trusted
                    (95% CI still too wide -> more games)
- games_from_counts -- rebuild one matchup's Results from aggregate win/draw/loss
                    counts (e.g. copied off a rate_adaptively batch log after an
                    interrupted run) instead of individual GameRecords -- exact
                    rating, close-but-not-identical bootstrap CI

Elo-vs-samples and Elo-vs-compute curves (plan Sec 4) are just fit_with_ci run
once per checkpoint; the plotting lives in the analysis notebook.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# One game's result: (white player name, black player name, White's score: 1 / 0.5 / 0).
Result = tuple[str, str, float]


def games_from_counts(a: str, b: str, wins: int, draws: int, losses: int) -> list[Result]:
    """Rebuild one matchup's Results from ``a``'s aggregate record against ``b``.

    Use this to recover a rating fit when the real per-game records are gone --
    most commonly, a rate_adaptively() call that was interrupted (Colab
    disconnect, or a manual stop) before it returned. It only prints each
    batch's win/draw/loss counts as it goes; nothing is saved until it returns,
    so an interrupt loses the actual GameRecords. Summing the printed counts
    across whatever batches did finish and passing them here recovers the same
    fit fit_with_ci would have produced from the real games.

    This is safe because the Elo model has no first-move/color term --
    expected_score depends only on the two ratings, so a decisive game
    contributes the same log-likelihood whether it's recorded as (a, b, 1.0) or
    (b, a, 0.0). fit_ratings only ever uses each player's TOTAL score and
    opponent list (never which color, never game order), so the point
    estimate -- the rating itself -- comes out exactly identical to fitting
    the real, individually-recorded games.

    The bootstrap CI (fit_with_ci) is a valid nonparametric CI from these
    counts, but won't be bit-for-bit identical to the CI the real per-game
    records would have given: its resampling is done by list position within
    a matchup, and this reconstruction necessarily orders the games
    differently than they were actually played (all wins first, then draws,
    then losses) -- the same way re-running fit_with_ci with a different seed
    lands on a different but equally valid CI. In practice the two are close;
    only the rating is guaranteed exact. What's lost for good is the per-game
    detail (PGNs, move latencies, exact color split), not the rating.
    """
    return [(a, b, 1.0)] * wins + [(a, b, 0.5)] * draws + [(b, a, 1.0)] * losses


@dataclass(frozen=True)
class RatingEstimate:
    """One player's fitted rating and how much to trust it."""

    name: str
    rating: float
    ci_low: float
    ci_high: float
    games: int
    score: float            # fraction of points won, 0..1
    at_bound: str | None    # None = rating is pinned by the data. Otherwise it's NOT a real
                            # estimate: "upper" = could be arbitrarily higher (e.g. won
                            # everything), "lower" = arbitrarily lower, "unanchored" = no
                            # usable link to the fixed ratings at all (see identifiability)


def expected_score(rating_a: float | np.ndarray, rating_b: float | np.ndarray) -> float | np.ndarray:
    """Elo formula: A's expected score vs B (0..1). 400 points = 10:1 odds."""
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(rating_b) - np.asarray(rating_a)) / 400.0))


def _index_games(results: Sequence[Result]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Results -> (player names, white idx array, black idx array, white score array)."""
    names = sorted({p for w, b, _ in results for p in (w, b)})
    idx = {n: i for i, n in enumerate(names)}
    w = np.array([idx[r[0]] for r in results], dtype=np.int64)
    b = np.array([idx[r[1]] for r in results], dtype=np.int64)
    s = np.array([r[2] for r in results], dtype=np.float64)
    return names, w, b, s


def identifiability(results: Sequence[Result], fixed_names: Iterable[str]) -> dict[str, str | None]:
    """Which players' ratings the games can actually pin down, relative to the fixed anchors.

    Output: {player: None | "upper" | "lower" | "unanchored"} for every non-fixed player.

    Core logic: draw an arrow X -> Y whenever X took any points off Y (a win or a
    draw). A player's rating is capped from above only if some fixed anchor
    reaches them along the arrows (anchor beat someone who beat someone... who
    took points off them), and floored from below only if they reach a fixed
    anchor. Missing the cap -> "upper"; missing the floor -> "lower"; missing
    both -> "unanchored". This catches the subtle case where a player has mixed
    results but its only route to the scale runs through someone who won or lost
    everything.
    """
    took_points_off: dict[str, set] = defaultdict(set)
    players = set()
    for w, b, s in results:
        players |= {w, b}
        if s > 0:
            took_points_off[w].add(b)
        if s < 1:
            took_points_off[b].add(w)
    reverse: dict[str, set] = defaultdict(set)
    for x, ys in took_points_off.items():
        for y in ys:
            reverse[y].add(x)

    def reach(start, edges):
        seen, stack = set(start), list(start)
        while stack:
            for nxt in edges[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    fixed_here = set(fixed_names) & players
    capped = reach(fixed_here, took_points_off)   # reachable FROM an anchor -> can't be infinitely strong
    floored = reach(fixed_here, reverse)          # can reach an anchor -> can't be infinitely weak
    flags = {}
    for p in players - fixed_here:
        if p in capped and p in floored:
            flags[p] = None
        elif p in floored:
            flags[p] = "upper"
        elif p in capped:
            flags[p] = "lower"
        else:
            flags[p] = "unanchored"
    return flags


def fit_ratings(results: Sequence[Result], fixed: dict[str, float], *,
                margin: float = 3000.0, tol: float = 0.01, max_iter: int = 1000,
                init: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, str | None]]:
    """Maximum-likelihood Elo for every player not in ``fixed``.

    Inputs:
      results -- (white, black, white_score) per game
      fixed   -- known ratings: the pin plus frozen members, e.g.
                 {"sf_1320": 1320, "sf_1500": 1490}; at least one player in the
                 results must be fixed (it anchors the scale)
      margin  -- ratings are searched within [min(fixed) - margin, max(fixed) + margin]
                 (wide on purpose: a random mover sits far below Stockfish's 1320 floor)
      init    -- optional starting guesses (speeds up the bootstrap re-fits)
    Outputs:
      (ratings for all players, flag per free player -- see identifiability())

    Core logic: under the Elo model a player's rating is right when their actual
    total score equals their expected total score against the opponents they
    played. We solve that one equation per free player (bisection -- the
    expected score only ever increases with rating), and cycle through the free
    players until nothing moves (each player's answer depends on the others').
    Some players have no finite answer (e.g. won every game); they drift to the
    edge of the search range, and identifiability() flags them so nobody mistakes
    that number for a real rating.
    """
    names, w, b, s = _index_games(results)
    fixed_here = {n: r for n, r in fixed.items() if n in names}
    if not fixed_here:
        raise ValueError("At least one player in the results must have a fixed rating.")
    free = [n for n in names if n not in fixed_here]
    idx = {n: i for i, n in enumerate(names)}

    lo = min(fixed_here.values()) - margin
    hi = max(fixed_here.values()) + margin
    r = np.full(len(names), float(np.mean(list(fixed_here.values()))))
    for n, v in (init or {}).items():
        if n in idx:
            r[idx[n]] = v
    for n, v in fixed_here.items():
        r[idx[n]] = v

    # Precompute, per free player: which games, who the opponent was, what they scored.
    per_player = {}
    for n in free:
        i = idx[n]
        as_w, as_b = w == i, b == i
        opp = np.concatenate([b[as_w], w[as_b]])
        score = np.concatenate([s[as_w], 1.0 - s[as_b]])
        per_player[n] = (opp, float(score.sum()), len(score))

    for _ in range(max_iter):
        biggest_move = 0.0
        for n in free:
            opp, total, count = per_player[n]
            opp_r = r[opp]
            if total <= 0.0:
                new = lo
            elif total >= count:
                new = hi
            else:
                a, c = lo, hi
                for _ in range(50):  # bisection on expected_total(rating) == actual_total
                    mid = 0.5 * (a + c)
                    if expected_score(mid, opp_r).sum() < total:
                        a = mid
                    else:
                        c = mid
                new = 0.5 * (a + c)
            biggest_move = max(biggest_move, abs(new - r[idx[n]]))
            r[idx[n]] = new
        if biggest_move < tol:
            break
    return {n: float(r[idx[n]]) for n in names}, identifiability(results, fixed_here)


def fit_with_ci(results: Sequence[Result], fixed: dict[str, float], *, n_boot: int = 1000,
                level: float = 0.95, seed: int = 0) -> dict[str, RatingEstimate]:
    """Ratings for every free player, each with a bootstrap confidence interval.

    Inputs: results and fixed ratings as in fit_ratings; n_boot resamples; CI level.
    Output: {player name: RatingEstimate} for the free (non-fixed) players.

    Core logic: re-fit the ratings on many resampled copies of the games. The
    resampling is done within each matchup (e.g. "agent vs sf_1320"), so every
    copy keeps the same mix of opponents -- only the luck of individual results
    varies. The spread of the re-fitted ratings is the CI.
    """
    results = list(results)
    point, bounds = fit_ratings(results, fixed)
    free = [n for n in point if n not in fixed]

    groups: dict[frozenset, list[int]] = defaultdict(list)
    for i, (wn, bn, _) in enumerate(results):
        groups[frozenset((wn, bn))].append(i)
    group_idx = [np.array(g) for g in groups.values()]

    rng = np.random.default_rng(seed)
    samples = {n: [] for n in free}
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in group_idx])
        boot, _ = fit_ratings([results[i] for i in pick], fixed, init=point)
        for n in free:
            samples[n].append(boot[n])

    alpha = (1.0 - level) / 2.0
    out = {}
    for n in free:
        games = [(wn, bn, sc) for wn, bn, sc in results if n in (wn, bn)]
        pts = sum(sc if wn == n else 1.0 - sc for wn, bn, sc in games)
        lo, hi = np.quantile(samples[n], [alpha, 1.0 - alpha])
        out[n] = RatingEstimate(name=n, rating=point[n], ci_low=float(lo), ci_high=float(hi),
                                games=len(games), score=pts / len(games), at_bound=bounds[n])
    return out


def pairs_needing_games(estimates: dict[str, RatingEstimate], results: Sequence[Result],
                        candidates: Iterable[str], *, ci_half_width: float = 100.0) -> set[frozenset]:
    """The adaptive rule (plan v3, Step 5): which matchups need more games.

    Inputs:
      estimates     -- fit_with_ci output for the current games
      results       -- the games so far
      candidates    -- players being rated for freezing (the rule only looks at their matchups)
      ci_half_width -- target: a candidate's 95% CI must be no wider than +/- this
    Output: set of matchups (frozenset of the two names) that need more games.
            Empty set = done.

    Core logic: a candidate whose 95% CI is still wider than the target, or whose
    rating is flagged off the scale, needs more games in ALL its matchups; a
    candidate that meets the target needs none. The CI is the only test because
    it measures the thing we care about -- is the number precise enough to
    freeze. Lopsided matchups (e.g. 90/10) do need more games, but that already
    shows up as a wider CI. A separate "score must be 20-80%" rule was dropped:
    a genuinely large gap never meets it, however many games are played.
    """
    need = set()
    for c in candidates:
        est = estimates[c]
        if est.at_bound is not None or (est.ci_high - est.ci_low) / 2 > ci_half_width:
            need |= {frozenset((w, b)) for w, b, _ in results if c in (w, b)}
    return need


def format_table(estimates: Iterable[RatingEstimate]) -> str:
    """Plain-text table of estimates, strongest first (for notebook printing)."""
    rows = sorted(estimates, key=lambda e: -e.rating)
    lines = [f"{'player':<16}{'elo':>7}{'95% CI':>18}{'games':>7}{'score':>7}  note"]
    for e in rows:
        note = {None: "", "upper": "OFF SCALE: could be much higher (only a lower bound)",
                "lower": "OFF SCALE: could be much lower (only an upper bound)",
                "unanchored": "OFF SCALE: no link to the fixed ratings -- number is meaningless"}[e.at_bound]
        lines.append(f"{e.name:<16}{e.rating:>7.0f}{f'[{e.ci_low:.0f}, {e.ci_high:.0f}]':>18}"
                     f"{e.games:>7}{e.score:>7.2f}  {note}")
    return "\n".join(lines)
