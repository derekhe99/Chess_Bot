"""Picks the positions the model will learn from, with a controlled mix of game phases.

Intent: in real games most positions are early -- every game has an opening,
fewer reach an endgame. Sampling positions uniformly would give a model that
mostly practiced openings. So each position is tagged opening / middlegame /
endgame, and the sample is filled to a target share per phase (configs/data.yaml).
Repeated positions (the same opening reached in many games) are kept only once.

Main pieces:
- PhaseRule / tag_phase  -- the rule that labels a position's phase
- position_key           -- what counts as "the same position" for dedup
- phase_quotas           -- target mix -> exact number of positions per phase
- sample_positions       -- walk games, fill each phase's quota, return records + stats
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import chess
import chess.pgn

PHASES = ("opening", "middlegame", "endgame")
_NONPAWN_VALUE = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


@dataclass(frozen=True)
class PhaseRule:
    """Thresholds for tagging a position's phase (a simple, common heuristic).

    endgame_max_material -- endgame if the non-pawn material left on the board
                            (both sides, N/B=3, R=5, Q=9; the start position has 62)
                            is at most this. Checked first.
    opening_max_ply      -- otherwise opening if fewer than this many half-moves
                            have been played; otherwise middlegame.
    """

    opening_max_ply: int = 20
    endgame_max_material: int = 13


def nonpawn_material(board: chess.Board) -> int:
    """Total knight/bishop/rook/queen value left on the board, both sides."""
    return sum(v * len(board.pieces(pt, color)) for pt, v in _NONPAWN_VALUE.items()
               for color in (chess.WHITE, chess.BLACK))


def tag_phase(board: chess.Board, rule: PhaseRule = PhaseRule()) -> str:
    """'opening', 'middlegame', or 'endgame' for this position."""
    if nonpawn_material(board) <= rule.endgame_max_material:
        return "endgame"
    return "opening" if board.ply() < rule.opening_max_ply else "middlegame"


def position_key(board: chess.Board) -> str:
    """The position for dedup purposes: pieces, side to move, castling, en passant.

    Leaves out the move counters, so the same position reached at different
    move numbers still counts as one.
    """
    return " ".join(board.fen().split()[:4])


def phase_quotas(num_positions: int, phase_mix: dict[str, float]) -> dict[str, int]:
    """Target fractions -> whole-number counts that add up to exactly ``num_positions``.

    Rounds each share down, then hands the leftover positions to the phases
    with the largest remainders.
    """
    if set(phase_mix) != set(PHASES):
        raise ValueError(f"phase_mix must have exactly the keys {PHASES}, got {sorted(phase_mix)}")
    total = sum(phase_mix.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"phase_mix must sum to 1, got {total}")
    raw = {p: num_positions * phase_mix[p] / total for p in PHASES}
    counts = {p: int(raw[p]) for p in PHASES}
    for p in sorted(PHASES, key=lambda p: raw[p] - counts[p], reverse=True)[: num_positions - sum(counts.values())]:
        counts[p] += 1
    return counts


def _game_id(game: chess.pgn.Game, fallback: int) -> str:
    """Lichess game id from the Site header (https://lichess.org/<id>), else a counter."""
    site = game.headers.get("Site", "")
    return site.rstrip("/").rsplit("/", 1)[-1] if "lichess.org/" in site else f"game{fallback}"


def _elo(game: chess.pgn.Game, side: str) -> int | None:
    """A player's rating from the headers, or None if missing ('?')."""
    value = game.headers.get(f"{side}Elo", "")
    return int(value) if value.isdigit() else None


def sample_positions(games: Iterable[chess.pgn.Game], num_positions: int, phase_mix: dict[str, float],
                     rule: PhaseRule = PhaseRule(), seed: int = 0) -> tuple[list[dict], dict]:
    """Fill a per-phase quota of distinct positions from a stream of games.

    Inputs:
      games         -- e.g. download.iter_games(path)
      num_positions -- total positions wanted
      phase_mix     -- {"opening": .., "middlegame": .., "endgame": ..}, sums to 1
      rule          -- phase thresholds
      seed          -- same games + same seed -> same sample
    Output:
      records -- one dict per position: fen, phase, ply, game_id, white_elo, black_elo
      stats   -- games read, targets vs. counts per phase, duplicates skipped, and
                 whether every quota was filled (``complete``)

    Core logic: for each game, list the positions it passed through (each one
    right before a move was played, so it always has a legal move to label) and
    group them by phase. Then take at most ONE random position per phase from
    that game, skipping positions already sampled from another game. Stop as
    soon as every phase's quota is full. One-per-phase-per-game spreads the
    sample across many games instead of many near-identical positions from a few.
    """
    rng = random.Random(seed)
    targets = phase_quotas(num_positions, phase_mix)
    buckets: dict[str, list[dict]] = {p: [] for p in PHASES}
    seen: set[str] = set()
    games_read = duplicates = 0

    for game in games:
        if all(len(buckets[p]) >= targets[p] for p in PHASES):
            break
        games_read += 1
        board = game.board()
        by_phase: dict[str, list[tuple[str, str, int]]] = {p: [] for p in PHASES}
        for move in game.mainline_moves():
            by_phase[tag_phase(board, rule)].append((position_key(board), board.fen(), board.ply()))
            board.push(move)
        gid = _game_id(game, games_read)
        for p in PHASES:
            if len(buckets[p]) >= targets[p] or not by_phase[p]:
                continue
            candidates = by_phase[p][:]
            rng.shuffle(candidates)
            for key, fen, ply in candidates:
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                buckets[p].append({"fen": fen, "phase": p, "ply": ply, "game_id": gid,
                                   "white_elo": _elo(game, "White"), "black_elo": _elo(game, "Black")})
                break

    records = [r for p in PHASES for r in buckets[p]]
    stats = {"games_read": games_read, "duplicates_skipped": duplicates,
             "targets": targets, "counts": {p: len(buckets[p]) for p in PHASES},
             "complete": all(len(buckets[p]) >= targets[p] for p in PHASES)}
    return records, stats
