"""Self-play: one player plays both sides, and every move is recorded for training.

Intent: produce the raw material of RL -- complete games where each position,
the move chosen there, and the game's final result are kept together. The
player is anything that implements a *move picker*: ``pick(board) -> (move,
info)``. The LLM (Step 7) picks by sampling from its masked move
probabilities; AlphaZero (Step 9) will pick from an MCTS search and put its
visit counts in ``info``. This file knows nothing about either.

Main pieces:
- MoveRecord        -- one move: position (FEN), move, who played it, the picker's info
- SelfPlayGame      -- one finished game: its moves, result, and how it ended
- play_selfplay_game  -- one game from a given opening
- play_selfplay_games -- n games, each from its own random short opening

Rules come from engine/board.py, exactly as in eval (repetition and the
50-move rule end the game; a ply cap adjudicates a draw). Opening moves are
played for variety and are NOT recorded as training moves.

Games are played one at a time. Batching the moves of many games into one GPU
call is the first speedup to make if self-play time dominates a round
(rl_log.csv separates self-play seconds from update seconds to tell).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import chess

from engine import board as B
from eval.match import random_opening

# pick(board) -> (UCI move, info). The board is a copy; info is any per-move data
# the trainer wants back (e.g. the value estimate, the move distribution's entropy).
MovePicker = Callable[[chess.Board], tuple[str, dict]]


@dataclass
class MoveRecord:
    """One recorded move. ``white`` = True if White played it."""

    fen: str
    move: str
    white: bool
    info: dict = field(default_factory=dict)


@dataclass
class SelfPlayGame:
    """One finished self-play game."""

    moves: list[MoveRecord]        # the player's moves only (not the opening)
    opening: list[str]
    white_score: float             # 1 White won, 0.5 draw, 0 Black won
    termination: str               # "checkmate", "threefold_repetition", "max_plies", ...
    plies: int                     # total half-moves, opening included
    final_fen: str = ""            # the position the game ended in (what Stockfish end scoring judges)

    def score_for(self, white: bool) -> float:
        """The result from one side's point of view: 1 win, 0.5 draw, 0 loss."""
        return self.white_score if white else 1.0 - self.white_score


def play_selfplay_game(pick: MovePicker, *, opening: list[str] | None = None,
                       max_plies: int = 300) -> SelfPlayGame:
    """Play one game with ``pick`` choosing every move for both sides.

    The picker must return a legal move (masked LLM and MCTS always do); an
    illegal one raises, because silently fixing it would corrupt training data.
    """
    board = B.new_board()
    opening = list(opening or [])
    for mv in opening:
        B.apply_move(board, mv)
    records: list[MoveRecord] = []
    while not B.is_terminal(board) and board.ply() < max_plies:
        move, info = pick(board.copy())
        if not B.is_legal(board, move):
            raise ValueError(f"picker returned illegal move {move!r} in {board.fen()}")
        records.append(MoveRecord(board.fen(), move, board.turn == chess.WHITE, dict(info)))
        B.apply_move(board, move)
    result = B.outcome(board)
    if result is None:
        white_score, termination = 0.5, "max_plies"
    else:
        white_score, termination = (result.white_value + 1) / 2, result.termination
    return SelfPlayGame(records, opening, white_score, termination, board.ply(), board.fen())


def play_selfplay_games(pick: MovePicker, n_games: int, *, opening_plies: int = 4, max_plies: int = 300,
                        seed: int = 0, log: Callable[[str], None] | None = None) -> list[SelfPlayGame]:
    """Play ``n_games`` games, each from its own seeded random opening of ``opening_plies`` half-moves."""
    rng = random.Random(seed)
    games = []
    for i in range(n_games):
        opening = random_opening(rng, opening_plies) if opening_plies else []
        games.append(play_selfplay_game(pick, opening=opening, max_plies=max_plies))
        if log is not None:
            g = games[-1]
            log(f"  self-play game {i + 1}/{n_games}: {g.white_score} ({g.termination}, {g.plies} plies)")
    return games
