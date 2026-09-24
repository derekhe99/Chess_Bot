"""Thin wrapper around python-chess: create positions, apply moves, list
legal moves, and detect terminal states/outcomes.

Every other module should go through this file for anything that involves
a *rule decision* (what counts as game over, what counts as a legal move
string). Reading plain board state (``board.turn``, ``board.fen()``) straight
from the python-chess object is fine.

Design decisions baked in here:
- Threefold repetition and the 50-move rule END the game automatically
  (normally a player has to claim them), so self-play and eval games can't
  wander forever. They trigger once the repetition / 50 moves has actually
  HAPPENED on the board -- deliberately not python-chess's
  ``outcome(claim_draw=True)``, which also ends the game if the side to move
  merely *could* play into a repetition, and would cut short games where
  the winning side has such a move available. (Fivefold repetition, the
  75-move rule, stalemate, and insufficient material are automatic anyway.)
- Results are reported from White's point of view (+1 / 0 / -1), with
  ``score_for`` to flip to either side (needed for RL rewards).
- A ply cap for runaway games is NOT a chess rule, so it lives in the
  self-play / match config, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

STARTING_FEN = chess.STARTING_FEN


@dataclass(frozen=True)
class GameResult:
    """Final result of a finished game."""

    winner: chess.Color | None  # chess.WHITE, chess.BLACK, or None for a draw
    termination: str            # e.g. "checkmate", "stalemate", "threefold_repetition"
    white_value: int            # +1 White won, 0 draw, -1 Black won


def new_board(fen: str | None = None) -> chess.Board:
    """Starting position, or the position given by ``fen``."""
    return chess.Board(fen) if fen else chess.Board()


def legal_moves(board: chess.Board) -> list[str]:
    """All legal moves in UCI notation, sorted so the order is stable."""
    return sorted(m.uci() for m in board.legal_moves)


def is_legal(board: chess.Board, move: str) -> bool:
    """True if ``move`` is a well-formed UCI string AND legal here.

    Never raises -- the unmasked LLM will produce garbage strings, and this
    is how we check them.
    """
    try:
        return chess.Move.from_uci(move.strip()) in board.legal_moves
    except (ValueError, AttributeError):
        return False


def apply_move(board: chess.Board, move: str) -> None:
    """Play ``move`` (UCI) on ``board`` IN PLACE. Raises ValueError if illegal.

    In place because copying the board every move is wasteful; callers that
    need the old position (e.g. MCTS) should call ``board.copy()`` first.
    """
    if not is_legal(board, move):
        raise ValueError(f"Illegal move {move!r} in position {board.fen()}")
    board.push_uci(move.strip())


def outcome(board: chess.Board) -> GameResult | None:
    """Result of the game, or None if it isn't over."""
    # Automatic endings first (checkmate takes priority over everything).
    o = board.outcome()
    if o is not None:
        if o.winner is None:
            value = 0
        else:
            value = 1 if o.winner == chess.WHITE else -1
        return GameResult(winner=o.winner, termination=o.termination.name.lower(), white_value=value)
    # Draws we treat as automatic -- only once they've actually happened.
    if board.is_repetition(3):
        return GameResult(winner=None, termination="threefold_repetition", white_value=0)
    if board.is_fifty_moves():
        return GameResult(winner=None, termination="fifty_moves", white_value=0)
    return None


def is_terminal(board: chess.Board) -> bool:
    """True if the game is over (see module docstring for the draw rules)."""
    return outcome(board) is not None


def score_for(result: GameResult, color: chess.Color) -> int:
    """Result from ``color``'s point of view: +1 win, 0 draw, -1 loss."""
    return result.white_value if color == chess.WHITE else -result.white_value
