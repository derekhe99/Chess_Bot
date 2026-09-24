"""Canonical move vocabulary: every UCI move that can ever be legal in
standard chess (1968 of them), with move <-> index mapping and a
legal-move mask.

Shared by the AlphaZero policy head, MCTS, and the LLM's masked decoding,
so every agent speaks the same move language.

How the 1968 comes about:
- 1792 "piece" moves: from every square, every destination a queen or a
  knight could reach on an empty board. This covers every non-promotion
  move of every piece, including castling (UCI writes castling as the king
  moving two squares, e.g. e1g1).
- 176 promotions: a pawn stepping or capturing onto the last rank
  (22 from-to pairs per side x 2 sides) x 4 pieces (q, r, b, n).

Moves are in the absolute board frame (White's view) -- nothing is flipped
for the side to move. If a later encoding flips the board for Black, it
must flip moves with it.

FROZEN: the order below defines what every policy-output index means. If it
ever changes, every trained checkpoint silently becomes garbage. The test
suite pins a hash of the vocabulary to catch accidental changes.
"""
from __future__ import annotations

import chess
import numpy as np

_PROMOTION_PIECES = ("q", "r", "b", "n")


def _build_vocab() -> tuple[str, ...]:
    """Takes no input. Returns a tuple of 1968 UCI move strings, e.g.
    ("a1a2", "a1a3", ..., "h2h1n"); a move's position in the tuple is its index.

    Moves are grouped by shape, not piece: every non-promotion move (pawn, king,
    castling included) is a straight line or an L. Promotions are separate only
    because they add a piece letter.
    """
    moves: list[str] = []

    # Straight-line (rank/file/diagonal) or L-shaped from->to pairs: 1792 moves.
    for from_sq in chess.SQUARES:
        ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
        for to_sq in chess.SQUARES:
            if to_sq == from_sq:
                continue
            df = chess.square_file(to_sq) - ff  # file offset
            dr = chess.square_rank(to_sq) - fr  # rank offset
            queen_like = df == 0 or dr == 0 or abs(df) == abs(dr)
            knight = {abs(df), abs(dr)} == {1, 2}
            if queen_like or knight:
                moves.append(chess.square_name(from_sq) + chess.square_name(to_sq))

    # Promotions: 7th->8th rank (White) or 2nd->1st (Black), straight or diagonal,
    # times q/r/b/n: 176 moves. Ranks are 0-indexed.
    for from_rank, to_rank in ((6, 7), (1, 0)):
        for ff in range(8):
            for df in (-1, 0, 1):
                tf = ff + df
                if not 0 <= tf < 8:  # diagonal would leave the board
                    continue
                base = chess.square_name(chess.square(ff, from_rank)) + chess.square_name(chess.square(tf, to_rank))
                moves.extend(base + p for p in _PROMOTION_PIECES)

    return tuple(moves)


MOVES: tuple[str, ...] = _build_vocab()
NUM_MOVES: int = len(MOVES)
MOVE_TO_INDEX: dict[str, int] = {m: i for i, m in enumerate(MOVES)}


def move_to_index(move: str) -> int:
    """Index of a UCI move string. Raises KeyError if it's not in the vocab."""
    try:
        return MOVE_TO_INDEX[move]
    except KeyError:
        raise KeyError(f"{move!r} is not in the move vocabulary") from None


def index_to_move(index: int) -> str:
    """UCI move string for a vocab index."""
    return MOVES[index]


def legal_indices(board: chess.Board) -> np.ndarray:
    """Sorted vocab indices of every legal move in this position."""
    return np.array(sorted(MOVE_TO_INDEX[m.uci()] for m in board.legal_moves), dtype=np.int64)


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean array of shape (NUM_MOVES,): True where the move is legal."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    mask[legal_indices(board)] = True
    return mask
