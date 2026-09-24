"""Board -> model input. Three representations:

- ``fen``        -- the standard FEN string (LLM, "FEN" arm of the toggle)
- ``structured`` -- every square written out explicitly (LLM, "structured"
                    arm of the toggle)
- planes         -- stacked 8x8 float arrays for the AlphaZero CNN

The two text encoders live in ``TEXT_ENCODERS`` so a config value
(``"fen"`` / ``"structured"``) picks one via ``encode_text``.

PROVISIONAL: the exact structured-text format is still an open decision in
the plan (Sec 0). The version here is a reasonable default -- lock or change
it before Step 6, because it changes prompt length (and so inference cost).
"""
from __future__ import annotations

from typing import Callable

import chess
import numpy as np

# ---------------------------------------------------------------- text ----

def to_fen(board: chess.Board) -> str:
    """Standard FEN. En passant square only shown if a capture is legal."""
    return board.fen()


def _castling_str(board: chess.Board) -> str:
    s = ""
    if board.has_kingside_castling_rights(chess.WHITE):
        s += "K"
    if board.has_queenside_castling_rights(chess.WHITE):
        s += "Q"
    if board.has_kingside_castling_rights(chess.BLACK):
        s += "k"
    if board.has_queenside_castling_rights(chess.BLACK):
        s += "q"
    return s or "-"


def _legal_ep_square(board: chess.Board) -> int | None:
    # python-chess sets ep_square after ANY double pawn push; FEN (and we)
    # only report it when an en passant capture is actually legal.
    return board.ep_square if board.has_legal_en_passant() else None

def to_structured_text(board: chess.Board) -> str:
    """Every square spelled out, one line per rank (8 down to 1).

    Uppercase = White, lowercase = Black, '.' = empty. The point versus FEN:
    FEN compresses empty runs into digits ("4P3"), so the model has to count
    to know where a piece is. Here every square is named explicitly.

    Example (start position, first lines):
        Side to move: white
        Castling: KQkq
        En passant: -
        8: a8=r b8=n c8=b d8=q e8=k f8=b g8=n h8=r
        7: a7=p b7=p ...
    """
    ep = _legal_ep_square(board)
    lines = [
        f"Side to move: {'white' if board.turn == chess.WHITE else 'black'}",
        f"Castling: {_castling_str(board)}",
        f"En passant: {chess.square_name(ep) if ep is not None else '-'}",
    ]
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            cells.append(f"{chess.square_name(sq)}={piece.symbol() if piece else '.'}")
        lines.append(f"{rank + 1}: " + " ".join(cells))
    return "\n".join(lines)


TEXT_ENCODERS: dict[str, Callable[[chess.Board], str]] = {
    "fen": to_fen,
    "structured": to_structured_text,
}


def encode_text(board: chess.Board, kind: str) -> str:
    """Text representation by config name: 'fen' or 'structured'."""
    try:
        return TEXT_ENCODERS[kind](board)
    except KeyError:
        raise ValueError(f"Unknown text encoding {kind!r}; choose from {sorted(TEXT_ENCODERS)}") from None


# -------------------------------------------------------------- planes ----

_PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# Plane layout (index: meaning)
#   0-5   White P N B R Q K       6-11  Black p n b r q k
#   12    side to move (all 1s if White to move)
#   13-16 castling rights: White K-side, White Q-side, Black K-side, Black Q-side (all 1s if held)
#   17    en passant target square (one-hot, only if a capture is legal)
#   18    halfmove clock / 100 (constant plane; how close the 50-move rule is)
# Cell [plane, r, f] is rank r+1, file f (a=0) -- absolute frame, not flipped
# for the side to move, matching engine/moves.py.
# No move-history planes (AlphaZero used 8 past positions); skipped for the
# small compute budget -- add later if the CNN needs repetition awareness.
NUM_PLANES = 19


def to_planes(board: chess.Board) -> np.ndarray:
    """float32 array of shape (NUM_PLANES, 8, 8). See layout above."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for color_offset, color in ((0, chess.WHITE), (6, chess.BLACK)):
        for i, piece_type in enumerate(_PIECE_ORDER):
            for sq in board.pieces(piece_type, color):
                planes[color_offset + i, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    if board.turn == chess.WHITE:
        planes[12] = 1.0
    for i, (color, kingside) in enumerate(
        ((chess.WHITE, True), (chess.WHITE, False), (chess.BLACK, True), (chess.BLACK, False))
    ):
        has_right = (
            board.has_kingside_castling_rights(color) if kingside
            else board.has_queenside_castling_rights(color)
        )
        if has_right:
            planes[13 + i] = 1.0

    ep = _legal_ep_square(board)
    if ep is not None:
        planes[17, chess.square_rank(ep), chess.square_file(ep)] = 1.0

    planes[18] = min(board.halfmove_clock, 100) / 100.0
    return planes
