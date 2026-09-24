"""Step 1 gate checks for engine/.

Run from the repo root:
    python -m pytest tests/test_engine.py -v

The Stockfish tests skip (not fail) if the binary isn't present -- run the
Stockfish cell in notebooks/00_setup.ipynb first so they actually run.
"""
import hashlib
import random

import chess
import numpy as np
import pytest

from engine import board as B
from engine import encoding as E
from engine import moves as M


def random_positions(n_games=20, max_plies=200, seed=0):
    """Yield every position from a handful of seeded random games."""
    rng = random.Random(seed)
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(max_plies):
            yield b.copy()
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(rng.choice(legal))


# --------------------------------------------------------------- moves ----

def test_vocab_size_and_uniqueness():
    assert M.NUM_MOVES == 1968
    assert len(set(M.MOVES)) == 1968


def test_vocab_is_frozen():
    # If this fails, the move order changed and every trained checkpoint's
    # policy indices would silently mean different moves. Don't just update
    # the hash -- make sure the change is intentional.
    digest = hashlib.sha256("\n".join(M.MOVES).encode()).hexdigest()
    assert digest == "79abef8ff5313d331d124b9921af08a1c8eca15a878586a7da403beadfc9606c"


def test_index_round_trip():
    for i in range(M.NUM_MOVES):
        assert M.move_to_index(M.index_to_move(i)) == i


def test_unknown_move_raises():
    with pytest.raises(KeyError):
        M.move_to_index("e2e9")


def test_legal_mask_matches_python_chess():
    n = 0
    for b in random_positions():
        legal = {m.uci() for m in b.legal_moves}
        mask = M.legal_mask(b)
        assert mask.shape == (M.NUM_MOVES,)
        assert mask.sum() == len(legal)
        assert {M.index_to_move(i) for i in np.flatnonzero(mask)} == legal
        n += 1
    assert n > 1000  # sanity: we really checked a lot of positions


@pytest.mark.parametrize("fen, expected", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", {"e1g1", "e1c1"}),         # White castles both ways
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", {"e8g8", "e8c8"}),         # Black castles both ways
    ("1n5k/P7/8/8/8/8/8/K7 w - - 0 1", {"a7a8q", "a7a8n", "a7b8r", "a7b8b"}),  # promotions incl. capture
    ("7k/8/8/8/8/8/1p6/R6K b - - 0 1", {"b2b1q", "b2a1n"}),              # Black promotion incl. capture
    ("k7/8/8/3pP3/8/8/8/K7 w - d6 0 1", {"e5d6"}),                       # en passant
])
def test_special_moves_in_vocab_and_mask(fen, expected):
    b = chess.Board(fen)
    mask = M.legal_mask(b)
    for mv in expected:
        assert mv in {m.uci() for m in b.legal_moves}, f"{mv} should be legal in {fen}"
        assert mask[M.move_to_index(mv)]


# --------------------------------------------------------------- board ----

def test_new_board_and_legal_moves():
    b = B.new_board()
    moves = B.legal_moves(b)
    assert len(moves) == 20 and moves == sorted(moves)
    assert B.new_board("8/8/8/8/8/8/8/K6k w - - 0 1").fen().startswith("8/8/8/8/8/8/8/K6k")


def test_is_legal_never_raises():
    b = B.new_board()
    assert B.is_legal(b, "e2e4")
    assert B.is_legal(b, " e2e4 ")
    assert not B.is_legal(b, "e2e5")
    assert not B.is_legal(b, "hello")
    assert not B.is_legal(b, "")
    assert not B.is_legal(b, None)


def test_apply_move():
    b = B.new_board()
    B.apply_move(b, "e2e4")
    assert b.turn == chess.BLACK
    with pytest.raises(ValueError):
        B.apply_move(b, "e2e4")  # that pawn already moved


def test_checkmate():
    b = B.new_board()
    for mv in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate
        B.apply_move(b, mv)
    assert B.is_terminal(b)
    r = B.outcome(b)
    assert r.winner == chess.BLACK and r.termination == "checkmate" and r.white_value == -1
    assert B.score_for(r, chess.BLACK) == 1 and B.score_for(r, chess.WHITE) == -1


def test_stalemate():
    b = B.new_board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    r = B.outcome(b)
    assert r.winner is None and r.termination == "stalemate" and r.white_value == 0


def test_threefold_repetition_ends_game():
    b = B.new_board()
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for mv in shuffle * 2:
        assert not B.is_terminal(b)
        B.apply_move(b, mv)
    # start position has now occurred 3 times -> claimable -> we treat as over
    assert B.is_terminal(b)
    assert B.outcome(b).termination == "threefold_repetition"


def test_fifty_move_rule_ends_game():
    b = B.new_board("k7/8/8/8/8/8/8/KR6 w - - 100 80")
    assert B.is_terminal(b)
    assert B.outcome(b).termination == "fifty_moves"


def test_draw_not_declared_a_move_early():
    # python-chess's claim_draw=True would already call these draws, because
    # a move exists that *would* reach the repetition / 50th move. We don't.
    b = B.new_board()
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        B.apply_move(b, mv)
    assert not B.is_terminal(b)
    assert not B.is_terminal(B.new_board("k7/8/8/8/8/8/8/KR6 w - - 99 80"))


def test_ongoing_game_has_no_outcome():
    assert B.outcome(B.new_board()) is None


# ------------------------------------------------------------ encoding ----

def test_fen_round_trip():
    for b in random_positions(n_games=3):
        assert chess.Board(E.to_fen(b)).fen() == b.fen()


def test_structured_text_lists_every_square_correctly():
    for b in random_positions(n_games=3):
        text = E.to_structured_text(b)
        cells = [tok for line in text.splitlines()[3:] for tok in line.split()[1:]]
        assert len(cells) == 64
        for cell in cells:
            name, sym = cell.split("=")
            piece = b.piece_at(chess.parse_square(name))
            assert sym == (piece.symbol() if piece else ".")


def test_structured_text_metadata():
    b = B.new_board()
    B.apply_move(b, "e2e4")
    lines = E.to_structured_text(b).splitlines()
    assert lines[0] == "Side to move: black"
    assert lines[1] == "Castling: KQkq"
    assert lines[2] == "En passant: -"  # no black pawn can capture on e3
    assert lines[3].startswith("8: a8=r")


def test_encode_text_registry():
    b = B.new_board()
    assert E.encode_text(b, "fen") == b.fen()
    assert E.encode_text(b, "structured") == E.to_structured_text(b)
    with pytest.raises(ValueError):
        E.encode_text(b, "image")


def test_planes_start_position():
    p = E.to_planes(B.new_board())
    assert p.shape == (E.NUM_PLANES, 8, 8) and p.dtype == np.float32
    assert p[0, 1].sum() == 8           # 8 white pawns on rank 2
    assert p[6, 6].sum() == 8           # 8 black pawns on rank 7
    assert p[5, 0, 4] == 1              # white king on e1
    assert p[11, 7, 4] == 1             # black king on e8
    assert p[12].all()                  # white to move
    assert p[13:17].all()               # all castling rights
    assert p[17].sum() == 0 and p[18].sum() == 0


def test_planes_piece_counts_match_board():
    for b in random_positions(n_games=3):
        p = E.to_planes(b)
        assert p[:12].sum() == len(b.piece_map())


def test_planes_side_to_move_castling_and_ep():
    b = B.new_board("k7/8/8/3pP3/8/8/8/K7 w - d6 0 1")
    p = E.to_planes(b)
    assert p[17, 5, 3] == 1             # ep target d6
    assert p[13:17].sum() == 0          # no castling rights
    B.apply_move(b, "a1b1")
    assert E.to_planes(b)[12].sum() == 0  # black to move


# ----------------------------------------------------------- stockfish ----

try:
    from engine.stockfish import Stockfish, find_stockfish
    find_stockfish()
    HAVE_SF = True
except (FileNotFoundError, ImportError):
    HAVE_SF = False

needs_sf = pytest.mark.skipif(not HAVE_SF, reason="Stockfish binary not found (run the setup notebook)")


@pytest.fixture(scope="module")
def sf():
    with Stockfish() as engine:
        yield engine


@needs_sf
def test_finds_mate_in_one(sf):
    a = sf.analyse(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"), depth=10)
    assert a.best_move == "a1a8"
    assert a.mate == 1 and a.cp is None and a.win_prob > 0.99


@needs_sf
def test_best_move_and_eval_for_any_fen(sf):
    positions = [b for b in random_positions(n_games=5, max_plies=60, seed=1) if any(b.legal_moves)]
    for b in positions[::10]:
        a = sf.analyse(b, depth=8)
        assert a.best_move in {m.uci() for m in b.legal_moves}
        assert (a.cp is None) != (a.mate is None)  # exactly one kind of score
        assert 0.0 <= a.win_prob <= 1.0
        assert a.depth is not None


@needs_sf
def test_eval_is_from_side_to_move(sf):
    # Same material (White up a queen), flip who's to move: the score must
    # flip sign, because it's always from the side to move's point of view.
    black_to_move = sf.analyse(chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1"), depth=10)
    white_to_move = sf.analyse(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"), depth=10)
    assert black_to_move.win_prob < 0.5 < white_to_move.win_prob


@needs_sf
def test_labels_are_reproducible(sf):
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    first = sf.analyse(b, depth=12)
    sf.analyse(chess.Board(), depth=12)  # something else in between
    assert sf.analyse(b, depth=12) == first


@needs_sf
def test_limited_strength_play():
    with Stockfish(elo=1500) as weak:
        assert weak.elo == 1500
        b = chess.Board()
        assert weak.play(b, time=0.05) in {m.uci() for m in b.legal_moves}
        with pytest.raises(ValueError):
            weak.set_strength(800)  # below the floor -> use weak anchors instead
        weak.set_strength(None)
        assert weak.elo is None


@needs_sf
def test_depth_one_play(sf):
    b = chess.Board()
    assert sf.play(b, depth=1) in {m.uci() for m in b.legal_moves}


@needs_sf
def test_finished_position_raises(sf):
    with pytest.raises(ValueError):
        sf.analyse(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), depth=5)  # stalemate


@needs_sf
def test_cpu_time_is_tracked():
    engine = Stockfish()
    before = engine.cpu_seconds
    engine.analyse(chess.Board(), depth=16)
    during = engine.cpu_seconds
    engine.close()
    assert during > before
    assert engine.cpu_seconds == during or engine.cpu_seconds >= during  # frozen after close
    assert engine.version.startswith("Stockfish")
