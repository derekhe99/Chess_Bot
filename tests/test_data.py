"""Step 3 checks for data/: download, phase sampling, Stockfish labeling, dataset.

Intent: prove each pipeline piece works on small, fast, offline cases before
spending Colab time on the real Lichess month. Games are generated here and
written as a real .pgn.zst file, so streaming is exercised without the network.
The labeling tests skip (not fail) without the Stockfish binary.

Run from the repo root:
    python -m pytest tests/test_data.py -v
"""
import random

import chess
import chess.pgn
import pandas as pd
import pytest
import zstandard

from data import download
from data.annotate import annotate, load_labels
from data.dataset import PositionDataset, fixed_split, load_datasets, near_win_report, split_by_game
from data.sample_positions import (PHASES, PhaseRule, game_ending, nonpawn_material, phase_quotas, position_key,
                                   sample_positions, tag_phase)


def random_game(seed: int, max_plies: int = 160) -> chess.pgn.Game:
    """A random legal game with Lichess-style headers -- long enough to reach endgames."""
    rng, board = random.Random(seed), chess.Board()
    while not board.is_game_over() and board.ply() < max_plies:
        board.push(rng.choice(list(board.legal_moves)))
    game = chess.pgn.Game.from_board(board)
    game.headers.update({"Site": f"https://lichess.org/g{seed:06d}", "WhiteElo": "1500", "BlackElo": "?"})
    return game


def write_zst(path, games) -> None:
    text = "\n\n".join(str(g) for g in games) + "\n"
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))


# ------------------------------------------------------------- download ----

def test_lichess_names():
    assert download.lichess_filename("2013-06") == "lichess_db_standard_rated_2013-06.pgn.zst"
    assert download.lichess_url("2013-06").endswith("/standard/lichess_db_standard_rated_2013-06.pgn.zst")
    for bad in ("2013-6", "2013-13", "june"):
        with pytest.raises(ValueError):
            download.lichess_filename(bad)


def test_parse_sha256sums():
    text = ("aa" * 32 + "  lichess_db_standard_rated_2013-01.pgn.zst\n"
            + "BB" * 32 + "  lichess_db_standard_rated_2013-06.pgn.zst\n")
    assert download.parse_sha256sums(text, "lichess_db_standard_rated_2013-06.pgn.zst") == "bb" * 32
    with pytest.raises(KeyError):
        download.parse_sha256sums(text, "lichess_db_standard_rated_2099-01.pgn.zst")


def test_ensure_month_uses_cache_and_checks_the_checksum(tmp_path, monkeypatch):
    cached = tmp_path / download.lichess_filename("2013-06")
    write_zst(cached, [random_game(1)])
    monkeypatch.setattr(download, "published_sha256", lambda month: download.sha256_file(cached))
    assert download.ensure_month("2013-06", tmp_path) == cached          # cached file: no download
    monkeypatch.setattr(download, "published_sha256", lambda month: "0" * 64)
    with pytest.raises(RuntimeError):
        download.ensure_month("2013-06", tmp_path)                          # corrupt or different file


def test_iter_games_streams_the_compressed_file(tmp_path):
    path = tmp_path / "games.pgn.zst"
    games = [random_game(s) for s in range(5)]
    write_zst(path, games)
    read = list(download.iter_games(path))
    assert len(read) == 5
    assert [m.uci() for m in read[2].mainline_moves()] == [m.uci() for m in games[2].mainline_moves()]
    assert len(list(download.iter_games(path, max_games=2))) == 2


def test_iter_months_reads_in_order_and_downloads_only_when_reached(tmp_path, monkeypatch):
    files = {}
    for k, month in enumerate(("2013-06", "2013-07", "2013-08")):
        files[month] = tmp_path / download.lichess_filename(month)
        write_zst(files[month], [random_game(10 * k + s) for s in range(3)])
    opened = []
    monkeypatch.setattr(download, "ensure_month", lambda m, d, verify=True: opened.append(m) or files[m])
    used = []
    games = download.iter_months(["2013-06", "2013-07", "2013-08"], tmp_path, used=used)
    first_four = [next(games) for _ in range(4)]
    assert opened == ["2013-06", "2013-07"]                                 # 2013-08 never fetched
    assert [u["month"] for u in used] == ["2013-06", "2013-07"]
    assert used[0]["sha256"] == download.sha256_file(files["2013-06"])
    assert first_four[3].headers["Site"].endswith("g000010")                # first game of 2013-07


# ------------------------------------------------------------- sampling ----

def test_tag_phase():
    rule = PhaseRule(opening_max_ply=20, endgame_max_material=13)
    assert nonpawn_material(chess.Board()) == 62
    assert tag_phase(chess.Board(), rule) == "opening"
    full_material_late = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 15")
    assert tag_phase(full_material_late, rule) == "middlegame"
    rook_vs_rook_and_bishop = chess.Board("4k3/8/8/8/8/8/2r5/R3KB2 w - - 0 40")   # 5 + 3 + 5 = 13
    assert tag_phase(rook_vs_rook_and_bishop, rule) == "endgame"


def test_position_key_ignores_move_counters():
    a = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 30")
    b = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 12 55")
    assert position_key(a) == position_key(b)


def test_phase_quotas_sum_exactly():
    q = phase_quotas(100, {"opening": 1 / 3, "middlegame": 1 / 3, "endgame": 1 / 3})
    assert sum(q.values()) == 100 and max(q.values()) - min(q.values()) <= 1
    with pytest.raises(ValueError):
        phase_quotas(10, {"opening": 0.5, "middlegame": 0.6, "endgame": 0.0})
    with pytest.raises(ValueError):
        phase_quotas(10, {"opening": 1.0})


def test_sample_fills_quotas_with_distinct_legal_positions():
    games = [random_game(s) for s in range(300)]
    mix = {"opening": 0.3, "middlegame": 0.4, "endgame": 0.3}
    records, stats = sample_positions(iter(games), 90, mix, seed=7)
    assert stats["complete"] and stats["counts"] == phase_quotas(90, mix)
    keys = [position_key(chess.Board(r["fen"])) for r in records]
    assert len(set(keys)) == len(keys)                                      # dedup
    for r in records:
        board = chess.Board(r["fen"])
        assert any(board.legal_moves)                                       # always something to label
        assert tag_phase(board) == r["phase"]
    per_game_phase = [(r["game_id"], r["phase"]) for r in records]
    assert len(set(per_game_phase)) == len(per_game_phase)                  # <= 1 per phase per game
    again, _ = sample_positions(iter(games), 90, mix, seed=7)
    assert again == records                                                 # reproducible


def test_game_ending():
    def game(moves, termination="Normal"):
        board = chess.Board()
        for m in moves:
            board.push_san(m)
        g = chess.pgn.Game.from_board(board)
        g.headers["Termination"] = termination
        return g, board
    assert game_ending(*game(["f3", "e5", "g4", "Qh4#"])) == "checkmate"
    assert game_ending(*game(["e4"], "Time forfeit")) == "time"
    assert game_ending(*game(["e4"])) == "resign_or_draw"
    assert game_ending(*game(["e4"], "Abandoned")) == "other"
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert game_ending(chess.pgn.Game(), stalemate) == "stalemate"
    assert game_ending(chess.pgn.Game(), chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")) == "insufficient_material"


def test_sample_records_result_and_ending():
    games = [random_game(s) for s in range(60)]
    records, _ = sample_positions(iter(games), 30, {p: 1 / 3 for p in PHASES}, seed=1)
    by_id = {g.headers["Site"].rsplit("/", 1)[-1]: g for g in games}
    for r in records:
        g = by_id[r["game_id"]]
        assert r["result"] == g.headers["Result"]
        assert r["ending"] == game_ending(g, g.end().board())


def test_sample_skips_excluded_games_and_positions():
    games = [random_game(s) for s in range(300)]
    mix = {"opening": 0.3, "middlegame": 0.4, "endgame": 0.3}
    base, _ = sample_positions(iter(games), 60, mix, seed=7)
    held_games = {r["game_id"] for r in base[:20]}
    held_keys = {position_key(chess.Board(r["fen"])) for r in base[:20]}
    held_keys.add(position_key(chess.Board()))                             # a position every game passes through
    records, stats = sample_positions(iter(games), 90, mix, seed=7, exclude_games=held_games, exclude_keys=held_keys)
    assert stats["complete"] and stats["games_excluded"] == len(held_games)
    assert not {r["game_id"] for r in records} & held_games
    assert not {position_key(chess.Board(r["fen"])) for r in records} & held_keys


def test_sample_reports_a_shortfall_instead_of_failing():
    games = [random_game(s) for s in range(5)]
    records, stats = sample_positions(iter(games), 1000, {p: 1 / 3 for p in PHASES})
    assert not stats["complete"] and stats["games_read"] == 5 and len(records) < 1000


# -------------------------------------------------------------- dataset ----

def _labels_df(n_games=10, per_game=3):
    rows = []
    for g in range(n_games):
        for k in range(per_game):
            rows.append({"fen": chess.STARTING_FEN, "best_move": "e2e4", "win_prob": 0.5 + k / 100,
                         "phase": "opening", "game_id": f"g{g}"})
    return pd.DataFrame(rows)


def test_split_by_game_never_shares_a_game():
    df = _labels_df()
    train, val = split_by_game(df, val_fraction=0.2, seed=1)
    assert len(train) + len(val) == len(df)
    assert set(train["game_id"]).isdisjoint(set(val["game_id"]))
    assert val["game_id"].nunique() == 2
    train2, _ = split_by_game(df, val_fraction=0.2, seed=1)
    assert train.equals(train2)


def _distinct_fen(i: int) -> str:
    """A different position for each i < 1000: black king on h8, white king and rook moved around."""
    board = chess.Board(None)
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(i % 40, chess.Piece(chess.KING, chess.WHITE))           # squares a1..h5
    board.set_piece_at(40 + i // 40 % 14, chess.Piece(chess.ROOK, chess.WHITE))  # a6..f7
    return board.fen()


def test_fixed_split_keeps_the_base_validation_set():
    base = _labels_df(n_games=10, per_game=2)
    base["fen"] = [_distinct_fen(i) for i in range(len(base))]
    _, expected_val = split_by_game(base, 0.2, 0)
    leak_game = expected_val["game_id"].iloc[0]
    new = pd.DataFrame({"fen": [_distinct_fen(100 + i) for i in range(5)] + [expected_val["fen"].iloc[1]],
                        "best_move": "a1a2", "win_prob": 0.5, "phase": "endgame",
                        "game_id": ["n0", "n1", "n2", leak_game, "n4", "n5"]})
    train, val = fixed_split(new, base, 0.2, 0)
    assert val.equals(expected_val)                                         # the same validation set as before
    assert list(train["game_id"]) == ["n0", "n1", "n2", "n4"]               # leaked game and position dropped


def test_near_win_report():
    df = pd.DataFrame({"fen": [chess.STARTING_FEN, chess.STARTING_FEN, "8/8/8/4k3/8/8/8/R3K3 w - - 0 1",
                               "8/8/8/4k3/8/8/8/R3K3 b - - 0 1"],
                       "win_prob": [0.5, 0.95, 0.99, 0.02], "phase": ["opening", "opening", "endgame", "endgame"],
                       "ending": ["time", "checkmate", "checkmate", "resign_or_draw"]})
    r = near_win_report(df)
    assert r.loc["opening", "positions"] == 2 and r.loc["opening", "near_win"] == 0.5
    assert r.loc["endgame", "near_win"] == 1.0 and r.loc["endgame", "simple_ending"] == 1.0
    assert r.loc["all", "positions"] == 4 and r.loc["all", "from_checkmate_game"] == 0.5
    assert "from_checkmate_game" not in near_win_report(df.drop(columns="ending"))


def test_position_dataset_records(tmp_path):
    df = _labels_df(n_games=2, per_game=1)
    item = PositionDataset(df, "structured")[0]
    assert item["board_text"].startswith("Side to move: white")
    assert item["label_move"] == "e2e4" and item["label_value"] == 0.5 and item["phase"] == "opening"
    assert PositionDataset(df, "fen")[0]["board_text"] == chess.STARTING_FEN
    with pytest.raises(ValueError):
        PositionDataset(df, "image")
    df.to_parquet(tmp_path / "labels.parquet", index=False)
    train, val = load_datasets(tmp_path / "labels.parquet", val_fraction=0.5)
    assert len(train) == 1 and len(val) == 1


# ------------------------------------------------------------- labeling ----

try:
    from engine.stockfish import find_stockfish
    find_stockfish()
    HAVE_SF = True
except FileNotFoundError:
    HAVE_SF = False

needs_sf = pytest.mark.skipif(not HAVE_SF, reason="Stockfish binary not found (run the setup notebook)")


@needs_sf
def test_annotate_labels_in_parallel_and_resumes(tmp_path):
    records, _ = sample_positions(iter([random_game(s) for s in range(40)]), 9,
                                  {p: 1 / 3 for p in PHASES}, seed=3)
    manifest = annotate(records, tmp_path, depth=4, workers=2, chunk_size=2, verbose=False)
    assert manifest["rows"] == 9 and manifest["labeled_this_run"] == 9
    assert manifest["stockfish_cpu_seconds"] > 0
    labels = load_labels(tmp_path)
    for _, row in labels.iterrows():
        assert chess.Move.from_uci(row["best_move"]) in chess.Board(row["fen"]).legal_moves
        assert 0.0 <= row["win_prob"] <= 1.0
    first_cpu = manifest["stockfish_cpu_seconds"]
    again = annotate(records, tmp_path, depth=4, workers=2, chunk_size=2, verbose=False)
    assert again["labeled_this_run"] == 0 and again["rows"] == 9           # resumed: nothing redone
    assert again["stockfish_cpu_seconds"] == pytest.approx(first_cpu)


@needs_sf
def test_labels_are_deterministic(tmp_path):
    records, _ = sample_positions(iter([random_game(s) for s in range(20)]), 3,
                                  {p: 1 / 3 for p in PHASES}, seed=5)
    a = annotate(records, tmp_path / "a", depth=6, workers=1, verbose=False)
    b = annotate(list(reversed(records)), tmp_path / "b", depth=6, workers=1, verbose=False)
    la = load_labels(tmp_path / "a").set_index("fen")["best_move"]
    lb = load_labels(tmp_path / "b").set_index("fen")["best_move"]
    assert a["rows"] == b["rows"] == 3 and la.sort_index().equals(lb.sort_index())
