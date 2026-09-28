"""Downloads one month of Lichess games and streams them out one game at a time.

Intent: SFT needs positions that real players actually reach. Lichess publishes
every rated game, one compressed file per month. Plan v3 (Sec 3.2) settles on a
single older month, pinned by name in configs/data.yaml, so the same file --
and so the same games -- is used every time the dataset is rebuilt.

Main pieces:
- lichess_filename / lichess_url -- month ("2013-06") -> file name / download URL
- parse_sha256sums   -- pull one file's checksum out of Lichess's sha256sums.txt
- ensure_month       -- download the month once (cached, e.g. on Drive) and check it
                        against Lichess's published checksum
- iter_games         -- stream games out of the .pgn.zst file without ever
                        decompressing the whole file into memory
"""
from __future__ import annotations

import hashlib
import io
import re
import urllib.request
from pathlib import Path
from typing import Iterator

import chess.pgn
import zstandard

LICHESS_BASE = "https://database.lichess.org/standard"
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def lichess_filename(month: str) -> str:
    """'2013-06' -> 'lichess_db_standard_rated_2013-06.pgn.zst' (rated standard games only)."""
    if not _MONTH.match(month):
        raise ValueError(f"month must look like '2013-06', got {month!r}")
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def lichess_url(month: str) -> str:
    """Download URL for one month's file."""
    return f"{LICHESS_BASE}/{lichess_filename(month)}"


def parse_sha256sums(text: str, filename: str) -> str:
    """Find ``filename``'s checksum in the text of Lichess's sha256sums.txt.

    Each line there is '<64 hex chars>  <filename>'. Raises if the file isn't listed.
    """
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == filename:
            return parts[0].lower()
    raise KeyError(f"{filename} is not listed in sha256sums.txt")


def published_sha256(month: str) -> str:
    """Lichess's own checksum for the month's file (fetched from the database site)."""
    with urllib.request.urlopen(f"{LICHESS_BASE}/sha256sums.txt", timeout=60) as r:
        return parse_sha256sums(r.read().decode("utf-8"), lichess_filename(month))


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, read in 1 MB pieces (never loads the whole file)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_month(month: str, cache_dir: str | Path, *, verify: bool = True) -> Path:
    """Make sure the month's file is in ``cache_dir`` (downloading it if not) and return its path.

    Inputs:  month like "2013-06"; cache_dir (e.g. <Drive>/data/raw); verify -- check
             the file against Lichess's published SHA-256.
    Output:  path to the .pgn.zst file.

    Core logic: download to '<name>.part' first and rename only when the download
    finished, so an interrupted session never leaves a truncated file that looks
    complete. Then compare checksums: a mismatch means a corrupt or different
    file, which would silently change the dataset -- so it raises.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / lichess_filename(month)
    if not path.exists():
        tmp = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(lichess_url(month), tmp)
        tmp.replace(path)
    if verify:
        expected, actual = published_sha256(month), sha256_file(path)
        if expected != actual:
            raise RuntimeError(f"{path.name}: checksum {actual} != published {expected}. "
                               "Delete the file and download it again.")
    return path


def iter_games(path: str | Path, max_games: int | None = None) -> Iterator[chess.pgn.Game]:
    """Yield games from a .pgn.zst file one at a time, decompressing as it goes.

    Inputs: path to the file; max_games -- stop early (None = the whole file).
    Output: chess.pgn.Game objects, in file order (so a fixed seed gives a fixed sample).
    Games python-chess couldn't fully parse are skipped.
    """
    count = 0
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                encoding="utf-8", errors="replace")
        while max_games is None or count < max_games:
            game = chess.pgn.read_game(text)
            if game is None:
                return
            if game.errors:
                continue
            count += 1
            yield game
