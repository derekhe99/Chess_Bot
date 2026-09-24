"""Starts and manages a Stockfish process.

Two jobs:
1. Labeling (``analyse``): best move + evaluation for a position, at a fixed
   depth or node count. The evaluation is also converted to a win
   probability -- the SFT value target.
2. Playing (``play``): a move at full strength, capped depth, or a limited
   ``UCI_Elo`` -- the eval anchors.

It also tracks the CPU time the engine process has used (``cpu_seconds``) so
labeling cost can be charged to the compute budget, per the plan.

Usage:
    with Stockfish() as sf:
        a = sf.analyse(board, depth=12)
        a.best_move, a.cp, a.win_prob, sf.cpu_seconds

One Stockfish object = one engine process. For parallel labeling (Step 3),
run several, each with threads=1.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import psutil

# Where notebooks/00_setup.ipynb puts the binary, relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BINARY = _REPO_ROOT / "stockfish" / "stockfish-linux-x86-64-universal"

# Converts centipawns to a win probability. "lichess" is the logistic curve
# Lichess uses for its win% display (the same formula DeepMind's
# searchless-chess paper used for its value targets). It ignores draws, so
# the result is best read as "expected score".
WIN_PROB_MODEL = "lichess"


def find_stockfish(path: str | os.PathLike | None = None) -> str:
    """Locate the Stockfish binary.

    Order: explicit ``path`` -> $STOCKFISH_PATH -> <repo>/stockfish/... (where
    the setup notebook downloads it) -> ``stockfish`` on PATH.
    """
    candidates = [path, os.environ.get("STOCKFISH_PATH"), _DEFAULT_BINARY, shutil.which("stockfish")]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    raise FileNotFoundError(
        "Stockfish binary not found. Run the Stockfish cell in notebooks/00_setup.ipynb, "
        "or pass path=... / set STOCKFISH_PATH."
    )


@dataclass(frozen=True)
class Analysis:
    """Engine verdict on one position. Every score is from the point of view
    of the side to move (positive = good for whoever is about to move)."""

    best_move: str        # UCI
    cp: int | None        # centipawns; None when a forced mate was found
    mate: int | None      # moves to mate (+ = side to move mates, - = gets mated); None otherwise
    win_prob: float       # 0..1, from WIN_PROB_MODEL
    depth: int | None     # depth actually searched
    nodes: int | None     # nodes actually searched


def _limit(depth: int | None, nodes: int | None, time: float | None) -> chess.engine.Limit:
    if depth is None and nodes is None and time is None:
        raise ValueError("Give at least one of depth=, nodes=, time= (seconds)")
    return chess.engine.Limit(depth=depth, nodes=nodes, time=time)


class Stockfish:
    """One Stockfish process. Use as a context manager, or call ``close()``."""

    def __init__(self, path: str | None = None, threads: int = 1, hash_mb: int = 16,
                 elo: int | None = None):
        self.path = find_stockfish(path)
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self._engine.configure({"Threads": threads, "Hash": hash_mb})
        self._proc = psutil.Process(self._engine.transport.get_pid())
        self._final_cpu: float | None = None
        self.version: str = self._engine.id.get("name", "unknown")
        self.elo: int | None = None
        self.set_strength(elo)

    # ---- strength ----------------------------------------------------------

    @property
    def elo_range(self) -> tuple[int, int]:
        """(min, max) UCI_Elo this Stockfish build accepts. Min is ~1320,
        which is why the plan adds weaker anchors below it."""
        opt = self._engine.options["UCI_Elo"]
        return int(opt.min), int(opt.max)

    def set_strength(self, elo: int | None) -> None:
        """Limit playing strength to ``elo``, or ``None`` for full strength.

        Note: Stockfish calibrates UCI_Elo against time-based search, so when
        playing at a limited Elo, prefer ``play(board, time=...)`` over a
        depth limit.
        """
        if elo is None:
            self._engine.configure({"UCI_LimitStrength": False})
        else:
            lo, hi = self.elo_range
            if not lo <= elo <= hi:
                raise ValueError(f"UCI_Elo must be in [{lo}, {hi}], got {elo}. "
                                 "Use the weak anchors in eval/anchors.py for lower ratings.")
            self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self.elo = elo

    # ---- the two jobs ------------------------------------------------------

    def analyse(self, board: chess.Board, *, depth: int | None = None, nodes: int | None = None,
                time: float | None = None, fresh: bool = True) -> Analysis:
        """Label a position: best move + evaluation.

        ``fresh=True`` clears the engine's hash table first, so a position's
        label doesn't depend on which positions were analysed before it
        (keeps labels reproducible regardless of order).
        """
        if not any(board.legal_moves):
            raise ValueError(f"No legal moves to analyse (checkmate/stalemate): {board.fen()}")
        info = self._engine.analyse(board, _limit(depth, nodes, time),
                                    game=object() if fresh else None)
        pv = info.get("pv")
        best = pv[0].uci() if pv else self.play(board, depth=depth, nodes=nodes, time=time)
        score = info["score"].relative
        return Analysis(
            best_move=best,
            cp=score.score(),
            mate=score.mate(),
            win_prob=score.wdl(model=WIN_PROB_MODEL).expectation(),
            depth=info.get("depth"),
            nodes=info.get("nodes"),
        )

    def play(self, board: chess.Board, *, depth: int | None = None, nodes: int | None = None,
             time: float | None = None) -> str:
        """Pick a move (UCI) at the current strength setting."""
        if not any(board.legal_moves):
            raise ValueError(f"No legal moves to play (checkmate/stalemate): {board.fen()}")
        return self._engine.play(board, _limit(depth, nodes, time)).move.uci()

    # ---- compute accounting ------------------------------------------------

    @property
    def cpu_seconds(self) -> float:
        """Total CPU seconds (user + system, all threads) used by this engine
        process since it started -- includes the one-off NNUE network load."""
        if self._final_cpu is not None:
            return self._final_cpu
        t = self._proc.cpu_times()
        return t.user + t.system

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._final_cpu is None:
            self._final_cpu = self.cpu_seconds  # read before the process exits
            self._engine.quit()

    def __enter__(self) -> "Stockfish":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
