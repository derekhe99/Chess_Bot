"""The reference set: the frozen opponents every agent is rated against.

Intent: a set of opponents whose ratings never change, so an agent's Elo can be
read off from how it scores against them (plan v4, Sec 0 "Eval"). Exactly one
rating is pinned by definition: Stockfish UCI_Elo 1320 = 1320. Every other
member is rated once -- from games among the members -- and then frozen:
Stockfish 1500 and 1700 in Step 2; random, greedy, and the SFT checkpoint
step_20000 in Step 5 (random and greedy lose every game to Stockfish 1320,
so they couldn't be rated until a checkpoint bridged the gap).

Main pieces:
- RandomAgent        -- plays a uniformly random legal move
- GreedyAgent        -- takes the most valuable piece it can capture, else random
- StockfishAgent     -- Stockfish at a chosen strength AND search limit
- AnchorSpec         -- plain-data description of one member (swap settings here)
- REFERENCE_SET      -- the starting members; SMOKE_TEST_OPPONENT -- Stockfish
                        depth 1, used only to smoke-test the harness
- build_agents / close_agents -- turn specs into playable agents, and shut them down
- fixed_ratings      -- pinned ratings + frozen ratings: what the Elo fit holds fixed
- build_reference_agents -- playable agents for reference rows (incl. checkpoint rungs)
- ResultsTable       -- results/results.csv, the one file of record: the pin, the frozen
                        reference set, and every evaluated model (plan v4, Sec 2.2)

Every member follows the harness's Agent interface (eval/match.py): a ``name``
and ``select_move(board) -> UCI string``.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess

from engine.stockfish import SF_VERSION, Stockfish

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class RandomAgent:
    """Plays a uniformly random legal move -- the weakest member of the set."""

    def __init__(self, name: str = "random", seed: int = 0):
        """Seeded so a given game can be replayed exactly."""
        self.name = name
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> str:
        """Any legal move, uniformly at random."""
        return self._rng.choice(list(board.legal_moves)).uci()


class GreedyAgent:
    """Captures the most valuable piece available; otherwise plays randomly.

    No lookahead at all -- it will happily take a pawn and lose its queen.
    Ties are broken at random.
    """

    def __init__(self, name: str = "greedy", seed: int = 0):
        """Seeded so a given game can be replayed exactly."""
        self.name = name
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> str:
        """Highest-value capture if there is one, else a random legal move."""
        best, best_value = [], 0
        for move in board.legal_moves:
            if board.is_en_passant(move):
                value = _PIECE_VALUES[chess.PAWN]
            else:
                captured = board.piece_at(move.to_square)
                value = _PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
            if value > best_value:
                best, best_value = [move], value
            elif value == best_value and value > 0:
                best.append(move)
        pool = best if best else list(board.legal_moves)
        return self._rng.choice(pool).uci()


class StockfishAgent:
    """Stockfish as an opponent: a strength setting plus a search limit.

    ``limit`` decides how long it thinks per move, as one of
    {"time": seconds}, {"depth": plies}, or {"nodes": count}. The reference set
    uses {"time": 0.1} (Stockfish calibrated UCI_Elo under time limits); it's
    plain data, so switching is a one-line change to the spec.
    """

    _LIMIT_KEYS = {"time", "depth", "nodes"}

    def __init__(self, name: str, limit: dict, elo: int | None = None, path: str | None = None,
                 threads: int = 1, hash_mb: int = 16):
        """Start one Stockfish process; ``elo=None`` means full strength."""
        if not limit or not set(limit) <= self._LIMIT_KEYS:
            raise ValueError(f"limit must use keys from {sorted(self._LIMIT_KEYS)}, got {limit!r}")
        self.name = name
        self.limit = dict(limit)
        self._sf = Stockfish(path=path, threads=threads, hash_mb=hash_mb, elo=elo)

    def select_move(self, board: chess.Board) -> str:
        """Stockfish's move under this member's strength and search limit."""
        return self._sf.play(board, **self.limit)

    def close(self) -> None:
        """Shut down the engine process."""
        self._sf.close()


@dataclass(frozen=True)
class AnchorSpec:
    """Plain-data description of one member. Edit these, not the classes.

    kind   -- "random", "greedy", or "stockfish"
    elo    -- Stockfish UCI_Elo setting (1320..3190); None = full strength
    limit  -- Stockfish search limit, e.g. {"time": 0.1} or {"depth": 1}
    seed   -- for the random/greedy members
    pin    -- rating fixed by definition (only Stockfish 1320 has one); every
              other member gets its rating by being measured, then frozen
    """

    name: str
    kind: str
    elo: int | None = None
    limit: dict = field(default_factory=dict)
    seed: int = 0
    pin: float | None = None


# The starting reference set (plan v3). To change a Stockfish level, edit its
# elo/name here; to change which rating is the pin, move `pin=`. Checkpoint
# rungs live in results.csv (ResultsTable.add_reference), not here.
_SF_LIMIT = {"time": 0.1}
REFERENCE_SET: tuple[AnchorSpec, ...] = (
    AnchorSpec("random", "random", seed=1),
    AnchorSpec("greedy", "greedy", seed=2),
    AnchorSpec("sf_1320", "stockfish", elo=1320, limit=_SF_LIMIT, pin=1320.0),
    AnchorSpec("sf_1500", "stockfish", elo=1500, limit=_SF_LIMIT),
    AnchorSpec("sf_1700", "stockfish", elo=1700, limit=_SF_LIMIT),
)

# Not a member (it measured ~1700, duplicating sf_1700). Used only in the Step 2
# smoke test: random should lose almost every game to it.
SMOKE_TEST_OPPONENT = AnchorSpec("sf_depth1", "stockfish", limit={"depth": 1})


def build_anchor(spec: AnchorSpec):
    """AnchorSpec -> a playable agent."""
    if spec.kind == "random":
        return RandomAgent(spec.name, seed=spec.seed)
    if spec.kind == "greedy":
        return GreedyAgent(spec.name, seed=spec.seed)
    if spec.kind == "stockfish":
        return StockfishAgent(spec.name, limit=spec.limit, elo=spec.elo)
    raise ValueError(f"Unknown anchor kind {spec.kind!r}")


def build_agents(specs=REFERENCE_SET) -> dict:
    """Build every agent in ``specs``; returns {name: agent}. Call close_agents when done."""
    return {s.name: build_anchor(s) for s in specs}


def close_agents(agents: dict) -> None:
    """Shut down any agent that holds a process (the Stockfish ones)."""
    for agent in agents.values():
        if hasattr(agent, "close"):
            agent.close()


def fixed_ratings(specs=REFERENCE_SET, frozen: dict[str, float] | None = None) -> dict[str, float]:
    """Ratings the Elo fit holds fixed: the pin(s) from ``specs`` plus any frozen ratings.

    Inputs: member specs; ``frozen`` = {name: rating}, e.g. ResultsTable.ratings().
    Output: {name: rating}. A frozen rating may never overwrite a pin.
    """
    ratings = {s.name: float(s.pin) for s in specs if s.pin is not None}
    clash = set(ratings) & set(frozen or {})
    if clash:
        raise ValueError(f"{sorted(clash)} are pinned; they can't also have a frozen rating")
    ratings.update(frozen or {})
    return ratings


def build_reference_agents(results: "ResultsTable", names, artifact_root: str | Path, *, device=None) -> dict:
    """{name: playable agent} for reference members, from their results.csv rows.

    Stockfish / random / greedy members come from REFERENCE_SET; checkpoint rungs
    (spec kind "sft_checkpoint") are loaded from
    <artifact_root>/checkpoints/sft/<run>/checkpoints/step_NNNNNN. Call close_agents when done.
    """
    specs = {s.name: s for s in results.specs}
    agents = {}
    for name in names:
        if results.role(name) not in ("pin", "reference"):
            raise ValueError(f"{name!r} is not a reference member in {results.path}")
        spec = results.spec(name)
        if spec.get("kind") == "sft_checkpoint":
            from train.sft import load_checkpoint  # lazy: pulls in torch/transformers
            step_dir = Path(artifact_root) / "checkpoints" / "sft" / spec["run"] / "checkpoints" / f"step_{spec['step']:06d}"
            agents[name] = load_checkpoint(step_dir, device=device, name=name)
        else:
            agents[name] = build_anchor(specs[name])
    return agents


RESULTS_COLUMNS = ["name", "role", "rating", "ci_low", "ci_high", "games", "run", "step", "seed",
                   "train_gpu_hours", "label_cpu_hours", "sec_per_move", "illegal_rate", "gpu",
                   "stockfish_version", "date", "spec", "note"]
ROLES = ("pin", "reference", "model")


class ResultsTable:
    """``results/results.csv``: the project's one file of record (plan v4, Sec 2.2).

    One row per player. ``role`` says what the row is:
      pin       -- fixed by definition (sf_1320 = 1320); always taken from ``specs``
      reference -- rated once, then frozen: the opponents every model is rated against
      model     -- an evaluated checkpoint; overwritten after every eval batch, so a
                   stopped run keeps its latest Elo + CI. What the Step 11 charts plot.
    Reference and pin rows are never overwritten. A model can later be promoted to a
    reference (a new rung) with ``add_reference`` under the same name.

    Every rating depends on the exact Stockfish build, so loading refuses a file whose
    reference rows were measured with a different version.
    """

    def __init__(self, path: str | Path, specs=REFERENCE_SET):
        """Open ``path`` (created on first save). Pin rows always exist, even before a save."""
        self.path = Path(path)
        self.specs = specs
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, newline="") as f:
                for row in csv.DictReader(f):
                    v = row.get("stockfish_version") or None
                    if row["role"] != "model" and v is not None and v != SF_VERSION:
                        raise RuntimeError(f"{row['name']!r} was rated with {v!r}, but the code pins "
                                           f"{SF_VERSION!r}; its rating doesn't carry over")
                    self.rows[row["name"]] = {k: row.get(k, "") for k in RESULTS_COLUMNS}
        for s in specs:
            if s.pin is not None:
                self.rows[s.name] = self._row(s.name, "pin", s.pin, s.pin, s.pin, "",
                                              spec=asdict(s), stockfish_version=SF_VERSION)

    @staticmethod
    def _row(name, role, rating, ci_low, ci_high, games, **fields) -> dict:
        row = {k: "" for k in RESULTS_COLUMNS}
        row.update(name=name, role=role, rating=round(float(rating), 1), ci_low=round(float(ci_low), 1),
                   ci_high=round(float(ci_high), 1), games=games, date=_dt.date.today().isoformat())
        for k, v in fields.items():
            if k not in RESULTS_COLUMNS:
                raise KeyError(f"unknown results column {k!r}")
            row[k] = json.dumps(v) if k == "spec" and isinstance(v, dict) else v
        return row

    def role(self, name: str) -> str | None:
        """"pin", "reference", "model", or None if ``name`` has no row."""
        row = self.rows.get(name)
        return row["role"] if row else None

    def ratings(self) -> dict[str, float]:
        """{name: rating} for reference rows (pins excluded -- fixed_ratings adds those)."""
        return {n: float(r["rating"]) for n, r in self.rows.items() if r["role"] == "reference"}

    def fixed(self) -> dict[str, float]:
        """Everything an Elo fit holds fixed: pin(s) + reference ratings."""
        return fixed_ratings(self.specs, self.ratings())

    def spec(self, name: str) -> dict:
        """The JSON ``spec`` cell of a row, decoded (how to rebuild that player)."""
        cell = self.rows[name]["spec"]
        return json.loads(cell) if cell else {}

    def add_reference(self, name: str, rating: float, ci_low: float, ci_high: float, games: int, *,
                      spec: dict, note: str = "") -> None:
        """Freeze a new reference member. Refuses an existing pin or reference name."""
        if self.role(name) in ("pin", "reference"):
            raise ValueError(f"{name!r} is already a {self.role(name)}; frozen ratings are never re-rated")
        sf = SF_VERSION if spec.get("kind") == "stockfish" else ""
        self.rows[name] = self._row(name, "reference", rating, ci_low, ci_high, games, spec=spec,
                                    note=note, stockfish_version=sf)

    def log_model(self, estimate, records=(), **fields) -> dict:
        """Write (or overwrite) a model's row from an Elo estimate and its eval games.

        Inputs: ``estimate`` -- a RatingEstimate (eval.elo); ``records`` -- the eval
        GameRecords, used for sec_per_move and illegal_rate; ``fields`` -- any other
        RESULTS_COLUMNS (run, step, seed, train_gpu_hours, label_cpu_hours, gpu, note).
        Call it after every eval batch, then ``save()``: stopping early keeps the last batch.
        """
        name = estimate.name
        if self.role(name) in ("pin", "reference"):
            raise ValueError(f"{name!r} is a {self.role(name)} row; it can't be overwritten by a model eval")
        moves = illegal = 0
        seconds = 0.0
        for r in records:
            side = "white" if r.white == name else "black" if r.black == name else None
            if side:
                moves += len(r.latency[side])
                seconds += sum(r.latency[side])
                illegal += r.illegal_attempts[side]
        if moves:
            fields.setdefault("sec_per_move", round(seconds / moves, 4))
            fields.setdefault("illegal_rate", round(illegal / moves, 4))
        if estimate.at_bound:
            fields["note"] = (f"OFF SCALE ({estimate.at_bound}) " + str(fields.get("note", ""))).strip()
        self.rows[name] = self._row(name, "model", estimate.rating, estimate.ci_low, estimate.ci_high,
                                    estimate.games, **fields)
        return self.rows[name]

    def save(self) -> None:
        """Write every row: pins, then references, then models."""
        order = {r: i for i, r in enumerate(ROLES)}
        rows = sorted(self.rows.values(), key=lambda r: order[r["role"]])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(self.path)  # never leaves a half-written file behind
