"""The reference set: the frozen opponents every agent is rated against.

Intent: a set of opponents whose ratings never change, so an agent's Elo can be
read off from how it scores against them (plan v3, Sec 0 "Eval"). Exactly one
rating is pinned by definition: Stockfish UCI_Elo 1320 = 1320. Every other
member is rated once -- from games among the members -- and then frozen:
Stockfish 1500 and 1700 in Step 2; random, greedy, and ~3 early SFT
checkpoints in Step 5 (random and greedy lose every game to Stockfish 1320,
so they can't be rated until the checkpoints bridge the gap).

Main pieces:
- RandomAgent        -- plays a uniformly random legal move
- GreedyAgent        -- takes the most valuable piece it can capture, else random
- StockfishAgent     -- Stockfish at a chosen strength AND search limit
- AnchorSpec         -- plain-data description of one member (swap settings here)
- REFERENCE_SET      -- the starting members; SMOKE_TEST_OPPONENT -- Stockfish
                        depth 1, used only to smoke-test the harness
- build_agents / close_agents -- turn specs into playable agents, and shut them down
- fixed_ratings      -- pinned ratings + frozen ratings: what the Elo fit holds fixed
- FrozenRegistry     -- the saved, never-re-rated ratings (JSON on Drive); later
                        Steps add checkpoint rungs to it

Every member follows the harness's Agent interface (eval/match.py): a ``name``
and ``select_move(board) -> UCI string``.
"""
from __future__ import annotations

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
# rungs are added in Step 5 via FrozenRegistry, not here.
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

    Inputs: member specs; ``frozen`` = {name: rating}, e.g. FrozenRegistry.ratings().
    Output: {name: rating}. A frozen rating may never overwrite a pin.
    """
    ratings = {s.name: float(s.pin) for s in specs if s.pin is not None}
    clash = set(ratings) & set(frozen or {})
    if clash:
        raise ValueError(f"{sorted(clash)} are pinned; they can't also have a frozen rating")
    ratings.update(frozen or {})
    return ratings


@dataclass(frozen=True)
class FrozenMember:
    """One frozen member of the reference set: its rating and how it was measured.

    spec  -- enough to rebuild and play it exactly as it was rated: an AnchorSpec
             as a dict, or (Step 5) a checkpoint's path, weights hash, and locked
             play settings
    notes -- anything else worth keeping (e.g. opponents, games per pair)
    """

    name: str
    rating: float
    ci_low: float
    ci_high: float
    games: int
    spec: dict
    frozen_on: str = field(default_factory=lambda: _dt.date.today().isoformat())
    notes: dict = field(default_factory=dict)


class FrozenRegistry:
    """The saved ratings of the reference set's measured members (JSON, lives on Drive).

    Members are only ever added -- never re-rated or overwritten -- so every
    later rating is measured against exactly the same numbers. The file also
    records the Stockfish version, because the Stockfish members' strength (and
    so every frozen number) depends on the exact engine.
    """

    def __init__(self, path: str | Path):
        """Open the registry at ``path`` (created on first save if it doesn't exist)."""
        self.path = Path(path)
        self.members: dict[str, FrozenMember] = {}
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            if data.get("stockfish") != SF_VERSION:
                raise RuntimeError(f"Registry was frozen with {data.get('stockfish')!r}, "
                                   f"but the code pins {SF_VERSION!r}; its ratings don't carry over")
            self.members = {n: FrozenMember(**m) for n, m in data["members"].items()}

    def __contains__(self, name: str) -> bool:
        """Whether ``name`` is already frozen."""
        return name in self.members

    def ratings(self) -> dict[str, float]:
        """{name: frozen rating} -- pass to fixed_ratings()."""
        return {n: m.rating for n, m in self.members.items()}

    def add(self, member: FrozenMember) -> None:
        """Freeze a new member. Refuses a name that's already frozen (never re-rate)."""
        if member.name in self.members:
            raise ValueError(f"{member.name!r} is already frozen at {self.members[member.name].rating:.0f}; "
                             "frozen members are never re-rated")
        self.members[member.name] = member

    def save(self) -> None:
        """Write the registry to its JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"stockfish": SF_VERSION, "members": {n: asdict(m) for n, m in self.members.items()}}
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
