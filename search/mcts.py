"""PUCT Monte Carlo tree search over any model with ``evaluate(boards)``, plus
batched self-play: many games at once, one network call per simulation step.

Intent: AlphaZero (Step 9) picks each move by searching -- it looks ahead with
its own network instead of trusting the network's first guess. The search is
model-agnostic so the LLM + MCTS arm can plug an LLM into the same code later.

The interface the model must expose (keep it stable; plan Step 9 (b)):

    evaluate(boards: list[chess.Board]) -> list[tuple[dict[str, float], float]]

For each board: priors over its legal moves (UCI -> probability, summing to 1)
and a value = expected score for the SIDE TO MOVE in [0, 1] (win 1, draw 0.5,
loss 0) -- the LLM value head's convention (``Decision.win_prob``). The model
works in the absolute frame (real UCI moves); any flipping is its own business.

How one search works (one tree per game):
  1. Expand the root: one model call gives priors for its moves.
  2. Repeat ``simulations`` times:
     - Select: from the root, keep stepping into the child with the highest
         Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
       Q = average result of that move so far (exploit), the second term =
       prior-weighted bonus for moves tried rarely (explore). An unvisited
       child's Q is the parent's current Q (a neutral guess, not "lost").
     - Evaluate the leaf: a finished game gets its EXACT result from
       engine.board.outcome (never the network); otherwise the model gives its
       priors (the leaf is expanded) and a value.
     - Back up: add the value to every node on the path, flipping it at each
       level (1 - v), because a position good for one side is bad for the other.
  3. The root's visit counts are the search's answer: the most-visited move,
     or (self-play, early plies) a move sampled in proportion to visits.

Batching: in every simulation step each game descends to one leaf, and all
games' leaves go to the model in a single ``evaluate`` call -- the GPU gets
batches of ~n_games positions instead of one. No virtual loss is needed
because each tree gets only one leaf per step.

Main pieces:
- MCTSConfig     -- search settings (configs/alphazero.yaml fills them)
- Node           -- one tree node
- run_search     -- batched search over several root positions (one tree each)
- visit_distribution / choose_move -- read the answer off a searched root
- search_move    -- one position -> best move (the eval agent's move picker)
- play_games     -- batched self-play games with search; returns SelfPlayGame records
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Protocol

import chess
import numpy as np

from engine import board as B
from eval.match import random_opening
from search.selfplay import MoveRecord, SelfPlayGame


class Evaluator(Protocol):
    """Anything MCTS can search with: priors over legal moves + value for the side to move."""

    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict[str, float], float]]: ...


@dataclass(frozen=True)
class MCTSConfig:
    """Search settings. Defaults are the plan's provisional values (Step 9 (b), (e))."""

    simulations: int = 100        # simulations per move (after expanding the root)
    c_puct: float = 1.5           # exploration weight
    dirichlet_alpha: float = 0.3  # root noise, self-play only
    dirichlet_weight: float = 0.25
    temperature_plies: int = 30   # self-play: sample by visits while board.ply() < this, then most-visited

    def __post_init__(self):
        if self.simulations < 1 or self.c_puct <= 0:
            raise ValueError("simulations must be >= 1 and c_puct > 0")
        if not 0 <= self.dirichlet_weight <= 1 or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_weight must be in [0, 1] and dirichlet_alpha > 0")


class Node:
    """One position in the tree, reached by ``move`` from its parent.

    ``W`` sums values from the point of view of the player who played ``move``
    (the parent's side to move), so the parent reads a child's Q = W / N directly
    and picks the max. ``terminal`` is the exact value for the side to move here
    if the game is over at this node (None = not over, or not checked yet).
    Boards aren't stored: a simulation copies the root board and replays moves.
    """

    __slots__ = ("move", "prior", "children", "N", "W", "terminal", "checked")

    def __init__(self, move: str | None = None, prior: float = 1.0):
        self.move = move
        self.prior = prior
        self.children: list[Node] = []
        self.N = 0
        self.W = 0.0
        self.terminal: float | None = None
        self.checked = False

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        """Q of this node for its PARENT's side to move (0.5 if unvisited)."""
        return self.W / self.N if self.N else 0.5


def _terminal_value(board: chess.Board, max_plies: int | None) -> float | None:
    """Exact value for the side to move if the game is over here, else None.

    Uses engine.board.outcome (checkmate, stalemate, repetition, 50-move rule, ...);
    the self-play ply cap, when given, ends the game as a draw just as the game loop does.
    """
    result = B.outcome(board)
    if result is not None:
        return (B.score_for(result, board.turn) + 1) / 2
    if max_plies is not None and board.ply() >= max_plies:
        return 0.5
    return None


def _expand(node: Node, board: chess.Board, priors: dict[str, float]) -> None:
    """Give ``node`` one child per legal move, with the model's priors (renormalized)."""
    moves = B.legal_moves(board)
    p = [max(float(priors.get(m, 0.0)), 0.0) for m in moves]
    total = sum(p)
    if total <= 0:
        p, total = [1.0] * len(moves), float(len(moves))
    node.children = [Node(m, x / total) for m, x in zip(moves, p)]


def _add_noise(node: Node, cfg: MCTSConfig, rng: np.random.Generator) -> None:
    """Self-play only: mix Dirichlet noise into the root priors, so every move gets some tries."""
    noise = rng.dirichlet([cfg.dirichlet_alpha] * len(node.children))
    w = cfg.dirichlet_weight
    for child, n in zip(node.children, noise):
        child.prior = (1 - w) * child.prior + w * float(n)


def _select(node: Node, c_puct: float) -> Node:
    """The child with the highest PUCT score (first one wins ties, so it's deterministic)."""
    sqrt_n = math.sqrt(node.N)
    fpu = 1.0 - node.value() if node.N else 0.5   # parent's own Q, from its side to move
    best, best_score = None, -math.inf
    for child in node.children:
        q = child.W / child.N if child.N else fpu
        score = q + c_puct * child.prior * sqrt_n / (1 + child.N)
        if score > best_score:
            best, best_score = child, score
    return best


def _backup(path: list[Node], value: float) -> None:
    """``value`` is for the side to move at the leaf (path[-1]); flip it at every level up."""
    v = 1.0 - value                     # the leaf's own W is from its mover's point of view
    for node in reversed(path):
        node.N += 1
        node.W += v
        v = 1.0 - v


def run_search(boards: list[chess.Board], evaluator: Evaluator, cfg: MCTSConfig, *,
               add_noise: bool = False, rng: np.random.Generator | None = None,
               max_plies: int | None = None) -> list[Node]:
    """Search every board in ``boards`` (each its own tree); returns the searched roots.

    Every model call carries one position per game: first all the roots, then one
    leaf per game per simulation (fewer when a game's simulation ends in a finished
    position, which needs no model). Root visit counts sum to ``cfg.simulations``.
    ``add_noise`` = Dirichlet noise at the roots (self-play); ``rng`` makes it reproducible.
    ``max_plies``: positions at or past it count as drawn (the self-play ply cap).
    """
    if add_noise and rng is None:
        raise ValueError("add_noise needs an rng (so the search is reproducible)")
    for b in boards:
        if _terminal_value(b, max_plies) is not None:
            raise ValueError(f"can't search a finished position: {b.fen()}")
    roots = [Node() for _ in boards]
    for root, b, (priors, value) in zip(roots, boards, evaluator.evaluate(boards)):
        _expand(root, b, priors)
        root.checked = True
        root.N, root.W = 1, 1.0 - float(value)       # the expansion counts as the root's first visit
        if add_noise:
            _add_noise(root, cfg, rng)

    for _ in range(cfg.simulations):
        pending = []                                   # (path, board) of leaves that need the model
        for root, root_board in zip(roots, boards):
            board = root_board.copy()
            node, path = root, [root]
            while node.expanded:
                node = _select(node, cfg.c_puct)
                board.push_uci(node.move)
                path.append(node)
            if not node.checked:
                node.terminal = _terminal_value(board, max_plies)
                node.checked = True
            if node.terminal is not None:
                _backup(path, node.terminal)           # exact result, no model call
            else:
                pending.append((path, board))
        if pending:
            outs = evaluator.evaluate([b for _, b in pending])
            for (path, board), (priors, value) in zip(pending, outs):
                _expand(path[-1], board, priors)
                _backup(path, float(value))
    return roots


def visit_distribution(root: Node) -> dict[str, float]:
    """{move: share of the root's visits} -- the self-play policy target (sums to 1)."""
    total = sum(c.N for c in root.children)
    return {c.move: c.N / total for c in root.children} if total else {}


def root_value(root: Node) -> float:
    """The search's estimate of the root position, for its side to move (0..1)."""
    return 1.0 - root.value()


def choose_move(root: Node, temperature: float = 0.0, rng: random.Random | None = None) -> str:
    """Most-visited move (temperature 0; ties -> higher prior), or sampled by visits^(1/temperature)."""
    if temperature <= 0:
        return max(root.children, key=lambda c: (c.N, c.prior)).move
    if rng is None:
        raise ValueError("sampling needs an rng")
    weights = [c.N ** (1.0 / temperature) for c in root.children]
    return rng.choices([c.move for c in root.children], weights=weights)[0]


def search_move(board: chess.Board, evaluator: Evaluator, cfg: MCTSConfig) -> str:
    """Eval play: search one position, no noise, return the most-visited move."""
    return choose_move(run_search([board], evaluator, cfg)[0])


def play_games(evaluator: Evaluator, n_games: int, cfg: MCTSConfig, *, opening_plies: int = 4,
               max_plies: int = 300, seed: int = 0,
               log: Callable[[str], None] | None = None) -> list[SelfPlayGame]:
    """Self-play ``n_games`` games at once with search, all trees batched into each model call.

    Each game starts from its own random opening of ``opening_plies`` half-moves (not
    recorded, as in search/selfplay.py), then every move is chosen by a noisy root
    search: sampled by visit counts while ``board.ply() < cfg.temperature_plies``,
    most-visited after. A game ends by the rules in engine/board.py or as a draw at
    ``max_plies`` (opening plies included, as in the LLM runs).

    Each MoveRecord's info holds ``visits`` (the visit distribution: the policy
    target) and ``root_value`` (the search's estimate for the mover).
    Returns SelfPlayGame records (same type as LLM self-play), in game order.
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    games = []
    for _ in range(n_games):
        board = B.new_board()
        opening = random_opening(py_rng, opening_plies) if opening_plies else []
        for mv in opening:
            B.apply_move(board, mv)
        games.append({"board": board, "opening": opening, "records": []})

    def over(g) -> bool:
        return _terminal_value(g["board"], max_plies) is not None

    active = [g for g in games if not over(g)]
    while active:
        boards = [g["board"] for g in active]
        roots = run_search(boards, evaluator, cfg, add_noise=True, rng=np_rng, max_plies=max_plies)
        for g, root in zip(active, roots):
            board = g["board"]
            temp = 1.0 if board.ply() < cfg.temperature_plies else 0.0
            move = choose_move(root, temp, py_rng)
            g["records"].append(MoveRecord(board.fen(), move, board.turn == chess.WHITE,
                                           {"visits": visit_distribution(root), "root_value": root_value(root)}))
            B.apply_move(board, move)
        active = [g for g in active if not over(g)]

    out = []
    for i, g in enumerate(games):
        board = g["board"]
        result = B.outcome(board)
        if result is None:
            white_score, termination = 0.5, "max_plies"
        else:
            white_score, termination = (result.white_value + 1) / 2, result.termination
        out.append(SelfPlayGame(g["records"], g["opening"], white_score, termination, board.ply(), board.fen()))
        if log is not None:
            log(f"  self-play game {i + 1}/{n_games}: {white_score} ({termination}, {board.ply()} plies)")
    return out
