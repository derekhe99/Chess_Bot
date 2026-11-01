"""The search switch: any network, rated with or without search (plan v5 Step 12c).

Intent: the same trained network can play two ways --
- **no search:** play the legal move its policy rates highest (one model call per move);
- **search:** run MCTS (search/mcts.py) guided by the network's priors and values, and
  play the most-visited move.
Rating both on the same checkpoint measures what search adds to that network (plan v5
Q3). The network only has to expose ``evaluate(boards) -> [(priors over legal UCI
moves, expected score for the side to move)]`` (search.mcts.Evaluator): AlphaZero and
CNN SFT checkpoints via model.az_net.AZEvaluator; LLM checkpoints via
model/llm_evaluator.py (Step 15).

These generalize AlphaZero's PolicyAgent / AlphaZeroAgent (train/az_train.py), which
stay as they are: same rules, so they play the same moves (tested), and stage 1-2's
ratings stand.

Main pieces:
- NoSearchAgent -- top-prior legal move; ties go to the first in sorted UCI order
- MCTSAgent     -- search_move at N simulations: no root noise, most-visited move
- make_agent    -- one switch: simulations 0 -> NoSearchAgent, N -> MCTSAgent(N)
- agent_name    -- the results.csv row name for a search setting (plan v5 Sec 2.2)

Every agent follows the harness's Agent interface (eval/match.py): ``name`` and
``select_move(board) -> UCI``. Its search budget is its inference cost: it shows up in
the eval games' seconds per move, and so in results.csv's sec_per_move.
"""
from __future__ import annotations

import dataclasses

import chess

from search.mcts import Evaluator, MCTSConfig, search_move


class NoSearchAgent:
    """The network alone: plays the legal move its policy rates highest."""

    def __init__(self, evaluator: Evaluator, name: str):
        self.name = name
        self.evaluator = evaluator

    def select_move(self, board: chess.Board) -> str:
        """Top-prior legal move (ties: first in sorted UCI order, so it is deterministic)."""
        (priors, _), = self.evaluator.evaluate([board])
        return max(sorted(priors), key=lambda m: priors[m])


class MCTSAgent:
    """The network guiding a tree search: no root noise, plays the most-visited move.

    The board is the match's copy with its move history, so the search sees repetitions.
    """

    def __init__(self, evaluator: Evaluator, name: str, cfg: MCTSConfig):
        if cfg.simulations < 1:
            raise ValueError(f"MCTSAgent needs at least 1 simulation, got {cfg.simulations}")
        self.name = name
        self.evaluator = evaluator
        self.cfg = cfg

    def select_move(self, board: chess.Board) -> str:
        """Search this position at cfg.simulations and play the most-visited move."""
        return search_move(board, self.evaluator, self.cfg)


def agent_name(base: str, simulations: int) -> str:
    """Row name for a search setting: '<base>_nosearch' (0) or '<base>_mcts<N>'.

    ``base`` is the checkpoint's usual name (e.g. 'run@15000'); the suffix goes on the
    run part, so 'run@15000' -> 'run_mcts16@15000', matching AlphaZero's
    '<run>_nosearch@rN' rows.
    """
    suffix = "_nosearch" if simulations == 0 else f"_mcts{simulations}"
    run, sep, step = base.partition("@")
    return f"{run}{suffix}{sep}{step}"


def make_agent(evaluator: Evaluator, name: str, simulations: int,
               cfg: MCTSConfig | None = None) -> NoSearchAgent | MCTSAgent:
    """The switch: 0 simulations = no search; N = MCTS with N simulations per move.

    ``cfg`` gives the other search settings (default: MCTSConfig(), the AlphaZero
    settings); its simulation count is replaced by ``simulations``.
    """
    if simulations < 0:
        raise ValueError(f"simulations must be >= 0, got {simulations}")
    if simulations == 0:
        return NoSearchAgent(evaluator, name)
    return MCTSAgent(evaluator, name, dataclasses.replace(cfg or MCTSConfig(), simulations=simulations))
