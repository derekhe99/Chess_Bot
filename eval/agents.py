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
- load_player   -- a saved checkpoint of any arm, by a plain spec, as a playable agent
                   (the head-to-head and later re-evals load players this way)
- rate          -- rate a spec against reference opponents into a results row, every game
                   saved (e.g. an LLM checkpoint with search: Step 15/17's control)
- rerate        -- (Step 12e) re-rate an existing model row with every game saved; the
                   new row replaces the old one and its note keeps the old rating

Every agent follows the harness's Agent interface (eval/match.py): ``name`` and
``select_move(board) -> UCI``. Its search budget is its inference cost: it shows up in
the eval games' seconds per move, and so in results.csv's sec_per_move.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

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


PLAYER_KINDS = ("sft", "llm_rl", "alphazero")


def checkpoint_dir(spec: dict, artifact_root: str | Path) -> Path:
    """Where a spec's checkpoint lives under ``artifact_root``/checkpoints."""
    kind, run, step = spec["kind"], spec["run"], int(spec["step"])
    if kind not in PLAYER_KINDS:
        raise ValueError(f"kind must be one of {PLAYER_KINDS}, got {kind!r}")
    sub = f"step_{step:06d}" if kind == "sft" else f"round_{step:04d}"
    return Path(artifact_root) / "checkpoints" / kind / run / "checkpoints" / sub


def load_player(spec: dict, artifact_root: str | Path, *, device=None):
    """A saved checkpoint as a playable agent, from a plain spec.

    spec: {"name": row name, "kind": "sft" | "llm_rl" | "alphazero", "run": run folder,
           "step": SFT step or RL / AlphaZero round, "simulations": search budget
           (AlphaZero default: configs/alphazero.yaml's; 0 = no search)}.
    An LLM checkpoint without "simulations" plays as in its evals (LLMPolicy, masked
    argmax); with it, the LLM drives the search through model.llm_evaluator (Step 15/17).
    Imports are lazy: loading an LLM pulls in transformers.
    """
    path = checkpoint_dir(spec, artifact_root)
    if not path.is_dir():
        raise FileNotFoundError(f"{spec['name']}: no checkpoint at {path}")
    if spec["kind"] in ("sft", "llm_rl"):
        from train.sft import load_checkpoint
        policy = load_checkpoint(path, device=device, name=spec["name"])
        if "simulations" not in spec:
            return policy
        from model.llm_evaluator import LLMEvaluator
        return make_agent(LLMEvaluator(policy), spec["name"], int(spec["simulations"]))
    from model.az_net import AZEvaluator
    from train.az_train import load_checkpoint, load_configs
    net = load_checkpoint(path, device)
    _, _, mcts_cfg, _ = load_configs()
    sims = int(spec.get("simulations", mcts_cfg.simulations))
    return make_agent(AZEvaluator(net, next(net.parameters()).device), spec["name"], sims, mcts_cfg)


# Row fields a re-rating carries over from the old row (everything that describes the
# checkpoint, not the eval): the Elo, CI, games, W/D/L, speed and date are re-measured.
CARRIED_FIELDS = ("run", "step", "seed", "train_gpu_hours", "label_cpu_hours", "dataset_size")


def is_rerated(results, name: str, games: int) -> bool:
    """True if ``name``'s row already has W/D/L (rated with records) and its full game count."""
    row = results.rows.get(name)
    return bool(row and row.get("wins") not in ("", None) and int(row["games"] or 0) >= games)


def rate(spec: dict, results, artifact_root, opponents: dict, *, fields: dict | None = None,
         games_per_opponent: int = 20, batch_games: int = 10, gpu: str = "", n_boot: int = 1000,
         load=None, verbose: bool = True):
    """Rate a spec against reference opponents into its results row (every game saved). Returns the estimate.

    ``fields``: the row's other columns (run, step, train_gpu_hours, label_cpu_hours,
    dataset_size, note, ...). After every batch the row is (over)written with the Elo, CI,
    W/D/L and sec/move so far, and its games replace any saved ones: a stop keeps the last
    batch, and a rerun of an unfinished rating starts it again from scratch.
    """
    from eval.match import evaluate_agent
    fields = dict(fields or {})
    agent = (load or load_player)(spec, artifact_root)

    def save(est, recs):
        results.log_model(est, recs, gpu=gpu, **fields)
        results.save()

    try:
        est, _ = evaluate_agent(agent, opponents, results.fixed(), games_per_opponent, batch_games=batch_games,
                                on_batch=save, n_boot=n_boot, verbose=verbose)
    finally:
        del agent
    return est


def rerate(spec: dict, results, artifact_root, opponents: dict, *, games_per_opponent: int = 20,
           batch_games: int = 10, gpu: str = "", n_boot: int = 1000, load=None, verbose: bool = True):
    """Re-rate one existing model row with full records (plan v5 Step 12e). Returns the estimate.

    spec: a load_player spec whose ``name`` is an existing model row; optional
    ``dataset_size`` fills that column (blank on rows rated before Step 12).
    opponents: {name: agent}, the same reference members the row was first rated against.
    The checkpoint plays ``games_per_opponent`` games against each (same openings as the
    first eval: seed 0), in batches; after every batch the row is overwritten (Elo, CI,
    W/D/L, sec/move, this pod's GPU) and its games replace any saved ones, so a stop keeps
    the last batch. Fields that describe the checkpoint (CARRIED_FIELDS) are carried over,
    and the note gains the old rating.
    """
    name = spec["name"]
    if results.role(name) != "model":
        raise ValueError(f"{name!r} is not a model row in {results.path}; nothing to re-rate")
    old = dict(results.rows[name])
    keep = {k: old[k] for k in CARRIED_FIELDS if old.get(k) not in ("", None)}
    if spec.get("dataset_size") is not None:
        keep["dataset_size"] = spec["dataset_size"]
    note = old.get("note", "")
    if "re-rated with records" not in note:   # first re-rating: remember the old rating
        note = (f"{note}; " if note else "") + (f"re-rated with records (was {old['rating']} "
                                                f"[{old['ci_low']}, {old['ci_high']}], {old['games']} games)")
    return rate(spec, results, artifact_root, opponents, fields={**keep, "note": note},
                games_per_opponent=games_per_opponent, batch_games=batch_games, gpu=gpu, n_boot=n_boot,
                load=load, verbose=verbose)
