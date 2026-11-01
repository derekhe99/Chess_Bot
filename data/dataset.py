"""Serves the labeled positions to training, in the chosen board representation.

Intent: the one place labeled positions become training examples. It hands out
plain records -- the board as text, Stockfish's move, and the win probability --
and deliberately stops there. Turning a record into the model's actual input
(prompt + move + end-of-text, tokenized) is done by the one shared prompt
template in model/llm_policy.py (Step 4), so training and play read exactly the
same text (plan v3, Sec 2.1).

Main pieces:
- split_by_game   -- train/validation split where no game lands on both sides
- fixed_split     -- (Step 13) any dataset's training rows + the ONE validation set
                     every size and architecture shares (the 50k's validation split)
- near_win_report -- (Step 13) per phase: how many positions are near-wins or simple endings
- PositionDataset -- PyTorch Dataset of plain records for one representation
- load_datasets   -- labels.parquet -> (train, validation) datasets
"""
from __future__ import annotations

import random
from pathlib import Path

import chess
import pandas as pd
from torch.utils.data import Dataset

from engine.encoding import TEXT_ENCODERS, encode_text


def split_by_game(df: pd.DataFrame, val_fraction: float = 0.1, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split positions into train / validation by GAME, not by position.

    Positions from one game are related (same players, same plan), so if some
    went to training and others to validation, the validation score would
    flatter the model. Here a whole game goes to one side. ``val_fraction`` is the
    share of games held out; the same seed always gives the same split.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    games = sorted(df["game_id"].unique())
    random.Random(seed).shuffle(games)
    val_games = set(games[: round(len(games) * val_fraction)])
    is_val = df["game_id"].isin(val_games)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def fixed_split(df: pd.DataFrame, base_df: pd.DataFrame, val_fraction: float = 0.1,
                seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows of ``df`` and the fixed validation set taken from ``base_df``.

    The validation set is exactly ``split_by_game(base_df, val_fraction, seed)``'s
    validation half -- the 50k dataset's, unchanged since Step 3 -- so models trained on
    any dataset size are scored on the same positions (plan v5 Step 13). Rows of ``df``
    from a validation game, or repeating a validation position, are dropped from
    training (the Step 13 sampler already skips them; this is the safety net).
    """
    from data.sample_positions import position_key  # local: keeps dataset.py light to import
    _, val_df = split_by_game(base_df, val_fraction, seed)
    val_games = set(val_df["game_id"])
    val_keys = {position_key(chess.Board(f)) for f in val_df["fen"]}
    keys = df["fen"].map(lambda f: position_key(chess.Board(f)))
    keep = ~df["game_id"].isin(val_games) & ~keys.isin(val_keys)
    return df[keep].reset_index(drop=True), val_df


def near_win_report(df: pd.DataFrame, near: float = 0.9, simple_pieces: int = 6) -> pd.DataFrame:
    """Per phase: positions, share near-won or near-lost, share of simple endings.

    near_win: Stockfish's win probability for the side to move is above ``near`` or
    below 1 - ``near`` (one side is close to winning; the label shows how to convert or
    defend). simple_ending: at most ``simple_pieces`` pieces on the board, kings
    included (e.g. king + rook vs. king + pawn). Also, if the dataset has the Step 13
    ``ending`` column, the share of positions from games that ended in checkmate.
    Used to decide whether a conversion-enriched variant is needed (plan v5 Step 13).
    """
    out = df.assign(near_win=(df["win_prob"] > near) | (df["win_prob"] < 1 - near),
                    simple_ending=df["fen"].map(lambda f: len(chess.Board(f).piece_map()) <= simple_pieces))
    cols = {"positions": ("fen", "size"), "near_win": ("near_win", "mean"),
            "simple_ending": ("simple_ending", "mean")}
    if "ending" in df:
        out["from_checkmate_game"] = df["ending"] == "checkmate"
        cols["from_checkmate_game"] = ("from_checkmate_game", "mean")
    table = out.groupby("phase").agg(**cols)
    table.loc["all"] = out.agg({c: f for c, f in cols.values()}).rename(
        {c: name for name, (c, _) in cols.items()})
    return table.round(3)


class PositionDataset(Dataset):
    """Labeled positions as plain records, for one board representation.

    Each item: {"board_text": the position as text ("structured" or "fen"),
                "label_move": Stockfish's move (UCI, e.g. "e2e4"),
                "label_value": win probability for the side to move (0..1),
                "fen": the position, "phase": opening/middlegame/endgame}.
    """

    def __init__(self, df: pd.DataFrame, representation: str = "structured"):
        """``df`` needs columns fen, best_move, win_prob, phase (labels.parquet has them)."""
        if representation not in TEXT_ENCODERS:
            raise ValueError(f"representation must be one of {sorted(TEXT_ENCODERS)}, got {representation!r}")
        missing = {"fen", "best_move", "win_prob", "phase"} - set(df.columns)
        if missing:
            raise ValueError(f"labels are missing columns {sorted(missing)}")
        self.representation = representation
        self._fen = df["fen"].tolist()
        self._move = df["best_move"].tolist()
        self._value = df["win_prob"].astype(float).tolist()
        self._phase = df["phase"].tolist()

    def __len__(self) -> int:
        """Number of positions."""
        return len(self._fen)

    def __getitem__(self, i: int) -> dict:
        """One record; the board text is rendered on the fly from the stored FEN."""
        return {"board_text": encode_text(chess.Board(self._fen[i]), self.representation),
                "label_move": self._move[i], "label_value": self._value[i],
                "fen": self._fen[i], "phase": self._phase[i]}


def load_datasets(path: str | Path, representation: str = "structured", val_fraction: float = 0.1,
                  seed: int = 0) -> tuple[PositionDataset, PositionDataset]:
    """labels.parquet -> (train dataset, validation dataset), split by game."""
    train_df, val_df = split_by_game(pd.read_parquet(path), val_fraction, seed)
    return PositionDataset(train_df, representation), PositionDataset(val_df, representation)
