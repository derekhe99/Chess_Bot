"""Serves the labeled positions to training, in the chosen board representation.

Intent: the one place labeled positions become training examples. It hands out
plain records -- the board as text, Stockfish's move, and the win probability --
and deliberately stops there. Turning a record into the model's actual input
(prompt + move + end-of-text, tokenized) is done by the one shared prompt
template in model/llm_policy.py (Step 4), so training and play read exactly the
same text (plan v3, Sec 2.1).

Main pieces:
- split_by_game   -- train/validation split where no game lands on both sides
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
