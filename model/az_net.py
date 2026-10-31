"""AlphaZero's network: a small ResNet that looks at a position and outputs a
policy (how promising each move is) and a value (expected score for the side to move).

Intent: the search (search/mcts.py) needs two things from a model at every leaf --
which moves to look at first (priors) and how good the position is (value). AlphaZero
learns both from its own games. This file is the network plus the adapter that gives
the search exactly the ``evaluate(boards)`` interface it expects.

Shape (plan Step 9 (d); size provisional, configs/alphazero.yaml ``net``):
  input   to_planes_stm(board): 19 x 8 x 8, the position seen by the side to move
          (Black's positions are mirrored, so "my pieces" always move up the board)
  stem    3x3 conv -> batch norm -> ReLU, ``channels`` wide
  trunk   ``blocks`` residual blocks: two 3x3 convs with batch norm, plus the skip
          (each block refines the features; the skip makes deep nets trainable)
  policy  1x1 conv (2 filters) -> linear to 1,968 logits, one per move in the
          side-to-move frame; illegal moves are masked out before the softmax
  value   1x1 conv (1 filter) -> linear to 64 -> ReLU -> linear to 1 -> sigmoid
          = expected score for the side to move, in [0, 1]

``forward`` returns the value as a LOGIT (before the sigmoid): training uses
BCE-with-logits, which is the same loss as BCE after a sigmoid but numerically safe in
mixed precision. ``AZEvaluator`` applies the sigmoid.

Main pieces:
- NetConfig     -- size settings
- AZNet         -- the network
- masked_policy -- logits + legal mask -> move probabilities (illegal = 0)
- AZEvaluator   -- the MCTS adapter: real boards in, (priors over real moves, value) out
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import board as B
from engine.encoding import NUM_PLANES, to_planes_stm
from engine.moves import NUM_MOVES, stm_index, stm_legal_mask


@dataclass(frozen=True)
class NetConfig:
    """Network size. Provisional default 6 blocks x 64 channels (plan Sec 5 item 4)."""

    blocks: int = 6
    channels: int = 64

    def __post_init__(self):
        if self.blocks < 1 or self.channels < 1:
            raise ValueError("blocks and channels must be positive")

    @classmethod
    def from_dict(cls, d: dict | None) -> "NetConfig":
        """From a YAML block; unknown keys are an error (catches typos)."""
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown net settings {sorted(unknown)}")
        return cls(**d)


class _ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AZNet(nn.Module):
    """Policy-value ResNet. forward(planes [B, 19, 8, 8]) -> (policy logits [B, 1968], value logit [B])."""

    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.trunk = nn.Sequential(*[_ResBlock(c) for _ in range(cfg.blocks)])
        self.policy_conv = nn.Sequential(nn.Conv2d(c, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.policy_fc = nn.Linear(2 * 64, NUM_MOVES)
        self.value_conv = nn.Sequential(nn.Conv2d(c, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.stem(x))
        policy = self.policy_fc(self.policy_conv(h).flatten(1))
        value = self.value_fc(self.value_conv(h).flatten(1)).squeeze(1)
        return policy, value


def masked_policy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the legal moves only (illegal moves get probability 0). Computed in float32."""
    return torch.softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)


def encode_batch(boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
    """(planes [B, 19, 8, 8] float32, legal masks [B, 1968] bool), both in the side-to-move frame."""
    planes = np.stack([to_planes_stm(b) for b in boards])
    masks = np.stack([stm_legal_mask(b) for b in boards])
    return planes, masks


class AZEvaluator:
    """Wraps an AZNet for MCTS: ``evaluate(boards)`` -> [(priors over real UCI moves, value for side to move)].

    Each call is one batched forward pass. Black's positions are mirrored on the way
    in (to_planes_stm) and their move probabilities are mapped back to real moves on
    the way out (stm_index), so the search only ever sees real moves. The network runs
    in eval mode (batch norm uses its running statistics), under bfloat16 autocast on
    GPU (mixed precision: faster, and bfloat16 needs no loss scaling).
    """

    def __init__(self, net: AZNet, device: str | torch.device | None = None):
        self.net = net
        self.device = torch.device(device) if device is not None else next(net.parameters()).device

    @torch.inference_mode()
    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict[str, float], float]]:
        if not boards:
            return []
        planes, masks = encode_batch(boards)
        x = torch.from_numpy(planes).to(self.device)
        mask = torch.from_numpy(masks).to(self.device)
        self.net.eval()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits, value = self.net(x)
        probs = masked_policy(logits, mask).cpu().numpy()
        values = torch.sigmoid(value.float()).cpu().numpy()
        out = []
        for b, p, v in zip(boards, probs, values):
            white = b.turn == chess.WHITE
            out.append(({m: float(p[stm_index(m, white)]) for m in B.legal_moves(b)}, float(v)))
        return out
