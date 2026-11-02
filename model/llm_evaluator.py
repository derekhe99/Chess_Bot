"""An LLM checkpoint as a network the search can use (plan v5 Steps 15 and 17).

Intent: MCTS (search/mcts.py) needs one thing from a network: ``evaluate(boards) ->
[(priors over the legal moves, expected score for the side to move)]``. The LLM player
(model/llm_policy.py, unchanged) already has both pieces:
- **priors** -- it scores every legal move as log P(reply = move | prompt), the same
  score masked play ranks moves by; a softmax over the legal moves turns those scores
  into probabilities (temperature 1: the model's own distribution over legal moves);
- **value** -- its value head's win probability, trained on Stockfish's win estimates
  (SFT), read at the prompt's last token.

The point of this module is **batching**. Play scores one position at a time (~0.09 s,
mostly fixed overhead), and a search asks for ~17 positions per move. Self-play runs many
games at once (search.mcts.play_games sends all their leaf positions in one call), so
``evaluate`` scores a whole list of positions together:
1. every prompt in one forward pass (left-padded, with explicit position ids so each
   prompt sees exactly the positions it would see alone), keeping its keys/values;
2. every (position, legal move) candidate in chunks, each candidate reusing its own
   prompt's keys/values, so a prompt is never recomputed per move.
It returns the same numbers as scoring each position alone (tested), only faster.

Main pieces:
- LLMEvaluator  -- wraps an LLMPolicy; ``evaluate(boards)`` as above
- time_search   -- seconds for one batched search move across many boards (the timing
                   probe that sizes Step 17's self-play)
"""
from __future__ import annotations

import time

import chess
import torch

from engine import board as B


class LLMEvaluator:
    """An LLMPolicy as an ``evaluate()`` network for search.mcts.

    batch_positions -- positions per forward pass (memory: prompts are ~300 tokens)
    move_batch      -- candidate moves per forward pass in step 2; each carries a copy of
                       its prompt's keys/values (~34 MB per candidate for Qwen3-0.6B at
                       ~300 tokens), so 128 is ~4.5 GB on the GPU
    """

    def __init__(self, policy, *, batch_positions: int = 64, move_batch: int = 128):
        self.policy = policy
        self.batch_positions = batch_positions
        self.move_batch = move_batch
        self._move_ids: dict[str, list[int]] = {}   # UCI move -> token ids (a few thousand distinct)

    @property
    def device(self):
        return self.policy.device

    def _ids(self, move: str) -> list[int]:
        if move not in self._move_ids:
            self._move_ids[move] = self.policy.move_ids(move)
        return self._move_ids[move]

    @torch.inference_mode()
    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict[str, float], float]]:
        """[(priors over legal UCI moves, win probability for the side to move)], one per board."""
        out = []
        for start in range(0, len(boards), self.batch_positions):
            out += self._evaluate_batch(boards[start:start + self.batch_positions])
        return out

    def _evaluate_batch(self, boards: list[chess.Board]) -> list[tuple[dict[str, float], float]]:
        from transformers import DynamicCache
        policy, dev = self.policy, self.device
        policy.model.eval()

        # 1. All prompts at once, left-padded so every prompt ends at the last position.
        prompts = [policy.prompt_ids(b) for b in boards]
        lengths = torch.tensor([len(p) for p in prompts], device=dev)
        width = int(lengths.max())
        ids = torch.full((len(boards), width), policy.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(boards), width), dtype=torch.long, device=dev)
        for i, p in enumerate(prompts):
            ids[i, width - len(p):] = torch.tensor(p, device=dev)
            mask[i, width - len(p):] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)   # each prompt counts from 0, as when alone
        out = policy.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True,
                           output_hidden_states=True, logits_to_keep=1)
        win_prob = torch.sigmoid(policy.value_head(out.hidden_states[-1][:, -1])).tolist()
        first_logp = torch.log_softmax(out.logits[:, -1].float(), dim=-1)          # (boards, vocab)
        kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]

        # 2. Every (board, legal move) candidate, in chunks, each on its own prompt's cache.
        legal = [B.legal_moves(b) for b in boards]
        pairs = [(i, m) for i, moves in enumerate(legal) for m in moves]
        scores: list[dict[str, float]] = [{} for _ in boards]
        for start in range(0, len(pairs), self.move_batch):
            chunk = pairs[start:start + self.move_batch]
            rows = torch.tensor([i for i, _ in chunk], device=dev)
            cand = [self._ids(m) for _, m in chunk]
            k = max(len(c) for c in cand)
            c_ids = torch.full((len(chunk), k), policy.pad_id, dtype=torch.long, device=dev)
            real = torch.zeros((len(chunk), k), dtype=torch.long, device=dev)
            for j, c in enumerate(cand):
                c_ids[j, :len(c)] = torch.tensor(c, device=dev)
                real[j, :len(c)] = 1
            cache = DynamicCache()
            for layer, (keys, values) in enumerate(kv):
                cache.update(keys.index_select(0, rows), values.index_select(0, rows), layer)
            logits = policy.model(input_ids=c_ids, attention_mask=torch.cat([mask[rows], real], dim=1),
                                  position_ids=lengths[rows].unsqueeze(1) + torch.arange(k, device=dev),
                                  past_key_values=cache, use_cache=True).logits
            # logp[j, t] = log P(token t of candidate j | its prompt + its tokens before t)
            logp = torch.cat([first_logp[rows].unsqueeze(1), torch.log_softmax(logits[:, :k - 1].float(), dim=-1)],
                             dim=1)
            tok = logp.gather(-1, c_ids.unsqueeze(-1)).squeeze(-1)
            for (i, m), s in zip(chunk, (tok * real).sum(dim=1).tolist()):
                scores[i][m] = s

        results = []
        for i, moves in enumerate(legal):
            s = torch.tensor([scores[i][m] for m in moves])
            p = torch.softmax(s, dim=0).tolist()
            results.append((dict(zip(moves, p)), win_prob[i]))
        return results


def time_search(evaluator, boards: list[chess.Board], simulations: int, cfg=None) -> dict:
    """Seconds for one search move on every board at once (one batched move of len(boards) games).

    The timing probe for Step 17: on the L4 it turns the throughput estimate into a
    measurement. Returns seconds per batched move, per game-move, and per evaluated position.
    """
    import dataclasses
    from search.mcts import MCTSConfig, run_search
    cfg = dataclasses.replace(cfg or MCTSConfig(), simulations=simulations)
    calls = []
    inner = evaluator.evaluate

    def counting(bs):
        calls.append(len(bs))
        return inner(bs)

    evaluator.evaluate = counting
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_search([b.copy() for b in boards], evaluator, cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    finally:
        evaluator.evaluate = inner
    positions = sum(calls)
    return {"boards": len(boards), "simulations": simulations, "seconds": round(seconds, 3),
            "seconds_per_game_move": round(seconds / len(boards), 4),
            "positions_evaluated": positions, "seconds_per_position": round(seconds / max(positions, 1), 5),
            "model_calls": len(calls)}
