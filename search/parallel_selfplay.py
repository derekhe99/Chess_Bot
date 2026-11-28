"""Step 19: AlphaZero self-play on several CPU cores at once.

Intent: Step 18's self-play left the GPU ~98% idle. The search (search/mcts.py) is pure Python,
and one Python process runs one thread at a time (the GIL), so one CPU core did all the tree
work while the GPU waited for it. Threads can't fix that; processes can: each has its own
interpreter and its own core.

How: ``n_workers`` processes, started once and kept for the whole run (starting one costs
~10 s: importing torch, opening CUDA). Each round the parent sends every worker the current
network weights (~3 MB for 6 x 64) and its share of the games; each worker plays them with the
existing batched search (``play_games``, unchanged) on its own copy of the network on the GPU,
and sends the games back. The parent trains on them exactly as before. Nothing about the
games changes except which process plays them.

Details:
- ``spawn``, not ``fork``: a process forked after CUDA is initialized can't use CUDA.
- Each worker uses 1 torch thread, so 5 workers don't each start 6 math threads on 6 cores.
- Seeds: worker i plays with ``worker_seed(seed, i)``; games are split by ``split_games``.
  For a given seed and worker count the split and seeds are fixed, so a resumed run replays
  the same work (GPU arithmetic is not bit-exact, as before).

Main pieces:
- split_games      -- how many of the round's games each worker plays
- worker_seed      -- each worker's seed for the round
- ParallelSelfPlay -- the worker pool; ``play(net, n_games, ...)`` -> SelfPlayGame records,
                      in worker order (same type as play_games)
"""
from __future__ import annotations

import traceback

import torch
import torch.multiprocessing as mp

from model.az_net import AZEvaluator, AZNet, NetConfig
from search.mcts import MCTSConfig, play_games
from search.selfplay import SelfPlayGame


def split_games(n_games: int, n_workers: int) -> list[int]:
    """Games per worker, as even as possible (the first ones take the remainder): 128 over 5 -> 26,26,26,25,25."""
    if n_games < 0 or n_workers < 1:
        raise ValueError("n_games must be >= 0 and n_workers >= 1")
    q, r = divmod(n_games, n_workers)
    return [q + (i < r) for i in range(n_workers)]


def worker_seed(seed: int, i: int) -> int:
    """Worker i's seed for a round whose seed is ``seed``: distinct per worker, fixed for a given round."""
    return seed * 1_009 + i


def _worker_main(conn, net_cfg: dict, mcts_cfg: dict, device: str) -> None:
    """A worker's loop: (weights, n, opening_plies, max_plies, seed) in -> ("ok", games) or ("error", text) out; None stops it."""
    torch.set_num_threads(1)
    dev = torch.device(device)
    net = AZNet(NetConfig(**net_cfg)).to(dev).eval()
    evaluator = AZEvaluator(net, dev)
    mcts = MCTSConfig(**mcts_cfg)
    conn.send(("ready", None))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        state_dict, n, opening_plies, max_plies, seed = msg
        try:
            net.load_state_dict(state_dict)
            net.eval()
            conn.send(("ok", play_games(evaluator, n, mcts, opening_plies=opening_plies, max_plies=max_plies, seed=seed)))
        except Exception:
            conn.send(("error", traceback.format_exc()))
    conn.close()


class ParallelSelfPlay:
    """A pool of self-play worker processes. Use as a context manager (or call ``close``)."""

    def __init__(self, n_workers: int, net_cfg: NetConfig, mcts_cfg: MCTSConfig, device: str | torch.device):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        import dataclasses
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(child, dataclasses.asdict(net_cfg), dataclasses.asdict(mcts_cfg),
                                                       str(device)), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        for c in self.conns:                      # wait until every worker has its network
            status, _ = c.recv()
            if status != "ready":
                raise RuntimeError("a self-play worker failed to start")

    @property
    def n_workers(self) -> int:
        return len(self.conns)

    def play(self, net: AZNet, n_games: int, *, opening_plies: int = 4, max_plies: int = 300,
             seed: int = 0) -> list[SelfPlayGame]:
        """``n_games`` self-play games with ``net``'s current weights, split over the workers."""
        weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        jobs = [(c, n, i) for i, (c, n) in enumerate(zip(self.conns, split_games(n_games, self.n_workers))) if n]
        for c, n, i in jobs:
            c.send((weights, n, opening_plies, max_plies, worker_seed(seed, i)))
        games, errors = [], []
        for c, _, i in jobs:                      # collect every reply, even after an error, so the pipes stay in step
            status, out = c.recv()
            if status == "ok":
                games.extend(out)
            else:
                errors.append(f"worker {i}:\n{out}")
        if errors:
            raise RuntimeError("self-play worker failed:\n" + "\n".join(errors))
        return games

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.conns, self.procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
