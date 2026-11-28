"""Step 19 checks for search/parallel_selfplay.py and az_train's ``selfplay_workers``.

What they prove:
- split_games spreads a round's games evenly and loses none; worker seeds differ per worker.
- A 2-worker pool returns every game, with the search's visit targets, in a fixed order, and
  the same seed gives the same games (on the CPU, where arithmetic is exact).
- A worker's error reaches the parent instead of hanging it.
- az_train runs rounds with workers, writes the usual log, and a run_config written before the
  setting existed still reads as 1 worker (so Steps 9 and 18 resume unchanged).

    python -m pytest tests/test_parallel_selfplay.py -v
"""
import json

import pytest
import torch

from model.az_net import AZNet, NetConfig
from search.mcts import MCTSConfig
from search.parallel_selfplay import ParallelSelfPlay, split_games, worker_seed
from train.az_train import AZConfig, train

TINY_NET = NetConfig(blocks=1, channels=8)
TINY_MCTS = MCTSConfig(simulations=3, temperature_plies=6)


def test_split_games_is_even_and_complete():
    assert split_games(128, 5) == [26, 26, 26, 25, 25]
    assert split_games(3, 5) == [1, 1, 1, 0, 0]
    assert sum(split_games(1000, 7)) == 1000
    with pytest.raises(ValueError):
        split_games(10, 0)


def test_worker_seeds_differ():
    assert len({worker_seed(42, i) for i in range(5)}) == 5
    assert worker_seed(42, 0) != worker_seed(43, 0)


def test_pool_plays_every_game_reproducibly():
    torch.manual_seed(0)
    net = AZNet(TINY_NET).eval()
    with ParallelSelfPlay(2, TINY_NET, TINY_MCTS, "cpu") as pool:
        a = pool.play(net, 3, opening_plies=2, max_plies=10, seed=7)
        b = pool.play(net, 3, opening_plies=2, max_plies=10, seed=7)
    assert len(a) == 3 and all(g.moves and "visits" in g.moves[0].info for g in a)
    assert [[r.move for r in g.moves] for g in a] == [[r.move for r in g.moves] for g in b]


def test_worker_error_reaches_the_parent():
    wrong = AZNet(NetConfig(blocks=1, channels=16))          # weights that don't fit the workers' network
    with ParallelSelfPlay(1, TINY_NET, TINY_MCTS, "cpu") as pool:
        with pytest.raises(RuntimeError, match="worker failed"):
            pool.play(wrong, 1, max_plies=6)


def test_settings_check_workers():
    with pytest.raises(ValueError):
        AZConfig(selfplay_workers=0)


def test_train_with_workers_and_old_config_reads_as_one(tmp_path):
    cfg = AZConfig(games_per_round=3, max_plies=12, buffer_positions=500, batch_size=8, samples_per_position=1.0,
                   checkpoint_every=1, selfplay_workers=2)
    run = tmp_path / "par"
    s = train(cfg, TINY_NET, TINY_MCTS, run, device="cpu", max_rounds=1, log=lambda *_: None)
    assert s["round"] == 1 and s["games"] == 3
    assert json.loads((run / "run_config.json").read_text())["train"]["selfplay_workers"] == 2

    old = tmp_path / "old"                       # a serial run whose run_config predates the setting
    serial = AZConfig(games_per_round=2, max_plies=10, buffer_positions=500, batch_size=8, samples_per_position=1.0,
                      checkpoint_every=1)
    train(serial, TINY_NET, TINY_MCTS, old, device="cpu", max_rounds=1, log=lambda *_: None)
    saved = json.loads((old / "run_config.json").read_text())
    del saved["train"]["selfplay_workers"]
    (old / "run_config.json").write_text(json.dumps(saved))
    s = train(serial, TINY_NET, TINY_MCTS, old, device="cpu", max_rounds=2, log=lambda *_: None)
    assert s["round"] == 2
