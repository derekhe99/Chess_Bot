"""Step 5 checks for train/sft.py, on the same tiny offline Qwen3 stand-in as test_llm_policy.py.

What they prove: the batched training loss is exactly -log P(move) as play
scores it (padding and the logits shortcut change nothing); the value head reads
the same hidden state in training and play; training actually lowers the loss;
an interrupted-and-resumed run ends with the same weights as an uninterrupted
one; and the checkpoint screen plays and rates every checkpoint.

    python -m pytest tests/test_sft.py -v
"""
import json
import random

import chess
import pandas as pd
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from data.dataset import PositionDataset
from engine import board as B
from eval.anchors import GreedyAgent, RandomAgent
from model.llm_policy import IGNORE_INDEX, LLMPolicy, PolicyConfig
from tests.test_llm_policy import positions, tiny_model, tiny_tokenizer
from train.sft import (TrainConfig, _load_adapters, batch_losses, collate, encode, list_checkpoints,
                       run_name, screen_checkpoints, subset, train)


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


def make_policy(tok, lora_seed=0, **cfg):
    """Same tiny base model every time (like re-downloading Qwen); ``lora_seed`` varies the fresh LoRA init."""
    model = tiny_model(len(tok), seed=0)
    torch.manual_seed(1_000 + lora_seed)
    return LLMPolicy(model, tok, PolicyConfig(model_id="tiny", **cfg))


def dataset(n=24, seed=0, representation="structured"):
    """Labeled positions with made-up (but legal) label moves and win probabilities."""
    rng = random.Random(seed)
    boards = positions(n_games=4, plies=40, seed=seed)[:n]
    df = pd.DataFrame({"fen": [b.fen() for b in boards],
                       "best_move": [rng.choice(B.legal_moves(b)) for b in boards],
                       "win_prob": [rng.random() for _ in boards],
                       "phase": ["middlegame"] * len(boards),
                       "game_id": [i // 10 for i in range(len(boards))]})
    return PositionDataset(df, representation)


def small_cfg(**changes):
    base = dict(train_positions=None, val_positions=8, val_top1_positions=4, epochs=2, batch_size=4,
                learning_rate=1e-2, checkpoint_every=3, log_every=1, seed=0)
    return TrainConfig(**{**base, **changes})


# ---------------------------------------------------------------- config ----

def test_sft_yaml_train_block_matches_the_defaults():
    import yaml
    block = yaml.safe_load(open("configs/sft.yaml"))["train"]
    assert TrainConfig.from_dict(block) == TrainConfig()
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"lr": 1e-4})
    with pytest.raises(ValueError):
        TrainConfig(value_loss="huber")


def test_run_name_and_subset():
    assert run_name(PolicyConfig(), TrainConfig()) == "structured_masked_n5000_e10_s0"
    a, b = subset(100, 10, seed=3), subset(100, 10, seed=3)
    assert a == b and len(set(a)) == 10 and subset(100, 10, seed=4) != a
    assert subset(5, 10, 0) == list(range(5)) and subset(5, None, 0) == list(range(5))


# ------------------------------------------------------------------ loss ----

def test_batched_loss_is_exactly_minus_log_p_of_the_move(tok):
    """Padding + asking for only the move positions' logits = the plain per-example loss = play's score."""
    policy = make_policy(tok)
    torch.nn.init.normal_(policy.value_head.linear.weight, std=0.5)  # a non-trivial value head
    ds = dataset(12)
    records = [ds[i] for i in range(len(ds))]
    # add a promotion so move lengths differ within the batch
    promo = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    records.append({"fen": promo.fen(), "label_move": "a7a8q", "label_value": 0.9})
    examples = [encode(policy, r) for r in records]
    assert len({len(e["input_ids"]) for e in examples}) > 1   # really mixed lengths
    batch = collate(examples, policy.pad_id)
    with torch.no_grad():
        move_nll, value, win_logit = batch_losses(policy, batch, "bce")
        for i, (rec, ex) in enumerate(zip(records, examples)):
            ids = torch.tensor([ex["input_ids"]])
            logits = policy.model(input_ids=ids).logits.float()[0, :-1]
            labels = torch.tensor(ex["labels"][1:])
            ref = torch.nn.functional.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")
            assert abs(move_nll[i].item() - ref.item()) < 1e-4
            board = chess.Board(rec["fen"])
            scores, win_prob = policy.score_moves(board, B.legal_moves(board))
            assert abs(move_nll[i].item() + scores[rec["label_move"]]) < 1e-4   # = -score in play
            assert abs(torch.sigmoid(win_logit[i]).item() - win_prob) < 1e-5     # same value-head input
    target = batch["label_value"]
    assert torch.allclose(value, torch.nn.functional.binary_cross_entropy_with_logits(
        win_logit, target, reduction="none"), atol=1e-6)


# -------------------------------------------------------------- training ----

def test_training_lowers_the_loss(tok, tmp_path):
    policy = make_policy(tok)
    ds = dataset(16)
    summary = train(policy, ds, ds, small_cfg(epochs=15, checkpoint_every=100), tmp_path / "run", log=lambda *_: None)
    assert summary["finished"] and summary["total_steps"] == 15 * 4
    log = pd.read_csv(tmp_path / "run" / "train_log.csv")
    # The tiny frozen random model can only move a little (it goes ~23 -> ~21.5 here);
    # the real learning check is the Colab run. This just proves gradients reach the adapters.
    assert log["train_move_loss"].iloc[-5:].mean() < log["train_move_loss"].iloc[:5].mean() - 0.5
    assert log["train_value_loss"].iloc[-5:].mean() < log["train_value_loss"].iloc[:5].mean()
    val = pd.read_csv(tmp_path / "run" / "val_log.csv")
    assert list(val["step"]) == [0, 60]                               # untrained + final
    assert val["val_move_loss"].iloc[-1] < val["val_move_loss"].iloc[0]
    assert summary["positions_seen"] == 15 * 16 and summary["tokens_seen"] > 0


def test_resume_gives_the_same_weights(tok, tmp_path):
    ds, cfg = dataset(14), small_cfg()           # 14 positions / batch 4 -> 4 steps per epoch, 8 total
    quiet = lambda *_: None
    straight = make_policy(tok)
    train(straight, ds, ds, cfg, tmp_path / "a", log=quiet)

    first = make_policy(tok)
    s = train(first, ds, ds, cfg, tmp_path / "b", max_steps=3, log=quiet)   # "Colab disconnects" at step 3
    assert not s["finished"] and s["step"] == 3
    resumed = make_policy(tok, lora_seed=7)           # a fresh session: different random LoRA init, overwritten on resume
    s = train(resumed, ds, ds, cfg, tmp_path / "b", log=quiet)
    assert s["finished"] and s["step"] == 8
    for p, q in zip(straight.trainable_parameters(), resumed.trainable_parameters()):
        assert torch.equal(p, q)
    a_log = pd.read_csv(tmp_path / "a" / "train_log.csv")
    b_log = pd.read_csv(tmp_path / "b" / "train_log.csv")
    assert list(a_log["step"]) == list(b_log["step"])                        # no duplicated rows
    assert a_log["train_move_loss"].tolist() == b_log["train_move_loss"].tolist()
    assert [s for s, _ in list_checkpoints(tmp_path / "b")] == [0, 3, 6, 8]


def test_a_run_folder_refuses_other_settings(tok, tmp_path):
    ds = dataset(8)
    train(make_policy(tok), ds, ds, small_cfg(epochs=1), tmp_path / "run", log=lambda *_: None)
    with pytest.raises(ValueError):
        train(make_policy(tok), ds, ds, small_cfg(epochs=1, learning_rate=1e-3), tmp_path / "run",
              log=lambda *_: None)


def test_step_zero_is_the_untrained_model(tok, tmp_path):
    ds = dataset(8)
    train(make_policy(tok), ds, ds, small_cfg(epochs=1), tmp_path / "run", log=lambda *_: None)
    step0 = dict(list_checkpoints(tmp_path / "run"))[0]
    cfg = json.loads((step0 / "policy_config.json").read_text())
    assert cfg["representation"] == "structured" and cfg["masked"]
    fresh, loaded = make_policy(tok), make_policy(tok, lora_seed=7)
    _load_adapters(loaded, step0)
    board = chess.Board()
    assert fresh.score_moves(board, B.legal_moves(board)) == loaded.score_moves(board, B.legal_moves(board))


# -------------------------------------------------------------- screening ----

def test_screen_rates_every_checkpoint(tok, tmp_path):
    ds = dataset(8)
    train(make_policy(tok), ds, ds, small_cfg(epochs=2), tmp_path / "run", log=lambda *_: None)

    def loader(step_dir):
        p = make_policy(tok)
        _load_adapters(p, step_dir)
        return p

    opponents = {"random": RandomAgent(seed=1), "greedy": GreedyAgent(seed=2)}
    estimates, records = screen_checkpoints(tmp_path / "run", opponents, {"greedy": 500.0}, n_games=2,
                                            loader=loader, verbose=False, max_plies=30)
    steps = [s for s, _ in list_checkpoints(tmp_path / "run")]
    assert set(estimates) == {f"step_{s}" for s in steps} | {"random"}
    assert len(records) == len(steps) * 2 * 2
