"""Step 4 checks for model/llm_policy.py.

Intent: prove the wrapper's mechanics on a tiny, random, offline stand-in for
Qwen3 (same architecture class, a few thousand parameters, a character-level
tokenizer with Qwen's chat format), so these run in seconds with no download:
prompts, legal-only masked play, move scores that match the SFT loss exactly,
unmasked retries and the three-strikes forfeit, and save/load.

The real gate (plan v3, Step 4) runs the actual Qwen3-0.6B and is opt-in,
because it downloads ~1.2 GB. In Colab (GPU runtime), from the repo root:
    RUN_MODEL_TESTS=1 python -m pytest tests/test_llm_policy.py -v -s -k gate

Everything else:
    python -m pytest tests/test_llm_policy.py -v
"""
import os
import random

import chess
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from engine import board as B
from eval.anchors import RandomAgent
from eval.match import play_game
from model.llm_policy import IGNORE_INDEX, LLMPolicy, PolicyConfig, build_prompt, parse_move

# A stand-in for Qwen3's chat template with thinking off (same shape of output).
CHAT_TEMPLATE = (
    "{%- for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}{%- endif %}"
    "{%- endif %}"
)
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def tiny_tokenizer():
    """Character-level tokenizer: every printable ASCII character is one token."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast
    chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
    vocab = {t: i for i, t in enumerate(["<unk>"] + chars)}
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    raw.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", eos_token="<|endoftext|>",
                                  pad_token="<|endoftext|>")
    tok.add_special_tokens({"additional_special_tokens": SPECIALS[1:]})
    tok.chat_template = CHAT_TEMPLATE
    return tok


def tiny_model(vocab_size, seed=0):
    """A 2-layer Qwen3 with random weights -- same code path as the real one, tiny."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=2048,
                      tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


def make_policy(tok, seed=0, **cfg):
    return LLMPolicy(tiny_model(len(tok), seed), tok, PolicyConfig(model_id="tiny", **cfg))


def positions(n_games=6, plies=60, seed=0):
    """Positions (with a legal move) from a few random games, including promotions etc."""
    rng, out = random.Random(seed), []
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(plies):
            if B.is_terminal(b):
                break
            out.append(b.copy())
            b.push(rng.choice(list(b.legal_moves)))
    return out


# --------------------------------------------------------------- prompt ----

def test_prompt_template(tok):
    board = chess.Board()
    for rep in ("structured", "fen"):
        p = build_prompt(board, tok, rep, "Play.", enable_thinking=False)
        assert "Play.\n" in p and p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert "\nMove:<|im_end|>" in p
        assert "e2e4" not in p                                  # legal moves are never listed
    assert chess.STARTING_FEN in build_prompt(board, tok, "fen", "Play.")
    assert "Side to move: white" in build_prompt(board, tok, "structured", "Play.")
    with pytest.raises(ValueError):
        PolicyConfig(representation="image")


def test_config_from_yaml_block():
    c = PolicyConfig.from_dict({"representation": "fen", "masked": False, "lora_target_modules": ["q_proj"]})
    assert c.representation == "fen" and not c.masked and c.lora_target_modules == ("q_proj",)
    with pytest.raises(ValueError):
        PolicyConfig.from_dict({"lora_rank": 8})


def test_sft_yaml_policy_block_matches_the_defaults():
    import yaml
    block = yaml.safe_load(open("configs/sft.yaml"))["policy"]
    assert PolicyConfig.from_dict(block) == PolicyConfig()   # one set of defaults, not two


def test_parse_move_is_strict():
    assert parse_move("e2e4") == "e2e4"
    assert parse_move("  e7e8q.\nand then") == "e7e8q"
    assert parse_move('"g1f3"') == "g1f3"
    assert parse_move("Nf3") == "Nf3"        # not converted: SAN is an illegal answer here
    assert parse_move("") == ""


# ------------------------------------------------------------ the model ----

def test_lora_starts_as_a_no_op_and_value_head_says_half(tok):
    base = tiny_model(len(tok))
    ids = torch.tensor([tok("hello board", add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        before = base(input_ids=ids).logits
    policy = LLMPolicy(base, tok, PolicyConfig(model_id="tiny"))
    with torch.no_grad():
        after = policy.model(input_ids=ids).logits
    assert torch.allclose(before, after)                     # untrained = base model
    _, win_prob = policy.score_moves(chess.Board(), ["e2e4"])
    assert win_prob == pytest.approx(0.5)                    # zero-init head
    names = [n for n, p in policy.model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)       # only adapters train
    assert len(policy.trainable_parameters()) == len(names) + 2   # + value head weight, bias


def test_masked_play_is_always_legal(tok):
    policy = make_policy(tok)
    for board in positions():
        d = policy.act(board)
        assert d.legal and B.is_legal(board, d.move)
        assert set(d.move_logprobs) == set(B.legal_moves(board))
        assert d.move == max(d.move_logprobs, key=d.move_logprobs.get)
        assert all(s < 0 for s in d.move_logprobs.values())
        assert 0.0 < d.win_prob < 1.0


def test_prefix_cache_gives_the_same_scores(tok):
    cached = make_policy(tok, score_batch_size=7)            # several chunks per position
    full = cached.variant(prefix_cache=False)
    for board in positions(n_games=2, plies=30):
        a, va = cached.score_moves(board, B.legal_moves(board))
        b, vb = full.score_moves(board, B.legal_moves(board))
        assert va == pytest.approx(vb, abs=1e-5)
        for m in a:
            assert a[m] == pytest.approx(b[m], abs=1e-4)


def test_move_score_equals_the_sft_loss(tok):
    """Scoring a move and training on it measure the same number: log P = -(summed cross-entropy)."""
    policy = make_policy(tok)
    board = chess.Board("r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 8")
    for move in ("e1g1", "d4e5", "a1b1"):
        ex = policy.training_example(board, move)
        assert ex["input_ids"][-1] == policy.end_id and ex["labels"][ex["value_index"]] == IGNORE_INDEX
        assert ex["labels"][ex["value_index"] + 1:] == policy.move_ids(move)
        ids, labels = torch.tensor([ex["input_ids"]]), torch.tensor([ex["labels"]])
        with torch.no_grad():
            logits = policy.model(input_ids=ids).logits
        loss_sum = F.cross_entropy(logits[0, :-1].float(), labels[0, 1:], ignore_index=IGNORE_INDEX, reduction="sum")
        score, _ = policy.score_moves(board, [move])
        assert score[move] == pytest.approx(-loss_sum.item(), abs=1e-4)
        assert ids.shape[1] - 1 - ex["value_index"] == len(policy.move_ids(move))


def test_representations_give_different_prompts_same_legality(tok):
    policy = make_policy(tok)
    fen = policy.variant(representation="fen")
    board = chess.Board()
    assert policy.prompt(board) != fen.prompt(board)
    assert fen.model is policy.model                         # variants share weights
    assert fen.act(board).legal


# ----------------------------------------------------------- unmasked ----

def test_unmasked_retries_sample_then_forfeit(tok):
    """A random tiny model writes garbage: the harness forfeits it on the third attempt."""
    policy = make_policy(tok, masked=False)
    board = chess.Board()
    d1, d2, d3 = policy.act(board), policy.act(board), policy.act(board)
    assert (d1.attempt, d2.attempt, d3.attempt) == (1, 2, 3)
    assert not d1.sampled and d2.sampled and d3.sampled
    assert 0.0 < d1.win_prob < 1.0 and isinstance(d1.raw_text, str)
    board.push_uci("e2e4")
    assert policy.act(board).attempt == 1                     # new position: greedy again

    fresh = make_policy(tok, masked=False)
    game = play_game(fresh, RandomAgent(seed=1), max_plies=40)
    assert game.termination == "illegal_move" and game.white_score == 0.0
    assert game.illegal_attempts["white"] == 3 and len(fresh.illegal_log) == 3


def test_unmasked_legal_output_is_played(tok, monkeypatch):
    """When the model does write a legal move, it's parsed and played."""
    policy = make_policy(tok, masked=False)
    real_decode = tok.decode
    monkeypatch.setattr(policy.tokenizer, "decode", lambda *a, **k: " e2e4 is best")
    d = policy.act(chess.Board())
    monkeypatch.setattr(policy.tokenizer, "decode", real_decode)
    assert d.move == "e2e4" and d.legal and not policy.illegal_log


def test_greedy_unmasked_is_deterministic(tok):
    a = make_policy(tok, masked=False).act(chess.Board())
    b = make_policy(tok, masked=False).act(chess.Board())
    assert a.raw_text == b.raw_text


# ------------------------------------------------------ the 4 toggles ----

def test_all_four_toggle_combinations_finish_a_game(tok):
    """The Step 4 gate in miniature: every combination completes a game vs. random."""
    policy = make_policy(tok)
    for rep in ("structured", "fen"):
        for masked in (True, False):
            agent = policy.variant(representation=rep, masked=masked)
            game = play_game(agent, RandomAgent(seed=3), max_plies=24)
            assert game.termination in {"max_plies", "illegal_move", "checkmate", "stalemate"}
            if masked:
                assert game.illegal_attempts["white"] == 0


# ---------------------------------------------------------- save/load ----

def test_save_and_load_round_trip(tok, tmp_path):
    base_dir = tmp_path / "base"
    tiny_model(len(tok), seed=4).save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    config = PolicyConfig(model_id=str(base_dir))
    policy = LLMPolicy.from_pretrained(config, device="cpu")
    with torch.no_grad():                                    # pretend training changed things
        for p in policy.trainable_parameters():
            p.add_(0.05 * torch.randn_like(p))
    board = chess.Board()
    before, v_before = policy.score_moves(board, B.legal_moves(board))
    policy.save(tmp_path / "ckpt")
    assert not any((tmp_path / "ckpt").glob("model*.safetensors"))   # base weights not saved
    loaded = LLMPolicy.from_pretrained(config, device="cpu", adapter_dir=tmp_path / "ckpt")
    after, v_after = loaded.score_moves(board, B.legal_moves(board))
    assert v_after == pytest.approx(v_before) and v_before != pytest.approx(0.5)
    for m in before:
        assert after[m] == pytest.approx(before[m], abs=1e-5)


# ------------------------------------------------ the real gate (opt-in) ----

@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1",
                    reason="downloads Qwen3-0.6B; set RUN_MODEL_TESTS=1 (Colab, GPU runtime)")
def test_gate_untrained_qwen_plays_all_four_combinations():
    """Plan v3, Step 4 gate: the untrained base model plays a full game against the
    random mover in all 4 toggle combinations without crashing (a forfeit by
    illegal moves counts as a finished game)."""
    import time
    policy = LLMPolicy.from_pretrained(PolicyConfig())
    print(f"\nloaded {policy.config.model_id} on {policy.device} ({next(policy.model.parameters()).dtype})")
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    a, _ = policy.score_moves(board, B.legal_moves(board))
    b, _ = policy.variant(prefix_cache=False).score_moves(board, B.legal_moves(board))
    worst = max(abs(a[m] - b[m]) for m in a)
    print(f"prefix cache vs. full recompute: largest score difference {worst:.4f}")
    assert worst < 0.1                                       # same numbers up to fp16/bf16 rounding
    for rep in ("structured", "fen"):
        for masked in (True, False):
            agent = policy.variant(representation=rep, masked=masked)
            start = time.perf_counter()
            game = play_game(agent, RandomAgent(seed=7), max_plies=400)
            secs = time.perf_counter() - start
            first = agent.act(chess.Board())
            print(f"{agent.name:24s} {game.termination:14s} {game.plies:3d} plies  "
                  f"illegal attempts {game.illegal_attempts['white']}  "
                  f"{sum(game.latency['white']) / max(1, len(game.latency['white'])):.2f} s/move  ({secs:.0f} s)")
            print(f"{'':24s} opening reply: {first.raw_text!r} -> {first.move}  win prob {first.win_prob:.2f}"
                  if not masked else f"{'':24s} opening move: {first.move}  win prob {first.win_prob:.2f}")
            assert game.termination in {"max_plies", "illegal_move", "checkmate", "stalemate", "insufficient_material",
                                        "threefold_repetition", "fifty_moves", "fivefold_repetition", "seventyfive_moves"}
            if masked:
                assert game.illegal_attempts["white"] == 0
