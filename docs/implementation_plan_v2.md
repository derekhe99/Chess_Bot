# Chess AI — Implementation Plan v2 (Level 1 / Level 2)

**Version:** v2, Sep 23, 2026. Supersedes v1 (archived outside the repo at `ChessAI/archive/chess_ai_implementation_plan_v1.md`).
**Changes from v1:** Stockfish download corrected to the Stockfish 19 `universal` build (Sec 3.2); repo tree updated to `Chess_Bot/` with `CLAUDE.md` and `docs/` (Sec 2).
**Source:** Framing_Docs_updated.docx, the project plan, and the architecture map v2
**Scope:** Infrastructure, repo structure, dependencies, and execution order. Pseudocode and the modularity/config design come next.

---

## 0. Locked assumptions (from the framing doc)

These decisions are already made. Everything below depends on them.

| Decision | Choice |
|---|---|
| LLM family | **Pretrained small open-weight LLM**, fine-tuned (not a transformer trained from scratch) |
| Legal-move toggle | **Hard masking / constrained decoding** (on) vs. free generation (off). Listing legal moves in the prompt is *not* the toggle. |
| Board representation toggle | FEN text vs. structured text (piece-per-square tokens). No image input. |
| LLM design | Phase 1: 2x2 (mask × representation), all trained with SFT warm-start + RL. Phase 2: best combo retrained under all 3 regimes (self-play RL only / SFT only / SFT + RL). |
| AlphaZero | Small CNN policy-value net + MCTS, pure self-play |
| Terminology | **Game** = one full game. **Round** = a batch of self-play games (e.g., 50 games), followed by one training update. SFT has no rounds; it trains on a static dataset. |
| Eval | Rated only against fixed anchors: Stockfish at limited strength levels plus weak anchors below its ~1320 floor |
| Compute accounting | GPU-hours (train + inference) as the shared currency. Stockfish labeling CPU time is charged to SFT and warm-start. Inference compute per move is capped in eval. |
| Rigor | 2–3 seeds per RL arm. Start with a small number of eval games, then scale up for final numbers. |
| Sequencing | SFT-on-Stockfish baseline end to end first |

**Still open (decide before Step 6):** the exact compute budget number; the exact structured-text format.

**Decided:** LLM checkpoint is `Qwen/Qwen3-0.6B` (Apache-2.0). Run with `enable_thinking=False` for the baseline; chosen over `Qwen2.5-0.5B-Instruct` because that flag is a hard, same-checkpoint switch, so the optional chain-of-thought stretch arm (Section 4, after Step 11) needs only `enable_thinking=True` later, not a model swap. Caveat: at 0.6B there's no dedicated non-thinking-specialist release (unlike the 4B+ `-Instruct-2507` checkpoints) -- this is the original hybrid model with thinking off, not a specialist -- worth a line in the eventual methodology writeup.

---

## 1. Hardware and service provider

**Primary: Google Colab, with code in GitHub and checkpoints/data in Google Drive.**

| Use | Tier | GPU | Why |
|---|---|---|---|
| Development and debugging (Steps 1–5) | Colab free | NVIDIA T4, 16 GB | Free. Good enough for smoke tests and small SFT runs. No bf16, so use fp16. |
| **All reported experiments** | **Colab Pro (~$10/mo, 100 compute units)** | **NVIDIA L4, 24 GB** | bf16 support, more memory, and a cheap burn rate (roughly 1.7 compute units/hr, so about 58 GPU-hours per 100 units). |
| Fallback if Colab is unreliable | Vast.ai or RunPod | RTX 4090, 24 GB | ~$0.35–0.50/hr. Only if you move **all** arms there. |

**Rules that follow from the fairness principle:**
- Every run whose numbers you report must use the **same GPU type**. Colab doesn't guarantee which GPU you get, so log `nvidia-smi` at the start of every run and discard or rerun anything not on the L4.
- Don't use A100/H100 for reported runs. They'd break comparability and burn units about 3–5x faster.
- Colab gives you only a few CPU cores (check with `nproc`). Stockfish labeling and eval games are **CPU-bound**, so time them in Step 2 before you plan dataset and eval sizes.
- Sessions die (12-hr cap on free, idle disconnects). **Checkpoint every round / every N steps to Drive, and make every script resumable.**

---

## 2. Repository and notebook structure

**Pattern: one GitHub repo of plain `.py` modules plus several thin Colab notebooks.** Each notebook clones the repo, mounts Drive, and calls into the modules. Separate notebooks per stage, not one giant notebook, because each stage runs in its own session, fails independently, and resumes independently. All logic lives in `.py` files; notebooks only set configs and call functions.

```
Chess_Bot/
  README.md
  CLAUDE.md          # context for Claude Code; imports this plan
  docs/
    implementation_plan_v2.md   # this file
  requirements.txt
  configs/
    base.yaml          # paths, seed, GPU type, compute budget
    data.yaml          # dataset size, phase mix, Stockfish label depth
    sft.yaml           # SFT hyperparameters
    llm_rl.yaml        # LLM self-play RL hyperparameters
    alphazero.yaml     # network size, MCTS sims, games/round, rounds
    eval.yaml          # anchors, games per matchup, per-move budget
  engine/
    board.py
    moves.py
    encoding.py
    stockfish.py
  data/
    download.py
    sample_positions.py
    annotate.py
    dataset.py
  model/
    llm_policy.py
    az_net.py
  search/
    mcts.py
    selfplay.py
  train/
    sft.py
    llm_rl.py
    az_train.py
  eval/
    anchors.py
    match.py
    elo.py
  utils/
    config.py
    budget.py
    logging.py
    checkpoint.py
  notebooks/
    00_setup.ipynb
    01_build_dataset.ipynb
    02_train_sft.ipynb
    03_evaluate.ipynb
    04_train_llm_rl.ipynb
    05_train_alphazero.ipynb
    06_analysis.ipynb
  tests/
    test_engine.py
    test_mcts.py
```

The original plan's `serve/` folder (UCI bot / API) is deferred. It isn't needed to answer the research question.

### 2.1 What each file does

**`engine/` — the chess backbone**
- **`board.py`** — Thin wrapper around `python-chess`: create positions, apply moves, list legal moves, and detect terminal states and outcomes (win/loss/draw, including repetition and the 50-move rule). Every other module touches chess rules only through this file.
- **`moves.py`** — Defines one canonical move vocabulary (all ~1.9k possible UCI moves) with move↔index mapping and a function that returns a legal-move mask for a position. Shared by the AlphaZero policy head, MCTS, and the LLM's masked decoding, so all agents speak the same move language.
- **`encoding.py`** — Turns a board into each representation: FEN string, structured piece-per-square text, and tensor planes for the CNN. Each representation is a named, swappable function so a config toggle picks it.
- **`stockfish.py`** — Starts and manages Stockfish processes. Two jobs: labeling positions (best move plus evaluation at a set depth or node count) and playing at a limited strength for eval. Records the CPU time it uses so labeling cost can be charged to the budget.

**`data/` — SFT data pipeline**
- **`download.py`** — Fetches raw games from Lichess (a monthly PGN dump or the Hugging Face mirror, see Section 3) and streams them without loading everything into memory. Caches the raw files to Drive.
- **`sample_positions.py`** — Extracts positions from games, tags each as opening, middlegame, or endgame, and samples to a target phase mix set in config. Removes duplicate positions.
- **`annotate.py`** — Runs Stockfish over the sampled positions in parallel to attach the best move and the evaluation (converted to a win probability). Writes a Parquet file to Drive and logs the total CPU time spent.
- **`dataset.py`** — PyTorch `Dataset` that turns labeled positions into model inputs and targets for the chosen board representation. Handles train/validation split and batching.

**`model/`**
- **`llm_policy.py`** — Loads the pretrained model and tokenizer, attaches LoRA adapters and a small value head (predicts win likelihood from the final hidden state), and builds prompts. Implements move selection both ways: **masked** (score only legal moves and pick from them) and **unmasked** (free generation, then parse; illegal output is logged and handled per config).
- **`az_net.py`** — Small ResNet-style CNN for AlphaZero with a policy head over the shared move vocabulary and a value head. Size (blocks, channels) is set in config.

**`search/`**
- **`mcts.py`** — PUCT Monte Carlo tree search that works with any model exposing `(policy, value) = evaluate(positions)`. Batches leaf evaluations into single GPU calls. Written model-agnostic so an optional LLM + MCTS arm can reuse it later.
- **`selfplay.py`** — Runs many self-play games concurrently (concurrency is a config value) and records training examples: (position, move or search policy, final outcome). Used by both AlphaZero and LLM self-play RL.

**`train/`**
- **`sft.py`** — Supervised fine-tuning on Stockfish labels: cross-entropy on the move plus a value loss on the win probability. Logs positions seen, GPU time, and tokens, and saves checkpoints for eval curves.
- **`llm_rl.py`** — LLM self-play RL: play a round of games, assign the final win/loss/draw result as the reward to every move in that game, and update with a policy-gradient loss (with a baseline, plus a KL penalty toward the starting model to keep it stable). Can start from the base model (pure RL) or an SFT checkpoint (warm-start).
- **`az_train.py`** — AlphaZero loop: self-play round → replay buffer → gradient updates → checkpoint, repeated for N rounds. Rounds, games per round, and MCTS sims come from config.

**`eval/`**
- **`anchors.py`** — Defines the fixed opponents: random mover, material-greedy player, Stockfish capped at depth 1, and Stockfish at several `UCI_Elo` levels. Weak anchors are rated once by playing the Stockfish ladder, then frozen.
- **`match.py`** — Plays N games of agent vs. anchor with alternating colors, enforcing the per-move inference budget (time, or MCTS sims). Records results, PGNs, move latency, and illegal-move attempts.
- **`elo.py`** — Estimates an agent's Elo from its results against the anchors with bootstrap confidence intervals. Also produces Elo-vs-samples and Elo-vs-compute curves from checkpoint evals.

**`utils/`**
- **`config.py`** — Loads and merges YAML configs and applies notebook overrides. Every tunable number lives in config, never in code.
- **`budget.py`** — Compute meter: GPU-seconds, Stockfish CPU-seconds, tokens processed, and a rough FLOPs estimate per run. Can stop training when the budget is spent, which enforces equal compute across arms.
- **`logging.py`** — Writes metrics per run to CSV in Drive, optionally to Weights & Biases. Every run gets an ID, and its config and GPU type are saved with the results.
- **`checkpoint.py`** — Saves and resumes model, optimizer, replay buffer, RNG state, and budget meter to Drive with safe writes. This is what makes Colab disconnects survivable.

**`notebooks/`** — Thin drivers, one per stage: setup, build dataset, train SFT, evaluate, train LLM RL, train AlphaZero, analysis/plots.

**`tests/`** — Quick checks run before any long run: move vocabulary round-trips, legal masks match `python-chess`, and MCTS finds mate-in-1.

---

## 3. Dependencies and external resources

### 3.1 Python packages
| Package | Purpose |
|---|---|
| `python-chess` | Rules, move generation, PGN parsing, UCI engine interface |
| `torch` | Models and training |
| `transformers`, `peft`, `accelerate` | Load the open-weight LLM, LoRA fine-tuning |
| `datasets`, `huggingface_hub` | Model and dataset download |
| `zstandard` | Decompress Lichess `.pgn.zst` dumps |
| `pandas`, `pyarrow` | Labeled dataset storage (Parquet) |
| `pyyaml` | Configs |
| `wandb` (optional) | Experiment dashboards |

### 3.2 External resources and how to get them

| Resource | How to get it | Notes |
|---|---|---|
| **Open-weight LLM** | Hugging Face Hub via `AutoModelForCausalLM.from_pretrained(...)`. Create a free HF account and a read token, and store it in Colab Secrets. | **Decided: `Qwen/Qwen3-0.6B`.** Apache-2.0, ungated, fits comfortably on an L4 with LoRA. See Section 0 for why Qwen3 over Qwen2.5. |
| **Stockfish (oracle and eval anchor)** | Download the official Linux binary from the Stockfish GitHub releases page into the Colab runtime (Stockfish 19+ ships one `stockfish-linux-x86-64-universal` build; there is no separate `avx2` asset anymore). Fallback: `apt-get install stockfish` (older version). Drive it from Python with `chess.engine.SimpleEngine.popen_uci`. | Pin the version and log it. Use a fixed depth or node count for labeling. Use `UCI_LimitStrength` + `UCI_Elo` for eval levels (minimum is around 1320, which is why the weak anchors exist). |
| **Lichess games (SFT source)** | Option A: monthly rated-standard PGN dumps from `database.lichess.org`. Use an **older month** (early years are hundreds of MB rather than tens of GB). Option B: stream the Lichess games dataset on the Hugging Face Hub with `datasets` (`streaming=True`). | You only need tens of thousands of positions, so streaming or a small month is plenty. |
| **Precomputed Lichess evals** (optional shortcut) | `database.lichess.org` also publishes a Stockfish evaluation database of positions. | Saves labeling time **but hides labeling cost**. Either label yourself (preferred) or charge an estimated labeling cost to the budget. |
| **GitHub** | Repo for the code, cloned in each notebook | Private repo is fine; use a token in Colab Secrets. |
| **Google Drive** | `drive.mount()` in each notebook | Datasets, checkpoints, logs, results |

**No paid model APIs are needed.** The Lichess API is not needed (no bot account). A prompted frontier-model baseline would add an API key, but it isn't in the current scope.

---

## 4. Execution order

Each step ends with a **gate**: don't move on until it passes.

**Step 0 — Setup** (`00_setup.ipynb`)
Create the repo skeleton, requirements, and configs. Confirm in Colab: GPU detected and logged, Drive mounted, Stockfish binary runs, HF model loads and generates text.
*Gate:* one notebook cell brings up a working environment from scratch.

**Step 1 — Chess backbone** (`engine/`, `tests/test_engine.py`)
Build `board.py`, `moves.py`, `encoding.py`, `stockfish.py`.
*Gate:* tests pass; Stockfish returns a best move and evaluation for any FEN; all three encodings print correctly.

**Step 2 — Eval harness with anchors** (`eval/`, `03_evaluate.ipynb`)
Build it *before* any model so every later model is measured the same way. Validate on known cases: random vs. Stockfish depth 1 should lose almost every game, and Stockfish at two `UCI_Elo` levels should come out roughly as far apart as their settings. Rate and freeze the weak anchors. Time how long a game takes on Colab's CPUs.
*Gate:* the harness outputs sensible Elo estimates with confidence intervals for the anchors themselves.

**Step 3 — SFT data pipeline** (`data/`, `01_build_dataset.ipynb`)
Download → sample by phase → label with Stockfish → Parquet in Drive. Start with ~5–10k positions to check timing, then build the full set (tens of thousands).
*Gate:* labeled dataset in Drive, phase mix matches config, labeling CPU time logged.

**Step 4 — LLM policy wrapper** (`model/llm_policy.py`)
Load the model with LoRA and a value head; implement masked and unmasked move selection for both representations.
*Gate:* the untrained base model plays a full game against the random mover in all 4 toggle combinations without crashing.

**Step 5 — SFT baseline end to end** (`train/sft.py`, `02_train_sft.ipynb`) — **Day-1 milestone**
Train one config (FEN + masking on) on a small slice, evaluate the checkpoints in the harness.
*Gate:* an Elo number with a confidence interval that beats the untrained model. This validates the whole stack.

**Step 6 — Lock the compute budget**
Using timings from Steps 2–5, set the fixed GPU-hour budget per arm and the per-move inference cap, and write them to `base.yaml` / `eval.yaml`. Move to the L4 for all reported runs from here.

**Step 7 — LLM self-play RL** (`search/selfplay.py`, `train/llm_rl.py`, `04_train_llm_rl.ipynb`)
Start from the SFT checkpoint (warm-start) and confirm Elo improves over the SFT baseline across a few rounds.
*Gate:* stable training (no collapse), and Elo is flat or rising.

**Step 8 — LLM Phase 1: 2x2 sweep**
Run all 4 combos (mask × representation) under SFT + RL at equal budget, 2–3 seeds each. Pick the best combo.

**Step 9 — AlphaZero arm** (`model/az_net.py`, `search/mcts.py`, `train/az_train.py`, `05_train_alphazero.ipynb`)
Start with `tests/test_mcts.py` (mate-in-1), then run parallel self-play training at the same budget, 2–3 seeds. This is independent of Steps 7–8, so it can run in a separate session alongside them.

**Step 10 — LLM Phase 2: regime comparison**
Retrain the best combo under pure self-play RL, SFT only, and SFT + RL at equal budget, 2–3 seeds each.

**Step 11 — Final eval and analysis** (`06_analysis.ipynb`)
Scale up eval games for final numbers. Produce Elo with confidence intervals, Elo-vs-samples curves (sample efficiency), Elo-vs-compute, training time, and inference time per move for every arm. These map directly to the presentation slides.

**Optional stretch:** LLM + MCTS arm, reusing `search/mcts.py` with the LLM's policy and value heads.

---

## 5. Next alignment points
1. Pseudocode for each file
2. Modularity and config design: which parameters go in which YAML (sample count, phase mix, Stockfish depth for labeling and eval, MCTS sims, games per round, rounds, seeds, per-move budget), and the shared interfaces (`evaluate(positions)`, `select_move(board)`) that let arms swap in and out
