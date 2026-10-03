# Chess AI — Implementation Plan v3 (Level 1 / Level 2)

**Version:** v3, Sep 24, 2026. Supersedes v2 (archived outside the repo at `ChessAI/archive/chess_ai_implementation_plan_v2.md`, next to v1).
**Source:** `docs/framing.md` (v2, now the canonical framing), the project plan, and the architecture map v2

**Changes from v2:**
1. **Evaluation uses a frozen reference set** (Sec 0 Eval row, Sec 2.1 `eval/`, Steps 2 and 5). Stockfish `UCI_Elo` 1320 is the only pinned rating. Stockfish 1500 and 1700 are rated against it and frozen. Random and greedy stay in the set but are unrated until Step 5, when ~3 frozen SFT checkpoints connect them to the scale. Each new model is rated against the frozen set; only its own rating is fitted. Dropped from the set: Stockfish depth 1 (measured at ~1700, duplicating a rung; kept as a smoke-test opponent only) and Stockfish 2000.
2. **Checkpoint rungs** (Step 5): picked by strength, not by training step. **This selection rule is tentative; Derek may revise it later.** Rungs are frozen with adaptive game counts, starting at 20 games per neighbor pair.
3. **Illegal moves** (new Sec 0 row): three consecutive illegal attempts on one move forfeits the game, in eval and in self-play.
4. **Unmasked arms self-play unmasked** (Sec 0 LLM design row, Step 7): the masking toggle stays meaningful through RL, and illegal-move rate is tracked over training.
5. **Dollar cost** (Sec 0 Compute accounting row, `budget.py`, Steps 6 and 11): hours are converted at a pinned market rental rate (not the Colab bill). Reported as dollars to reach a given Elo, inference cost per game, and amortized training cost per game.
6. **Step 2 gate reduced** to what can be measured before any model exists.

**In-place edit, Sep 27, 2026:** the adaptive game-count rule now stops on the 95% CI alone; the "score must be 20–80%" trigger was dropped (Step 5, *Adaptive game counts*).
**In-place edit, Sep 27, 2026 (Step 3 start):** SFT games come from one Lichess monthly dump (Sec 3.2, Option A); `dataset.py` yields plain records and the prompt is built only in Step 4's `llm_policy.py` (Sec 2.1); phase-mix sampling and dedup stay in the Step 3 gate.

**In-place edit, Sep 30, 2026 (Step 5 freeze):** the Day-2 checkpoint-rung freeze (Sec 4, Step 5 "Freeze checkpoint rungs") was interrupted mid-run and reconstructed from aggregate win/draw/loss counts (`eval.elo.games_from_counts`) instead of a completed `rate_adaptively` run. Two departures from the rule below, discussed with Derek and accepted as provisional (revisit if there's a spare Colab window, not blocking): every CI came in wider than the +/-100 target (~110-125 Elo half-width, fewer games than a full adaptive run); and `step_15000` was dropped as a rung (too close to `step_20000` to be worth its own member), leaving a single ~600-Elo gap between `step_20000` (719) and `sf_1320` (1320) -- past the "<=300 Elo between neighbors" selection heuristic, which the rule below already marks tentative and Derek's to revise. Frozen: `random` = 525 [384, 624], `greedy` = 641 [495, 724], `step_20000` = 719 [581, 799], in `results/reference_set.json`.

**Scope:** Infrastructure, repo structure, dependencies, and execution order. Pseudocode and the modularity/config design come next.

---

## 0. Locked assumptions (from the framing doc)

These decisions are already made. Everything below depends on them.

| Decision | Choice |
|---|---|
| LLM family | **Pretrained small open-weight LLM**, fine-tuned (not a transformer trained from scratch) |
| Legal-move toggle | **Hard masking / constrained decoding** (on) vs. free generation (off). Listing legal moves in the prompt is *not* the toggle. |
| Illegal moves | An agent that makes **three consecutive illegal attempts on the same move forfeits the game** (it loses; termination `illegal_move`). Applies wherever games are played: eval matches and self-play. Masked agents can't trigger it. Every illegal attempt is logged. |
| Board representation toggle | FEN text vs. structured text (piece-per-square tokens). No image input. |
| LLM design | Phase 1: 2x2 (mask × representation), all trained with SFT warm-start + RL. Phase 2: best combo retrained under all 3 regimes (self-play RL only / SFT only / SFT + RL). The masking toggle applies in self-play too: unmasked arms play their self-play games unmasked (see Illegal moves), so how often they attempt illegal moves over RL training is itself an observed outcome. |
| AlphaZero | Small CNN policy-value net + MCTS, pure self-play |
| Board orientation | **LLM arms never flip**: they see absolute FEN/UCI, matching the notation the pretrained model learned from. **AlphaZero flips**: the CNN always sees the position from the side to move (board mirrored + colors swapped when Black moves), the standard AlphaZero practice, so it learns one set of patterns for both colors instead of two. Not flipping would handicap the AlphaZero arm's sample efficiency and bias the comparison against it. The Step 1 move vocabulary and planes stay absolute; the flip is a thin layer added in Step 9. |
| Terminology | **Game** = one full game. **Round** = a batch of self-play games (e.g., 50 games), followed by one training update. SFT has no rounds; it trains on a static dataset. |
| Eval | **Frozen reference set.** One pin: Stockfish `UCI_Elo` 1320 = 1320. The other members are rated once from games among themselves and then frozen: Stockfish 1500 and 1700 (Step 2); random mover, material-greedy player, and ~3 early SFT checkpoints that fill the gap between greedy and 1320 (Step 5). Each new model is rated against the frozen set and only its own rating is fitted, so the scale never drifts. A model can later be frozen into the set as a new rung. Stockfish members play at 0.1 s per move. |
| Compute accounting | GPU-hours (train + inference) as the shared currency. Stockfish labeling CPU time is charged to SFT and warm-start. Inference compute per move is capped in eval. **Also reported in dollars:** hours × a pinned **market rental rate** (one public hourly price per resource, with source and date, pinned in Step 6), not the Colab bill. Derived metrics: dollars to reach a given Elo; inference cost per game (the agent's own compute only, never the opponent's); amortized training cost per game at stated lifetime game counts (e.g. 1k / 100k / 10M). |
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
    framing.md                  # research intent (canonical)
    implementation_plan_v3.md   # this file
  requirements.txt
  configs/
    base.yaml          # paths, seed, GPU type, compute budget
    data.yaml          # dataset size, phase mix, Stockfish label depth
    sft.yaml           # SFT hyperparameters
    llm_rl.yaml        # LLM self-play RL hyperparameters
    alphazero.yaml     # network size, MCTS sims, games/round, rounds
    eval.yaml          # reference set, games per matchup, per-move budget
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
- **`sample_positions.py`** — Extracts positions from games, tags each as opening, middlegame, or endgame, and samples to a target phase mix set in config. Removes duplicate positions. Phase rule (thresholds in config): endgame if little non-pawn material is left, else opening if early in the game, else middlegame. At most one position per phase per game, so no single game dominates.
- **`annotate.py`** — Runs Stockfish over the sampled positions in parallel to attach the best move and the evaluation (converted to a win probability). Writes Parquet to Drive in chunks, so a dropped Colab session resumes where it stopped, and logs the total Stockfish CPU time spent.
- **`dataset.py`** — PyTorch `Dataset` that yields plain records for the chosen board representation: the board as text (structured or FEN), the label move (UCI), and the label win-probability. Splits train/validation **by game**, so positions from one game never land on both sides. It does not build prompts or tokenize: the one shared prompt template lives in `model/llm_policy.py` (Step 4), so training and inference read identical text.

**`model/`**
- **`llm_policy.py`** — Loads the pretrained model and tokenizer, attaches LoRA adapters and a small value head (predicts win likelihood from the final hidden state), and builds prompts. Implements move selection both ways: **masked** (score only legal moves and pick from them) and **unmasked** (free generation, then parse; an illegal move is logged and retried, and the third consecutive illegal attempt forfeits the game, per Sec 0).
- **`az_net.py`** — Small ResNet-style CNN for AlphaZero with a policy head over the shared move vocabulary and a value head. Size (blocks, channels) is set in config.

**`search/`**
- **`mcts.py`** — PUCT Monte Carlo tree search that works with any model exposing `(policy, value) = evaluate(positions)`. Batches leaf evaluations into single GPU calls. Written model-agnostic so an optional LLM + MCTS arm can reuse it later.
- **`selfplay.py`** — Runs many self-play games concurrently (concurrency is a config value) and records training examples: (position, move or search policy, final outcome). Used by both AlphaZero and LLM self-play RL.

**`train/`**
- **`sft.py`** — Supervised fine-tuning on Stockfish labels: cross-entropy on the move plus a value loss on the win probability. Logs positions seen, GPU time, and tokens, and saves checkpoints for eval curves.
- **`llm_rl.py`** — LLM self-play RL: play a round of games, assign the final win/loss/draw result as the reward to every move in that game, and update with a policy-gradient loss (with a baseline, plus a KL penalty toward the starting model to keep it stable). Can start from the base model (pure RL) or an SFT checkpoint (warm-start). Unmasked arms self-play unmasked: a forfeit by illegal moves counts as a loss, and the illegal-move rate is logged per round.
- **`az_train.py`** — AlphaZero loop: self-play round → replay buffer → gradient updates → checkpoint, repeated for N rounds. Rounds, games per round, and MCTS sims come from config.

**`eval/`**
- **`anchors.py`** — Defines the reference set: random mover, material-greedy player, and Stockfish at `UCI_Elo` 1320 / 1500 / 1700 (0.1 s per move), plus a frozen registry (one flat CSV, `results/reference_set.csv` on Drive — revised Oct 3, 2026, was JSON) that trained checkpoints are added to as rungs (weights hash + locked play settings, JSON-encoded in the row's own cell). Stockfish 1320 is the only pinned rating (its own row, no interval); every other member is rated once, then frozen. Stockfish at depth 1 is kept as a smoke-test opponent only.
- **`match.py`** — Plays N games of agent vs. opponent with alternating colors, enforcing the per-move inference budget (time, or MCTS sims) and the three-strikes illegal-move forfeit. Records results, PGNs, move latency, and illegal-move attempts.
- **`elo.py`** — Fits Elo ratings with bootstrap confidence intervals, holding the pinned and frozen ratings fixed, and flags ratings the games can't pin down (e.g. an agent that lost every game). Used to rate the reference set once (with adaptive game counts when freezing a rung) and then to rate each new model against it, fitting only the new model. `append_rating_log` saves each such evaluation (one row per model, per time it's evaluated) to `results/model_ratings.csv` -- the running history Elo-vs-samples and Elo-vs-compute curves are built from (joined to `results/experiments.csv` on run/step), distinct from `anchors.py`'s frozen, never-re-rated registry.

**`utils/`**
- **`config.py`** — Loads and merges YAML configs and applies notebook overrides. Every tunable number lives in config, never in code.
- **`budget.py`** — Compute meter: GPU-seconds, Stockfish CPU-seconds, tokens processed, and a rough FLOPs estimate per run. Can stop training when the budget is spent, which enforces equal compute across arms. Converts hours to dollars with the pinned market-rate price table (Sec 0).
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
| **Stockfish (oracle and eval reference)** | Download the official Linux binary from the Stockfish GitHub releases page into the Colab runtime (Stockfish 19+ ships one `stockfish-linux-x86-64-universal` build; there is no separate `avx2` asset anymore). Fallback: `apt-get install stockfish` (older version). Drive it from Python with `chess.engine.SimpleEngine.popen_uci`. | Pin the version and log it. Use a fixed depth or node count for labeling. Use `UCI_LimitStrength` + `UCI_Elo` for eval levels (minimum is 1320, which is why the reference set extends below it with random, greedy, and frozen checkpoints). |
| **Lichess games (SFT source)** | Option A: monthly rated-standard PGN dumps from `database.lichess.org`. Use an **older month** (early years are hundreds of MB rather than tens of GB). Option B: stream the Lichess games dataset on the Hugging Face Hub with `datasets` (`streaming=True`). | **Decided: Option A.** One older month (default `2013-06`, ~225k games), pinned by name in `configs/data.yaml`, checked against Lichess's published `sha256sums.txt`, and cached in Drive `data/raw/`. Same file, same games, every time. You only need tens of thousands of positions. |
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

**Step 2 — Eval harness with the reference set** (`eval/`, `03_evaluate.ipynb`)
Build it *before* any model so every later model is measured the same way. Only the Stockfish end of the reference set can be rated now: random and greedy lose every game to Stockfish 1320, so their ratings can't be fitted until the Step 5 checkpoints connect them to the scale. Smoke-test the harness on known cases (random vs. Stockfish depth 1 loses almost every game; an always-illegal agent forfeits on its third attempt). Play Stockfish 1320 / 1500 / 1700 against each other, rate 1500 and 1700 with 1320 pinned (same adaptive game counts as Step 5), and freeze them. Time how long a game takes on Colab's CPUs.
*Gate:* smoke tests pass; Stockfish 1500 and 1700 are rated with sensible confidence intervals and frozen; game timings recorded. The weak end of the scale is deferred to Step 5.

**Step 3 — SFT data pipeline** (`data/`, `01_build_dataset.ipynb`)
Download → sample by phase → label with Stockfish → Parquet in Drive. Start with ~5–10k positions to check timing, then build the full set (tens of thousands). Phase-mix sampling and dedup are in scope for the Day-1 dataset.
*Gate:* labeled dataset in Drive, phase mix matches config, labeling CPU time logged.

**Step 4 — LLM policy wrapper** (`model/llm_policy.py`)
Load the model with LoRA and a value head; implement masked and unmasked move selection for both representations.
*Gate:* the untrained base model plays a full game against the random mover in all 4 toggle combinations without crashing (a forfeit by illegal moves counts as a finished game).

**Step 5 — SFT baseline end to end** (`train/sft.py`, `02_train_sft.ipynb`) — **Day-1 milestone**
Train one config (FEN + masking on) on a small slice, evaluate the checkpoints in the harness.

**Freeze checkpoint rungs** (completes the reference set). Pick ~3 of the checkpoints training already saves to fill the gap between greedy and Stockfish 1320:
- **Selection (tentative; Derek may revise this later):** pick **by strength, not by training step**. Score the saved checkpoints quickly and choose ones spread evenly through the gap, so neighboring rungs score 20–80% against each other (about 150–300 Elo apart). Fixed step intervals would bunch the rungs together, because learning is fast early and slow late.
- **How many:** enough that no gap between neighbors exceeds ~300 Elo; 3 is the starting guess.
- **Adaptive game counts:** start at 20 games per neighbor pair; add 20 more to every matchup of a rung whose 95% CI is still wider than ±100 Elo (or whose rating is off the scale), up to a cap of 100 games per matchup. A matchup that hits the cap is reported for a manual call, not frozen silently. Why the CI alone: lopsided matchups do need more games, but that already shows up as a wider CI; a separate "score must be 20–80%" rule can never be met by a genuinely large gap, so it would only burn games to the cap. The CI matters here beyond absolute numbers: each rung's error lands in the spacing between rungs, and so in the shape of the Elo-vs-training curve.
- **Freezing:** record the rating, hash the weights, lock the play settings (always the top legal move), and never retrain or re-rate it. Random and greedy get their ratings in the same fit, through the new rungs.
- **Known bias:** checkpoint rungs play like an LLM, which may slightly favor the LLM arms. The non-LLM members keep this in check (Step 9 can add an AlphaZero rung).

*Gate (revised Oct 1, 2026 -- see note below):* a reference set connected from random up to Stockfish 1700, with every member rated, a confidence interval, and frozen.

> **In-place edit, Oct 1, 2026:** dropped "an Elo number with a confidence interval that beats the untrained model" from this gate. Derek's call: the thing Step 5 actually needs to hand off to Step 7 is a reference set with CIs to rate RL checkpoints against, not a beat-the-untrained-model check on the SFT run itself -- that's a useful sanity signal, but not a gate on whether the reference set is usable.

**Step 6 — Lock the compute budget**
Using timings from Steps 2–5, set the fixed GPU-hour budget per arm and the per-move inference cap, pin the market-rate price table (Sec 0, Compute accounting), and write them to `base.yaml` / `eval.yaml`. Move to the L4 for all reported runs from here.

**Step 7 — LLM self-play RL** (`search/selfplay.py`, `train/llm_rl.py`, `04_train_llm_rl.ipynb`)
Start from the SFT checkpoint (warm-start) and confirm Elo improves over the SFT baseline across a few rounds.
Log the illegal-move rate per round. Risk to watch: an unmasked arm that starts RL without SFT may forfeit most early games by illegal moves, producing short games with little chess signal, and stall.
*Gate:* stable training (no collapse), and Elo is flat or rising.

**Step 8 — LLM Phase 1: 2x2 sweep**
Run all 4 combos (mask × representation) under SFT + RL at equal budget, 2–3 seeds each. Pick the best combo.

**Step 9 — AlphaZero arm** (`model/az_net.py`, `search/mcts.py`, `train/az_train.py`, `05_train_alphazero.ipynb`)
Start with `tests/test_mcts.py` (mate-in-1), then run parallel self-play training at the same budget, 2–3 seeds. Before training, add side-to-move flipping for the CNN (Sec 0, Board orientation): mirror the board with `board.mirror()` before `to_planes`, and map the network's move outputs back through a fixed 1968-entry mirror table (e.g. `e7e5` ↔ `e2e4`); flip the stored search policies the same way. Add a test that flipping twice returns the original position and move, and that flipped legal masks still match python-chess. This is independent of Steps 7–8, so it can run in a separate session alongside them. Optionally, freeze an early AlphaZero checkpoint into the reference set (same procedure as Step 5) to balance the LLM checkpoint rungs.

**Step 10 — LLM Phase 2: regime comparison**
Retrain the best combo under pure self-play RL, SFT only, and SFT + RL at equal budget, 2–3 seeds each.

**Step 11 — Final eval and analysis** (`06_analysis.ipynb`)
Scale up eval games for final numbers. Produce Elo with confidence intervals, Elo-vs-samples curves (sample efficiency), Elo-vs-compute (in GPU-hours and dollars), training time, inference time per move, inference cost per game, amortized training cost per game, and illegal-move rate over training (unmasked arms) for every arm. These map directly to the presentation slides.

**Optional stretch:** LLM + MCTS arm, reusing `search/mcts.py` with the LLM's policy and value heads.

---

## 5. Next alignment points
1. Pseudocode for each file
2. Modularity and config design: which parameters go in which YAML (sample count, phase mix, Stockfish depth for labeling and eval, MCTS sims, games per round, rounds, seeds, per-move budget), and the shared interfaces (`evaluate(positions)`, `select_move(board)`) that let arms swap in and out
