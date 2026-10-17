# Chess AI — Implementation Plan v4 (Level 1 / Level 2)

**Version:** v4, Oct 10, 2026. Supersedes v3 (archived outside the repo at `ChessAI/archive/chess_ai_implementation_plan_v3.md`, next to v1 and v2).
**Source:** `docs/framing.md` (canonical framing), plan v3, and the Oct 10 80/20 review with Derek.

**Guiding principle (new in v4): this is an MVP, run 80/20.** Track the few numbers the final slides need, in as few files as possible. No metric, log file, or rigor step stays unless it feeds one of the outputs in Step 11.

**Changes from v3:**
1. **No strict equal compute budgets** (Sec 0 Compute accounting). Each arm trains until it plateaus or hits a loose safety cap, and fairness comes from plotting **Elo vs. compute** curves. Equal budgets made the training-cost bars identical by construction. The old "locked budget" (7.69 GPU-h) was also the SFT run's own spend, which left nothing for RL in the SFT + RL arm.
2. **One seed per arm** (Sec 0 Rigor). More seeds can be added later for the final comparison if time allows.
3. **One results file** (Sec 0 Eval, `eval/anchors.py`). `results/results.csv` holds the pin, the frozen reference set, and every evaluated model checkpoint, with a `role` column. It replaces `reference_set.csv`, `model_ratings.csv`, and `experiments.csv`.
4. **Simpler eval of new models** (Sec 0 Eval, Steps 7–11). 20 games against each of the 2–3 nearest reference opponents, played in batches of 10, with the row saved after every batch, so stopping early always keeps the latest number. There is no CI target, and no more rungs get frozen unless a model's CI is unusable (wider than ±200).
5. **Cut:** the per-move inference cap and the inference benchmark, the market-rate price table, `utils/budget.py`, `utils/logging.py`, `utils/checkpoint.py`, `utils/config.py`, FLOPs and token counting, W&B, Stockfish wall-clock logging, the optional AlphaZero rung, and the provisional/frozen status mechanism. Seconds per move now come from the eval games themselves. Dollars are hours × one hourly rate, computed in the analysis notebook.
6. **Compute moves to RunPod** (Sec 1–2) for Step 6 onward: an on-demand L4 pod, a network volume for artifacts, and headless notebooks. Colab and Drive served Steps 0–5.

---

## 0. Locked assumptions

These decisions are made. Everything below depends on them.

| Decision | Choice |
|---|---|
| LLM family | **Pretrained small open-weight LLM**, fine-tuned (not a transformer trained from scratch). `Qwen/Qwen3-0.6B` (Apache-2.0), `enable_thinking=False`. At 0.6B this is the hybrid model with thinking switched off, not a non-thinking specialist; note that in the writeup. |
| Legal-move toggle | **Hard masking / constrained decoding** (on) vs. free generation (off). Listing legal moves in the prompt is *not* the toggle. |
| Illegal moves | **Three consecutive illegal attempts on the same move forfeit the game** (termination `illegal_move`), in eval and in self-play. Masked agents can't trigger it. |
| Board representation toggle | FEN text vs. structured text (piece-per-square tokens). No image input. |
| LLM design | Phase 1: 2x2 (mask × representation), all SFT warm-start + RL. Phase 2: the best combo under 3 regimes (self-play RL only / SFT only / SFT + RL). Unmasked arms self-play unmasked, so their illegal-move rate over RL is an observed outcome. |
| AlphaZero | Small CNN policy-value net + MCTS, pure self-play. |
| Board orientation | LLM arms never flip (absolute FEN/UCI). AlphaZero flips to the side to move (board mirrored + colors swapped when Black moves), added as a thin layer in Step 9. |
| Terminology | **Game** = one full game. **Round** = a batch of self-play games followed by one training update. SFT has no rounds. |
| Eval | **Frozen reference set, complete as of Step 5:** `sf_1320` = 1320 (the pin), `sf_1500`, `sf_1700`, `random`, `greedy`, `step_20000`. Each new model is rated against it and only its own rating is fitted. Stockfish members play at 0.1 s per move. **New models:** 20 games against each of the 2–3 reference opponents nearest its expected strength, in batches of 10; after each batch the model's row in `results.csv` is overwritten with the latest Elo + 95% CI. There is no CI target: report what you get, and add +20 games by hand only if a CI is wider than ±200. A model may later be frozen as a new rung (role `model` → `reference`) only if CIs in the 719–1320 gap turn out unusable. |
| Source of truth | **One CSV, `results/results.csv`** on the artifact volume. One row per player, with `role` = `pin` / `reference` / `model`. The eval code reads the pin + reference rows as fixed opponents; `model` rows are what gets plotted. Each run's own folder keeps scratch logs (`train_log.csv`, `val_log.csv`) for loss curves; nothing else reads them. |
| Compute accounting | **No equal budgets.** Each arm trains until Elo plateaus or it hits a loose safety cap (`max_gpu_hours_per_arm` in `base.yaml`, set in Step 7 once RL speed is known). Fairness comes from **Elo vs. training GPU-hours curves**: 3–4 checkpoints per run are evaluated, and each row carries the cumulative training GPU-hours at that checkpoint (an in-code timer). Stockfish labeling CPU-hours (0.99 h for the 50k dataset) are charged to every SFT-using arm. **Dollars** = GPU-hours × `gpu_usd_per_hour` + CPU-hours × `cpu_usd_per_hour` (RunPod on-demand prices, source + date, in `base.yaml`), computed in the analysis notebook. **Inference cost** = seconds per move measured in the eval games × the same rate. There is no per-move inference cap; AlphaZero's MCTS simulations per move is a config value and shows up in its sec/move. |
| Rigor | **1 seed per arm.** Seeds can be added later for the final comparison if time allows. |
| Sequencing | SFT-on-Stockfish baseline end to end first (done, Step 5). |

**Still open:** the exact structured-text format is settled in code (Step 4). `max_gpu_hours_per_arm` is set in Step 7.

---

## 1. Hardware and service provider

**Step 6 onward: RunPod.** Steps 0–5 ran on Google Colab, with artifacts on Google Drive.

| Use | Where | GPU | Why |
|---|---|---|---|
| Steps 0–5 (done) | Colab (free T4 / Pro L4) | T4 / L4 | Dev, the SFT baseline, and the reference set. The SFT run's 7.69 GPU-h were on an L4. |
| **Step 6 onward (all reported runs)** | **RunPod, on-demand pod** | **NVIDIA L4, 24 GB** | Runs unattended for many hours and survives closing the laptop. Same GPU type as the SFT run, so SFT hours and RL hours add up on one curve. On-demand rather than spot, so pods aren't killed mid-run. |

**Rules:**
- **Artifacts live on a RunPod network volume** mounted at `/workspace`: `artifact_root = /workspace/chess-ai` (`data/processed`, `checkpoints/`, `results/`). The volume outlives the pod. Copy `results/results.csv` to the laptop now and then as a backup.
- **Code lives in GitHub only.** Clone once onto the volume (`/workspace/Chess_Bot`), then `git pull` at the start of each session. Code is never edited on the pod.
- **A pod must always have the network volume attached.** Create the volume first (Storage page), then deploy the pod from it, in the same datacenter. A pod without a volume is terminated, with its data lost, when the balance hits $0 (this happened once). Check the Storage page after any long idle spell.
- **Long jobs run as headless notebooks** inside `tmux`, with a budget cap so the balance never reaches $0. Set `BAL` (balance now), `FLOOR` (never go below, ~$2, enough for storage) and `RATE` (the L4 $/hr), then:
  ```
  HOURS=$(python -c "print(round(($BAL-$FLOOR)/$RATE*0.9, 2))")
  timeout ${HOURS}h jupyter nbconvert --to notebook --execute notebooks/XX.ipynb --output-dir /workspace/chess-ai/logs --ExecutePreprocessor.timeout=-1
  runpodctl stop pod $RUNPOD_POD_ID
  ```
  Closing the browser or laptop doesn't stop it, the `timeout` ends the job at the floor, and the stop line always runs, since an idle pod still bills. RunPod has no API we rely on for the balance, so `BAL` is typed by hand at launch. Turn on the low-balance email alert (no auto-pay).
- **Secrets** (`HF_TOKEN`, a GitHub token if the repo is private) are RunPod environment variables.
- Log the GPU name in every results row. Discard or flag any reported run that isn't on an L4.
- Every training script is resumable (checkpoint every N steps / every round to the volume).
- Eval games are CPU-bound, and the pod's vCPU count differs from Colab's. Time one 20-game match in Step 6 before planning eval sizes.

---

## 2. Repository and notebook structure

**Pattern: one GitHub repo of plain `.py` modules plus thin notebooks.** All logic lives in `.py` files; notebooks set parameters and call functions. Each notebook runs interactively in JupyterLab on the pod, or headless for long jobs.

```
Chess_Bot/
  README.md
  CLAUDE.md          # context for Claude Code; imports this plan
  docs/
    framing.md                  # research intent (canonical)
    implementation_plan_v4.md   # this file
    jobs_to_be_done.md          # the JTBD view of this plan
  requirements.txt
  configs/
    base.yaml          # artifact_root, gpu_type, hourly rates, max_gpu_hours_per_arm
    data.yaml          # dataset month, size, phase mix, Stockfish label depth
    sft.yaml           # policy + SFT hyperparameters
    llm_rl.yaml        # LLM self-play RL hyperparameters
    alphazero.yaml     # network size, MCTS sims, games/round, rounds
    eval.yaml          # games per opponent, batch size
  engine/   board.py  moves.py  encoding.py  stockfish.py
  data/     download.py  sample_positions.py  annotate.py  dataset.py
  model/    llm_policy.py  az_net.py
  search/   mcts.py  selfplay.py
  train/    sft.py  llm_rl.py  az_train.py
  eval/     anchors.py  match.py  elo.py
  notebooks/
    00_setup.ipynb  01_build_dataset.ipynb  02_train_sft.ipynb  03_evaluate.ipynb
    04_train_llm_rl.ipynb  05_train_alphazero.ipynb  06_analysis.ipynb
  tests/    test_engine.py  test_data.py  test_llm_policy.py  test_sft.py  test_eval.py  test_mcts.py
```

There is no `utils/` folder (cut in v4). Checkpoint/resume logic lives in each `train/*.py`, and configs are read with `yaml.safe_load`.

### 2.1 What each file does

**`engine/` — the chess backbone**
- **`board.py`** — Thin wrapper around `python-chess`: positions, moves, legal moves, and terminal states (incl. repetition and the 50-move rule).
- **`moves.py`** — One canonical move vocabulary (1,968 UCI moves) with move↔index mapping and a legal-move mask. Shared by AlphaZero, MCTS, and the LLM's masked decoding.
- **`encoding.py`** — Board → FEN text, structured piece-per-square text, or CNN tensor planes.
- **`stockfish.py`** — Starts and drives the pinned Stockfish 19: labeling (best move + eval) and limited-strength play. Records labeling CPU time.

**`data/` — SFT data pipeline**
- **`download.py`** — Fetches the pinned Lichess month (`2013-06`), checks its sha256, and streams it.
- **`sample_positions.py`** — Positions tagged by phase, sampled to the configured phase mix, deduplicated, at most one per phase per game.
- **`annotate.py`** — Parallel, resumable Stockfish labeling to Parquet. Logs total CPU time in `manifest.json`.
- **`dataset.py`** — Plain records (board text, label move, label win-probability), split train/validation by game. Prompts are built only in `llm_policy.py`.

**`model/`**
- **`llm_policy.py`** — Qwen3-0.6B + LoRA + a value head, the shared prompt, and masked/unmasked move selection.
- **`az_net.py`** — Small ResNet-style CNN with policy (shared vocabulary) and value heads.

**`search/`**
- **`mcts.py`** — PUCT MCTS over any `(policy, value) = evaluate(positions)` model, with batched leaf evaluation.
- **`selfplay.py`** — Concurrent self-play games recording (position, move or search policy, outcome). Used by LLM RL and AlphaZero.

**`train/`**
- **`sft.py`** — SFT on Stockfish labels (move cross-entropy + value loss), resumable. Writes `train_log.csv` / `val_log.csv` (scratch, per run), including the cumulative `gpu_seconds_train` the results row needs.
- **`llm_rl.py`** — Self-play policy-gradient RL (game result as reward, a value baseline, a KL penalty toward the start model), from the base model or an SFT checkpoint, resumable. Its per-round log in the run folder includes the illegal-move rate (unmasked arms) and cumulative training GPU-seconds.
- **`az_train.py`** — AlphaZero loop: self-play round → replay buffer → updates → checkpoint, resumable, with the same cumulative GPU-seconds counter.

**`eval/`**
- **`anchors.py`** — The reference agents (random, greedy, Stockfish at UCI_Elo 1320/1500/1700, 0.1 s/move) and **`ResultsTable`**, the only reader/writer of `results/results.csv`. It returns the fixed ratings (pin + reference rows), upserts a model's row after each eval batch, and refuses to overwrite a pin or reference row.
- **`match.py`** — Plays games with color-swapped pairs and the three-strikes forfeit, recording results, latency, and illegal attempts. `evaluate_agent` rates one model against the reference set in batches and calls back after each batch. `rate_adaptively` (used to build the reference set in Steps 2 and 5) stays for any future rung.
- **`elo.py`** — Elo fit with bootstrap CIs, holding fixed ratings fixed and flagging off-scale ratings.

**`notebooks/`** — Thin drivers: setup, build dataset, train SFT (+ evaluate its checkpoints), evaluate, train LLM RL, train AlphaZero, analysis/plots.

**`tests/`** — Quick checks before long runs: move vocabulary round-trips, legal masks match `python-chess`, SFT loss and resume, eval harness and the results table, and MCTS finds mate-in-1.

### 2.2 `results/results.csv` schema

| column | meaning |
|---|---|
| `name` | player id, e.g. `sf_1320`, `step_20000`, `sft_structured_masked_s0@25000` |
| `role` | `pin` / `reference` / `model` |
| `rating`, `ci_low`, `ci_high`, `games` | Elo, 95% bootstrap CI, games played |
| `run`, `step`, `seed` | training run folder name, checkpoint step, seed (blank for non-trained players) |
| `train_gpu_hours` | cumulative training GPU-hours at this checkpoint |
| `label_cpu_hours` | Stockfish labeling CPU-hours charged (0.99 for SFT-using arms, else 0) |
| `sec_per_move` | mean seconds per move in its eval games |
| `illegal_rate` | illegal attempts ÷ moves in its eval games |
| `gpu`, `stockfish_version`, `date` | provenance |
| `spec` | JSON: how to rebuild the player (reference rows; e.g. a rung's run, step, weights hash) |
| `note` | free text |

---

## 3. Dependencies and external resources

### 3.1 Python packages
| Package | Purpose |
|---|---|
| `python-chess` | Rules, move generation, PGN parsing, UCI engine interface |
| `torch` | Models and training |
| `transformers`, `peft`, `accelerate`, `torchao` | Load the LLM, LoRA fine-tuning |
| `datasets`, `huggingface_hub` | Model download |
| `zstandard` | Decompress the Lichess `.pgn.zst` dump |
| `pandas`, `pyarrow`, `numpy` | Dataset storage (Parquet), results table, analysis |
| `pyyaml` | Configs |
| `jupyter`, `nbconvert` | Headless notebook runs on RunPod (usually preinstalled in RunPod PyTorch images) |

### 3.2 External resources

| Resource | How to get it | Notes |
|---|---|---|
| **LLM** | Hugging Face Hub, `Qwen/Qwen3-0.6B`; `HF_TOKEN` as a RunPod env var | Apache-2.0, ungated. |
| **Stockfish** | `engine.stockfish.ensure_stockfish()` downloads the official Stockfish 19 Linux binary into the repo folder | Pinned; never upgrade without discussing it. |
| **Lichess games** | One monthly dump from `database.lichess.org` (`2013-06`), sha256-checked | Already labeled (50k positions); the raw dump isn't needed again. |
| **GitHub** | Repo cloned once onto the volume | Token as an env var if private. |
| **RunPod** | On-demand L4 pod + network volume (same datacenter) | `runpodctl` for file transfer (`send`/`receive`) and stopping pods. |

No paid model APIs are needed.

---

## 4. Execution order

Each step ends with a **gate**: don't move on until it passes.

**Step 0 — Setup** — *done (Colab).* One notebook cell brings up a working environment.

**Step 1 — Chess backbone** (`engine/`, `tests/test_engine.py`) — *done.* Tests pass; Stockfish labels any FEN; all three encodings print.

**Step 2 — Eval harness with the reference set** (`eval/`, `03_evaluate.ipynb`) — *done.* Smoke tests pass; `sf_1500` = 1580 [1500, 1673] and `sf_1700` = 1731 [1654, 1834] are frozen against the `sf_1320` pin.

**Step 3 — SFT data pipeline** (`data/`, `01_build_dataset.ipynb`) — *done.* 50,000 positions from Lichess `2013-06`, exact phase mix, train/val split by game; Stockfish labeling CPU 0.99 h.

**Step 4 — LLM policy wrapper** (`model/llm_policy.py`) — *done.* The untrained model finished a game vs. random in all 4 toggle combinations.

**Step 5 — SFT baseline end to end** (`train/sft.py`, `02_train_sft.ipynb`) — *done.*
Structured text + masking on, trained on the full 44,982-position train split, stopped at step 25,000 (7.69 L4 GPU-hours). Frozen checkpoint rung: `step_20000` = 719 [581, 799]; `random` = 525 [384, 624] and `greedy` = 641 [495, 724] were rated through it. CIs are ~±115 (wider than the original ±100 target) and there is a ~600-Elo gap from 719 to 1320. Both are accepted: v4 drops the CI target and adds rungs only if needed.
*Gate (met):* a reference set connected from random up to Stockfish 1700, every member rated with a CI.
*Carry-over to Step 6:* the SFT arm's 3–4 checkpoints are evaluated into `results.csv` as the first points on its Elo-vs-compute curve.

**Step 6 — Move to RunPod and the single results file** (`eval/anchors.py`, `eval/match.py`, setup cells in every notebook, `configs/base.yaml`)
1. Code: `results.csv` with `role` replaces the three old CSVs; batched `evaluate_agent`; cuts listed in Changes from v3 #5.
2. RunPod: create a network volume and an on-demand L4 pod (PyTorch image), clone the repo to `/workspace/Chess_Bot`, set `HF_TOKEN`.
3. Migrate artifacts from Drive to `/workspace/chess-ai`: the processed dataset, the SFT run folder (at least `step_20000`, `step_25000`, and the run's logs), and `reference_set.csv` (converted once to `results.csv`).
4. Pin `gpu_usd_per_hour` (RunPod L4 on-demand) and `cpu_usd_per_hour`, with source + date, in `base.yaml`.
5. Smoke test on the pod: `00_setup` passes; evaluate 3–4 SFT checkpoints (`02_train_sft`'s eval section) into `results.csv`; time a 20-game match.
*Gate:* `results.csv` on the volume holds the pin, 5 reference rows, and the SFT checkpoints with Elo, CI, train GPU-hours, and sec/move, all produced on the pod.

**Step 7 — LLM self-play RL** (`search/selfplay.py`, `train/llm_rl.py`, `04_train_llm_rl.ipynb`)
Warm-start from the SFT checkpoint (structured + masked). Play rounds of self-play, update, and checkpoint each round. Log the illegal-move rate and cumulative GPU-seconds per round in the run folder. Evaluate 3–4 checkpoints into `results.csv`. Set `max_gpu_hours_per_arm` from the observed RL speed. Risk to watch: an unmasked arm without SFT may forfeit most early games and stall.
*Gate:* stable training (no collapse), and Elo is flat or rising across evaluated checkpoints.

**Step 8 — LLM Phase 1: 2x2 sweep**
All 4 combos (mask × representation) under SFT + RL, 1 seed each, each trained to plateau or the cap. The 3 new combos need their own SFT run first. Evaluate 3–4 checkpoints per run. Pick the best combo from the Elo-vs-compute curves.

**Step 9 — AlphaZero arm** (`model/az_net.py`, `search/mcts.py`, `train/az_train.py`, `05_train_alphazero.ipynb`)
Start with `tests/test_mcts.py` (mate-in-1). Add side-to-move flipping: `board.mirror()` before `to_planes`, a fixed 1,968-entry move mirror table, and stored search policies flipped the same way, plus a test that flipping twice round-trips and that flipped masks match python-chess. Then train by pure self-play, 1 seed, to plateau or the cap, and evaluate 3–4 checkpoints. This is independent of Steps 7–8, so it can run on a separate pod.

**Step 10 — LLM Phase 2: regime comparison**
The best combo under pure self-play RL, SFT only, and SFT + RL, 1 seed each, to plateau or the cap, with 3–4 checkpoints evaluated per run.

**Step 11 — Final analysis** (`06_analysis.ipynb`, reads only `results.csv`)
- Elo ± CI bar chart, one bar per arm (best checkpoint).
- Elo vs. training GPU-hours (and dollars) curves per arm: sample efficiency.
- Dollars to train (GPU + labeling CPU) and inference dollars per game (sec/move × moves/game × rate) per arm.
- Illegal-move rate for the unmasked arms.
- Optional: rescale by playing more games for the arms on the final slides.

**Optional stretch:** an LLM + MCTS arm, reusing `search/mcts.py` with the LLM's policy and value heads.

---

## 5. Next alignment points
1. Self-play RL design details (round size, KL weight, baseline), at the start of Step 7.
