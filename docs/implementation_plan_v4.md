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

**Update, Oct 17, 2026 (edited in place, still v4):**
7. **Focus on structured + masked.** The LLM work concentrates on one configuration (structured text, masking on) and on the questions that matter most: SFT vs. self-play RL, and LLM + MCTS as a scaffold. The 2x2 sweep (Step 8) is **deferred**, not cut. The unmasked and FEN code paths stay in the repo.
8. **LLM + MCTS is promoted** from optional stretch to a planned arm, after AlphaZero (it reuses `search/mcts.py`). Its design is an alignment point (Sec 5).
9. **Step 7 no longer logs illegal moves** (a masked model can't make one) **and no longer sets the safety cap**; `max_gpu_hours_per_arm` is decided later.
10. **Step 7's RL design** is set as provisional defaults in `configs/llm_rl.yaml` (see Step 7), to revisit after the test run.

**Update, Oct 21, 2026 (edited in place, still v4):**
11. **Step 7 merges into `main` before run 3's evals finish** (runs 1-2 already met its gate). Each RL run's code is tagged so its results can be reproduced.
12. **AlphaZero (Step 9) is specified in implementation detail** (Step 9 (a)-(h)) so it can be built in a separate session, on `step-9-alphazero` cut from `main`, and trained on a second pod while the LLM work continues.
13. **AlphaZero's cap is staged: 9, then 18, then 30 GPU-hours**, each extension by resuming the same run. 18 matches the LLM's SFT + RL total; 30 is a placeholder for the LLM + MCTS arm's total.
14. **`results.csv` saves merge instead of overwrite** (lock file + re-read), so two eval jobs on two pods can't erase each other's rows.

**Update, Oct 25, 2026 (edited in place, still v4):**
15. **Step 7 gets a 4th run** (Derek's call), beyond the original 3: run 3's config with a 5x learning rate, to test whether update size, not the reward, is the bottleneck. Run 3's result is recorded (flat).

**Update, Oct 31, 2026 (edited in place, still v4):**
16. **Run 4 is flat too:** no LLM RL configuration (runs 1-4) moved Elo off the SFT plateau.
17. **AlphaZero (Step 9) is built and stage 1 is done; its gate is met.** Settings confirmed after the smoke run with 50 simulations per move (not 100): search is CPU-bound on the pod. Stage 1 (9 GPU-h, no data, no Stockfish): Elo 653 -> 729, level with the best LLM checkpoints.
18. **Next, one step at a time:** AlphaZero no-search evals (how much of its Elo is search), then stage 2 (18 GPU-h, same opponents), then the LLM + MCTS arm as play-time search first (option A, Sec 5 item 2).

---

## 0. Locked assumptions

These decisions are made. Everything below depends on them.

| Decision | Choice |
|---|---|
| LLM family | **Pretrained small open-weight LLM**, fine-tuned (not a transformer trained from scratch). `Qwen/Qwen3-0.6B` (Apache-2.0), `enable_thinking=False`. At 0.6B this is the hybrid model with thinking switched off, not a non-thinking specialist; note that in the writeup. |
| Legal-move toggle | **Hard masking / constrained decoding** (on) vs. free generation (off). Listing legal moves in the prompt is *not* the toggle. |
| Illegal moves | **Three consecutive illegal attempts on the same move forfeit the game** (termination `illegal_move`), in eval and in self-play. Masked agents can't trigger it. |
| Board representation toggle | FEN text vs. structured text (piece-per-square tokens). No image input. |
| LLM design | **Current focus (Oct 17): structured text + masking on.** On that config: 3 regimes (SFT only / SFT + RL / self-play RL only), then LLM + MCTS. The 2x2 (mask × representation, Phase 1) is deferred; if it returns, unmasked arms self-play unmasked, so their illegal-move rate over RL is an observed outcome. |
| AlphaZero | Small CNN policy-value net + MCTS, pure self-play. |
| Board orientation | LLM arms never flip (absolute FEN/UCI). AlphaZero flips to the side to move (board mirrored + colors swapped when Black moves), added as a thin layer in Step 9. |
| Terminology | **Game** = one full game. **Round** = a batch of self-play games followed by one training update. SFT has no rounds. |
| Eval | **Frozen reference set, complete as of Step 5:** `sf_1320` = 1320 (the pin), `sf_1500`, `sf_1700`, `random`, `greedy`, `step_20000`. Each new model is rated against it and only its own rating is fitted. Stockfish members play at 0.1 s per move. **New models:** 20 games against each of the 2–3 reference opponents nearest its expected strength, in batches of 10; after each batch the model's row in `results.csv` is overwritten with the latest Elo + 95% CI. There is no CI target: report what you get, and add +20 games by hand only if a CI is wider than ±200. A model may later be frozen as a new rung (role `model` → `reference`) only if CIs in the 719–1320 gap turn out unusable. |
| Source of truth | **One CSV, `results/results.csv`** on the artifact volume. One row per player, with `role` = `pin` / `reference` / `model`. The eval code reads the pin + reference rows as fixed opponents; `model` rows are what gets plotted. Each run's own folder keeps scratch logs (`train_log.csv`, `val_log.csv`) for loss curves; nothing else reads them. |
| Compute accounting | **No equal budgets.** Each arm trains until Elo plateaus or it hits a loose safety cap (set per arm: Step 7 uses `max_gpu_hours_per_arm` in `base.yaml`, AlphaZero its own `max_gpu_hours` in `configs/alphazero.yaml` (Step 9); `null` = no cap). Fairness comes from **Elo vs. training GPU-hours curves**: 3–4 checkpoints per run are evaluated, and each row carries the cumulative training GPU-hours at that checkpoint (an in-code timer). Stockfish labeling CPU-hours (0.99 h for the 50k dataset) are charged to every SFT-using arm. **Dollars** = GPU-hours × `gpu_usd_per_hour` + CPU-hours × `cpu_usd_per_hour` (RunPod on-demand prices, source + date, in `base.yaml`), computed in the analysis notebook. **Inference cost** = seconds per move measured in the eval games × the same rate. There is no per-move inference cap; AlphaZero's MCTS simulations per move is a config value and shows up in its sec/move. |
| Rigor | **1 seed per arm.** Seeds can be added later for the final comparison if time allows. |
| Sequencing | SFT-on-Stockfish baseline end to end first (done, Step 5). |

**Still open:** the exact structured-text format is settled in code (Step 4). Caps: Step 7 = 17.7 and AlphaZero = staged 9 / 18 / 30 (Oct 21) are set, stage 2's 18 matching the LLM's SFT + RL runs (16.5-18.2; Oct 31); Step 10 and LLM + MCTS are still open. AlphaZero's settings were confirmed after its smoke run (Oct 31, Sec 5 item 4): the provisional defaults, except 50 simulations per move.

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
- **Long jobs run as headless notebooks** inside `tmux`, with a budget cap so the balance never reaches $0. At launch a watchdog timer is started (`sleep` for `(BAL - FLOOR) / RATE * 0.9` hours, then `runpodctl stop pod $RUNPOD_POD_ID`), with `FLOOR` ~$2 (enough for storage). If the job finishes before the cap, nothing happens and the pod stays up for further work; at the cap the pod is stopped. Exact commands are in `docs/runpod_runbook.md`. RunPod has no API we rely on for the balance, so `BAL` is typed by hand at launch. Turn on the low-balance email alert (no auto-pay).
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
- **`mcts.py`** — PUCT MCTS over any `(policy, value) = evaluate(positions)` model, with batched leaf evaluation, plus batched self-play: many games at once, one network call per simulation step across all of them (Step 9 (b)). Model-agnostic, so the LLM + MCTS arm reuses both.
- **`selfplay.py`** — Self-play games through a pluggable move picker, recording (position, move or search policy, outcome). Used by LLM RL; AlphaZero plays its games batched in `mcts.py` instead (aligned Oct 30) and only reuses this file's record types. Games run one at a time for now; batching moves across games is the first speedup if self-play time dominates.

**`train/`**
- **`sft.py`** — SFT on Stockfish labels (move cross-entropy + value loss), resumable. Writes `train_log.csv` / `val_log.csv` (scratch, per run), including the cumulative `gpu_seconds_train` the results row needs.
- **`llm_rl.py`** — Self-play policy-gradient RL (game result as reward, a value baseline, a KL penalty toward the start model), from the base model or an SFT checkpoint, resumable. Its per-round log (`rl_log.csv`, scratch) holds game results, losses, KL, policy entropy, and cumulative training GPU-seconds.
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

**Step 7 — LLM self-play RL** (`search/selfplay.py`, `train/llm_rl.py`, `configs/llm_rl.yaml`, `04_train_llm_rl.ipynb`)
Warm-start from the SFT checkpoint (structured + masked, run `structured_masked_nall_e10_s0`, step 25,000). Each round: play self-play games, make one update pass, save resume state; keep a checkpoint every few rounds. Evaluate 3–4 checkpoints into `results.csv`; each row's training GPU-hours = SFT's 7.69 h + the RL hours so far.
RL design (provisional defaults in `configs/llm_rl.yaml`; revisit after the test run):
- *Self-play:* 64 games per round, each from a 4-ply random opening. Moves are sampled from the masked distribution (softmax over the legal moves' scores, temperature 1.0) so games differ and the model explores. Draw at 300 plies.
- *Reward:* every move gets its game's result from the mover's side, as a score (win 1, draw 0.5, loss 0).
- *Baseline:* the value head's own prediction at that position, when the move was played. Advantage = score − prediction.
- *Loss:* policy gradient (−advantage × log P(move)) + KL penalty toward the SFT model on the played move (weight 0.1) + value loss (BCE vs. the score, weight 1.0). One pass over the round's positions, 256 positions per optimizer step, learning rate 1e-5.
- *Compute:* RL GPU time = wall-clock of self-play + updates (both occupy the GPU).
- *Logged per round* (scratch `rl_log.csv`): W/D/L, plies, losses, KL, policy entropy (the collapse check), self-play vs. update seconds.
*Runs and budget (Oct 17):* up to 3 configs (a 4th added Oct 25: run 4), 1 seed each: run 1 = the defaults above, runs 2-3 = changes chosen from run 1's results. Each run gets 10 RL GPU-hours as a ceiling, not a target (`max_gpu_hours_per_arm` = 17.7, since the cap counts total training hours, SFT's 7.69 included): stop early on a plateau or collapse, extend (resume) if Elo is still rising at the cap. Question: does RL from the SFT model raise Elo above the SFT plateau (~690), and at what compute? Hypotheses: H1 Elo rises beyond the CI; H0 flat (outcome-only reward from 64 games/round may be too weak a signal); H2 degrades (entropy crash, KL growth; a config problem, not a result). Eval: after each run, up to 4 kept checkpoints (evenly spaced, incl. the last), same opponents and games as the SFT evals; `rl_log.csv` is the health check during the run. Stop rules and the rerun policy are set after run 1, from its data.

*Run 1 (Oct 18, the defaults):* 20 rounds, 10.47 RL GPU-h (18.16 total), stopped by the cap. Healthy but flat: entropy ~1.9 throughout, KL 0.03-0.075 and settling near 0.04; Elo r5 647, r10 714, r15 681, r20 689 vs. SFT 25k 681 [638, 714], every CI overlapping (H0). Why: ~67% of games are real draws (almost none at the ply cap), and the value head stayed at chance (BCE ~ ln 2, prediction ~0.5), so a draw gives every move ~0 advantage and only about a third of the games carried any signal. The policy also barely moved (KL ~0.04 at learning rate 1e-5).

*Run 2 (Oct 18): Stockfish-scored endings, the only change from run 1.* Hypothesis: the missing signal is the bottleneck. A game that ends by repetition, the 50-move rule, or the ply cap is scored by Stockfish's evaluation of its final position (Stockfish 19, fixed node count, converted to a win probability with the same Lichess curve as the SFT value labels) instead of 0.5. Checkmate, stalemate, and insufficient material keep their exact result. Every move still gets its game's score, and the value head trains on it. Everything else as run 1 (learning rate 1e-5, KL 0.1, 300-ply cap, 64 games per round, 10 RL GPU-h ceiling). Config: `end_scoring: stockfish`, `tag: sfend` (run folder `..._s0_sfend`; the tag keeps a changed config out of an earlier run's folder). This is its own regime and is labeled as one: the reward uses an engine's judgment of where self-play ended up, not only the game's result, which matters for the comparison with the AlphaZero arm (results only). New per-round logging (both runs' code): how games ended (checkmate, repetition, 50-move rule, stalemate, insufficient material, cap), how many were re-scored, how many re-scored games came out lopsided (score < 0.3 or > 0.7), Stockfish seconds. Run 3 is chosen from run 2's logs: a shorter cap if long no-progress draws dominate, or a higher learning rate if the policy still barely moves.

*Run 2, first rounds (Oct 20):* the code works (Stockfish ~0.2 s per round), but the draws are mostly insufficient material (~25 per round) and stalemate (~10), not repetition (3-5), the 50-move rule (2-6), or the cap (0-1). Those exact draws aren't re-scored, so only 7-10 games per round were re-scored, 2-3 of them lopsided: the reward changed for ~5% of games, and run 2 is expected to look like run 1. The stalemates suggest a winning side failing to mate: the signal exists in the middlegame and is gone by the final position. Run 2 still runs to the cap and is evaluated as planned (Derek's call), as a clean "final-position scoring" data point.

*Run 2 result (Oct 20):* 20 rounds, 10.40 RL GPU-h (18.09 total), stopped by the cap; healthy throughout (entropy ~1.9, KL ~0.04-0.05 with one transient 0.15 at round 9). Elo r5 689 [655, 731], r10 655 [621, 689], r15 689 [647, 731], r20 664 [621, 706] vs. SFT 25k 681 [638, 714]: flat (H0), every CI overlapping SFT's, zig-zagging opposite to run 1 at the same rounds (noise). Endings stayed the same all run (~21 mates, ~25 insufficient material, ~13 stalemates, 3-11 re-scored per round). Conclusion: scoring only the final position adds almost no signal, because the advantage is gone by the end.

*Run 3 (Oct 20): a two-phase curriculum, run after run 2.* Rounds 1-10: a 120-ply cap, and every game still going at ply 120 is scored by Stockfish (plus repetition and 50-move endings, as in run 2), so the reward reflects who stood better in the middlegame, before the advantage decays into stalemate or bare kings. Rounds 11 to the cap: back to 300 plies with run 2's scoring, so the model also trains on full games and gets the real result for converting an advantage (a checkmate rewards the conversion; a failed conversion is a plain draw, so it isn't penalized). Everything else as runs 1-2. Config: `max_plies: 120`, `late_from_round: 11`, `max_plies_late: 300`, `tag: curric120`. Evals at r5 and r10 measure phase 1, r15 and r20 phase 2 (does a middlegame gain survive full games). Budget: ~3.7 GPU-h for phase 1 + ~11 rounds of phase 2 at ~0.55 h reaches the 10 RL GPU-h cap near round 21. Risk: 10 rounds may be too few to move the policy (run 1's KL stayed ~0.04 over 20 rounds). Labeled "self-play with Stockfish adjudication".

*Run 3, rounds 1-12 (Oct 20, running):* phase 1 re-scored 44-56 games per round (run 2: 3-11), most of them lopsided; the value loss fell to 0.56-0.61, below ln 2 for the first time (the value head learns); KL rose 0.02 to 0.05, entropy steady ~1.9. Round 11 (300 plies): value loss 0.73, because phase 1's confident "this side is winning" predictions meet exact draws (32 insufficient material, 14 stalemates): the model reaches won positions it can't convert. Round 12: checkmates 10 to 29, stalemates 14 to 6, average length 168 to 145 plies, value loss back to 0.66, KL 0.06. Self-play logs can't show strength (one model plays both sides); Elo comes from the r5-r20 evals after training.

*Run 3 result (Oct 21):* 23 rounds, 10.36 RL GPU-h (18.05 total), stopped by the cap; healthy throughout (entropy ~1.9, KL 0.04-0.06). Elo r5 697 [664, 731], r10 706 [664, 756], r15 647 [603, 689], r20 655 [612, 706] vs. SFT 25k 681 [638, 714]: flat (H0), every CI overlapping SFT's. The drop from r10 (end of phase 1) to r15 fits "full games washed out a phase-1 gain" but is inside the noise. Phase 2 settled into run 2's pattern (~17-30 mates, ~8-13 stalemates, ~21-32 insufficient material per round), with the value loss back to ~0.66-0.69 and the average advantage shrinking from ~0.32 to ~0.13-0.21. Across runs 1-3, the 12 RL checkpoints average ~678 vs. SFT's 681: about 30 RL GPU-h moved Elo by nothing measurable. Diagnosis: phase 1 fixed the signal, but the policy barely moved (KL ~0.05 in every run); evals play the single most likely move, so small probability shifts rarely change it; and ~1,300 games per run is small.

*Run 4 (Oct 25, Derek's call; a 4th run beyond the original 3): run 3's config with a 5x learning rate, the only change.* Hypothesis: update size, not the reward, is the bottleneck. Learning rate 1e-5 to 5e-5; everything else as run 3 (rounds 1-10 at a 120-ply cap scored by Stockfish, then 300 plies with run 2's scoring; KL weight 0.1; 64 games per round; 256 positions per step; the same 17.7 total cap, about 23 rounds; the learning rate doesn't change round time). Config: `learning_rate: 5.0e-5`, `tag: curric120_lr5x` (run folder `..._s0_curric120_lr5x`). What to read in `rl_log.csv`: KL rising clearly above run 3's ~0.05 means the bigger step moves the policy; entropy falling toward ~1.0 or KL above ~0.5 would be warning signs. Derek reads them by hand after rounds 2-3; no stop rules, no new diagnostics (Oct 25). Evals: r5, r10, r15, r20, same opponents and games, compared with run 3 at the same rounds. Branch `step-7-run4-lr5x`, cut from `main` (run 3's code); after its evals it merges into `main`, then `main` into `step-9-alphazero`; its code commit is tagged `run4-lr5x`.

*Run 4 result (Oct 31, from `results.csv`):* Elo r5 689 [647, 739], r10 697 [664, 731], r15 697 [647, 748], r20 681 [647, 714] at 9.52 / 11.37 / 13.99 / 16.56 total GPU-h, vs. SFT 25k 681 [638, 714]: flat (H0), like runs 1-3. A 5x larger update didn't move Elo either. Across runs 1-4, all 16 RL checkpoints sit at 647-714.

*Rejected (Oct 20): Stockfish as the critic* (advantage = result - Stockfish's expected score at every position). It would penalize failed conversions directly, but it grades every move with the engine, which is engine-guided RL close to SFT, not self-play. The self-learned route to per-move credit is the model's own value estimates plus search (the AlphaZero and LLM + MCTS arms).

*Elo-vs-compute plot (pulled forward from Step 11, Oct 18):* `eval/plots.py` draws Elo vs. total training GPU-hours from `results.csv` only: SFT checkpoints (step_20000 included), each RL run in its own color joined to its starting checkpoint, 95% CIs, nearby reference ratings as lines. Notebooks 02 and 04 save it to `results/elo_vs_compute.png` after their evals.

*Stop rules:* none for now (the cap only); revisit later. *Rerun policy:* none fixed; decided per run.

*Merge and tags (aligned Oct 21):* Step 7 merges into `main` while run 3 is still running (its gate was already met by runs 1-2; run 3's evals are recorded in this Step when they finish). Tags: `run1-default` = `608e187`, `run2-sfend` = `0091940`, `run3-curric` = `04e245a` (the code each run used). Don't `git pull` on the pod while run 3 is running.

*Gate:* stable training (no collapse: entropy doesn't crash, KL stays bounded), and Elo is flat or rising across evaluated checkpoints.

**Step 8 — LLM Phase 1: 2x2 sweep** — *deferred (Oct 17).*
All 4 combos (mask × representation) under SFT + RL, 1 seed each, each trained to plateau or the cap. The 3 new combos need their own SFT run first. Evaluate 3–4 checkpoints per run. Pick the best combo from the Elo-vs-compute curves.

**Step 9 — AlphaZero arm** — *stage 1 done, gate met (Oct 31)* (`search/mcts.py`, `model/az_net.py`, `train/az_train.py`, `configs/alphazero.yaml`, `05_train_alphazero.ipynb`; tests in `tests/test_mcts.py`, `tests/test_engine.py`, `tests/test_eval.py`, plus a new `tests/test_az.py`)
Pure self-play from a randomly initialized network, 1 seed: no SFT data, no Stockfish, and no engine scoring anywhere in training (game results only). Independent of Steps 7-8, so it can run on its own pod. The files above exist as stubs (docstrings + TODO). This spec (Oct 21) is written so another session can implement it without the conversation behind it: details not given here are implementation choices; anything that would change a decision below goes back to Derek first.

*Branching and build order (aligned Oct 21):*
1. Step 7 is merged into `main` and each run's code tagged (Step 7, "Merge and tags").
2. `step-9-alphazero` is cut from `main` (Derek runs git; Claude suggests the commands).
3. Build in this order, each piece tested before the next: (a) safe `results.csv` saves, (b) MCTS, (c) side-to-move flipping, (d) the network, (e) the training loop, (f) the eval agent, (g) notebook 05, (h) the smoke run.
4. Running alongside an LLM job means a second pod on the same volume, with its own watchdog; while two pods run, each watchdog uses `RATE` = the sum of both pods' rates (Derek sets it). Never `git pull` on a pod that is running a job.

*(a) Safe concurrent writes to `results.csv` (`eval/anchors.py`; the branch's first commit).* Today `ResultsTable` loads the whole file when created, and `save()` rewrites the whole file from that in-memory copy through one fixed `.tmp` name. Two eval jobs that overlap (e.g. an LLM eval and an AlphaZero eval on two pods sharing the volume) therefore erase each other's new rows, even with different names. Fix: `ResultsTable` records which rows it changed (`log_model`, `add_reference`); `save()` takes a lock file next to `results.csv` (created with `O_CREAT | O_EXCL`, retried with a short sleep, always removed afterwards; a lock older than ~10 minutes counts as stale), re-reads the file from disk, overwrites only the rows this instance changed, writes a temp file with a unique name (pid + random suffix), and replaces the original. A lock file rather than `fcntl.flock`, because flock may not work on the RunPod network volume. Callers don't change (notebooks keep calling `save()` after every batch). Tests: two tables load the same file, each logs a different model, both save, both rows are present; a row one table changed isn't reverted by the other's save; existing `tests/test_eval.py` still passes. *Done (Oct 21):* as specified, plus two details: `save()` refuses to overwrite a row that another job froze as a reference after this table loaded it, and after a save the table holds the merged file (so it also sees the other job's rows).

*(b) MCTS (`search/mcts.py`, `tests/test_mcts.py`).* PUCT search over any model exposing `evaluate(boards: list[chess.Board]) -> list[tuple[dict[str, float], float]]`: for each board, priors over its legal moves (UCI string to probability, summing to 1) and a value = the expected score for the side to move, in [0, 1] (win 1, draw 0.5, loss 0). This is the LLM value head's convention (`Decision.win_prob`), so the LLM + MCTS arm can later plug an LLM wrapper into the same search; this interface is the part to keep stable. Values are backed up negamax-style (a child's value seen from its parent is 1 - v). Terminal positions use the exact result from `engine.board.outcome`, never the network. Selection: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`; an unvisited child's Q defaults to its parent's current Q. Self-play only: Dirichlet noise at the root (alpha 0.3, weight 0.25), and the move is sampled from visit counts (temperature 1) for the first 30 plies of a game, then the most-visited move. Eval: most-visited move, no noise. Batching: many games play at once, each with its own tree; at each step every game descends to one leaf and all leaves go to the network in one call (no virtual loss needed). Reusing the chosen child's subtree after a move is optional. *Where batched self-play lives (aligned Oct 30):* in `search/mcts.py`, next to the search, because it only needs the `evaluate()` interface and the LLM + MCTS arm will reuse it; `train/az_train.py` calls it, and `search/selfplay.py` (LLM code) is unchanged. Gate tests: a dummy model (uniform priors, value 0.5) finds a mate-in-1 from a constructed position (this proves the terminal handling and backup without a trained network); visit counts at the root sum to the simulation count; the search is deterministic for a fixed seed. *Done (Oct 30):* `run_search` (batched search, one tree per board), `search_move` (eval: no noise, most-visited), `play_games` (batched self-play, returning `selfplay.py`'s `SelfPlayGame` records with the visit distribution and root value per move). Implementation choices: the root's expansion counts as its first visit, so `sqrt(N_parent)` is never 0 and the priors steer from the first simulation; an unvisited child's Q = the parent's running mean for its side to move; positions at the self-play ply cap count as drawn inside the search; the 30 sampled plies count from the game's start (the 4 opening plies included, as the 300-ply cap does); no subtree reuse (optional; a later speedup). Measured: ~340 µs per simulation for tree + rules alone (dev container CPU), so ~11 min per round at the provisional defaults before network time; to revisit with the smoke run (Sec 5 item 4).

*(c) Side-to-move flipping (`engine/encoding.py` / `engine/moves.py`, `tests/test_engine.py`).* The network always sees the position from the side to move. When Black is to move: `board.mirror()` (ranks flipped, colors swapped) before `to_planes`, and moves pass through a fixed 1,968-entry mirror table (rank r to 7 - r on both from and to squares, promotion piece kept). The network's policy is in the flipped frame and is mapped back before MCTS sees it; training policy targets are stored in the flipped frame (what the network predicts). Tests: the mirror table is a permutation and its own inverse; flipping twice round-trips; on random positions, the mirrored legal mask equals python-chess's legal mask of the mirrored board. LLM code doesn't change (it never flips). *Done (Oct 30):* `engine/moves.py` gains `mirror_move`, `MIRROR_INDEX` (the 1,968-entry table), `stm_index(move, white_to_move)` (a real move's index in the network's frame) and `stm_legal_mask(board)`; `engine/encoding.py` gains `stm_board(board)` and `to_planes_stm(board)`. The frozen vocabulary and the absolute-frame `to_planes` are unchanged. `board.mirror()` keeps the 50-move clock and flips castling rights and the en passant square; it drops the move history, which the planes don't use (the search keeps playing on the real board).

*(d) Network (`model/az_net.py`).* Input: `to_planes` of the (possibly mirrored) board, 19 x 8 x 8. ResNet: a 3x3 conv stem, then N residual blocks (two 3x3 convs with batch norm and ReLU, plus the skip). Policy head: 1x1 conv, then a linear layer to 1,968 logits, illegal moves masked with `legal_mask` before the softmax. Value head: 1x1 conv, linear to 64, ReLU, linear to 1, sigmoid = expected score for the side to move. An `evaluate()` adapter wraps the network for MCTS (batching, masking, flipping back). Mixed precision on GPU. Provisional size: 6 blocks x 64 channels (~0.5M parameters, small enough that self-play rather than the network dominates time). *Done (Oct 30):* `AZNet`, `NetConfig` (size from `configs/alphazero.yaml` `net`), and `AZEvaluator` (the `evaluate()` adapter: one batched forward pass, Black's positions mirrored in via `to_planes_stm` / `stm_legal_mask`, probabilities mapped back to real moves via `stm_index`). Details the spec left open: the policy head's 1x1 conv has 2 filters and the value head's 1 (as in AlphaZero); `forward` returns the value as a logit so training can use BCE-with-logits (the same loss, safe in mixed precision), and the adapter applies the sigmoid; mixed precision = bfloat16 autocast on GPU (no loss scaling needed). At 6 x 64 the net has ~0.71M parameters, not ~0.5M: the policy head's linear layer alone is ~0.25M. Measured on the dev CPU: encoding a position costs ~140 µs, on top of ~340 µs of tree work per simulation, so self-play is CPU-bound (~15 min per round at the defaults) whatever the network costs on the L4; for the smoke run.

*(e) Training loop (`train/az_train.py`, `configs/alphazero.yaml`).* Each round: play G self-play games with MCTS (each from a 4-ply random opening, as in the LLM self-play and evals; a 300-ply cap that ends the game as a draw, like the LLM runs; otherwise the usual endings in `engine/board.py`, all scored by their exact result); store every position's planes, MCTS visit distribution, and the game's final score from the side to move in a replay buffer of the most recent positions; then K gradient steps on minibatches sampled from the buffer. Loss = cross-entropy(policy logits, visit distribution) + value loss (BCE vs. the score in [0, 1], as the LLM value head) + weight decay. Save resume state (network, optimizer, buffer, RNG states, counters) every round, atomically; keep a checkpoint every `checkpoint_every` rounds. Run folder `checkpoints/alphazero/<run_name>` (e.g. `az_b6c64_s0`, plus an optional `tag` as in Step 7) with `run_config.json`; a resume with different settings is refused, as in Step 7. GPU time = wall-clock of self-play + updates, as in Step 7. Log per round to scratch `az_log.csv`: W/D/L and game endings (same columns as `rl_log.csv`), average plies, policy loss, value loss, policy entropy, buffer size, self-play vs. update seconds, GPU-hours so far. Provisional defaults (confirm with Derek after the smoke run, before the first real run; Sec 5 item 4): 128 games per round, 100 simulations per move, `c_puct` 1.5, buffer = the most recent ~200k positions, K = enough minibatches of 256 that each new position is sampled about twice, Adam with learning rate 1e-3 and weight decay 1e-4, `checkpoint_every` set from the smoke run's round time so each cap stage has 4+ kept checkpoints. *Done (Oct 30):* `train/az_train.py` (`AZConfig`, `load_configs`, `ReplayBuffer`, `train`, `load_checkpoint`, `list_checkpoints`, `gpu_hours_at`); all settings in `configs/alphazero.yaml` (`max_gpu_hours`, `net`, `mcts`, `train`), every value provisional. Implementation choices: the buffer stores positions compactly and losslessly (planes as uint8 with the 50-move plane as its integer count; each position's legal moves as side-to-move indices + visit shares, which double as the training mask), ~2.2 KB per position, ~450 MB at 200k, saved with the latest round only; every round's randomness is seeded from (seed, round), so a resume replays exactly (tested: 2 + 2 rounds = 4 rounds straight, identical weights) and no RNG state is saved; the logged policy entropy is the network's policy on the training minibatches; K = ceil(new positions x 2 / 256); mixed precision = bfloat16 autocast (no loss scaling). As in Step 7, the cap is checked at the start of each round, so a stage can overshoot its cap by up to one round; the round it stops at is recorded in `kept_rounds.json` and never pruned.

*Compute cap, staged 9 / 18 / 30 GPU-hours (aligned Oct 21).* `max_gpu_hours` in `configs/alphazero.yaml` is passed to `train()` as an argument, as in Step 7, and is not saved in `run_config.json`, so raising it and rerunning resumes the same run. Stage 1: cap 9, then evaluate. Stage 2 (as needed): raise to 18, matching the LLM's SFT + RL runs (16.5-18.2 total GPU-h across runs 1-4; Derek, Oct 31), then evaluate. Stage 3 (as needed): raise to 30, a placeholder for the LLM + MCTS arm's total compute, replaced by that arm's real number once it is designed, then evaluate. Whenever training stops at a cap, that round becomes a kept checkpoint (never pruned when the run is later resumed), so every stage ends on an evaluable checkpoint at the matched compute. No stop rules (as in Step 7): Derek decides each extension from the evals. No SFT and no labeling: `train_gpu_hours` starts at 0 and `label_cpu_hours` = 0.

*(f) Eval (`05_train_alphazero.ipynb`'s eval section, modeled on notebook 04's).* An `AlphaZeroAgent` with `select_move` = MCTS at the eval simulation count (provisionally the same as training; it shows up in sec/move, the inference cost), most-visited move, no noise. It is rated with `eval.match.evaluate_agent` exactly like every other arm: 20 games against each of 2-3 reference opponents nearest its strength, in batches of 10, the row saved after every batch; an already-evaluated checkpoint is skipped. Per stage: up to 4 kept checkpoints, evenly spaced, always including the stage's last. Row fields: name `az_<run_name>@rN`, run `alphazero/<run_name>`, step = round, seed, train_gpu_hours = GPU-hours at that round, label_cpu_hours 0, gpu, note "pure self-play AlphaZero, MCTS <sims> sims/move". Opponents: start with `random`, `greedy`, `step_20000`, and move up (`sf_1320`, `sf_1500`, `sf_1700`) as it gets stronger; a rating at the edge of the set is flagged `OFF SCALE` by `log_model`, a cue to move the opponents. Plot: `eval/plots.py` draws the AlphaZero run as its own series starting at 0 GPU-hours (its RL-note regex must not drop it; add a test to `tests/test_plots.py`). *Done (Oct 30):* in `train/az_train.py` (next to `load_checkpoint`, as Step 7's agents come from `train/sft.py`): `AlphaZeroAgent` (an `eval.match.Agent`: `search_move`, no noise, most-visited; the match hands it a board copy with its move history, so the search sees repetitions), `load_agent`, and `eval_rounds` (up to 4 kept checkpoints per cap stage, evenly spaced, each stage's last included; rounds after the last cap stop form an open stage, rated from its regular kept checkpoints). Row names are `<run_name>@rN`, e.g. `az_b6c64_s0@r5`: `run_name` already starts with `az_`, so the literal `az_<run_name>` would read `az_az_...`. `eval/plots.py` gains `run_kind` (`llm_rl/` = RL, `alphazero/` = AlphaZero, else SFT), so an AlphaZero run is its own series labeled by round, at its own GPU-hours from 0, never joined to an SFT point; tested in `tests/test_plots.py`. The notebook's eval section is (g).

*(g) Notebook and (h) smoke run.* `05_train_alphazero.ipynb` follows notebook 04's layout: setup cell (`BRANCH`), parameters cell (`SMOKE_TEST`, `RUN_TRAINING`, `EVAL_ROUNDS`, `OPPONENTS`), train, eval, plot to `results/elo_vs_compute.png`. `SMOKE_TEST = True` plays 2 tiny rounds (few games, few simulations) into its own folder, no eval. Run headless on the pod per `docs/runpod_runbook.md`. Report the smoke run's seconds per round (self-play vs. update) to Derek with the provisional defaults; together they set how many rounds the 9-GPU-h stage buys, before the real run starts. *(g) done (Oct 30):* `notebooks/05_train_alphazero.ipynb`, in notebook 04's layout: setup (`BRANCH`, warns if the pod is on another branch), parameters (`RUN_TRAINING`, `SMOKE_TEST`, `MAX_ROUNDS`, `EVAL_ROUNDS`, `OPPONENTS`; everything else from `configs/alphazero.yaml`), train, training curves (value loss drawn against ln 2), eval (`eval_rounds`, rated checkpoints skipped on a rerun), plot. The smoke test (2 rounds of 8 games, 8 simulations, 40-ply cap, own `_smoke` folder, no eval) adds a timing probe at the real settings, because tiny rounds don't scale: one move of `games_per_round` games at the real simulation count, plus one update step at the real batch size, projected to a round and to the number of rounds the 9-GPU-h stage buys at 80 / 150 / 250 moves per game. Tested here by running its cells as a script (smoke path; and a tiny non-smoke run through eval, plot, and a rerun that skips rated checkpoints). *(h) instructions (Oct 31):* `docs/runpod_runbook.md` section F (pod, watchdog, checkout, tests, headless run, which numbers to report, then the stage-1 budget: ~12-13 pod-hours with evals, `BAL` ~$10+). Run 4 has finished, so the smoke run uses one pod and the one clone, with the usual single-pod watchdog. Open for later (to decide before AlphaZero trains alongside another LLM job): the repo on the volume is shared by every pod, so a checkout from one pod changes the code under another pod's job; the fix would be a separate AlphaZero clone (one line in notebook 05's `REPO_DIR`), and each watchdog would then budget for both pods. *Smoke run (Oct 31, L4 pod, 24-core EPYC with a ~5-CPU container quota):* the loop, saves, resume and the GPU path all ran; round 1 entropy 3.24 (about uniform over the legal moves) and value loss 0.694 (ln 2), as a random network should. Timing probe: one move of 128 games at 100 simulations = 3.7 s (286 µs per game-simulation); an update step (batch 256) = 5 ms, peak GPU memory 0.11 GB, so a round is almost all CPU-bound search and the GPU mostly idles. A random network rarely mates (7 of 8 smoke games hit the cap), so early games will run to ~300 plies: ~18 min per round at 100 simulations, ~30 rounds (~3,800 games) in stage 1. *Decision (Derek, Oct 31): option B, 50 simulations per move* (training and eval; ~2x the games, ~60 rounds / ~7,500 games in stage 1; weaker search targets per move). Considered: keeping 100; parallel self-play across CPU workers (~3-4x with the ~5-CPU quota, new code; held in reserve, and as a changed setting it would start a new run); a shorter ply cap (rejected: it matches the LLM runs). The rest of the provisional defaults are confirmed (Sec 5 item 4); `checkpoint_every` 5 gives ~12 kept checkpoints in stage 1.

*Gate (proposed Oct 21, confirmed by Derek Oct 31):* all tests pass (mate-in-1, flipping, concurrent saves); training is stable (finite losses, policy entropy doesn't collapse in the early rounds, value loss below ln 2 after the first rounds); Elo rises above `random` across the stage-1 checkpoints. **Met (Oct 31).**

*Stage 1 result (Oct 31):* run `az_b6c64_s0`, 53 rounds (6,784 games), 9.16 GPU-h (the cap was checked at round 53's start, at 8.98), ~10 min per round. Training stayed healthy: policy loss 3.11 -> 2.68, entropy 3.12 -> 2.70 (no collapse), value loss 0.51 in round 1 (small buffer), ~0.61 once the buffer filled (round 10), then easing to 0.57; games hitting the cap fell from 29 to ~4-9 per round, average length ~165 -> ~135-145 plies, ~70-90 checkmates per round; repetition draws grew to 10-16 and stalemates to 6-14 per round. Elo (60 games each vs. `random`, `greedy`, `step_20000`; MCTS 50 sims, 0.09 s/move, the same as one LLM forward pass):

| checkpoint | train GPU-h | Elo [95% CI] |
|---|---|---|
| r5 | 0.98 | 653 [604, 697] |
| r20 | 4.01 | 716 [678, 755] |
| r40 | 7.20 | 722 [684, 755] |
| r53 | 9.16 | 729 [696, 762] |

Reading: a real rise from r5 to r53 (+75; the intervals just touch, but the difference is ~3 standard errors), most of it by r20 (~4 GPU-h), then nearly flat. Level with the best LLM checkpoints (SFT 15k 706 [664, 748]; every RL checkpoint 647-714), with no data and no Stockfish. Per opponent, r53 scored 18W 2D 0L vs. `random`, 1W 18D 1L vs. `greedy`, 1W 16D 3L vs. `step_20000`: it rarely loses but rarely converts against anything that defends, so its Elo is held down by draws (a draw counts as equal strength). The same conversion weakness as the LLM runs. Bracketing is fine at this strength (45% vs. `step_20000`, the top opponent).

*Next, decided Oct 31 (Derek), one step at a time:*
1. **No-search evals (an addition to (f), before stage 2).** A `PolicyAgent` plays the network's top legal move with no search; the same kept checkpoints are rated with it as a separate series (`az_<run>_nosearch@rN`, run `alphazero/<run>_nosearch`), at the same training GPU-hours. The gap to the search series is what search adds to the same network; with the LLM rows (no search) and LLM + MCTS option A it fills a 2x2 (architecture x search). No retraining; rerun notebook 05 with `RUN_TRAINING = False` (rated rows are skipped). *Built (Oct 31):* `PolicyAgent` / `load_policy_agent` in `train/az_train.py` (top-prior legal move, ties to the first in sorted UCI order, so deterministic); notebook 05's `EVAL_NO_SEARCH = True` rates each chosen checkpoint both ways, the no-search rows noted "no search (policy head's top move)". Tested: the agent plays the network's top move and only legal moves; a run rated with search, then rerun with `RUN_TRAINING = False`, adds only the no-search rows, at the same training GPU-hours, as a second series on the chart.
2. **Stage 2:** cap 18 (above). Opponents stay `random`, `greedy`, `step_20000`, so the whole AlphaZero curve is rated against one set; a game is most informative when its outcome is uncertain, and at ~730-850 these three bracket the model (`random` still carries ~75% of an even game's information, while `sf_1320` would be a near-certain loss). A stronger opponent is added only if a checkpoint goes above ~900 (about 75% vs. `step_20000`) or is flagged `OFF SCALE`; models living in the 800-1300 gap would call for a new frozen rung (Sec 0 Eval).
3. **Parallel pods:** running AlphaZero alongside the LLM + MCTS work needs AlphaZero's own clone of the repo on the volume (runbook section F; one line in notebook 05's `REPO_DIR`), done when that step comes.

**Step 10 — LLM Phase 2: regime comparison**
Structured + masked under pure self-play RL, SFT only, and SFT + RL, 1 seed each, to plateau or the cap, with 3–4 checkpoints evaluated per run. SFT only (Step 5) and SFT + RL (Step 7) already exist for this config, so the new run is pure self-play RL from the untrained model.

**Step 11 — Final analysis** (`06_analysis.ipynb`, reads only `results.csv`)
- Elo ± CI bar chart, one bar per arm (best checkpoint).
- Elo vs. training GPU-hours (and dollars) curves per arm: sample efficiency.
- Dollars to train (GPU + labeling CPU) and inference dollars per game (sec/move × moves/game × rate) per arm.
- Illegal-move rate for the unmasked arms (only if the 2x2 returns).
- Optional: rescale by playing more games for the arms on the final slides.

**LLM + MCTS arm** (planned, Oct 17; after Step 9, before Step 11): the structured + masked LLM with MCTS, reusing `search/mcts.py` with the LLM's policy and value heads through Step 9's `evaluate` interface (priors over legal moves + expected score for the side to move). Design is aligned before building (Sec 5). Its total training compute replaces AlphaZero's 30-GPU-h placeholder.
*Decided Oct 31 (Derek):* **option A first: search at play time only.** An LLM checkpoint is wrapped as the search's evaluator (priors = softmax over its masked move scores; value = its value head's win probability, `score_moves`), and rated with MCTS like any other agent; no new training, so its training compute is the checkpoint's own. Base: **SFT step 25,000** (681): every LLM checkpoint is within noise of it (RL runs 1-4 all flat), and its value head was trained on Stockfish win-probability labels, the best candidate leaf evaluator (RL value heads were near chance in runs 1-2). Simulations: **0** (the existing SFT 25k row) **and 16**; more later if 16 helps. Cost context: one LLM evaluation is ~0.09 s (~300x AlphaZero's), so 16 simulations is ~1.5 s/move and ~2 h per 60-game eval. Option B (search inside the self-play training loop, Derek's earlier lean for comparability with AlphaZero) is decided after A's result: if search doesn't help at play time it is unlikely to help in training, and A tells us whether the value head is a usable leaf evaluator. Built on its own branch from `main`; `model/llm_policy.py` stays unchanged.

---

## 5. Next alignment points
1. ~~Self-play RL design details (round size, KL weight, baseline)~~ — provisional defaults set Oct 17 (Step 7); run 1 done Oct 18 (flat); run 2 changes only how a non-checkmate ending is scored (Stockfish, Step 7).
2. LLM + MCTS arm design: MCTS only at play time vs. inside the self-play training loop (the earlier lean was the latter, for comparability with AlphaZero), simulations per move, how its compute is counted, and its cap. Fixed by Step 9 (Oct 21): the search interface (`evaluate(boards)` returns priors over legal moves and an expected score for the side to move). Input from Step 7: whether the value head is a usable leaf evaluator (at chance in runs 1-2; below ln 2 in run 3's early rounds). **Decided Oct 31:** MCTS at play time first (option A: SFT 25k, 0 and 16 simulations; LLM + MCTS section); in-training search (B) decided after A's result. Still open: B, and if B, its simulations, compute counting and cap.
3. ~~`max_gpu_hours_per_arm`~~ set Oct 17 for Step 7 (17.7 total = 10 RL GPU-h per run, up to 3 configs). Stop rules: none for now, cap only (Oct 18; revisit later). Rerun policy: none fixed, decided per run (Oct 18). AlphaZero (Step 9, Oct 21): staged 9 / 18 / 30 GPU-h, extended by resuming. Still open: the caps for pure RL (Step 10) and LLM + MCTS.
4. ~~AlphaZero provisional defaults~~ — confirmed Oct 31 from the smoke run (Step 9 (h)): simulations per move 100 -> 50 (training and eval), everything else as provisional (6 x 64 net, 128 games per round, `c_puct` 1.5, 200k buffer, ~2 samples per position, Adam 1e-3 / weight decay 1e-4, `checkpoint_every` 5). Parallel self-play across the pod's CPUs held in reserve (a new run if used).
