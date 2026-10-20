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
| Compute accounting | **No equal budgets.** Each arm trains until Elo plateaus or it hits a loose safety cap (`max_gpu_hours_per_arm` in `base.yaml`; the value is decided later, once RL speed is known; `null` = no cap). Fairness comes from **Elo vs. training GPU-hours curves**: 3–4 checkpoints per run are evaluated, and each row carries the cumulative training GPU-hours at that checkpoint (an in-code timer). Stockfish labeling CPU-hours (0.99 h for the 50k dataset) are charged to every SFT-using arm. **Dollars** = GPU-hours × `gpu_usd_per_hour` + CPU-hours × `cpu_usd_per_hour` (RunPod on-demand prices, source + date, in `base.yaml`), computed in the analysis notebook. **Inference cost** = seconds per move measured in the eval games × the same rate. There is no per-move inference cap; AlphaZero's MCTS simulations per move is a config value and shows up in its sec/move. |
| Rigor | **1 seed per arm.** Seeds can be added later for the final comparison if time allows. |
| Sequencing | SFT-on-Stockfish baseline end to end first (done, Step 5). |

**Still open:** the exact structured-text format is settled in code (Step 4). `max_gpu_hours_per_arm` (decided later). Step 7's RL hyperparameters are provisional defaults until the test run.

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
- **`mcts.py`** — PUCT MCTS over any `(policy, value) = evaluate(positions)` model, with batched leaf evaluation.
- **`selfplay.py`** — Self-play games through a pluggable move picker, recording (position, move or search policy, outcome). Used by LLM RL and AlphaZero. Games run one at a time for now; batching moves across games is the first speedup if self-play time dominates.

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
*Runs and budget (Oct 17):* up to 3 configs, 1 seed each: run 1 = the defaults above, runs 2-3 = changes chosen from run 1's results. Each run gets 10 RL GPU-hours as a ceiling, not a target (`max_gpu_hours_per_arm` = 17.7, since the cap counts total training hours, SFT's 7.69 included): stop early on a plateau or collapse, extend (resume) if Elo is still rising at the cap. Question: does RL from the SFT model raise Elo above the SFT plateau (~690), and at what compute? Hypotheses: H1 Elo rises beyond the CI; H0 flat (outcome-only reward from 64 games/round may be too weak a signal); H2 degrades (entropy crash, KL growth; a config problem, not a result). Eval: after each run, up to 4 kept checkpoints (evenly spaced, incl. the last), same opponents and games as the SFT evals; `rl_log.csv` is the health check during the run. Stop rules and the rerun policy are set after run 1, from its data.

*Run 1 (Oct 18, the defaults):* 20 rounds, 10.47 RL GPU-h (18.16 total), stopped by the cap. Healthy but flat: entropy ~1.9 throughout, KL 0.03-0.075 and settling near 0.04; Elo r5 647, r10 714, r15 681, r20 689 vs. SFT 25k 681 [638, 714], every CI overlapping (H0). Why: ~67% of games are real draws (almost none at the ply cap), and the value head stayed at chance (BCE ~ ln 2, prediction ~0.5), so a draw gives every move ~0 advantage and only about a third of the games carried any signal. The policy also barely moved (KL ~0.04 at learning rate 1e-5).

*Run 2 (Oct 18): Stockfish-scored endings, the only change from run 1.* Hypothesis: the missing signal is the bottleneck. A game that ends by repetition, the 50-move rule, or the ply cap is scored by Stockfish's evaluation of its final position (Stockfish 19, fixed node count, converted to a win probability with the same Lichess curve as the SFT value labels) instead of 0.5. Checkmate, stalemate, and insufficient material keep their exact result. Every move still gets its game's score, and the value head trains on it. Everything else as run 1 (learning rate 1e-5, KL 0.1, 300-ply cap, 64 games per round, 10 RL GPU-h ceiling). Config: `end_scoring: stockfish`, `tag: sfend` (run folder `..._s0_sfend`; the tag keeps a changed config out of an earlier run's folder). This is its own regime and is labeled as one: the reward uses an engine's judgment of where self-play ended up, not only the game's result, which matters for the comparison with the AlphaZero arm (results only). New per-round logging (both runs' code): how games ended (checkmate, repetition, 50-move rule, stalemate, insufficient material, cap), how many were re-scored, how many re-scored games came out lopsided (score < 0.3 or > 0.7), Stockfish seconds. Run 3 is chosen from run 2's logs: a shorter cap if long no-progress draws dominate, or a higher learning rate if the policy still barely moves.

*Run 2, first rounds (Oct 20):* the code works (Stockfish ~0.2 s per round), but the draws are mostly insufficient material (~25 per round) and stalemate (~10), not repetition (3-5), the 50-move rule (2-6), or the cap (0-1). Those exact draws aren't re-scored, so only 7-10 games per round were re-scored, 2-3 of them lopsided: the reward changed for ~5% of games, and run 2 is expected to look like run 1. The stalemates suggest a winning side failing to mate: the signal exists in the middlegame and is gone by the final position. Run 2 still runs to the cap and is evaluated as planned (Derek's call), as a clean "final-position scoring" data point.

*Run 2 result (Oct 20):* 20 rounds, 10.40 RL GPU-h (18.09 total), stopped by the cap; healthy throughout (entropy ~1.9, KL ~0.04-0.05 with one transient 0.15 at round 9). Elo r5 689 [655, 731], r10 655 [621, 689], r15 689 [647, 731], r20 664 [621, 706] vs. SFT 25k 681 [638, 714]: flat (H0), every CI overlapping SFT's, zig-zagging opposite to run 1 at the same rounds (noise). Endings stayed the same all run (~21 mates, ~25 insufficient material, ~13 stalemates, 3-11 re-scored per round). Conclusion: scoring only the final position adds almost no signal, because the advantage is gone by the end.

*Run 3 (Oct 20): a two-phase curriculum, run after run 2.* Rounds 1-10: a 120-ply cap, and every game still going at ply 120 is scored by Stockfish (plus repetition and 50-move endings, as in run 2), so the reward reflects who stood better in the middlegame, before the advantage decays into stalemate or bare kings. Rounds 11 to the cap: back to 300 plies with run 2's scoring, so the model also trains on full games and gets the real result for converting an advantage (a checkmate rewards the conversion; a failed conversion is a plain draw, so it isn't penalized). Everything else as runs 1-2. Config: `max_plies: 120`, `late_from_round: 11`, `max_plies_late: 300`, `tag: curric120`. Evals at r5 and r10 measure phase 1, r15 and r20 phase 2 (does a middlegame gain survive full games). Budget: ~3.7 GPU-h for phase 1 + ~11 rounds of phase 2 at ~0.55 h reaches the 10 RL GPU-h cap near round 21. Risk: 10 rounds may be too few to move the policy (run 1's KL stayed ~0.04 over 20 rounds). Labeled "self-play with Stockfish adjudication".

*Rejected (Oct 20): Stockfish as the critic* (advantage = result - Stockfish's expected score at every position). It would penalize failed conversions directly, but it grades every move with the engine, which is engine-guided RL close to SFT, not self-play. The self-learned route to per-move credit is the model's own value estimates plus search (the AlphaZero and LLM + MCTS arms).

*Elo-vs-compute plot (pulled forward from Step 11, Oct 18):* `eval/plots.py` draws Elo vs. total training GPU-hours from `results.csv` only: SFT checkpoints (step_20000 included), each RL run in its own color joined to its starting checkpoint, 95% CIs, nearby reference ratings as lines. Notebooks 02 and 04 save it to `results/elo_vs_compute.png` after their evals.

*Stop rules:* none for now (the cap only); revisit later. *Rerun policy:* none fixed; decided per run.

*Gate:* stable training (no collapse: entropy doesn't crash, KL stays bounded), and Elo is flat or rising across evaluated checkpoints.

**Step 8 — LLM Phase 1: 2x2 sweep** — *deferred (Oct 17).*
All 4 combos (mask × representation) under SFT + RL, 1 seed each, each trained to plateau or the cap. The 3 new combos need their own SFT run first. Evaluate 3–4 checkpoints per run. Pick the best combo from the Elo-vs-compute curves.

**Step 9 — AlphaZero arm** (`model/az_net.py`, `search/mcts.py`, `train/az_train.py`, `05_train_alphazero.ipynb`)
Start with `tests/test_mcts.py` (mate-in-1). Add side-to-move flipping: `board.mirror()` before `to_planes`, a fixed 1,968-entry move mirror table, and stored search policies flipped the same way, plus a test that flipping twice round-trips and that flipped masks match python-chess. Then train by pure self-play, 1 seed, to plateau or the cap, and evaluate 3–4 checkpoints. This is independent of Steps 7–8, so it can run on a separate pod.

**Step 10 — LLM Phase 2: regime comparison**
Structured + masked under pure self-play RL, SFT only, and SFT + RL, 1 seed each, to plateau or the cap, with 3–4 checkpoints evaluated per run. SFT only (Step 5) and SFT + RL (Step 7) already exist for this config, so the new run is pure self-play RL from the untrained model.

**Step 11 — Final analysis** (`06_analysis.ipynb`, reads only `results.csv`)
- Elo ± CI bar chart, one bar per arm (best checkpoint).
- Elo vs. training GPU-hours (and dollars) curves per arm: sample efficiency.
- Dollars to train (GPU + labeling CPU) and inference dollars per game (sec/move × moves/game × rate) per arm.
- Illegal-move rate for the unmasked arms (only if the 2x2 returns).
- Optional: rescale by playing more games for the arms on the final slides.

**LLM + MCTS arm** (planned, Oct 17; after Step 9, before Step 11): the structured + masked LLM with MCTS, reusing `search/mcts.py` with the LLM's policy and value heads. Design is aligned before building (Sec 5).

---

## 5. Next alignment points
1. ~~Self-play RL design details (round size, KL weight, baseline)~~ — provisional defaults set Oct 17 (Step 7); run 1 done Oct 18 (flat); run 2 changes only how a non-checkmate ending is scored (Stockfish, Step 7).
2. LLM + MCTS arm design: MCTS only at play time vs. inside the self-play training loop (the earlier lean was the latter, for comparability with AlphaZero), simulations per move, and how its compute is counted.
3. ~~`max_gpu_hours_per_arm`~~ set Oct 17 for Step 7 (17.7 total = 10 RL GPU-h per run, up to 3 configs). Stop rules: none for now, cap only (Oct 18; revisit later). Rerun policy: none fixed, decided per run (Oct 18). Still open: the cap for pure RL (Step 10) and the other arms.
