# Chess AI — Implementation Plan v5 (Level 1 / Level 2)

**Version:** v5, Oct 31, 2026. Supersedes v4 (archive it outside the repo at
`ChessAI/archive/chess_ai_implementation_plan_v4.md`, next to v1–v3). v4 holds the full
build detail and run-by-run history of Steps 0–9; v5 summarizes those and specifies the
work from here on.
**Source:** `docs/framing.md` (canonical framing), plan v4, and the Oct 31 results review
with Derek.

**Guiding principle (unchanged): this is an MVP, run 80/20.** Track the few numbers the
final slides need, in as few files as possible. Nothing gets built unless it feeds a
Step 11 output.

**Update, Oct 31, 2026 (edited in place, still v5):** Sec 5 items 1–4 decided with
Derek: a per-game file behind `results.csv`; re-rate 10 headline checkpoints; a
head-to-head of each arm's best-rated checkpoint at 40 games per pair; 500k positions
first. CNN size: start at 6 × 64. New Sec 4.1 says what can run in parallel and how.
Step numbers are labels, not the running order (Sec 4).
Later the same day: Step 13's small code prep runs before Step 12's build, so the 500k
labeling (the long pole) runs while Step 12 is built (Derek, Oct 31; Sec 4). Step 16's
config is set (cap 18). Step 13 design (Derek, Oct 31): the 500k set is sampled fresh with
the same rules (not nested), keeps the 50k's validation set, records each game's result
and ending, and reports its share of near-win positions before any enrichment is
considered; one CPU-only pod labels it (no multi-pod chunking).
**Update, Nov 1, 2026:** the 500k set is labeled (3.10 CPU-h, Step 13). LLM SFT on 500k:
capped at 2 passes (Derek; Sec 5 item 8), the Step 5 recipe otherwise unchanged, on its own
L4 pod while AlphaZero stage 2 runs (Step 14).

---

## Why v5: what the results say (Oct 31)

1. **LLM SFT is limited by its data, not its compute.** Validation move loss bottoms at
   step 15k (2.84) and then rises to 3.05 at 25k, while training loss keeps falling
   (2.53 → 2.16) in an epoch-by-epoch staircase: after ~5 passes over 45k positions it
   memorizes. Elo peaks at the same checkpoint (15k, 706), though within noise. No SFT
   trend after 5k steps is statistically real (slope z ≈ 0.9).
2. **LLM RL from SFT is flat in all four runs** (16 checkpoints at 647–714, mean 681 =
   SFT 25k; their spread matches eval noise). Run 4 moved the policy ~3× further than
   run 3 (KL ~0.15 vs ~0.05) with no Elo change, so update size isn't the bottleneck.
   The value head is near chance on full games (BCE ~0.67–0.69), so RL gets almost no
   per-move credit.
3. **AlphaZero is the only arm with a real upward trend:** 653 → 729 over 9.2 GPU-h
   (change z ≈ 2.6, slope z ≈ 2.4), level with the best LLM checkpoints, from no data and
   no Stockfish. Its losses haven't plateaued.
4. **Implication.** At this budget, imitating an oracle (a correct answer on every
   position) should be far more compute-efficient than learning from one game result per
   ~150 moves. So v5 scales imitation for **both** architectures on the same labeled
   data, measures what **play-time search** adds on top, and keeps AlphaZero as the
   teacher-free comparison. Evals are upgraded first, so every new number comes with
   wins/draws/losses and how games ended.

## Changes from v4

1. **Eval v2 (new Step 12):** a per-game record (result, how it ended, plies, sec/move),
   W/D/L per checkpoint, search on/off as a switch for any network, and a head-to-head
   round robin between each arm's best model.
2. **Scaled dataset (new Step 13):** more Stockfish-labeled Lichess positions from the
   existing pipeline, labeled once and shared by both architectures. Database first;
   labeling the model's own positions (DAgger) is parked.
3. **New arm, CNN SFT (new Step 14):** AlphaZero's network trained on the same Stockfish
   labels as the LLM. LLM SFT is rerun on the larger dataset. Both are compared on Elo
   vs. GPU-hours and Elo vs. dataset size.
4. **Search at play time for both imitation networks (new Step 15).** This absorbs the
   LLM + MCTS arm's option A (decided Oct 31) and adds the CNN counterpart, giving an
   architecture × search grid.
5. **AlphaZero stage 2 (new Step 16)** runs in the background when a pod is free.
6. **Deferred / parked:** Step 8 (2×2 sweep, still deferred), Step 10 (pure LLM
   self-play from scratch; proposed deferral, Sec 5), LLM + MCTS option B (search inside
   training), retuning AlphaZero's settings, DAgger, richer per-move labels.
7. **Numbering:** done steps keep their numbers; new work is Steps 12–16; Step 11 (final
   analysis) stays Step 11 and runs last. Execution order is in Sec 4.

---

## 0. Locked assumptions

| Decision | Choice |
|---|---|
| LLM family | **Pretrained small open-weight LLM**, fine-tuned: `Qwen/Qwen3-0.6B` (Apache-2.0), `enable_thinking=False` (the hybrid model with thinking off; note in the writeup). LoRA + a value head. |
| Legal-move toggle | Hard masking / constrained decoding (on) vs. free generation (off). Listing legal moves in the prompt is *not* the toggle. |
| Illegal moves | Three consecutive illegal attempts on one move forfeit the game, in eval and in self-play. Masked agents can't trigger it. |
| Board representation toggle | FEN text vs. structured text. No image input. |
| LLM design | **Structured text + masking on** (focus since Oct 17). The 2×2 (Step 8) stays deferred; the unmasked and FEN code paths stay in the repo. |
| Architectures | **LLM** (above, ~600M parameters) and **CNN** = AlphaZero's policy-value ResNet (`model/az_net.py`, 6 blocks × 64 channels, ~0.71M parameters), one network definition shared by the AlphaZero and CNN SFT arms. The LLM is ~850× the CNN (about three orders of magnitude), which is why arms are compared on GPU-hours, not only on data. |
| Training regimes | **Imitation (SFT):** LLM and CNN, on the same Stockfish-labeled positions. **Self-play RL:** AlphaZero (CNN + MCTS, game results only); LLM RL from SFT (Step 7, done, flat). |
| Search | PUCT MCTS (`search/mcts.py`) over any model exposing `evaluate(boards) -> [(priors over legal moves, expected score for the side to move)]`. In training only for AlphaZero; at play time it is a switch on any network agent (Step 12). No per-move inference cap: simulations per move is a setting and shows up in sec/move. |
| Board orientation | LLM arms never flip. CNN arms (AlphaZero and CNN SFT) flip to the side to move (`to_planes_stm`, `stm_index`). |
| Terminology | **Game** = one full game. **Round** = a batch of self-play games then one training update. SFT has no rounds. |
| Eval | **Frozen reference set:** `sf_1320` = 1320 (pin), `sf_1500`, `sf_1700`, `random`, `greedy`, `step_20000`. Each new model is rated against it; only its own rating is fitted. Stockfish members play at 0.1 s per move. New models: 20 games against each of the 2–3 nearest reference opponents, in batches of 10, row saved after each batch. No CI target; +20 games by hand only if a CI is wider than ±200. A model may be frozen as a new rung only if CIs in the gap above `step_20000` become unusable. **New in v5:** every eval game is recorded (Step 12), and head-to-head matches between models are reported as their own table, not fed into the reference fit. |
| Source of truth | **`results/results.csv` is the one summary file** (one row per player, `role` = pin / reference / model, now with W/D/L): every chart and the Step 11 slides read their headline numbers from it, and only `eval.anchors.ResultsTable` writes it. **`results/games.csv` (v5, decided Oct 31) is the supporting detail behind it:** one row per eval or head-to-head game, used to build summaries (W/D/L, draw types, head-to-head tables), never as a second source of headline numbers. Each run folder keeps scratch logs for loss curves. |
| Compute accounting | No equal budgets. Each arm trains to plateau or a loose cap; fairness comes from **Elo vs. training GPU-hours** curves, 3–4 checkpoints per run, each row carrying cumulative training GPU-hours. **Stockfish labeling CPU-hours are charged to every arm that trains on the labels, for the dataset size it used** (0.99 h for the 50k set, 3.10 h for the 500k set). Dollars = GPU-h × `gpu_usd_per_hour` + CPU-h × `cpu_usd_per_hour` (RunPod on-demand, source + date in `base.yaml`), in the analysis notebook. Inference cost = sec/move from eval games × the same rate. |
| Rigor | 1 seed per arm; more later if time allows. |
| Engine | Stockfish 19, pinned. Every rating depends on it; never upgrade without discussing it. |

**Still open:** see Sec 5.

---

## 1. Hardware and service provider (unchanged from v4)

Steps 0–5 ran on Colab with artifacts on Drive. **Step 6 onward: RunPod**, on-demand
NVIDIA L4 (24 GB) pods with a network volume mounted at `/workspace`.

- Artifacts on the volume: `artifact_root = /workspace/chess-ai` (`data/processed`,
  `checkpoints/{sft,llm_rl,alphazero,cnn_sft}`, `results/`). Back up `results/` to the
  laptop after every run.
- Code lives in GitHub only: clone once to `/workspace/Chess_Bot`, `git pull` per
  session, never edit on the pod. Never `git pull` on a pod that is running a job.
- A pod always has the network volume attached (create the volume first, deploy the pod
  from it, same datacenter).
- Long jobs run headless in `tmux` with the budget watchdog (`docs/runpod_runbook.md`).
  Two pods at once: each watchdog uses the sum of both rates, and a second job uses its
  own clone of the repo on the volume (the shared clone's checkout would change code
  under the other job).
- Secrets (`HF_TOKEN`, GitHub token) are pod environment variables.
- Log the GPU name in every results row; discard or flag runs not on an L4.
- Every training script is resumable. Eval games and MCTS self-play are CPU-bound (the
  pod container has ~5 CPUs); Stockfish labeling is CPU work too, so plan labeling and
  CPU-heavy evals so they don't fight for the same pod.

---

## 2. Repository and notebook structure

Pattern unchanged: plain `.py` modules plus thin notebooks. Additions in v5 are marked
**(v5)**; names of new files are provisional (implementation choices).

```
Chess_Bot/
  CLAUDE.md  README.md  requirements.txt
  docs/  framing.md  implementation_plan_v5.md  jobs_to_be_done.md  runpod_runbook.md
  configs/  base.yaml  data.yaml  sft.yaml  llm_rl.yaml  alphazero.yaml  eval.yaml
            cnn_sft.yaml (v5)
  engine/   board.py  moves.py  encoding.py  stockfish.py
  data/     download.py  sample_positions.py  annotate.py  dataset.py
  model/    llm_policy.py  az_net.py  llm_evaluator.py (v5)
  search/   mcts.py  selfplay.py
  train/    sft.py  llm_rl.py  az_train.py  cnn_sft.py (v5)
  eval/     anchors.py  match.py  elo.py  plots.py  agents.py (v5)
  notebooks/
    00_setup  01_build_dataset  02_train_sft  03_evaluate  04_train_llm_rl
    05_train_alphazero  06_analysis  07_train_cnn_sft (v5)  08_head_to_head (v5)
  tests/    test_engine  test_data  test_llm_policy  test_sft  test_eval  test_mcts
            test_az  test_plots  test_cnn_sft (v5)  test_agents (v5)
```

### 2.1 What each new or changed file does (v5)

- **`data/annotate.py`** — already runs one Stockfish process per CPU core and resumes
  after a stop; unchanged. Checked Oct 31: it can't split work across pods (it numbers
  its output files by counting the existing ones, so two pods in one folder would
  collide), so Step 13 labels on one many-core pod instead (Sec 4.1).
- **`eval/match.py`** — also records every eval game (Step 12a) and gains
  `head_to_head(players, games_per_pair)`: color-swapped pairs over shared openings,
  every game recorded.
- **`eval/anchors.py`** — `ResultsTable` gains W/D/L columns (Step 12b); the lock-file
  merge-on-save from v4 Step 9 (a) also covers the per-game record.
- **`eval/agents.py`** (v5) — the search switch: wraps any `evaluate()` model as
  `NoSearchAgent` (top-prior legal move, deterministic tie-break) or `MCTSAgent`
  (`search_move` at N simulations, most-visited, no noise). Generalizes AlphaZero's
  `PolicyAgent` / `AlphaZeroAgent`.
- **`model/llm_evaluator.py`** (v5) — wraps an LLM checkpoint as an `evaluate()` model:
  priors = softmax over its masked move scores, value = its value head's win
  probability. `model/llm_policy.py` stays unchanged.
- **`data/`** (Step 13) — `download.py` streams several months in order;
  `sample_positions.py` can skip given games and positions (the fixed validation set)
  and records each game's result and how it ended; `dataset.py` gives the fixed
  validation set; `data.yaml` gains a `scaled` block (500k, the months to read).
  Notebook 01 builds either set (`BUILD`) and runs from whichever repo clone it is opened in.
- **`train/cnn_sft.py`** (v5) — supervised training of `AZNet` on the Stockfish labels:
  policy cross-entropy vs. the label move (legal-masked, side-to-move frame), value BCE
  vs. the label win probability. Resumable; writes `train_log.csv` / `val_log.csv` with
  cumulative `gpu_seconds_train`, like `train/sft.py`.
- **`eval/plots.py`** — adds Elo vs. dataset size and a search-gain view; CNN SFT runs
  are their own series (`run_kind` gains `cnn_sft/`).

### 2.2 `results/results.csv` schema

As in v4 (`name`, `role`, `rating`, `ci_low`, `ci_high`, `games`, `run`, `step`,
`seed`, `train_gpu_hours`, `label_cpu_hours`, `sec_per_move`, `illegal_rate`, `gpu`,
`stockfish_version`, `date`, `spec`, `note`), plus **(v5)** `wins`, `draws`, `losses`
(from the model's side, summed over its eval games) and **`dataset_size`** (labeled
positions the run trained on; blank for self-play arms). Search variants are separate
rows: name suffix `_nosearch` or `_mcts<N>` and the same training GPU-hours as the
checkpoint they wrap.

---

## 3. Dependencies and external resources (unchanged from v4)

`python-chess`, `torch`, `transformers`, `peft`, `accelerate`, `torchao>=0.16.0`,
`datasets`, `huggingface_hub`, `zstandard`, `pandas`, `pyarrow`, `numpy`, `pyyaml`,
`jupyter`, `nbconvert`. Qwen3-0.6B from Hugging Face; Stockfish 19 via
`ensure_stockfish()`; Lichess monthly dumps (sha256-pinned); RunPod. No paid model APIs.

---

## 4. Execution order

Each step ends with a **gate**. Order from here:
**Step 9 no-search evals → Step 12 → Step 13 → Step 14 → Step 15 → Step 11**, with
**Step 16** run in the background whenever a pod is free. Several pieces overlap; Sec
4.1 says which. **Build order (Derek, Oct 31):** Step 13's code prep (fresh
500k sample, labeling-pod setup) comes before Step 12's build, so labeling starts on a
CPU-only pod while Step 12 is written; everything else follows the order above.

**Step numbers are labels, not the running order.** Steps that exist keep their numbers,
so references in this plan, CLAUDE.md, notes and git history ("Step 9 gate met") stay
valid; new work took the next free numbers (12–16); final analysis was already Step 11
and always runs last. A renumbering in running order was considered and not done
(Oct 31).

### 4.1 What can run in parallel (Oct 31)

**Two constraints decide what can overlap:**
- **Evals are CPU-bound, not GPU-bound.** Game rules, the reference opponents
  (Stockfish, random, greedy) and MCTS run on the CPU; the GPU only speeds up the
  network's own moves (it matters most for the LLM).
- **A pod's container has ~5 CPUs. Rule: one CPU-heavy job per pod.** Stockfish
  labeling, eval matches, MCTS self-play (AlphaZero) and LLM search evals are all
  CPU-heavy; two of them on one pod slow each other down and distort sec/move.

**What overlaps:**

| Work | Depends on | Where it runs | Can overlap with |
|---|---|---|---|
| Step 13 labeling (500k) | nothing; can start now | a CPU-only pod (cheaper; no GPU needed) | everything else |
| Step 12 build (eval v2) | nothing | Derek's laptop (code) | everything else |
| Step 12 reruns + head-to-head | Step 12 built | a GPU pod (the LLM rows) | labeling, Step 16 |
| Step 9 no-search evals | nothing (built) | a GPU pod | labeling |
| CNN SFT on the existing 50k (Step 14, first point) | `train/cnn_sft.py` only | a GPU pod | labeling, Step 12 build |
| LLM SFT 25k with search (Step 15) | Step 12 (c) search switch + `llm_evaluator.py` | a GPU pod | labeling |
| Step 16, AlphaZero stage 2 | nothing | its own pod and its own repo clone | everything else |

**What has to wait:** training on 500k (CNN and LLM) needs Step 13's labels; search on
those new models needs them trained; the head-to-head adds new arms as they finish.

**Labeling in parallel.** Every position is labeled independently, so the work splits
cleanly. `annotate.py` already uses one Stockfish process per core and is resumable, so
the only limit is cores: 500k ≈ 10 CPU-h is ~2 h on ~5 CPUs, or ~20–40 min on a 32-vCPU
CPU pod. **Decided Oct 31:** one many-core CPU-only pod, not several (splitting across
pods would need chunked output, Sec 2.1). Sampling the positions first is
single-threaded (~1–2 h of reading games). Parallelism only shortens wall time: the
CPU-hours charged to the arms stay ~10 either way.

**Several pods on one network volume.** A RunPod network volume can be attached to
several pods at once in the same datacenter. Rules: (1) each pod writes its own output
files; `results.csv` and `games.csv` are safe under concurrent writes (lock + merge),
labeling output is not (one labeling pod per dataset folder); (2) one repo clone per pod when they run
different code, so a checkout on one doesn't change the code under the other's job;
(3) each budget watchdog uses the combined hourly rate of all running pods.

### Done (detail in the archived v4)

**Steps 0–4** — setup, chess backbone, eval harness + reference set, 50k-position SFT
dataset (Lichess `2013-06`, Stockfish labeling 0.99 CPU-h), LLM policy wrapper.

**Step 5 — SFT baseline** — structured + masked, 44,982 training positions, stopped at
step 25,000 (7.69 L4 GPU-h). Frozen rung `step_20000` = 719; `random` = 525, `greedy` =
641.

**Step 6 — RunPod + `results.csv`** — SFT checkpoints rated: 5k 655 [621, 689], 10k 689
[655, 722], 15k 706 [664, 748], 25k 681 [638, 714]; ~0.09 s/move.

**Step 7 — LLM self-play RL from SFT** — four runs (defaults; Stockfish-scored endings;
middlegame-first curriculum; curriculum with 5× learning rate), ~10 RL GPU-h each, all
flat (647–714). Code tags `run1-default`, `run2-sfend`, `run3-curric`, `run4-lr5x`.

**Step 9 — AlphaZero, stage 1** — run `az_b6c64_s0`, 50 simulations per move, 53 rounds
(6,784 games), 9.16 GPU-h: r5 653 [604, 697], r20 716 [678, 755], r40 722 [684, 755],
r53 729 [696, 762]; 0.09 s/move. It rarely loses but rarely converts (r53: 1W 18D 1L vs.
`greedy`, 1W 16D 3L vs. `step_20000`). Gate met.

### Step 9, finishing: AlphaZero no-search evals — *done (Oct 31)*

Built Oct 31 (`PolicyAgent`, notebook 05 `EVAL_NO_SEARCH = True`). Rate the kept stage-1
checkpoints without search as their own series (`<run>_nosearch@rN`), at the same
training GPU-hours. Once Step 12 lands, these rows are re-run through the generic
`NoSearchAgent` only if their numbers would otherwise differ (they shouldn't: same rule).

*Result (Oct 31), 60 games each vs. `random`, `greedy`, `step_20000`:*

| round | GPU-h | with search (50 sims) | no search | search adds |
|---|---|---|---|---|
| r5 | 0.98 | 653 [604, 697] | 567 [522, 611] | +86 |
| r20 | 4.01 | 716 [678, 755] | 623 [580, 666] | +93 |
| r40 | 7.20 | 722 [684, 755] | 623 [580, 666] | +99 |
| r53 | 9.16 | 729 [696, 762] | 641 [611, 672] | +88 |

Reading: search adds a steady ~90 Elo at every checkpoint, and the network alone
improves too (567 → 641, z ≈ 2.7), so AlphaZero's rise is real learning, not only
search. Without search it is below LLM SFT 25k (641 vs. 681, z ≈ 1.6): the LLM is the
stronger raw policy, and AlphaZero's edge comes from search. Without search it plays in
~0.002 s/move vs. 0.09 s with search (and ~0.09 s for one LLM forward pass). r20 and r40
no-search rows are identical to the decimal, including the CI; check with Step 12's
per-game records whether that is coincidence or the same games/checkpoint.

### Step 12 — Eval v2: richer records, search switch, head-to-head

*Serves:* every Step 11 output (W/D/L and draw types explain why ratings are held down);
Q1–Q4 below. *Branch:* `step-12-eval-v2` from `main`.

- **(a) Record every eval game.** One record per game: date, white, black, result,
  termination (checkmate, stalemate, repetition, 50-move rule, insufficient material,
  ply cap, illegal-move forfeit), plies, sec/move per side, illegal attempts, opening
  id, and which eval or match it belonged to. **Decided Oct 31:** a separate file,
  `results/games.csv`, one row per game, appended safely under concurrent writes (same
  lock + merge pattern as `results.csv`). It is supporting detail; `results.csv` stays
  the summary.
- **(b) W/D/L in `results.csv`.** `wins`, `draws`, `losses` per model row, updated with
  each batch.
- **(c) Search switch for any network** (`eval/agents.py`): `NoSearchAgent` and
  `MCTSAgent(N)` over any `evaluate()` model, so CNN (AlphaZero or CNN SFT) and LLM
  (via `model/llm_evaluator.py`, Step 15) checkpoints can each be rated both ways.
- **(d) Head-to-head round robin** (`head_to_head`, notebook `08_head_to_head`):
  color-swapped pairs over shared openings among each arm's champion, every game
  recorded. Reported per pair as score %, W/D/L, the implied Elo difference
  (−400·log10(1/score − 1)) with a bootstrap CI, and draw types. Kept out of the
  reference fit, so every model's rating still comes from the frozen set alone.
  **Decided Oct 31:** each arm is represented by its **best-rated** checkpoint ("the
  best it got"), at **40 games per pair**, colors swapped. First round:
  `structured_masked_nall_e10_s0@15000` (SFT, 706),
  `rl_structured_masked_sft25000_s0_curric120_lr5x@r15` (RL run 4, 697; tied with r10,
  the more trained one taken), `az_b6c64_s0@r53` (AlphaZero, 729). 3 pairs × 40 = 120
  games. New arms (Steps 14–16) join with their best-rated checkpoint as they finish; if
  stage 2 produces a better AlphaZero checkpoint, it replaces r53.
- **(e) Re-evaluate the headline checkpoints with full records** (decided Oct 31,
  10 checkpoints): SFT 15k (best) and 25k (last); the last checkpoint (r20) of each RL
  run 1–4; AlphaZero r5, r20, r40, r53. Same opponents (`random`, `greedy`,
  `step_20000`) and games as their first evals; only what is saved changes (every game,
  with how it ended). Their Elo will shift slightly from fresh-game noise; the new rows
  replace the old ones. The other model rows keep their existing ratings without W/D/L.

*Tests:* a match writes one record per game with the right termination; W/D/L in the
row equals the sum of its records; concurrent writers lose nothing; `NoSearchAgent` and
`MCTSAgent` play only legal moves and are deterministic for a fixed seed;
`head_to_head` swaps colors and covers every pair.
*Gate:* tests pass; the 10 headline checkpoints re-rated with full records; the first
head-to-head round (3 pairs × 40 games) played and tabulated.

### Step 13 — Scaled imitation dataset

*Serves:* Q1 and Q2 (data is SFT's binding limit). *Branch:* `step-13-data-scale`.

- Same pipeline and settings as Step 3: Stockfish 19 at the same label depth, same label
  format (best move + win probability for the side to move), same phase mix, at most one
  position per phase per game, deduplicated.
- **Sampled fresh, not nested (Derek, Oct 31).** The 500k set is drawn from scratch with
  the same rules, so the 50k is not a subset of it (a clean rerun of the same code,
  rather than extending the old sample; costs ~1 CPU-h to relabel positions the 50k
  already had). Both sizes come from the same games and rules, so 50k vs. 500k still
  isolates dataset size. **Decided Oct 31: build 500k now.** Whether to build ~2M is
  decided after CNN SFT's 50k-vs-500k result (Step 14): if 500k clearly helps, 2M is
  worth its cost; if not, it is skipped.
- **Can start immediately**, in parallel with Step 12 (Sec 4.1), on a CPU-only pod
  (`docs/runpod_runbook.md` section G; notebook 01 with `BUILD = "scaled"`).
- **One fixed validation set:** the 50k's existing validation positions (~5k, its
  by-game split), shared by every size and both architectures. The 500k sample skips
  those games and those positions, so no validation game or position appears in any
  training set; all 500k positions are training data.
- **Source:** `2013-06` (~225k games) can't fill 500k at one position per phase per game
  (the endgame quota, ~167k, is the limit, since many games end before an endgame). The
  sampler reads `2013-06`, then `2013-07`, `2013-08`, … in order until every quota is
  full, downloading each month only when reached. Each file is checked against Lichess's
  published sha256, and the months used and their checksums are recorded in the
  dataset's `sampling_stats.json`.
- **Near-win positions (Derek, Oct 31, option A: measure first).** Goal: models should
  learn to convert won positions, not stalemate them. The sampler controls the phase
  mix but not how decisive a position is, which is only known after labeling. So each
  position also records its game's result and how the game ended (checkmate, stalemate,
  insufficient material, time, resignation/agreement); after labeling, notebook 01
  reports per phase the share of near-win positions (win probability > 0.9 or < 0.1)
  and of simple endings (≤ 6 pieces). Enrichment (e.g. a bucket from the last moves of
  games that ended in checkmate) is considered only if these shares and Step 12's draw
  types show a gap, as a separate dataset variant, so the 50k-vs-500k comparison stays
  a pure size comparison.
- **Cost:** estimated ~1 CPU-h per 50k labels (the Colab rate); actual for 500k: **3.10
  CPU-h** (the RunPod CPU labels ~3× faster per position), so 2M ≈ 12 CPU-h. Logged in each
  dataset's `manifest.json`; charged to every arm that trains on that size.
- Own-position labeling (DAgger: label the positions a trained model reaches in its own
  games) is parked until Step 14's results show whether drift is a problem.

*Gate:* the 500k set is labeled with its CPU-hours in the manifest; the phase mix
matches; no validation game or position in it; near-win and simple-ending shares
reported per phase.

*Result (Oct 31):* `sft_scaled_d12_n500000`: 500,000 positions (opening 166,650 /
middlegame 166,700 / endgame 166,650); Stockfish 3.10 CPU-h (11,150 s, 0.022 s/position);
labeling 13.7 min wall on a 16-vCPU CPU pod. Gate checks and the near-win table: from
the notebook's printed output (to record here).

### Step 14 — Imitation at scale: CNN SFT and LLM SFT

*Serves:* Q1 (architecture under identical data), Q2 (data scaling), and Q4's imitation
side. *Branch:* `step-14-sft-scale`.

- **CNN SFT (new arm):** `AZNet` at AlphaZero's size (6 × 64), so CNN SFT vs. AlphaZero
  compares two training regimes on one network. `train/cnn_sft.py`, `configs/cnn_sft.yaml`,
  notebook 07. Train on 50k and 500k (and 2M if built), stopping on validation loss,
  since overfitting is the known failure. It is cheap per position, so it runs first:
  the 50k run can start before Step 13 finishes (Sec 4.1), and its 50k-vs-500k result
  decides whether 2M is built. **Network size (Oct 31):** start at 6 × 64 (~0.71M
  parameters, AlphaZero's size). Add one larger network, 10 × 128 (~3.2M, ~4.5× larger,
  still ~190× smaller than the LLM), only if CNN SFT plateaus, to test whether network
  capacity is the limit.
- **LLM SFT on 500k:** the Step 5 recipe unchanged (same prompt, LoRA, value head, loss,
  learning rate, batch size), only the data grows. Cost estimate from Step 5 (~52k
  examples per GPU-h): one pass over 500k ≈ 9.6 GPU-h; over 2M ≈ 38 GPU-h.
  **Cap (Derek, Nov 1): 2 passes** (~62k steps, ~19 GPU-h at most, matching AlphaZero
  stage 2's 18 and the RL runs' ~17–18 totals). The learning rate is constant, so the 1-pass
  checkpoint (~31k steps) is the same model a 1-pass run would give: both levels are
  rated, and either can seed RL later. Validation is the fixed 50k split (`val_from`), so
  its losses compare directly with the Step 5 run's (best 2.84 at 15k). Config
  `configs/sft_500k.yaml` (Step 5's `configs/sft.yaml` untouched); run folder
  `structured_masked_nall_e2_s0_scaled_d12_n500000` (any non-default dataset is named in
  the folder, so it can't resume the Step 5 run). Notebook 02 (`SFT_CONFIG`,
  `RUN_TRAINING`, `RUN_EVAL` from the shell) on its own L4 pod and repo clone
  (`docs/runpod_runbook.md` section H). Training examples are stored as int64 arrays
  (same values, ~3 GB instead of ~12 GB of RAM for 500k).
- **Eval:** 3–4 checkpoints per run, no search (`NoSearchAgent` for the CNN; the existing
  masked argmax for the LLM, which is the same rule), with full records, into
  `results.csv`. Pick the lowest-validation-loss checkpoint plus evenly spaced ones.
  Opponents move up the reference set (`sf_1320` and above) as models strengthen; a
  model in the 800–1300 gap may need a new frozen rung (Sec 0 Eval).
- **Plots:** Elo vs. training GPU-hours (labeling CPU-hours in the dollar view) and Elo
  vs. dataset size, CNN and LLM on the same axes. GPU-hours matter here: the LLM is ~850×
  larger than the CNN, so per position it costs far more.

*Gate:* CNN SFT rated at 50k and 500k; LLM SFT rated at 500k; both curves drawn.

### Step 15 — Search at play time on the imitation networks

*Serves:* Q3 (what search adds, by architecture). Absorbs the LLM + MCTS arm's option A
(decided Oct 31). *Branch:* `step-15-search` (or the LLM + MCTS branch, if already cut).

- **`model/llm_evaluator.py`:** an LLM checkpoint as an `evaluate()` model (priors =
  softmax over masked move scores; value = value head). The LLM's value head was trained
  on Stockfish win probabilities, which makes SFT checkpoints the best leaf evaluators.
- **Ratings, with full records:** the best CNN SFT and best LLM SFT checkpoint at each
  data size, each with and without MCTS; also SFT 25k (50k data) at 16 simulations, as
  decided Oct 31. Defaults: CNN at 50 simulations (as AlphaZero); LLM at 16 (one LLM
  evaluation ~0.09 s, so ~1.5 s/move, ~2 h per 60-game eval). Matched simulations vs.
  matched sec/move: open (Sec 5 item 6).
- **Output:** an architecture × search grid per data size, {CNN, LLM} × {no search,
  search}, alongside AlphaZero's two series (self-play, with and without search). Search
  adds no training compute: each search row carries its checkpoint's GPU-hours, and its
  cost shows up in sec/move.
- LLM + MCTS option B (search inside the training loop) stays parked; revisit after
  this step's results.

*Gate:* every grid cell rated with sec/move; the search gain per network tabulated.

### Step 16 — AlphaZero stage 2 (background)

*Serves:* Q4 (the self-play curve at the LLM's compute). As specified in v4 Step 9:
raise `max_gpu_hours` to 18 and resume `az_b6c64_s0` with unchanged settings (a changed
setting would start a new run); rate up to 4 kept checkpoints of the stage, with and
without search, against the same opponents (`random`, `greedy`, `step_20000`; add
stronger ones only above ~900 or if flagged `OFF SCALE`). Runs whenever a pod is free;
alongside another job it needs its own repo clone and watchdog budgeting (Sec 1). Stage
3 (30 GPU-h) is decided after stage 2. Retuning AlphaZero (parallel self-play across
CPU workers, a bigger network, more simulations) is parked: it would start a new run,
and stage 1–2 is the clean baseline.

*Set up (Oct 31):* `configs/alphazero.yaml` has `max_gpu_hours: 18`; notebook 05 evaluates
both ways (`EVAL_NO_SEARCH = True`); launch per `docs/runpod_runbook.md` section F (stage 2).

*Gate:* the stage-2 checkpoints rated both ways; it joins the head-to-head.

### Deferred

- **Step 8 — 2×2 sweep** (mask × representation), deferred since Oct 17.
- **Step 10 — LLM regime comparison:** its only new run is pure self-play RL from the
  untrained model. Proposed: defer (Sec 5 item 7), since RL from a stronger starting
  point was flat in four runs.

### Step 11 — Final analysis (`06_analysis.ipynb`, reads only the results files)

- Elo ± CI per arm (best checkpoint), with W/D/L and draw types.
- Elo vs. training GPU-hours (and dollars) per arm: sample efficiency.
- **(v5)** Elo vs. dataset size, CNN SFT vs. LLM SFT.
- **(v5)** Search gain: the architecture × search grid (Step 15) and AlphaZero with vs.
  without search, against sec/move.
- **(v5)** Head-to-head table between arm champions.
- Dollars to train (GPU + labeling CPU) and inference dollars per game (sec/move ×
  moves/game × rate).
- Illegal-move rate for unmasked arms, only if the 2×2 returns.
- Optional: more games for the arms on the final slides.

### Research questions v5 serves (framing map)

- **Q1 Architecture:** CNN vs. LLM trained on identical imitation data, on Elo vs.
  GPU-hours.
- **Q2 Data scaling:** does more labeled data lift SFT past its plateau, for each
  architecture?
- **Q3 Search:** how much play-time MCTS adds to each network, for how much sec/move.
- **Q4 Regime:** imitation vs. pure self-play on the same CNN (CNN SFT vs. AlphaZero);
  the gap is the cost of learning without a teacher. (LLM SFT vs. SFT + RL: answered,
  flat.)

---

## 5. Open items (to align before the step that needs them)

**Decided Oct 31 (Derek):**
1. ~~Per-game record~~ — `results/games.csv` (one row per game) behind `results.csv`,
   which gains W/D/L and stays the single summary file (Step 12a).
2. ~~Re-eval scope~~ — the 10 headline checkpoints (Step 12e).
3. ~~Head-to-head~~ — best-rated checkpoint per arm, 40 games per pair, colors swapped
   (Step 12d).
4. ~~Dataset sizes~~ — 500k now, sampled fresh with the 50k's validation set; ~2M
   decided after CNN SFT's 50k-vs-500k result (Step 13).
5. **CNN size** — start at 6 × 64; add 10 × 128 only if CNN SFT plateaus (Step 14).
   The larger network itself is decided then.

**Still open (decide when the step comes):**
6. **Search budget (Step 15).** Matched simulations (cleaner algorithmic comparison) vs.
   matched sec/move (fair on inference cost; the LLM would get ~1–2 simulations at
   AlphaZero's 0.09 s).
7. **Step 10.** Defer (proposed) or keep the pure LLM self-play run.
8. **Caps.** LLM SFT at 500k decided Nov 1: 2 passes (Step 14). Still open: CNN SFT, LLM
   SFT at 2M, AlphaZero stage 3.
9. **Parked:** richer labels (Stockfish's score for every legal move, which needs
   relabeling); DAgger; LLM + MCTS option B; AlphaZero retune; Step 8.
10. **Prices:** fill `gpu_usd_per_hour` / `cpu_usd_per_hour` in `base.yaml` (source +
    date).
11. **Framing:** `docs/framing.md` should gain the CNN SFT arm, the data-scaling
    question, and the search grid (an in-place edit marked `> Update (v5)`).
