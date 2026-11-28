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
L4 pod while AlphaZero stage 2 runs (Step 14). Later: Step 16 done (AlphaZero to 18 GPU-h,
r109 798 with search). Step 12 (a)–(b) built; **the 500k SFT checkpoints are rated next,
before Step 12 (c)–(e)** (Derek), with (a)–(b)'s per-game records: `step-12-eval-v2` is
merged into `step-14-sft-scale` for that.
**Update, Nov 26, 2026:** Steps 12 (with the eval panel) and 17 done and merged into `main`;
results recorded in Steps 12, 14 and 17. The project's goals are restated as three
"powers" (scaffolding, SFT, RL; see the research questions after Step 11). Remaining work:
CNN SFT (Step 14b, revised) and CNN RL from CNN SFT (new Step 18), run as one overnight
chain, then Step 11 (Sec 4.0).
**Update, Nov 27, 2026:** Steps 14b and 18 done (one overnight chain); results in Sec 4.0 and
Steps 14 and 18. CNN RL is the project's first positive RL result (+76 with search over its
SFT start), but more SFT (pass 2) still matched or beat it. A GPU-utilization measurement for
the headline rows is added as Step 11 support (Derek): AlphaZero self-play ran at ~0-2% GPU.

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
| Eval | **Frozen reference set:** `sf_1320` = 1320 (pin), `sf_1500`, `sf_1700`, `random`, `greedy`, `step_20000`. Each new model is rated against it; only its own rating is fitted. Stockfish members play at 0.1 s per move. New models: 20 games against each of the 2–3 nearest reference opponents, in batches of 10, row saved after each batch. No CI target; +20 games by hand only if a CI is wider than ±200. A model may be frozen as a new rung only if CIs in the gap above `step_20000` become unusable. **New in v5:** every eval game is recorded (Step 12), and head-to-head matches between models are reported as their own table, not fed into the reference fit. **Changed Nov 1 (Step 12f, Derek): one panel for every headline model**, replacing the 2–3-nearest rule: `random`, `greedy`, `step_20000`, `sf_1320` + two frozen model rungs, AlphaZero r109 and SFT 500k @62.5k, 20 games each; every new model is rated on the full panel. |
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
  train/    sft.py  llm_rl.py  az_train.py  cnn_sft.py (v5)  llm_mcts.py (v5, Step 17)
  eval/     anchors.py  match.py  elo.py  plots.py  agents.py (v5)
  notebooks/
    00_setup  01_build_dataset  02_train_sft  03_evaluate  04_train_llm_rl
    05_train_alphazero  06_analysis  07_train_cnn_sft (v5)  08_head_to_head (v5)
    09_rerate_with_records (v5)  10_train_llm_mcts (v5, Step 17)
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
**Nov 1 (Derek):** after Step 12 (a)–(b), the 500k LLM SFT checkpoints are rated first (with
full records), then Step 12 (c)–(e).

**Step numbers are labels, not the running order.** Steps that exist keep their numbers,
so references in this plan, CLAUDE.md, notes and git history ("Step 9 gate met") stay
valid; new work took the next free numbers (12–16); final analysis was already Step 11
and always runs last. A renumbering in running order was considered and not done
(Oct 31).

### 4.0 Where things stand and what runs next (Nov 26, Derek; updated Nov 27)

**Done:** Steps 0–7, 9, 12 (incl. the eval panel), 13, 14 (LLM SFT on 500k + CNN SFT), 16, 17,
18. Steps 12 and 17 merged into `main` (Nov 26); Steps 14 and 18 merge next (Nov 27).

**Headline results, every row on the full panel (Step 12f):**

| Model | Training GPU-h | No search | With search |
|---|---|---|---|
| LLM SFT 500k @62.5k (2 passes) | 19.27 | 775 [744, 810] (rung) | **895** [846, 949] (16 sims) |
| LLM SFT 500k @30k (1 pass) | 9.25 | 735 [705, 764] | — |
| Self-play LLM (from @30k, round 4) | 19.58 | 687 [662, 713] | 814 [779, 855] (16 sims) |
| LLM SFT 50k @15k | 4.61 | 669 [636, 702] | — |
| AlphaZero CNN r109 (from scratch) | 18.04 | 666 [633, 698] | 733 [707, 761] (rung, 50 sims) |
| LLM RL run 4 @r20 | 16.56 | 662 [633, 691] | — |
| **CNN SFT 500k, pass 2** (Step 14b) | 0.008 | 742 [705, 779] | **890** [846, 938] (50 sims) |
| **CNN RL from CNN SFT pass 1, r136** (Step 18) | 19.33 | 698 [673, 727] | **859** [822, 904] (50 sims) |
| **CNN SFT 500k, pass 1** (Step 14b) | 0.003 | 662 [633, 691] | 783 [742, 822] (50 sims) |

**Order from here** (supersedes the order above and the Nov 1 table):

| # | Work | Pod | Gate / output | Est. |
|---|---|---|---|---|
| 1 | ~~**Step 14b CNN SFT + Step 18 CNN RL, one overnight chain**~~ — done Nov 27; both gates met | one L4 pod | merge into `main` | ~24 h |
| 2 | **GPU utilization of the headline rows** (Step 11 support, Derek Nov 27): `eval/gpu_util.py` | one L4 pod, restarted | one line per row in Sec 4.0 | ~30 min, ~$0.30 |
| 3 | **Step 11:** final analysis + slides | laptop | slides | 0 |

Dropped (Derek, Nov 26): a 50k CNN SFT run, a ~2M dataset, more re-rates of old curve
points, a second LLM self-play run. Watchdogs: whenever the number of running pods changes,
rerun the runbook's watchdog block on every running pod with the current balance and
RATE = $0.59 × running pods.

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

### Step 12 — Evals revamp: records, search switch, head-to-head, one panel

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
  **Changed Nov 1 (Derek):** AlphaZero r109 (798) replaces r53; the SFT entry is the 500k
  run's step 30k (`structured_masked_nall_e2_s0_scaled_d12_n500000@30000`, end of pass 1)
  instead of the 50k run's 15k: the latest SFT, chosen by validation loss (2.59 vs. 2.84)
  before its Elo exists. It is also in the 500k eval list, so it gets a rating too.
  **Round 1 result (Nov 1), 40 games per pair:**

  | Pair (a vs b) | a W/D/L | Elo gap a−b [95% CI] |
  |---|---|---|
  | AZ r109 vs RL run 4 r15 | 10/27/3 | +61 [+9, +127] |
  | AZ r109 vs SFT 500k @30k | 0/34/6 | −53 [−98, −17] |
  | RL run 4 r15 vs SFT 500k @30k | 3/32/5 | −17 [−61, +35] |

  SFT 500k @30k never lost to AZ r109, although the ratings put AZ ~110 above it (798 vs.
  689): the ratings came from different opponent sets (AlphaZero never played above
  `step_20000`) and strength isn't transitive. 55–65% of games end in threefold
  repetition. This mismatch is why (f) exists. Later rounds use RL run 4 **r20**
  (Derek, Nov 1: the same last-checkpoint rule as the panel).
- **(e) Re-evaluate the headline checkpoints with full records.** **Scope (Derek, Nov 1;
  was 10 on Oct 31): 7 checkpoints:** SFT 15k (the 50k run's best); the last checkpoint
  (r20) of each RL run 1–4; AlphaZero r5 and r53 (the ends of stage 1). Dropped: SFT 25k,
  AlphaZero r20 and r40 (saves ~3 h of the ~9). Same opponents and game counts as their
  first evals (LLM rows: `greedy`, `step_20000`, `sf_1320`; AlphaZero: `random`, `greedy`,
  `step_20000`; 20 games each), same openings (seed 0); only what is saved changes (every
  game, with how it ended). Games against deterministic opponents may replay exactly;
  games against time-limited Stockfish won't, so ratings can shift slightly. The new rows
  replace the old ones, and each new row's note keeps the old rating. The other model rows
  keep their existing ratings without W/D/L. Notebook `09_rerate_with_records`
  (`eval.agents.rerate`), resumable: a checkpoint with a full set of saved games is
  skipped. ~6 h of L4 time (~1 h per LLM checkpoint, ~30–45 min per AlphaZero one).
  *Done (Nov 1).*
- **(f) One eval panel for every headline model (added Nov 1, Derek).** Why: ratings
  against different opponents don't compare (the references' own ratings are uncertain,
  ±~100, and that error cancels only between models that faced the same ones), and
  nothing sits between `step_20000` (719) and `sf_1320`, where every model is. A game
  says most when the result is uncertain (information ∝ p(1−p)): vs. `sf_1320` a ~700
  model gets ~1/8 of an even game's information, so the gap needs rungs, not just
  `sf_1320` everywhere.
  - **The panel (6):** `random`, `greedy`, `step_20000`, `sf_1320` (pin), and two frozen
    model rungs: `az_b6c64_s0@r109` (AlphaZero, 50 simulations) and
    `structured_masked_nall_e2_s0_scaled_d12_n500000@62500` (SFT 500k, end of pass 2;
    lowest validation loss, 2.40). Not SFT @30k: it rates next to `step_20000` (no gap
    filled), and it is Step 17's start, whose self-play could learn to exploit it; it stays
    a fully rated model instead. No Stockfish rung below 1320: `UCI_Elo`'s floor is 1320,
    and a node- or depth-limited Stockfish isn't reliably weaker. `sf_1500`/`sf_1700` stay
    frozen, off the panel.
  - **Stage 1, calibrate the rungs:** each plays the 4 existing members and the other
    rung, **40 games per pairing** (a rung's error passes to every model rated against
    it; the original members got 160–200 games). Both fitted together with the 4 held
    fixed, then frozen. AZ r109's model row (798) becomes a reference row; its note keeps
    the old rating.
  - **Stage 2, rate the headline models on the panel, 20 games per opponent:** rows with
    saved games play only the opponents they lack (old games stay valid: same checkpoint,
    engine and frozen opponents), then are refitted on all their games. **Trimmed (Derek,
    Nov 1, cost): one row per story:** SFT 50k @15k, SFT 500k @30k, RL run 4 @r20 (play
    `random` + the 2 rungs: 60 games each); AZ r109 without search (no saved games: all 6,
    120 games). AZ r109 with search is the rung itself, so its search gain is rung rating −
    no-search rating. Dropped: RL runs 1–3 (same flat story), AZ r53.
  - **Stage 3, consistency check:** the ratings' predicted head-to-head score vs. the
    actual one (AZ r109 vs. SFT 500k @30k from round 1); a standing column in notebook
    08's table from now on. Large gaps mean the panel can't yet be trusted.
  - **Code:** `panel:` in `configs/eval.yaml` (the one opponent list every notebook uses;
    20 games per opponent, 40 for rungs); AlphaZero checkpoint rungs and model-to-
    reference promotion (`eval/anchors.py`); `calibrate_rungs()` and `extend()`
    (`eval/agents.py`); a panel-coverage check per row (Step 11 flags rows off the panel);
    a predicted-score column in `head_to_head_table`; notebook `11_eval_panel` (stages
    1–3, resumable); notebooks 04/05/08 use the panel (02 after Step 14 merges, since that
    branch changed notebook 02; 10 on Step 17's branch); tests. New logic lives in
    `eval/panel.py`, which no other open branch touches, so the merges stay clean.
  - **Where:** one GPU L4 pod for all of it (most games involve an LLM; keeps sec/move
    GPU-measured), ~9 h (stage 1 ~4.5, stage 2 ~4.5). Every new model now costs 120 games (was 60).

*Tests:* a match writes one record per game with the right termination; W/D/L in the
row equals the sum of its records; concurrent writers lose nothing; `NoSearchAgent` and
`MCTSAgent` play only legal moves and are deterministic for a fixed seed;
`head_to_head` swaps colors and covers every pair.
*Gate:* tests pass; the 7 headline checkpoints re-rated with full records; the first
head-to-head round (3 pairs × 40 games) played and tabulated (**met Nov 1**); (f): both
rungs calibrated and frozen, every headline row rated on the full panel, the consistency
check reported. Then `step-12-eval-v2` merges into `main`.

*Result (f), Nov 2 (notebook 11, 80 min on one L4):* rungs frozen at **AZ r109 733 [707, 761]**
and **SFT 500k @62.5k 775 [744, 810]** (200 calibration games each); on the panel (120 games):
SFT 500k @30k 735 (was 689), SFT 50k @15k 669 (was 714), AZ r109 no search 666 (was 666),
RL run 4 @r20 662 (was 681). AZ r109's old 798 was inflated by weak opponents. Consistency
check (AZ r109 vs. SFT @30k, round 1): predicted −2, observed −53 [−98, −17]: closer than the
old +110 but still outside the CI (34 of 40 games drawn). **Gate met; merged into `main`
Nov 26.** Follow-up: rung rows don't carry W/D/L yet (their games are in `games.csv` as
`kind = calib`); filled in by `calibrate_rungs` in Step 14b's build.

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
labeling 13.7 min wall on a 16-vCPU CPU pod. **Gate met** (all 7 checks pass; the fixed
validation set is the 50k's 5,018 positions, none of them or their games in the 500k).

Near-win check (option A), share of positions:

| phase | near-win (win prob > 0.9 or < 0.1) | simple ending (≤ 6 pieces) | from a game that ended in checkmate |
|---|---|---|---|
| opening | 0.061 | 0.000 | 0.270 |
| middlegame | 0.373 | 0.003 | 0.274 |
| endgame | 0.541 | 0.119 | 0.212 |
| all | 0.325 | 0.041 | 0.252 |

Game endings: resignation/draw 41%, time 31%, checkmate 25%, insufficient material 1.2%,
stalemate 1.1%. Reading: ~162k near-decisive positions (~90k of them endgames) and ~20k
simple endings, each labeled with Stockfish's best move, so the data already holds many
conversion examples; no enrichment now. Revisit only if Step 12's draw types show
models still failing to convert won positions.

### Step 14 — Imitation at scale: CNN SFT and LLM SFT

*Serves:* Q1 (architecture under identical data), Q2 (data scaling), and Q4's imitation
side. *Branch:* `step-14-sft-scale`.

- **CNN SFT (Step 14b; revised Nov 26, Derek):** `AZNet` at AlphaZero's size (6 × 64, ~0.71M
  parameters; same network as AlphaZero, so CNN SFT vs. AlphaZero compares two training
  regimes on one network). Trained on the **500k** set only (the same labels and fixed 50k
  validation set as the LLM): move target = Stockfish's best move (cross-entropy over the
  legal moves, side-to-move frame), value target = Stockfish's win probability (BCE).
  **Batch 256, 2 passes (~1,950 steps per pass)**; kept checkpoints at the **end of pass 1
  and of pass 2** (the analogues of LLM @30k and @62.5k: the same data seen, counted in
  passes; steps differ only because the batch is 16× larger). Saved in AlphaZero's
  checkpoint format, so Step 18 starts from it directly. Training costs minutes of GPU: SFT
  on a network this small is limited by data, not compute, so it is reported at its own
  GPU-h rather than padded to 19 (filling 19 GPU-h would mean hundreds of passes and
  memorization). `train/cnn_sft.py`, `configs/cnn_sft.yaml`, notebook 07. Dropped (Nov 26):
  the 50k run, the 2M dataset, the 10 × 128 network.
- **CNN SFT evals:** pass 1 and pass 2, each **with and without search** (50 simulations,
  AlphaZero's setting), on the full panel. These also fill the CNN half of Step 15.
  *Done (Nov 27):* run `cnn_b6c64_n500000_e2_s0`, on the panel (120 games each):

  | checkpoint | training GPU-h | no search | with search (50 sims) | search gain |
  |---|---|---|---|---|
  | pass 1 | 0.003 | 662 [633, 691] | 783 [742, 822] | +121 |
  | pass 2 | 0.008 | 742 [705, 779] | **890** [846, 938] | +148 |

  The CNN learns from the same labels as well as the LLM: pass 2 with search (890) ties LLM
  SFT @62.5k with search (895) at ~2,500× less training compute. Without search the CNN is
  below the LLM at both passes (662 vs. 735; 742 vs. 775).
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
  *Done (Nov 1): 62,500 steps, 19.27 GPU-h.* Validation (the fixed 50k split):

  | step | passes | val move loss | val value loss | top-1 vs Stockfish |
  |---|---|---|---|---|
  | 5,000 | 0.16 | 3.157 | 0.550 | 24.0% |
  | 10,000 | 0.32 | 2.919 | 0.545 | 26.5% |
  | 15,000 | 0.48 | 2.800 | 0.537 | 30.0% |
  | 20,000 | 0.64 | 2.749 | 0.563 | 30.5% |
  | 25,000 | 0.80 | 2.661 | 0.530 | 31.0% |
  | 30,000 | 0.96 | 2.589 | 0.523 | 31.5% |
  | 62,500 | 2.00 | 2.402 | 0.511 | 40.0% |

  Already below the 50k run's best (2.84 at its step 15k, after ~5 passes) by step 15k, and
  still falling with no sign of overfitting: more data is helping, as v5 predicted. **On
  the panel:** @30k 735, @62.5k 775 (rung) vs. the 50k run's best 669: +66 from 10× data,
  +40 more from the second pass (CIs overlap for the second step).
  **Evals (Nov 1, Derek: before Step 12 (c)–(e)):** steps 10k, 20k, 30k (≈ 1 pass) first,
  vs. `greedy`, `step_20000`, `sf_1320` (20 games each, as the 50k SFT evals), with full
  records; later 45k and the final step (62.5k), plus the lowest-validation-loss one if it is
  not among them; `sf_1500` added if a checkpoint scores well against `sf_1320`. Notebook 02
  with `RUN_TRAINING=0 RUN_EVAL=1 EVAL_STEPS=...` on a separate pod (runbook section I).
- **Eval:** every rated checkpoint on the full panel (Step 12f), with full records.
- **Plots:** Elo vs. training GPU-hours (labeling CPU-hours in the dollar view) and Elo
  vs. dataset size, CNN and LLM on the same axes. GPU-hours matter here: the LLM is ~850×
  larger than the CNN, so per position it costs far more.

*Gate (revised Nov 26):* LLM SFT rated at 500k (met); CNN SFT pass 1 and pass 2 rated on the
panel, each with and without search.

### Step 15 — Search at play time on the imitation networks

*Serves:* Q3 (what search adds, by architecture). Absorbs the LLM + MCTS arm's option A
(decided Oct 31). *Branch:* `step-15-search` (or the LLM + MCTS branch, if already cut).

- **`model/llm_evaluator.py`:** an LLM checkpoint as an `evaluate()` model (priors =
  softmax over masked move scores; value = value head). The LLM's value head was trained
  on Stockfish win probabilities, which makes SFT checkpoints the best leaf evaluators.
  **Batched (Nov 1):** `evaluate(boards)` scores many positions in one forward pass, so
  Step 17's self-play (many games at once, as AlphaZero's `play_games`) keeps the GPU busy.
- **Ratings, with full records:** the best CNN SFT and best LLM SFT checkpoint at each
  data size, each with and without MCTS; also SFT 25k (50k data) at 16 simulations, as
  decided Oct 31. **First (Derek, Nov 1): the 500k run's step 30k at 16 simulations** (and
  optionally 8), against its own no-search rating. It is Step 17's control (search at
  play time, no extra training) and its go/no-go check: if 16 simulations add a clear Elo
  gain, the search's move targets are better than the policy, so they are worth training
  on. Defaults: CNN at 50 simulations (as AlphaZero); LLM at 16 (one LLM
  evaluation ~0.09 s, so ~1.5 s/move, ~2 h per 60-game eval). Matched simulations vs.
  matched sec/move: open (Sec 5 item 6).
- **Output:** an architecture × search grid per data size, {CNN, LLM} × {no search,
  search}, alongside AlphaZero's two series (self-play, with and without search). Search
  adds no training compute: each search row carries its checkpoint's GPU-hours, and its
  cost shows up in sec/move.
- LLM + MCTS option B (search inside the training loop): **un-parked Nov 1 as Step 17.**

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

*Result (Nov 1):* 109 rounds, 18.04 GPU-h. 60 games each vs. `random`, `greedy`,
`step_20000` (rated before Step 12, so no per-game records):

| round | GPU-h | with search (50 sims) | no search | search adds |
|---|---|---|---|---|
| r55 | 9.45 | 697 [659, 735] | 647 [611, 684] | +50 |
| r75 | 12.73 | 769 [722, 829] | 659 [629, 690] | +110 |
| r90 | 15.02 | 769 [722, 821] | 659 [617, 703] | +110 |
| r109 | 18.04 | 798 [755, 853] | 666 [629, 703] | +132 |

Reading: with search it keeps climbing, 729 (r53) → 798 (r109), z ≈ 2.3, now the
highest-rated model in the project; the network alone barely moves (641 → 666, within
noise). So stage 2's gain comes mostly from search getting more out of the network (search
adds ~90 at stage 1, ~130 by r109), consistent with a better value head guiding the tree.
r75 and r90 rate identically (twice) because they scored the same against each opponent
(their CIs differ, so the games did); with draw-heavy, 20-game pairings, equal scores are
not rare. r109 replaces r53 in the head-to-head (Step 12d). Stage 3 (30 GPU-h): open
(Sec 5 item 8).

*Gate:* the stage-2 checkpoints rated both ways (met Nov 1); it joins the head-to-head.

### Step 17 — Self-play LLM with search vs. more SFT, at equal compute

*Serves:* Q4 for the LLM (imitation vs. self-play from the same start, at equal compute)
and Q3 (what search adds, and whether training with it changes that). *Added Nov 1
(Derek); the high-priority item after the 500k evals. Branch:* `step-17-llm-mcts` (cut
from `step-12-eval-v2`, which has the search switch).

**The question:** starting from SFT 500k @30k (end of pass 1, 9.25 GPU-h), does the LLM go
further with 10 more GPU-h of **self-play with search** or 10 more GPU-h of **SFT** (pass
2, which ended at @62.5k)?

- **The two models compared (names used from here on):**
  - **Self-play LLM** (this step): AlphaZero-style training with the LLM as the network,
    continuing SFT @30k's LoRA adapters. Rounds of self-play games with 16-simulation
    MCTS; the policy learns the search's root visit distribution over legal moves
    (cross-entropy of the LLM's masked move distribution), the value head learns the game
    result for the side to move. Trained to **10.0 GPU-h** (pass 2's measured cost:
    19.27 − 9.25).
  - **SFT @62.5k:** pass 2, already trained (no extra cost).
- **Evals (Derek, Nov 1): a 2 × 2, training method × search, all on the full panel
  (Step 12f), 20 games per member:**

  | | no search | with search (16 sims) |
  |---|---|---|
  | **SFT @62.5k** | free: its rung rating from Step 12(f) calibration | new eval, 120 games, ~3 h |
  | **Self-play LLM** (final checkpoint) | new eval, 120 games, ~2 h | new eval, 120 games, ~3 h |

  Reading it: the *with search* column = which LLM + search system is better (the end
  product); the *no search* column = which training made the network itself better; each
  row's gap = what search adds to that model. Without the self-play no-search cell, a
  self-play win couldn't be split into "it learned more" vs. "search did the work" (the
  question Step 7 left open).
- **Control (the go/no-go):** SFT @30k with 16-simulation search, no extra training,
  rated against `greedy`, `step_20000`, `sf_1320` (the same opponents as its no-search
  rating, 689). *Halfway (Nov 1, 30 of 60 games): 809 [739, 885], 8W 12D 10L vs. 1W 40D
  19L without search; 1.72 s/move.* Not extended to the full panel: the +50 check is
  already like for like.
- **Dropped (Derek, Nov 1, cost):** mid-run checkpoint evals (checkpoints are kept, so
  they can be rated later), head-to-heads vs. SFT @30k (the panel already shows whether
  it improved; the self-play LLM plays SFT @62.5k directly as a panel member), extending
  the control to the panel.
- **Why it can work where Step 7 didn't.** Plain RL gave one game result per ~150 moves
  and a value head at chance, so no per-move credit (four flat runs). Search gives every
  position an improved target (its visit counts), a per-move signal like SFT's labels but
  produced by the model's own look-ahead: the mechanism behind AlphaZero's climb.
- **Simulations: 16 per move (Derek, Nov 1).** With ~30 legal moves, 8 simulations barely
  look one move ahead, so the targets mostly echo the policy (close to imitating itself);
  16 let the top candidates be searched 2–3 half-moves deep, enough to catch a blunder.
- **Throughput (estimate, Nov 1; measured by the timing probe).** One unbatched LLM
  evaluation ~0.09 s; batched ~15–25 ms per position. Per training position: self-play
  ~17 evaluations (~0.35 s batched) + the update (all legal moves scored with gradients,
  ~0.2 s). In 10 GPU-h: **~60k positions, ~600 games** (vs. pass 2's 500k labeled
  positions). SFT is the favorite at equal compute; the self-play LLM keeping up or
  winning would be a strong result, and losing measures how much less efficient
  self-taught search is than an oracle at this scale.
- **Code:** `model/llm_evaluator.py` (batched; done); `train/llm_mcts.py` (rounds of
  `search.mcts.play_games` with the LLM evaluator, replay buffer, policy + value update,
  resumable, GPU-hour cap, `mcts_log.csv`); `configs/llm_mcts.yaml`; notebook
  `10_train_llm_mcts`. `model/llm_policy.py` stays unchanged.
- **Order:** (1) evaluator + timing probe (done); (2) control (running Nov 1); (3) if the
  control clears +50: build the trainer, smoke-test it; (4) train the self-play LLM to
  10.0 GPU-h on its own pod and repo clone, in parallel with Step 12(f); (5) the 3 new
  evals above, once the panel's rungs are frozen. Cost (2)–(5): ~17 pod-h, ~$10.
- **Training settings (Derek, Nov 1; `configs/llm_mcts.yaml`):** 64 games per round
  (searched together); **no pull toward the starting model** (AlphaZero has none; the
  targets are per-move); replay buffer = the **last 3 rounds'** positions; per round, **2 x
  the new positions** in examples, SFT's learning rate (1e-4) and batch size (16). Each
  example = one buffer position + one move sampled by its visit share, trained exactly like
  an SFT example (move NLL + value BCE vs. the game result): an unbiased estimate of the
  cross-entropy to the visit distribution at one example's cost. Fixed as AlphaZero: root
  noise 0.3/0.25, sampled moves for 30 plies, draw at 300 plies. A round starts only if at
  least half of it fits the budget (ends within ~half a round of 10.0 GPU-h). Checkpoint
  every 2 rounds; only the final one is evaluated.
- **Runs unattended (Derek, Nov 1):** one chained command on its pod: smoke test (2 tiny
  rounds, separate folder) -> gate check (+50) -> training -> the 3 new evals (waits for the
  panel's rungs) -> the pod stops itself. Notebook 10 parts 3-5.

*Gate (before training):* the control shows a clear Elo gain from 16 simulations,
**defined as +50 Elo or more over the same checkpoint without search (Derek, Nov 1)**;
the timing probe sizes the rounds. *Gate (end):* the self-play LLM trained to 10.0 GPU-h;
the 2 × 2 above filled, every cell on the full panel.

*Result (Nov 2):* control (SFT @30k + 16 sims, 3 opponents) 818 vs. 689 = **+129: go.** Self-play
ran 4 rounds (~2.6 GPU-h each, almost all self-play: ~1.3 s per position vs. the ~0.35
estimate; ~25k positions), 19.58 total GPU-h. On the panel:

| | no search | with search (16 sims) | search adds |
|---|---|---|---|
| SFT @62.5k | 775 [744, 810] | **895** [846, 949] | +120 |
| Self-play LLM (round 4) | 687 [662, 713] | 814 [779, 855] | +127 |
| start: SFT @30k | 735 [705, 764] | — | (+129, control) |

Reading: at equal compute, **more SFT beat self-play with search** (+88 without search, CIs
apart; +81 with search, CIs touching), and self-play pushed the raw network **down** from 735
to 687 (borderline). Search adds a steady ~+125 to every LLM checkpoint; self-play training
did not change how much. Likely causes: only 4 rounds, noisy 16-sim targets (top move ~39% of
visits), the value head retrained on game results. One seed, one setting: at this compute
and scale, self-play with search loses to imitation. **Gate met; merged into `main` Nov 26.**

### Step 18 — CNN RL from CNN SFT (added Nov 26, Derek)

*Serves:* goal 3, the power of RL (does self-play improve on an SFT start?) for the CNN,
mirroring Step 17 for the LLM; also an SFT start vs. a random start for AlphaZero's recipe.
*Branch:* `step-14-sft-scale` (built and run together with Step 14b, in one chain).

- **Recipe:** AlphaZero's training (`train/az_train.py`), **settings unchanged** (128 games
  per round, 50 simulations, its buffer, learning rate and sampling), started from **CNN SFT
  end of pass 1** instead of random weights (the analogue of Step 17 starting from LLM
  @30k). Code: `az_train` gains a start-from-checkpoint setting (a new run folder; AlphaZero's
  run is untouched).
- **Budget:** total training compute (CNN SFT pass 1 + RL) **≈ 19.3 GPU-h**, matching the
  LLM's final models (19.27 and 19.58) and AlphaZero from scratch (18.04).
- **Evals:** the final checkpoint on the full panel, with and without search (50 sims).
- **Readout:** vs. its start (CNN SFT pass 1): did RL improve the network? vs. CNN SFT pass 2:
  RL or more SFT? vs. AlphaZero from scratch (733 / 666): what did the SFT start buy?
- **Runs in one chain with Step 14b (Derek, Nov 26):** smoke tests (both) → CNN SFT → **auto-check**
  (RL starts only if validation move loss fell during SFT and pass 1's no-search panel rating
  beats `random`, 525; else the chain skips RL and stops the pod) → CNN SFT evals (evals first,
  so they're ready while RL runs) → RL → final evals → the pod stops itself. This replaces
  the manual look at Step 14b's results before Step 18 starts.

*Gate:* trained to ≈ 19.3 total GPU-h; the final checkpoint rated on the panel with and without
search.

*Done (Nov 27): gate met.* Run `az_b6c64_s0_cnnsft_p1`, 136 rounds, 19.33 GPU-h (incl.
0.003 of CNN SFT). Final checkpoint r136 on the panel (120 games each):

| comparison | no search | with search (50 sims) |
|---|---|---|
| **CNN RL r136** | **698** [673, 727] | **859** [822, 904] |
| vs. its start, CNN SFT pass 1 (662 / 783) | +36 (borderline, ~1.8σ) | **+76** (~2.6σ) |
| vs. CNN SFT pass 2 (742 / 890) | −44 (~1.9σ) | −31 (within noise) |
| vs. AlphaZero from scratch r109 (666 / 733) | +32 | **+126** |

(σ: the difference over the combined standard error, both from the CIs.) Readout: RL improved
the network (the project's first positive RL result), mostly in play with search (search gain
+161 vs. +121 at its start: the value head, trained on game results, seems to help search
most); more SFT (pass 2) still matched or beat it at a tiny fraction of the GPU-h; an SFT
start is worth +126 with search over random weights at equal compute. Training signals: value
loss 0.59 → 0.48 by ~r10, policy loss rose then recovered (2.42 → 2.52 at ~r30 → 2.43 at
r66), entropy steady ~2.4–2.5, repetition endings rising (2 → 20–34 per 128 games); the final
checkpoint drew 48 of 120 panel games with search.
**Caveat (measured Nov 27):** during this run the GPU sat at ~0–2% utilization with one CPU
core at 100% (Python MCTS, `search/mcts.py`): the GPU-h bought mostly CPU time, so this is a
lower bound on what AlphaZero's recipe gets from 19 GPU-h. Utilization per headline row:
Step 11 support item; parallel self-play: Sec 5 item 13.

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
- **GPU utilization per headline row (support item, Derek Nov 27).** GPU-h is the compute
  unit of every comparison, but a GPU-hour isn't equally busy in every arm (AlphaZero
  self-play measured ~0–2%). Per row: utilization = (self-play h × u_selfplay + update h ×
  u_update) ÷ total h.
  - *Known* (logs on the volume): per round `selfplay_seconds`, `update_seconds`, `positions`
    in `checkpoints/llm_rl/<run>/rl_log.csv`, `checkpoints/llm_mcts/<run>/mcts_log.csv`,
    `checkpoints/alphazero/<run>/az_log.csv`; Step 7's `stockfish_seconds` (CPU only, counted
    at 0% GPU; it sits inside `selfplay_seconds` and inside GPU-h).
  - *Measured* (new): u_selfplay and u_update per arm, by replaying a short slice with the real
    config and checkpoint (full game count, ~20 moves; ~30 update steps) while `nvidia-smi`
    samples utilization every second.
  - *Assumed:* the slice is representative (checked: seconds per position vs. the log's mean;
    flagged if off by > 25%); the slice misses each round's small-batch tail, so u_selfplay is
    slightly high; `nvidia-smi` utilization = % of time any kernel runs (standard, generous);
    same GPU type (L4) as the original runs.
  - *Rows:* (1) LLM SFT 500k (@30k and @62.5k; measured: ~30 SFT steps from @62.5k; also used
    for the SFT parts of rows 2–3 and for LLM SFT 50k, which ran on Colab: an assumption);
    (2) self-play LLM, Step 17 round 4; (3) LLM RL run 4 @r20; (4) AlphaZero from scratch
    r109; (5) CNN RL from CNN SFT, r136. Not measured: CNN SFT (0.003–0.008 GPU-h).
  - *Code:* `eval/gpu_util.py` (`split`: logs only, CPU; `measure`: GPU replay; `combine`),
    run from a notebook cell on a restarted L4 pod (~30 min, ~$0.30). *Output:* one line per
    row in Sec 4.0 (no new results file).

### The project's goals (Derek, Nov 26) and the research questions behind them

1. **Power of scaffolding (search):** what play-time MCTS adds to each network (Q3). *LLM:
   +120–130 at every checkpoint (Steps 15, 17); AlphaZero CNN: +67. CNN SFT: +121 / +148
   (passes 1 / 2); CNN RL from SFT: +161 (Steps 14b, 18).*
2. **Power of SFT:** imitation of Stockfish's labels, and how it scales with data (Q2). *LLM:
   669 (50k) → 735 → 775 (500k, 1 and 2 passes). CNN (500k): 662 → 742 without search, 783 →
   890 with (passes 1 and 2), at 0.003–0.008 GPU-h (Step 14b).*
3. **Power of RL:** does learning from game results improve on an SFT start, at equal
   compute (Q4)? *LLM: no (plain RL flat; self-play with search lost to more SFT, Step 17).
   CNN: yes vs. its start (+36 / +76 without / with search), but more SFT matched or beat it
   (Step 18); AlphaZero's GPU-h were mostly idle GPU (Step 18 caveat).*

Plus Q1, architecture: CNN vs. LLM on identical data and compute (Step 14b vs. 14a, Step 18 vs.
17), on Elo vs. GPU-hours.

---

## 5. Open items (to align before the step that needs them)

**Decided Oct 31 (Derek):**
1. ~~Per-game record~~ — `results/games.csv` (one row per game) behind `results.csv`,
   which gains W/D/L and stays the single summary file (Step 12a).
2. ~~Re-eval scope~~ — 7 headline checkpoints (Step 12e; narrowed from 10 on Nov 1).
3. ~~Head-to-head~~ — best-rated checkpoint per arm, 40 games per pair, colors swapped
   (Step 12d).
4. ~~Dataset sizes~~ — 500k, sampled fresh with the 50k's validation set (Step 13); ~2M
   dropped (Derek, Nov 26).
5. ~~CNN size~~ — 6 × 64 only (AlphaZero's); 10 × 128 dropped (Derek, Nov 26).

**Still open (decide when the step comes):**
6. **Search budget (Step 15).** Matched simulations (cleaner algorithmic comparison) vs.
   matched sec/move (fair on inference cost; the LLM would get ~1–2 simulations at
   AlphaZero's 0.09 s).
7. **Step 10.** Defer (proposed) or keep the pure LLM self-play run.
8. ~~Caps~~ — LLM SFT 500k: 2 passes (Nov 1). CNN SFT 500k: 2 passes, batch 256; CNN RL:
   ≈ 19.3 total GPU-h (Nov 26). AlphaZero stage 3, LLM SFT at 2M: dropped.
9. **Parked:** richer labels (Stockfish's score for every legal move, which needs
   relabeling); DAgger; AlphaZero retune; Step 8. (LLM + MCTS option B: un-parked Nov 1
   as Step 17.)
10. **Prices:** fill `gpu_usd_per_hour` / `cpu_usd_per_hour` in `base.yaml` (source +
    date).
11. **Framing:** `docs/framing.md` should gain the CNN SFT arm, the data-scaling
    question, the search grid, and Step 17's imitation-vs-search-self-play comparison (an
    in-place edit marked `> Update (v5)`).
12. ~~Step 17's training details~~ — decided Nov 1 (Step 17: 64 games/round, no KL, last
    3 rounds, 2x per position at SFT's learning rate).
13. **Parallel self-play for the AlphaZero arms (found Nov 27).** AlphaZero self-play is
    single-threaded Python (one core at 100%, GPU ~0–2%; the L4 pod has 6 vCPUs). Options:
    (a) keep the results as they are and report the caveat; (b) parallel self-play across the
    cores (plausibly ~4–5× the games per GPU-h), then re-run both CNN RL arms (from scratch and
    from SFT) at ~19.3 GPU-h so they stay comparable (~2 × $11.40 + build time). Decide after
    the utilization numbers (Step 11 support).
