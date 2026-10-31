# Chess AI — Jobs to Be Done (plan v5)

Oct 31, 2026 · @Derek · regenerated from `implementation_plan_v5.md` (v5: scale imitation for both architectures on one labeled dataset, measure play-time search on each, keep AlphaZero as the teacher-free comparison; eval v2 first)

## What this view is

`implementation_plan_v5.md` gives the repo structure (what each file does) and the running order. This view describes, for each step, the capability it delivers (the L1 "Big picture") and the jobs to be done inside it (the L2 bullets), plain English first and jargon second. Step numbers are labels, not the running order: done steps keep their numbers, new work is Steps 12–16, and Step 11 (final analysis) runs last.

Status: 🟢 core, load-bearing work for that step's milestone · 🟡 breadth or optional, layered on a working 🟢 result. Steps 0–7 and 9 are done. Next: Step 13's dataset prep, Step 12 (eval v2), then Steps 14–15; Step 16 runs in the background.

**What the project reports at the end (Step 11):** Elo ± CI per arm with wins/draws/losses and how games ended; Elo vs. training compute (GPU-hours and dollars); Elo vs. dataset size for the two imitation arms; what search adds to each network, against its inference cost; a head-to-head table between each arm's best model; and dollars to train and to play. Headline numbers come from `results/results.csv`; `results/games.csv` holds the per-game detail behind them.

**The research questions:** **Q1** architecture (CNN vs. LLM on identical imitation data), **Q2** data scaling (does more labeled data lift imitation past its plateau), **Q3** search (what play-time search adds to each network, for how much time per move), **Q4** regime (imitation vs. pure self-play on the same CNN).

## Done: Steps 0–7 and 9 ✅

**Big picture:** a working chess backbone, a fixed Elo yardstick, a 50k-position Stockfish-labeled dataset, an LLM chess player trained by imitation (SFT), four self-play RL runs on it, and an AlphaZero agent trained from scratch. Full detail is in the archived plan v4.

**Jobs done:**

- **Rules, moves, and board encodings in one place:** python-chess wrapper, the shared 1,968-move list, text and tensor encodings, the pinned Stockfish 19. *(chess backbone, Step 1)* — `engine/` — 🟢
- **A fixed yardstick for Elo:** Stockfish 1320 pinned at 1320; 1500, 1700, random, greedy and an SFT checkpoint rated once and frozen; every new model rated against them in batches that survive a stop, into one results file. *(reference set + `results.csv`, Steps 2, 5, 6)* — `eval/` — 🟢
- **Imitation baseline:** the LLM trained to copy Stockfish's moves on 45k positions; best Elo 706 (step 15k), and validation loss shows it starts memorizing after ~5 passes. *(LLM SFT, Step 5)* — `train/sft.py` — 🟢
- **Self-play RL on the LLM:** four runs (plain results, Stockfish-scored endings, a middlegame-first curriculum, a 5× larger update), all flat at 647–714. *(LLM RL, Step 7)* — `train/llm_rl.py` — 🟢
- **AlphaZero from scratch:** tree search guided by a small network that learns only from its own games' results; 653 → 729 in 9.2 GPU-hours, level with the best LLM checkpoints. Without search the same network rates 567 → 641, so search adds ~90 Elo. *(AlphaZero stage 1 + no-search evals, Step 9)* — `search/mcts.py`, `model/az_net.py`, `train/az_train.py` — 🟢

## Step 12 — Eval v2: richer records, a search switch, head-to-head

**Big picture:** make every rating explainable (how many wins, draws and losses, and how the games ended), let any network be rated with or without search, and put each arm's best model against the others directly.

**Jobs to be done:**

- **Keep a record of every eval game:** one row per game (players, result, how it ended, length, time per move, which eval it belonged to), safe when two pods write at once. *(per-game record)* — `eval/match.py`, `results/games.csv` — 🟢
- **Show wins, draws and losses next to each rating:** W/D/L columns in the summary file, updated after every batch. — `eval/anchors.py` — 🟢
- **Rate any network with or without search:** one switch that wraps any model as "play its top move" or "search N moves ahead first", so CNN and LLM checkpoints can each be rated both ways. *(`NoSearchAgent` / `MCTSAgent`)* — `eval/agents.py` — 🟢
- **Re-rate the 10 headline checkpoints with full records:** SFT 15k and 25k, each RL run's last checkpoint, AlphaZero's four; same opponents, now with every game saved. — notebooks — 🟢
- **Play the arms against each other:** each arm's best-rated checkpoint (SFT 15k, RL run 4 r15, AlphaZero r53), 40 games per pair, colors swapped, reported as score, W/D/L, implied Elo difference and draw types, kept out of the reference ratings. *(head-to-head round robin)* — `eval/match.py`, `08_head_to_head.ipynb` — 🟢

## Step 13 — Scaled imitation dataset

**Big picture:** ten times more Stockfish-labeled positions (500k), labeled once and shared by both architectures, because data, not compute, is what limits imitation.

**Jobs to be done:**

- **Label 500k positions the same way as the first 50k:** same Stockfish depth, label format, phase mix, at most one position per phase per game; the 50k set is an exact subset of the 500k. *(nested dataset)* — `data/`, `configs/data.yaml` — 🟢
- **Keep one validation set for everything:** the existing by-game split, shared by every size and both architectures, with no validation game in any training set. — `data/dataset.py` — 🟢
- **Check there are enough games:** count `2013-06`'s games first and add later Lichess months if needed, each pinned by checksum. — `data/download.py` — 🟢
- **Label in parallel on a CPU-only pod:** ~10 CPU-hours, ~2 h on 5 CPUs or ~20 min on a 32-core pod, split into chunks if several pods share the work; CPU-hours logged and charged to every arm that trains on it. *(chunked labeling)* — `data/annotate.py` — 🟢
- **Decide on ~2M positions later:** only if CNN SFT gains clearly from 50k to 500k. — 🟡

## Step 14 — Imitation at scale: CNN SFT and LLM SFT

**Big picture:** train both architectures on the same labeled data and see which learns more per GPU-hour and per position (Q1, Q2), and how imitation compares with self-play on the same network (Q4).

**Jobs to be done:**

- **Train AlphaZero's network to imitate Stockfish:** the same 6 × 64 network as AlphaZero, trained on the labels instead of self-play, on 50k first (can start before the 500k labels exist), then 500k, stopping when validation loss turns up. *(CNN SFT, new arm)* — `train/cnn_sft.py`, `configs/cnn_sft.yaml`, `07_train_cnn_sft.ipynb` — 🟢
- **Retrain the LLM on 500k:** the Step 5 recipe unchanged, only more data (~9.6 GPU-hours per pass), stopping on validation loss. *(LLM SFT at scale)* — `train/sft.py` — 🟢
- **Rate 3–4 checkpoints per run without search,** with full records, moving opponents up the reference set as models get stronger. — `eval/agents.py` — 🟢
- **Draw both curves:** Elo vs. GPU-hours and Elo vs. dataset size, CNN and LLM on the same axes. — `eval/plots.py` — 🟢
- **Try a larger CNN (10 × 128) only if CNN SFT plateaus,** to test whether network size is the limit. — 🟡

## Step 15 — Search at play time on the imitation networks

**Big picture:** measure what search adds to each imitation-trained network, and at what cost in time per move (Q3), completing an architecture × search grid.

**Jobs to be done:**

- **Let the search use the LLM:** wrap an LLM checkpoint so the search can ask it for move suggestions (its move probabilities) and a position judgment (its value head, trained on Stockfish's win estimates), without changing the LLM code. *(LLM evaluator)* — `model/llm_evaluator.py` — 🟢
- **Rate the SFT LLM with search first:** SFT 25k at 16 simulations against its 681 without search; it needs only the search switch, not the new data. — `eval/agents.py` — 🟢
- **Fill the grid:** the best CNN SFT and best LLM SFT at each data size, each with and without search, next to AlphaZero's two series; each search row carries its checkpoint's GPU-hours, its cost shows in time per move. — notebooks — 🟢
- **Decide matched simulations vs. matched time per move** when this step comes (the LLM gets ~1–2 simulations in AlphaZero's 0.09 s). — 🟡
- **Search inside training (option B) stays parked** until this step's results. — 🟡

## Step 16 — AlphaZero stage 2 (background)

**Big picture:** extend the self-play curve to the LLM's training compute (18 GPU-hours) on a free pod, without changing any setting.

**Jobs to be done:**

- **Resume the same run to 18 GPU-hours:** raise the cap and rerun; it continues from round 53. *(staged cap)* — `configs/alphazero.yaml` — 🟢
- **Rate the new checkpoints with and without search,** against the same opponents as stage 1, adding stronger ones only if a checkpoint passes ~900. — `05_train_alphazero.ipynb` — 🟢
- **Join the head-to-head** if it beats r53. — 🟡

## Deferred

- **Step 8 — 2×2 sweep** (masking × board representation): deferred since Oct 17. — 🟡
- **Step 10 — pure LLM self-play from scratch:** proposed deferral, since RL from a stronger start was flat four times. — 🟡

## Step 11 — Final analysis

**Big picture:** turn the results files into the slides: how strong each approach got, why, and what it cost.

**Jobs to be done:**

- **Compare final strength:** best checkpoint per arm, Elo ± CI with W/D/L and draw types. — `06_analysis.ipynb` — 🟢
- **Compare learning per unit of compute:** Elo vs. GPU-hours (and dollars) per arm. — `eval/plots.py` — 🟢
- **Compare learning per unit of data:** Elo vs. dataset size, CNN SFT vs. LLM SFT. — `eval/plots.py` — 🟢
- **Show what search buys:** the architecture × search grid and AlphaZero with vs. without search, against time per move. — `06_analysis.ipynb` — 🟢
- **Show the head-to-head table** between arm champions. — `06_analysis.ipynb` — 🟢
- **Compare dollars:** training (GPU + labeling CPU) and inference per game. — `06_analysis.ipynb` — 🟢
- **Tighten the headline numbers** with more games if time allows. — 🟡
