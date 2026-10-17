# Chess AI — Jobs to Be Done (Steps 0–11)

Oct 17, 2026 · @Derek · regenerated from `implementation_plan_v4.md` (Oct 17 update: focus on structured + masked; 2×2 deferred; LLM + MCTS planned)

## What this view is

`implementation_plan_v4.md` gives the repo structure (what each file does) and the build order. This view describes, for each step, the capability it delivers (the L1 "Big picture") and the jobs to be done inside it (the L2 bullets), plain English first and jargon second.

Status: 🟢 core, load-bearing work for that step's milestone · 🟡 breadth or optional, layered on a working 🟢 result. Steps 0–5 are done.

**What the project reports at the end (Step 11):** Elo ± CI per arm; Elo vs. training compute (GPU-hours and dollars), which also shows sample efficiency; and dollars to train and inference dollars per game (plus the illegal-move rate of unmasked arms, if the deferred 2×2 returns). All of it is read from one file, `results/results.csv`.

## Step 0 — Setup ✅

**Big picture:** a code repository and a working cloud environment with a GPU, storage, Stockfish, and the pretrained model.

**Jobs to be done:**

- **Repo structure + dependency list:** Set up the folders and list every library the project needs, so later steps have a place for code and something to install. *(repo skeleton + `requirements.txt`)* — root files — 🟢
- **Keep adjustable settings out of the code:** Put every tunable number (dataset size, learning rate, games per opponent) in a few settings files, so changing one never means hunting through scripts. *(config-driven design)* — `configs/*.yaml` — 🟢
- **Bring up the environment end to end:** In one session, confirm the GPU is detected, storage is writable, Stockfish runs, and the language model downloads and generates text. — `00_setup.ipynb` — 🟢

## Step 1 — Chess backbone ✅

**Big picture:** one trustworthy way for every part of the system to apply chess rules, describe a board, and ask Stockfish for answers.

**Jobs to be done:**

- **Know the rules of chess in one place:** Create positions, apply moves, list legal moves, and detect when and how a game ended (checkmate, stalemate, repetition, the 50-move rule). *(board wrapper on `python-chess`)* — `engine/board.py` — 🟢
- **Give every agent the same move list:** One fixed numbered list of all 1,968 possible moves, plus a function that marks which are legal right now, so the LLM and AlphaZero speak the same move language. *(move vocabulary + legal-move mask)* — `engine/moves.py` — 🟢
- **Describe the board as structured text:** Turn a position into the piece-by-square text the model reads. *(structured encoding)* — `engine/encoding.py` — 🟢
- **Describe the board as FEN:** The standard compact notation, for the representation comparison. *(FEN encoding)* — `engine/encoding.py` — 🟡
- **Describe the board as a grid of numbers:** The input format AlphaZero's network needs. *(tensor planes)* — `engine/encoding.py` — 🟡
- **Ask Stockfish for the best move, or have it play at a set strength:** Start and talk to the pinned Stockfish 19 for labeling data and for playing evaluation games, tracking the CPU time labeling costs. — `engine/stockfish.py` — 🟢

## Step 2 — Evaluation harness ✅

**Big picture:** a fixed yardstick so every model's strength can be read off as one Elo number with an uncertainty range.

**Jobs to be done:**

- **Play games between any two players:** Run matches with colors swapped over shared openings, enforcing the rule that three illegal attempts in a row lose the game, and record results, time per move, and illegal attempts. *(match runner)* — `eval/match.py` — 🟢
- **Turn game results into an Elo number with a range:** Fit ratings from who beat whom, holding known ratings fixed, with a confidence interval that reflects how many games were played. *(Elo fit + bootstrap CI)* — `eval/elo.py` — 🟢
- **Fix the top of the scale:** Pin Stockfish 1320 at 1320, then rate Stockfish 1500 and 1700 against it and freeze them. *(reference set, Stockfish end)* — `eval/anchors.py` — 🟢

## Step 3 — SFT data pipeline ✅

**Big picture:** 50,000 real chess positions, each labeled by Stockfish with the right move and how good the position is.

**Jobs to be done:**

- **Download real games and pull positions from them:** Stream one pinned Lichess month without loading it all into memory. — `data/download.py` — 🟢
- **Balance and deduplicate the positions:** Sample openings, middlegames, and endgames in a set ratio, with no repeats and at most one position per phase per game. *(phase-mix sampling)* — `data/sample_positions.py` — 🟡
- **Label each position with Stockfish:** Record the best move and a win probability, and log the CPU time spent (0.99 h), which is charged to the SFT arms' cost. *(annotation)* — `data/annotate.py` — 🟢
- **Serve training records split by game:** Plain records in either representation, with no game on both sides of the train/validation split. — `data/dataset.py` — 🟢

## Step 4 — LLM policy wrapper ✅

**Big picture:** turn a generic pretrained language model into a chess player that picks a move and estimates who is winning.

**Jobs to be done:**

- **Load the model and add a small trainable layer set:** Qwen3-0.6B with LoRA adapters, so training touches only a few million weights. *(LoRA)* — `model/llm_policy.py` — 🟢
- **One prompt used everywhere:** A single template (instruction, board text, side to move, ending in `Move:`) shared by training and play. *(prompt template)* — `model/llm_policy.py` — 🟢
- **Estimate who is winning:** A small output layer that reads the model's internal state and gives the win probability for the side to move. *(value head)* — `model/llm_policy.py` — 🟢
- **Pick only from legal moves:** Score every legal move and pick the most likely, so an illegal move is impossible. *(masked decoding)* — `model/llm_policy.py` — 🟢
- **Let the model write any move:** Free generation, then parse and check legality, with the three-strikes rule applied by the harness. *(unmasked generation)* — `model/llm_policy.py` — 🟡
- **Cover all 4 toggle combinations:** Masked/unmasked × structured/FEN. — `model/llm_policy.py` — 🟡

## Step 5 — SFT baseline ✅

**Big picture:** train the first real model end to end, and use its checkpoints to connect the bottom of the Elo scale (random, greedy) to Stockfish.

**Jobs to be done:**

- **Train the model to imitate Stockfish and save checkpoints:** Supervised fine-tuning (structured + masked) on the 44,982 training positions, resumable after a disconnect, logging losses and cumulative GPU time per run. *(SFT)* — `train/sft.py` — 🟢
- **Bridge the weak end of the scale:** Rate a mid-strength checkpoint against random, greedy, and Stockfish 1320 and freeze it, which gives random (525), greedy (641), and `step_20000` (719) real ratings. *(checkpoint rung)* — `eval/anchors.py`, `eval/match.py` — 🟢

## Step 6 — Move to RunPod and one results file

**Big picture:** run everything from here on an unattended cloud GPU, and keep every number the final slides need in a single file.

**Jobs to be done:**

- **One file of record for every rating:** Merge the reference set and all model ratings into `results/results.csv`, where a `role` column tells fixed opponents (pin, reference) apart from models being measured. *(single source of truth)* — `eval/anchors.py` (`ResultsTable`) — 🟢
- **Rate a model in small batches that survive a stop:** Play 10 games per opponent at a time and overwrite the model's row after each batch, so stopping early always keeps the latest Elo and range. *(batched evaluation)* — `eval/match.py` (`evaluate_agent`) — 🟢
- **Record what each rating cost:** Each model row carries the training GPU-hours to reach that checkpoint, labeling CPU-hours, seconds per move, and illegal-move rate, the raw inputs for every cost chart. — `eval/anchors.py`, notebooks — 🟢
- **Bring up the RunPod environment:** An on-demand L4 pod with a persistent network volume for artifacts, the repo cloned onto it, the Hugging Face token as an environment variable, and long notebooks run headless in tmux. *(RunPod setup)* — notebook setup cells, `configs/base.yaml` — 🟢
- **Move the existing artifacts across:** Copy the labeled dataset, the SFT checkpoints, and the reference set from Google Drive to the volume, converting the reference set into `results.csv` once. — one-off — 🟢
- **Pin the hourly prices:** One GPU and one CPU dollar rate (RunPod on-demand, with source and date) that turn hours into dollars at analysis time. — `configs/base.yaml` — 🟢
- **Put the SFT run on its compute curve:** Evaluate 3–4 SFT checkpoints into `results.csv` on the pod, which also smoke-tests the whole setup, and time a 20-game match on the pod's CPUs. — `02_train_sft.ipynb` — 🟢

## Step 7 — LLM self-play RL

**Big picture:** keep improving the SFT model by having it play itself, and confirm the RL loop actually raises Elo before running the sweep.

**Jobs to be done:**

- **Play the model against itself:** Play self-play games from random short openings, sampling each move from the model's own move probabilities so games differ, and record each game's positions, moves, and result. Shared with AlphaZero; games run one at a time for now. *(self-play runner)* — `search/selfplay.py` — 🟢
- **Learn from wins and losses:** Give every move in a game that game's result as its reward, and update the model toward moves from won games, with a baseline to cut noise and a pull back toward the starting model so it doesn't drift too fast. *(policy gradient + value baseline + KL penalty)* — `train/llm_rl.py` — 🟢
- **Start from the SFT model and check Elo rises:** Warm-start RL from the SFT checkpoint, evaluate 3–4 checkpoints into `results.csv`, and confirm Elo is flat or rising. *(warm-start)* — `train/llm_rl.py`, `04_train_llm_rl.ipynb` — 🟢
- **Count the compute and watch for collapse:** Log per round the results, losses, drift from the SFT model, and how spread out the model's move choices are, plus the GPU time of self-play and updates, so a collapsing run is caught early and every checkpoint has its GPU-hours. *(KL, policy entropy)* — `train/llm_rl.py` — 🟢

## Step 8 — LLM Phase 1: 2×2 sweep (deferred, Oct 17)

**Big picture:** find which combination of move masking and board representation hill-climbs Elo best per unit of compute.

**Jobs to be done:**

- **Train all 4 combinations:** For each mask × representation combo, run SFT, then self-play RL, to plateau or the cap, with 1 seed. — `train/sft.py`, `train/llm_rl.py` — 🟢
- **Put every run on the same chart:** Evaluate 3–4 checkpoints per run into `results.csv`. — `eval/match.py` — 🟢
- **Pick the best combination:** Choose the combo from the Elo-vs-compute curves; it becomes "the LLM arm" for Steps 10–11. — 🟢
- **Add seeds if time allows:** Extra seeds for the leading combos. — 🟡

## Step 9 — AlphaZero arm

**Big picture:** train the classic search-plus-small-network agent from pure self-play, as the non-LLM point of comparison.

**Jobs to be done:**

- **Build the AlphaZero network:** A small convolutional network that outputs a move distribution over the shared move list and a win estimate. *(policy-value CNN)* — `model/az_net.py` — 🟢
- **Search ahead before moving:** Try promising moves more often, using the network to guide and judge the search, with network calls batched on the GPU. *(PUCT MCTS)* — `search/mcts.py` — 🟢
- **Check the search finds mate-in-1:** A sanity test before spending compute. — `tests/test_mcts.py` — 🟢
- **Always show the network the board from the mover's side:** Mirror the board when Black moves and map moves back through a fixed mirror table, so one set of patterns serves both colors. *(side-to-move flipping)* — `model/az_net.py`, `engine/moves.py` — 🟢
- **Test that flipping is reversible:** Flipping twice returns the original, and flipped legal moves still match python-chess. — `tests/test_engine.py` — 🟢
- **Train by self-play and put it on the chart:** Self-play training to plateau or the cap, 1 seed, with 3–4 checkpoints evaluated into `results.csv`. — `train/az_train.py`, `search/selfplay.py` — 🟢

## LLM + MCTS arm (planned, Oct 17; after Step 9)

**Big picture:** test whether a search scaffold makes the LLM much stronger, by putting the structured + masked LLM inside the same tree search AlphaZero uses.

**Jobs to be done:**

- **Align the design first:** Search only at play time vs. inside the self-play training loop, simulations per move, and how its compute is counted. — plan Sec 5 — 🟢
- **Reuse the AlphaZero search:** Drive `search/mcts.py` with the LLM's move scores as the prior and its value head as the leaf estimate. — `search/mcts.py` — 🟢
- **Put it on the chart:** 3–4 checkpoints evaluated into `results.csv`. — `eval/match.py` — 🟢

## Step 10 — LLM Phase 2: regime comparison

**Big picture:** on structured + masked, find out how much each training regime (imitation, self-play, or both) buys per unit of compute.

**Jobs to be done:**

- **Train structured + masked three ways:** Pure self-play RL, SFT only, and SFT + RL, each to plateau or the cap, 1 seed. SFT only (Step 5) and SFT + RL (Step 7) already exist, so only pure self-play RL is new. — `train/sft.py`, `train/llm_rl.py` — 🟢
- **Put every regime on the same chart:** Evaluate 3–4 checkpoints per run into `results.csv`. — `eval/match.py` — 🟢

## Step 11 — Final analysis

**Big picture:** turn `results.csv` into the slides: how strong each approach got, and what it cost.

**Jobs to be done:**

- **Compare final strength:** Bar chart of the best checkpoint's Elo ± CI per arm. — `06_analysis.ipynb` — 🟢
- **Compare learning per unit of compute:** Elo vs. training GPU-hours (and dollars) curves per arm, which show sample efficiency. *(Elo-vs-compute)* — `06_analysis.ipynb` — 🟢
- **Compare dollars:** Training dollars (GPU + labeling CPU) and inference dollars per game (sec/move × moves/game × rate) per arm. — `06_analysis.ipynb` — 🟢
- **Show whether self-play cures illegal moves:** Illegal-move rate for the unmasked arms, only if the deferred 2×2 returns. — `06_analysis.ipynb` — 🟡
- **Tighten the headline numbers:** Play more games for the arms on the final slides. — `eval/match.py` — 🟡
