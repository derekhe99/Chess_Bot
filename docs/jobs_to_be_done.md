# Chess AI — Jobs to Be Done (Steps 0–11)

Oct 21, 2026 · @Derek · regenerated from `implementation_plan_v4.md` (Oct 21 update: Step 7 merged to `main` with runs 1–3; Step 9 AlphaZero specified in implementation detail; staged AlphaZero cap 9 / 18 / 30 GPU-h; `results.csv` saves merge instead of overwrite)

## What this view is

`implementation_plan_v4.md` gives the repo structure (what each file does) and the build order. This view describes, for each step, the capability it delivers (the L1 "Big picture") and the jobs to be done inside it (the L2 bullets), plain English first and jargon second.

Status: 🟢 core, load-bearing work for that step's milestone · 🟡 breadth or optional, layered on a working 🟢 result. Steps 0–6 are done; Step 7's code is done and merged (run 3's evals pending); Step 9 is next.

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

- **Know the rules of chess in one place:** Create positions, apply moves, list legal moves, and detect when and how a game ended (checkmate, stalemate, repetition, the 50-move rule, insufficient material). *(board wrapper on `python-chess`)* — `engine/board.py` — 🟢
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

## Step 6 — Move to RunPod and one results file ✅

**Big picture:** run everything from here on an unattended cloud GPU, and keep every number the final slides need in a single file.

**Jobs to be done:**

- **One file of record for every rating:** Merge the reference set and all model ratings into `results/results.csv`, where a `role` column tells fixed opponents (pin, reference) apart from models being measured. *(single source of truth)* — `eval/anchors.py` (`ResultsTable`) — 🟢
- **Rate a model in small batches that survive a stop:** Play 10 games per opponent at a time and overwrite the model's row after each batch, so stopping early always keeps the latest Elo and range. *(batched evaluation)* — `eval/match.py` (`evaluate_agent`) — 🟢
- **Record what each rating cost:** Each model row carries the training GPU-hours to reach that checkpoint, labeling CPU-hours, seconds per move, and illegal-move rate, the raw inputs for every cost chart. — `eval/anchors.py`, notebooks — 🟢
- **Bring up the RunPod environment:** An on-demand L4 pod with a persistent network volume for artifacts, the repo cloned onto it, the Hugging Face token as an environment variable, long notebooks run headless in tmux, and a timer that stops the pod before the balance runs out. *(RunPod setup + budget watchdog)* — notebook setup cells, `configs/base.yaml`, `docs/runpod_runbook.md` — 🟢
- **Move the existing artifacts across:** Copy the labeled dataset, the SFT checkpoints, and the reference set from Google Drive to the volume, converting the reference set into `results.csv` once. — one-off — 🟢
- **Pin the hourly prices:** One GPU and one CPU dollar rate (RunPod on-demand, with source and date) that turn hours into dollars at analysis time. — `configs/base.yaml` — 🟢
- **Put the SFT run on its compute curve:** Evaluate 3–4 SFT checkpoints into `results.csv` on the pod, which also smoke-tests the whole setup, and time a 20-game match on the pod's CPUs. — `02_train_sft.ipynb` — 🟢

## Step 7 — LLM self-play RL (code done, merged to `main`; run 3's evals pending)

**Big picture:** find out whether letting the SFT model play itself raises its Elo above the SFT plateau (~690), and at what compute. Runs 1–2 were flat; run 3 tests whether a stronger training signal changes that.

**Jobs to be done:**

- **Play the model against itself:** Play self-play games from random short openings, sampling each move from the model's own move probabilities so games differ, and record each game's positions, moves, result, how it ended, and its final position. Games run one at a time. *(self-play runner)* — `search/selfplay.py` — 🟢
- **Learn from wins and losses:** Give every move in a game that game's score as its reward, and update the model toward moves that turned out better than the model itself expected, with a pull back toward the starting model so it doesn't drift too fast. *(policy gradient + value baseline + KL penalty)* — `train/llm_rl.py` — 🟢
- **Score unfinished games with Stockfish:** A game that ends by repetition, the 50-move rule, or the ply cap gets Stockfish's win probability for its final position instead of a flat draw, so a game where one side stood clearly better still teaches something; checkmate, stalemate, and insufficient material keep their exact result. Labeled as its own regime, since an engine judges where self-play ended up. *(Stockfish end-scoring, run 2 on)* — `train/llm_rl.py` — 🟡
- **Teach the middlegame first, then full games:** Rounds 1–10 stop games at 120 plies and let Stockfish judge who stood better; later rounds play to 300 plies so the model also learns to finish a won game. *(two-phase curriculum, run 3)* — `train/llm_rl.py`, `configs/llm_rl.yaml` — 🟡
- **Keep each run's results reproducible:** Each changed config gets its own run folder through a run-name tag, older runs still resume with newer code, and each run's code is tagged in git. *(run tags + backward-compatible resume)* — `train/llm_rl.py`, git tags `run1-default` / `run2-sfend` / `run3-curric` — 🟢
- **Count the compute and watch for collapse:** Log per round the results, how games ended, how many were re-scored, losses, drift from the SFT model, and how spread out the model's move choices are, plus the GPU time of self-play and updates; stop at the GPU-hour cap and resume from the last round after a stop. *(KL, policy entropy, safety cap)* — `train/llm_rl.py` — 🟢
- **Start from the SFT model and put each run on the chart:** Warm-start RL from the SFT checkpoint, evaluate up to 4 kept checkpoints per run into `results.csv`, and confirm Elo is flat or rising. *(warm-start)* — `04_train_llm_rl.ipynb` — 🟢
- **Draw Elo vs. compute after every run:** One reusable chart from `results.csv` only: SFT checkpoints, each RL run in its own color joined to its starting checkpoint, 95% ranges, and nearby reference ratings. *(Elo-vs-compute plot, pulled forward from Step 11)* — `eval/plots.py` — 🟢

## Step 8 — LLM Phase 1: 2×2 sweep (deferred, Oct 17)

**Big picture:** find which combination of move masking and board representation hill-climbs Elo best per unit of compute.

**Jobs to be done:**

- **Train all 4 combinations:** For each mask × representation combo, run SFT, then self-play RL, to plateau or the cap, with 1 seed. — `train/sft.py`, `train/llm_rl.py` — 🟢
- **Put every run on the same chart:** Evaluate 3–4 checkpoints per run into `results.csv`. — `eval/match.py` — 🟢
- **Pick the best combination:** Choose the combo from the Elo-vs-compute curves; it becomes "the LLM arm" for Steps 10–11. — 🟢
- **Add seeds if time allows:** Extra seeds for the leading combos. — 🟡

## Step 9 — AlphaZero arm (next; branch `step-9-alphazero` from `main`)

**Big picture:** train the classic search-plus-small-network agent from scratch by pure self-play (game results only, no Stockfish, no human data), as the non-LLM point of comparison, on its own pod if needed.

**Jobs to be done:**

- **Let two jobs save results at once without losing rows:** Before saving, take a lock, re-read `results.csv`, and write back only the rows this job changed, so an LLM eval and an AlphaZero eval on two pods can't erase each other's work. *(lock file + merge-on-save)* — `eval/anchors.py`, `tests/test_eval.py` — 🟢
- **Search ahead before moving:** Try promising moves more often, using a model's move suggestions to guide the search and its win estimate to judge positions, with many games' positions sent to the GPU in one batch. Written against a simple model interface (move probabilities + expected score for the side to move) so the LLM can later drive the same search. *(PUCT MCTS)* — `search/mcts.py` — 🟢
- **Check the search finds mate-in-1:** A sanity test, with a dummy model that knows nothing, that the search itself finds a forced mate before any compute is spent. — `tests/test_mcts.py` — 🟢
- **Always show the network the board from the mover's side:** Mirror the board when Black moves and map moves through a fixed mirror table, so one set of patterns serves both colors. *(side-to-move flipping)* — `engine/encoding.py`, `engine/moves.py` — 🟢
- **Test that flipping is reversible:** Flipping twice returns the original, and flipped legal moves still match python-chess. — `tests/test_engine.py` — 🟢
- **Build the AlphaZero network:** A small convolutional network that outputs a move distribution over the shared move list (illegal moves masked) and a win estimate for the side to move. *(policy-value ResNet, provisionally 6 blocks × 64 channels)* — `model/az_net.py` — 🟢
- **Improve by playing itself:** Each round, play self-play games with search, store every position with the search's move preferences and the game's result, and train the network on a buffer of recent positions to predict both; resumable every round, with per-round logs of results, endings, losses, and GPU time. *(self-play + replay buffer + visit-count targets)* — `train/az_train.py`, `configs/alphazero.yaml` — 🟢
- **Spend compute in stages that line up with the other arms:** Stop at 9 GPU-hours, then extend by resuming to 18 (the LLM's SFT + RL total) and 30 (a placeholder for LLM + MCTS) as needed, always keeping the checkpoint where each stage stops. *(staged safety cap)* — `train/az_train.py` — 🟢
- **Rate it the same way as every other arm:** A player that searches before each move, rated against the nearest reference opponents into `results.csv` (up to 4 checkpoints per stage, including each stage's last), and drawn on the Elo-vs-compute chart from 0 GPU-hours. — `05_train_alphazero.ipynb`, `eval/match.py`, `eval/plots.py` — 🟢
- **Test cheaply before the real run:** A 2-round smoke run on the pod that measures seconds per round, so Derek can confirm the provisional settings (network size, games per round, simulations per move, learning rate) before the 9-hour stage. *(smoke run)* — `05_train_alphazero.ipynb` — 🟢

## LLM + MCTS arm (planned, Oct 17; after Step 9)

**Big picture:** test whether a search scaffold makes the LLM much stronger, by putting the structured + masked LLM inside the same tree search AlphaZero uses.

**Jobs to be done:**

- **Align the design first:** Search only at play time vs. inside the self-play training loop, simulations per move, how its compute is counted, and its cap; aligned after run 3's evals show whether the LLM's win estimate is good enough to judge positions. — plan Sec 5 — 🟢
- **Reuse the AlphaZero search:** Drive `search/mcts.py` through Step 9's interface, with the LLM's move scores as the suggestions and its value head as the position judge. — `search/mcts.py` — 🟢
- **Put it on the chart:** 3–4 checkpoints evaluated into `results.csv`; its total compute replaces AlphaZero's 30-GPU-hour placeholder. — `eval/match.py` — 🟢

## Step 10 — LLM Phase 2: regime comparison

**Big picture:** on structured + masked, find out how much each training regime (imitation, self-play, or both) buys per unit of compute.

**Jobs to be done:**

- **Train structured + masked three ways:** Pure self-play RL, SFT only, and SFT + RL, each to plateau or the cap, 1 seed. SFT only (Step 5) and SFT + RL (Step 7) already exist, so only pure self-play RL is new. Its cap is still open. — `train/sft.py`, `train/llm_rl.py` — 🟢
- **Put every regime on the same chart:** Evaluate 3–4 checkpoints per run into `results.csv`. — `eval/match.py` — 🟢

## Step 11 — Final analysis

**Big picture:** turn `results.csv` into the slides: how strong each approach got, and what it cost.

**Jobs to be done:**

- **Compare final strength:** Bar chart of the best checkpoint's Elo ± CI per arm. — `06_analysis.ipynb` — 🟢
- **Compare learning per unit of compute:** Elo vs. training GPU-hours (and dollars) curves per arm, which show sample efficiency, built on the Step 7 chart helper. *(Elo-vs-compute)* — `06_analysis.ipynb`, `eval/plots.py` — 🟢
- **Compare dollars:** Training dollars (GPU + labeling CPU) and inference dollars per game (sec/move × moves/game × rate) per arm. — `06_analysis.ipynb` — 🟢
- **Show whether self-play cures illegal moves:** Illegal-move rate for the unmasked arms, only if the deferred 2×2 returns. — `06_analysis.ipynb` — 🟡
- **Tighten the headline numbers:** Play more games for the arms on the final slides. — `eval/match.py` — 🟡
