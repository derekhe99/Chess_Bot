# CLAUDE.md — Chess_Bot

## Source docs: the why and the what

Both are imported so they're in context every session.

**Framing** (the *why*: research question, arms, metrics, and the review notes behind
each decision):

@docs/framing.md

**Implementation plan** (the *what* and *how*: the spec for scope, design, and
execution order):

@docs/implementation_plan_v4.md

`docs/jobs_to_be_done.md` is a third view of the same plan (what each step delivers);
regenerate it when the plan changes.

Precedence: the plan beats the framing (it was written later and incorporates the
review notes). This file beats both for implementation details and for decisions made
after the plan was written.

Rules:

- **80/20, MVP.** Track only the few numbers the Step 11 slides need, in as few files
  as possible. Every metric, log, or rigor step has to feed a Step 11 output, or it
  doesn't get built.
- **Before any change**, check it against the plan: which Step it belongs to, whether it
  respects the locked decisions (plan Sec 0), and whether that Step's prerequisites and
  gates are met (plan Sec 4). Don't build ahead of the current Step without asking. Also
  check it serves the framing: which question or metric it supports.
- **Deviations and open items are discussed first.** Any change that departs from the
  plan, adds scope, skips a gate, or settles an open item: stop, flag it to Derek with
  options and trade-offs, and align with him. Don't decide unilaterally.
- **Order after aligning**: update the current plan, `docs/implementation_plan_v4.md`
  (and the Status section below), first, then write the code. Small updates edit the
  current version in place; a new version only when Derek says so, with the old one
  archived outside the repo in `ChessAI/archive/`.
- Cowork mirrors the plan, the framing, and the JTBD view in its Project docs. After
  editing any of them, remind Derek so the Cowork copy gets re-synced.

## Working with Derek

- Research project, and Derek is learning as he goes: explain the *why* from first
  principles, keep it concise, no filler.
- **Derek runs git himself.** Suggest exact commands (`git commit -m "..."`); don't
  commit, push, or rewrite history unless he explicitly asks.

## Status

- Steps 0–4 — done (Colab). Engine, eval harness, 50k-position Lichess `2013-06` dataset
  (Stockfish labeling 0.99 CPU-h), LLM policy wrapper (Qwen3-0.6B, LoRA, value head,
  masked/unmasked, structured/FEN).
- Step 5 — done (Colab, L4). SFT, structured + masked, full 44,982-position train split,
  stopped at step 25,000 = 7.69 GPU-h (run `structured_masked_nall_e10_s0`). Frozen
  reference set: `sf_1320` = 1320 (pin), `sf_1500` = 1580, `sf_1700` = 1731, `random` =
  525, `greedy` = 641, `step_20000` = 719 (CIs ~±80-125). Gate met.
- Step 6 — done (Oct 17): move to RunPod + the single results file. Gate met: `results.csv`
  has 10 rows (pin, 5 references, 4 SFT checkpoints on an L4 pod). Elo (CI), 60 games each:
  5k = 655 (621–689), 10k = 689 (655–722), 15k = 706 (664–748), 25k = 681 (638–714);
  train GPU-h 1.54 / 3.08 / 4.61 / 7.69; ~0.09 s/move; illegal rate 0.
  - Oct 17: Network volume `chess-ai` (EU-RO-1, 30 GB) mounted at `/workspace`; budget-cap watchdog in `docs/runpod_runbook.md`.
  - Focus (Oct 17, plan v4 update): structured + masked only. SFT vs. self-play RL, then LLM +
    MCTS (planned after AlphaZero). The 2x2 sweep (Step 8) is deferred; `max_gpu_hours_per_arm`
    is decided later.
- Step 7 — code written Oct 17 (branch `step-7-llm-rl`): `search/selfplay.py` (pluggable
  move picker, one game at a time), `train/llm_rl.py` (REINFORCE, value-head baseline, KL to
  the SFT model, resumable, `rl_log.csv`), `configs/llm_rl.yaml` (provisional defaults),
  `04_train_llm_rl.ipynb`. Warm-starts from SFT step 25,000. Step 6 merged in.
  - Run 1 (Oct 18, defaults): 20 rounds, 10.47 RL GPU-h (18.16 total), stopped by the cap.
    Healthy (entropy ~1.9, KL ~0.04) but flat: r5 647, r10 714, r15 681, r20 689 vs. SFT 25k
    681 (H0). ~67% draws and a value head at chance, so little signal. Gate met (stable, flat).
  - Run 2 (Oct 18, plan Step 7): Stockfish scores non-checkmate endings (repetition, 50-move,
    cap), nothing else changed; `tag: sfend`. Added: run-name `tag`, game-ending counts in
    `rl_log.csv`, `eval/plots.py` (Elo vs. compute, saved by notebooks 02 and 04).
  - Run 2 first rounds (Oct 20): draws are mostly insufficient material and stalemate (not
    re-scored), so only ~5% of games get a new reward. Result: 20 rounds, 18.09 total GPU-h;
    Elo r5 689, r10 655, r15 689, r20 664 vs. SFT 681: flat (H0), like run 1.
  - Run 3 (Oct 20, plan Step 7): curriculum, rounds 1-10 at a 120-ply cap scored by Stockfish,
    then 300 plies with run 2's scoring; `tag: curric120`. Launched Oct 20 after run 2 (don't
    `git pull` on the pod while a run is going). Rounds 1-12 (plan Step 7): far more games
    re-scored, value loss below ln 2 for the first time, checkmates up at round 12. Evals after
    training. Stockfish-as-critic was rejected (too SFT-like).
  - Branch per run (Derek, Oct 20): each RL run's code lives on its own branch so its results
    can be reproduced: `step-7-llm-rl` = run 2 (`0091940`), `step-7-run3-curric` = run 3.
    Older run folders still resume with newer code (`ADDED_SETTINGS` in `train/llm_rl.py`).
  - Merge (Oct 21, Derek's call): Step 7 merges into `main` before run 3's evals (gate met by
    runs 1-2). Run code tagged `run1-default` (`608e187`), `run2-sfend` (`0091940`),
    `run3-curric` (`04e245a`). No stop rules for now; reruns decided per run.
- Step 9 — next (Oct 21): AlphaZero on `step-9-alphazero`, cut from `main` after the Step 7
  merge. The full implementation spec is plan Step 9 (a)-(h); build in that order. First
  commit: `results.csv` saves merge under a lock file instead of overwriting (two pods can
  share the volume); (a) done Oct 21. Batched self-play lives in `search/mcts.py` (Oct 30,
  Derek), so `search/selfplay.py` is untouched; (b) MCTS done Oct 30. Cap staged 9 / 18 /
  30 GPU-h, extended by resuming. Hyperparameters are provisional until Derek confirms them
  after the smoke run (plan Sec 5 item 4). Can train on a second pod; each watchdog then uses
  the combined rate.

## How code runs (one source of truth per thing)

1. **Local (Windows, VS Code)** — where code is written. Repo `github.com/derekhe99/Chess_Bot`.
2. **RunPod** (Step 6 onward; Steps 0–5 ran on Colab) — where it executes. On-demand L4
   pod with a network volume at `/workspace`. Repo cloned once to `/workspace/Chess_Bot`,
   `git pull` each session; code is never edited on the pod. Long jobs run headless in
   tmux with a budget-cap watchdog (a background timer that runs
   `runpodctl stop pod $RUNPOD_POD_ID`; `docs/runpod_runbook.md`, plan Sec 1).
   Always attach the network volume to the pod. `HF_TOKEN` is a pod environment variable.
3. **Artifacts** — `artifact_root` in `configs/base.yaml` (`/workspace/chess-ai`):
   `data/processed`, `checkpoints/{sft,llm_rl,alphazero}`, `results/results.csv`.
   Artifacts never go to git (`.gitignore` covers `results/`, checkpoints, `*.pt`,
   `*.parquet`, `stockfish/`). `results.csv` is the one file every chart reads; only
   `eval.anchors.ResultsTable` writes it.

- **Branches**: `main` = the last Step whose gate passed. Each Step is built on its own
  branch (`step-N-name`) cut from `main` and merged after its gate passes. Notebooks have
  a `BRANCH` variable in the setup cell.
- LLM checkpoints = LoRA adapter + value head + optimizer state; never the frozen base
  model (re-download it from Hugging Face instead).
- Model loading: `Qwen/Qwen3-0.6B` with `enable_thinking=False`, passed to
  `tokenizer.apply_chat_template` (not to `generate`).

## Commands

- Tests: `python -m pytest tests/ -v` from the repo root. Stockfish tests skip (not fail)
  without the binary.
- Python: keep code 3.10+ compatible.

## Known gotchas

- **Line endings**: Windows checkout uses CRLF; Linux tools may show every file as
  modified. Check real changes with `git diff --ignore-cr-at-eol`.
- **Stockfish is pinned to 19** (`SF_VERSION = "sf_19"` in `engine/stockfish.py`;
  `ensure_stockfish()` asserts the version). Every rating in `results.csv` depends on the
  exact engine; never upgrade it without discussing it first.
- `torchao>=0.16.0` is pinned because Colab's preinstalled version broke peft's LoRA
  dispatch; harmless elsewhere.
