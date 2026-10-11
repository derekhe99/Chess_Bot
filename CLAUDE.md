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
- Step 6 — in progress: move to RunPod + the single results file.
  - Code done (Oct 10, plan v4): `eval.anchors.ResultsTable` over
    `results/results.csv` (role = pin / reference / model) replaces `FrozenRegistry`,
    `reference_set.csv`, `model_ratings.csv`, `experiments.csv`;
    `evaluate_agent(..., batch_games=10, on_batch=...)` saves after every batch; cut
    `utils/` entirely, `games_from_counts`, `append_rating_log`, `screen_checkpoints`, the
    provisional/frozen status. Notebook setup cells read `artifact_root` from
    `configs/base.yaml` (no Drive mount).
  - `results/results.csv` (gitignored) was built locally from the old `reference_set.csv`
    (pin + 5 reference rows; `step_20000`'s spec has run/step but no weights hash -- the
    local CSV never had one). Upload it to `/workspace/chess-ai/results/`.
    `model_ratings.csv` / `experiments.csv` on Drive are retired, not migrated.
  - Next: RunPod volume + pod, migrate artifacts from Drive, pin hourly rates in
    `base.yaml`, evaluate 3–4 SFT checkpoints into `results.csv` on the pod (Step 6 gate).

## How code runs (one source of truth per thing)

1. **Local (Windows, VS Code)** — where code is written. Repo `github.com/derekhe99/Chess_Bot`.
2. **RunPod** (Step 6 onward; Steps 0–5 ran on Colab) — where it executes. On-demand L4
   pod with a network volume at `/workspace`. Repo cloned once to `/workspace/Chess_Bot`;
   each session `git fetch && git checkout <branch> && git reset --hard origin/<branch>`
   (code is never edited on the pod). Long jobs run headless in tmux:
   `jupyter nbconvert --to notebook --execute notebooks/XX.ipynb --output-dir /workspace/chess-ai/logs`
   (executed copy goes to the volume, repo stays clean); end them with
   `runpodctl stop pod $RUNPOD_POD_ID`. `HF_TOKEN` is a pod environment variable.
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
