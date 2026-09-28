# CLAUDE.md — Chess_Bot

## Source docs: the why and the what

Both are imported so they're in context every session.

**Framing** (the *why*: research question, arms, metrics, and the review notes behind
each decision):

@docs/framing.md

**Implementation plan** (the *what* and *how*: the spec for scope, design, and
execution order):

@docs/implementation_plan_v3.md

Precedence: the plan beats the framing (it was written later and incorporates the
review notes). This file beats both for implementation details and for decisions made
after the plan was written.

Rules:

- **Before any change**, check it against the plan: which Step it belongs to, whether it
  respects the locked decisions (plan Sec 0), and whether that Step's prerequisites and
  gates are met (plan Sec 4). Don't build ahead of the current Step without asking. Also
  check it serves the framing: which question or metric it supports.
- **Deviations and open items are discussed first.** Any change that departs from the
  plan, adds scope, skips a gate, or settles an open item ("Still open" in Sec 0,
  "Next alignment points" in Sec 5): stop, flag it to Derek with options and trade-offs,
  and align with him. Don't decide unilaterally.
- **Order after aligning**: update the current plan, `docs/implementation_plan_v3.md`
  (and the Status section below), first, then write the code. Small updates edit the
  current version in place; a new version (v4, ...) only when Derek says so, with the old
  one archived outside the repo in `ChessAI/archive/`.
- Cowork mirrors the plan and framing in its Project docs. After editing either, remind
  Derek so the Cowork copy gets re-synced. From v2 on, `docs/framing.md` is the canonical
  framing and is edited directly (the original Word doc is no longer maintained); framing
  changes follow the same discuss, align, update order.

## Working with Derek

- Research project, and Derek is learning as he goes: explain the *why* from first
  principles, keep it concise, no filler.
- **Derek runs git himself.** Suggest exact commands (`git commit -m "..."`); don't
  commit, push, or rewrite history unless he explicitly asks.

## Status

- Step 0 (setup, `notebooks/00_setup.ipynb`) — done; gate passed in Colab.
- Step 1 (`engine/`, `tests/test_engine.py`) — done; 34 tests pass in Colab.
- Step 2 (`eval/`, `tests/test_eval.py`, `notebooks/03_evaluate.ipynb`) — done; gate
  passed in Colab (Sep 27). Frozen in Drive `results/reference_set.json`:
  sf_1500 = 1580 [1500, 1673], sf_1700 = 1731 [1654, 1834] (160 games each, sf_1320
  pinned at 1320, 0.1 s/move). Both land above nominal -- a finding, not an error.
  Adaptive game counts stop on the 95% CI alone (plan v3, Step 5).
- Next: Step 3 (`data/`, `notebooks/01_build_dataset.ipynb`) on branch `step-3-data`.
- Not wired up yet: `configs/base.yaml: drive_root` is unused; the setup notebook
  hardcodes `DRIVE_ROOT`. Resolve as part of the config-design alignment point.

## How code runs (one source of truth per thing)

1. **Local (Windows, VS Code)** — where code is written. Repo `github.com/derekhe99/Chess_Bot`, branch `main`.
2. **Colab** — where it executes. Each session opens a notebook from GitHub and clones
   the repo fresh to `/content/Chess_Bot`. Code is never edited in Colab or kept on Drive.
3. **Google Drive** — artifacts only, `DRIVE_ROOT = /content/drive/MyDrive/chess-ai`:
   `data/raw`, `data/processed`, `checkpoints/{sft,llm_rl,alphazero}`, `logs`, `results`.
   Artifacts never go to git (`.gitignore` covers checkpoints, `*.pt`, `*.parquet`, `stockfish/`).

- **Branches**: `main` = the last Step whose gate passed. Each Step is built on its own
  branch (`step-N-name`) cut from `main`, tested in Colab from that branch (set `BRANCH`
  in the notebook's setup cell), and merged to `main` only after its gate passes.

- LLM checkpoints = LoRA adapter + value head + optimizer state; never the frozen base
  model (re-download it from Hugging Face instead).
- Model loading: `Qwen/Qwen3-0.6B` with `enable_thinking=False`, passed to
  `tokenizer.apply_chat_template` (not to `generate`).

## Commands

- Tests: `python -m pytest tests/ -v` from the repo root. Stockfish tests skip (not fail)
  without the binary — run the setup notebook's Stockfish cell first in Colab.
- Python: Colab is 3.13; keep code 3.10+ compatible.

## Known gotchas

- **Line endings**: Windows checkout uses CRLF; Linux tools may show every file as
  modified. Check real changes with `git diff --ignore-cr-at-eol`.
- **Colab**: `%cd` persists across cells (hence the absolute `REPO_DIR` clone guard);
  "Restart session" does NOT wipe `/content` ("Disconnect and delete runtime" does);
  a missing `nvidia-smi` means the runtime isn't set to GPU.
- **Stockfish is pinned to 19** (`SF_VERSION = "sf_19"` in `engine/stockfish.py`; `ensure_stockfish()` asserts
  the version). Labels and eval anchors depend on the exact engine; never upgrade it
  without discussing it first.
