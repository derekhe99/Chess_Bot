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
- Step 3 (`data/`, `tests/test_data.py`, `notebooks/01_build_dataset.ipynb`) — done;
  gate passed in Colab (Sep 28) on branch `step-3-data`. 50,000 positions from Lichess
  `2013-06` (`configs/data.yaml`, checked against Lichess's sha256), exact phase-mix quota
  hit (opening 16665 / middlegame 16670 / endgame 16665), no repeated positions, every
  label move legal. Stockfish CPU time: 59.5 min (~0.071 s/position at depth 12) --
  charged to the SFT budget (plan v3, Sec 0). Train/validation split by game: 44982 / 5018
  positions, no game on both sides. `dataset.py` yields plain records (the prompt is
  Step 4's).
- Step 4 (`model/llm_policy.py`, `tests/test_llm_policy.py`) — done; gate passed in Colab
  (Sep 28): untrained Qwen3-0.6B finished a game vs random in all 4 toggle combinations.
  Settings in `configs/sft.yaml` `policy:`. Implementation choices: the prompt is the chat
  template (thinking off) around "instruction / board text / Move:"; a move's score is
  log P(move text + end-of-reply), with the prompt run once and its keys/values reused
  for every legal move; unmasked = greedy first try, sampled retries (the harness forfeits
  on the third illegal attempt); value head = one zero-initialized linear layer on the
  last prompt token. Offline tests (tiny random Qwen3) pass; the gate is opt-in:
  `RUN_MODEL_TESTS=1 python -m pytest tests/test_llm_policy.py -v -s -k gate` on a GPU
  runtime.
- Step 5 (`train/sft.py`, `tests/test_sft.py`, `utils/logging.py`, `tests/test_logging.py`,
  `notebooks/02_train_sft.ipynb`) — in progress on branch `step-5-sft`. Structured text +
  masking on (plan corrected Sep 28). Implementation choices: move loss = -log P(move text +
  end-of-reply) summed over the move's tokens (exactly play's score; a test checks they're
  equal); logits are computed only at the move positions (memory); resume from `resume.pt`
  reproduces an unbroken run exactly (data order seeded per epoch, dropout per step);
  checkpoint/resume code lives in `sft.py` for now (`utils/checkpoint.py` waits for the
  config-design alignment point). Labeling CPU: both the whole dataset's and this run's share
  are logged; which one is charged is Step 6's call. Part 1 (training + a rough screen of every
  checkpoint vs random / greedy / sf_1320) is built; part 2 (freeze ~3 checkpoint rungs) comes
  after seeing where the checkpoints land.
  - **Day-1 run (Sep 28), logged:** 5,000 train positions, 10 epochs, checkpoint every 500
    steps. Finding: real move-selection learning (val_top1 3% -> ~21%, value loss 0.69 -> 0.59)
    but no checkpoint ever scored a point off Stockfish 1320 in the screen (0W-0D every
    checkpoint, 10 games each) -- the reference set doesn't connect yet, so the Step 5 gate
    isn't met. `val_move_loss` degraded after ~epoch 5 while `val_top1` kept climbing:
    overfitting from repeating the same 5,000 positions 10 times, not a ranking problem.
    Treated as Day-1 done and logged as a finding, not a failure.
  - **Field rename:** the per-run counter was called `positions_seen` but is cumulative across
    epochs (`train_positions x epochs`, i.e. examples processed with repeats), not the count of
    distinct positions -- misleading, since the Day-1 run's log showed "50000" despite training
    on only 5,000 unique positions. Renamed to `examples_seen` everywhere (`train/sft.py`,
    `tests/test_sft.py`, `utils/logging.py`'s CSV schema).
  - **Next run (Sep 28), in `configs/sft.yaml`:** trains on the full labeled train split
    instead of a slice (`train_positions: null`, 44,982 positions) to fix the overfitting with
    more unique data rather than fewer epochs on the same 5,000; `checkpoint_every` raised to
    5,000 (was 500) to keep the checkpoint count sane at 10x the steps.
  - **New: `utils/logging.py`** -- `log_experiment()` reads a run's own `run_config.json` /
    `val_log.csv` / checkpoints folder and appends one summary row to the shared
    `results/experiments.csv` (created on first use); `read_experiments()` reads it back
    newest-first. This is what Step 11's Elo-vs-samples curves are built from. The notebook now
    has a one-time backfill cell (logs the Day-1 run above) and a final cell that logs each
    live run, leaving only the one-line `finding` for Derek to write. Deliberately stopping a
    run early (it never returns from `train()`) used to break this, since `summary.json` is
    only written on return; fixed by reading `examples_seen` off `val_log.csv` and the
    checkpoint count off the checkpoints folder directly instead, neither of which needs the
    run to have finished.
  - **Day-2 run (Sep 29-30), stopped early at step 25,000** of a 28,110-step budget (full
    44,982-position train split, not high value to run to completion once the checkpoints
    needed were already saved). Logged via the fix above.
  - **New: `games_from_counts()`** (`eval/elo.py`) -- rebuilds a matchup's individual game
    results from aggregate win/draw/loss counts (e.g. copied off a `rate_adaptively` batch
    log after an interrupted run). The rating it produces is exactly identical to fitting the
    real per-game records (the Elo fit only ever uses a player's total score and opponent
    list, never color or order); the bootstrap CI is valid but not bit-for-bit identical,
    since `fit_with_ci` resamples by list position and the reconstruction necessarily orders
    games differently than they were actually played. Tested in `tests/test_eval.py`.
  - **Checkpoint rungs frozen (Sep 30), completing the reference set** -- `random`, `greedy`,
    `step_20000` added to `results/reference_set.json` (`sf_1320` pinned, `sf_1500`/`sf_1700`
    already frozen in Step 2): `random` = 525 [384, 624], `greedy` = 641 [495, 724],
    `step_20000` = 719 [581, 799] (200 games each, `sf_1320` pinned at 1320; exact numbers
    depend on the live `n_boot` seed). **Reconstructed, not from a completed live run:** the
    `rate_adaptively` cell was interrupted (`KeyboardInterrupt`) during batch 2; frozen instead
    from `games_from_counts()` on the pasted per-batch counts (batch 0 + batch 1 complete, all
    10 matchups; batch 2 partial, 5 of 10). Two known departures from the plan's Sec 4 freeze
    rule, both accepted by Derek as provisional (Sep 30), to revisit if there's a spare Colab
    window, not blocking further work:
    - Every CI here is wider (~110-125 Elo half-width) than the plan's +/-100 target -- fewer
      games than a full adaptive run to convergence.
    - `step_15000` (693) was dropped as a separate rung -- too close to `step_20000` to be
      worth its own member -- leaving a single ~600-Elo gap between `step_20000` (719) and
      `sf_1320` (1320), well past the plan's own tentative "no gap >~300 Elo between
      neighbors" selection heuristic. `step_15000`'s games were kept in the joint fit (dropping
      them entirely nearly doubles the CI half-width), just not frozen as its own rung. Expected
      to close as later SFT/RL checkpoints get frozen in.
  - **Source of truth for the reference set (revised Oct 3, 2026): one flat CSV,
    `results/reference_set.csv` on Drive** (`eval.anchors.FrozenRegistry`) -- Derek's call,
    replacing the earlier JSON. One row per member, pin included (`sf_1320`'s row is the pin
    itself -- ci_low = ci_high = 1320.0, no interval, since it's fixed by definition, not
    measured); `spec`/`notes` are JSON-encoded into their own cell so nothing is lost (a
    checkpoint rung's weights hash, representation, masked toggle all round-trip) while the
    file stays one flat CSV throughout. Members are only ever added, never re-rated. This is
    the only thing the evaluation code actually reads (`eval.match.evaluate_agent(new_model,
    opponents_built_from_the_registry, fixed_ratings(...), n_games)` fits only the new
    model's rating, holding every reference-set member fixed) -- not a convention, a
    functional dependency: if this file doesn't have a member, nothing can be rated against
    it. `FrozenRegistry(path)` is the only way anything reads or writes it; nothing else
    should parse this CSV by hand.
  - **New: `eval.elo.append_rating_log`** -- a model being evaluated (an SFT/RL/AZ
    checkpoint, not a frozen anchor) isn't in the registry above; it's a running history
    instead, `results/model_ratings.csv` on Drive, one row per evaluation (re-evaluating the
    same model later adds a new row, never overwrites). This is what Step 11's
    Elo-vs-samples / Elo-vs-compute curves actually read (joined to `results/experiments.csv`
    on `run`/`step` for the training-side columns).
  - **Both CSVs live on Drive only** (`DRIVE_ROOT/results/`), per the existing artifacts-stay-
    on-Drive convention ("How code runs" below) -- not in git. A `results/reference_set.csv`
    briefly existed in the repo by mistake (Sep 30-Oct 1); `.gitignore` now excludes
    `results/` so that can't recur.
  - **Gate revised (Oct 1, 2026):** dropped "beats the untrained model" from the Step 5 gate
    (plan Sec 4) -- Derek's call. What Step 7 actually needs from Step 5 is a reference set
    with CIs to rate RL checkpoints against; a beat-the-untrained-model check is a useful
    sanity signal on the SFT run, not a gate on the reference set's usability.
  - **Not yet pushed to GitHub:** everything above from Sep 30 on (`eval/elo.py`,
    `tests/test_eval.py`, `notebooks/02_train_sft.ipynb`, this file, the plan doc) exists only
    in Derek's local working tree -- `git status` shows them modified, unstaged. Since Colab
    clones fresh from GitHub each session, **the live `results/reference_set.json` on Drive
    still only has Step 2's `sf_1500`/`sf_1700`** -- the freeze code was never actually run
    there. Commit + push, then rerun the notebook's freeze cells in Colab, to actually update
    Drive's copy.
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
