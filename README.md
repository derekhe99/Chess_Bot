# Chess_Bot

Research/engineering project comparing a fine-tuned pretrained LLM against an
AlphaZero-style CNN+MCTS agent for chess, evaluated against fixed Stockfish
and weak anchors (never agent-vs-agent).

Full plan: `chess_ai_implementation_plan.md` (Level 1/2) in the parent
ChessAI folder, plus `Opus_Feedback_v1_9_22_26.md` for the structural review
that shaped it and `LLM Chess Bot -- Research & Engineering Project Plan.md`
for the research framing.

## Status
Step 0 -- Setup. Scaffold created. Still open: GitHub remote for this repo,
Colab Pro (not needed until Step 6), a Hugging Face token, and the exact
model checkpoint. See `notebooks/00_setup.ipynb`.

## Structure
- `engine/` -- chess rules, move vocabulary, board encodings, Stockfish wrapper
- `data/` -- SFT data pipeline (download, sample, annotate, dataset)
- `model/` -- LLM policy wrapper, AlphaZero CNN
- `search/` -- MCTS, self-play
- `train/` -- SFT, LLM self-play RL, AlphaZero training loops
- `eval/` -- fixed-anchor Elo harness
- `utils/` -- config loading, compute budget meter, logging, checkpointing
- `notebooks/` -- one thin driver notebook per pipeline stage
- `configs/` -- YAML configs (paths/seed/budget, data, SFT, LLM RL, AlphaZero, eval)
- `tests/` -- fast checks run before any long run

## Setup
Run `notebooks/00_setup.ipynb` top to bottom on a fresh Colab runtime.
Gate: it should bring up a working environment from scratch -- GPU detected,
Drive mounted, Stockfish runs, HF model loads and generates.
