# Chess_Bot

_A little research project on different AI (especially transformer vs CNN) architectures and training techniques for chess._

Research/engineering project comparing a fine-tuned pretrained LLM against an
AlphaZero-style CNN+MCTS agent for chess, evaluated against fixed Stockfish
and weak anchors (never agent-vs-agent).

Spec: `docs/implementation_plan_v4.md` (the what and how), `docs/framing.md` (the why),
`docs/jobs_to_be_done.md` (what each step delivers). Status: see `CLAUDE.md`.

## Model
`Qwen/Qwen3-0.6B` (Apache-2.0), non-thinking (`enable_thinking=False`).

## Structure
- `engine/` -- chess rules, move vocabulary, board encodings, Stockfish wrapper
- `data/` -- SFT data pipeline (download, sample, annotate, dataset)
- `model/` -- LLM policy wrapper, AlphaZero CNN
- `search/` -- MCTS, self-play
- `train/` -- SFT, LLM self-play RL, AlphaZero training loops
- `eval/` -- reference-set Elo harness + `results/results.csv` (the one results file)
- `notebooks/` -- one thin driver notebook per pipeline stage
- `configs/` -- YAML configs (paths/costs, data, SFT, LLM RL, AlphaZero, eval)
- `tests/` -- fast checks run before any long run

## Setup
RunPod (Step 6 on): on-demand L4 pod + network volume at `/workspace`; clone this repo to
`/workspace/Chess_Bot`, set `HF_TOKEN`, run `notebooks/00_setup.ipynb`. Long jobs:
`jupyter nbconvert --to notebook --execute notebooks/XX.ipynb --output-dir /workspace/chess-ai/logs --ExecutePreprocessor.timeout=-1` inside tmux.
