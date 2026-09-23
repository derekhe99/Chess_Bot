"""Supervised fine-tuning on Stockfish labels: cross-entropy on the move
plus a value loss on the win probability. Logs positions seen, GPU time,
tokens; saves checkpoints for eval curves.

TODO: implement (Step 5 -- Day-1 milestone).
"""
