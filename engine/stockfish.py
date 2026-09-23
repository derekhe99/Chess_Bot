"""Starts and manages Stockfish processes.

Two jobs: (1) labeling positions (best move + evaluation at a set depth or
node count), (2) playing at limited strength for eval. Records the CPU
time it uses so labeling cost can be charged to the compute budget.

TODO: implement (Step 1).
"""
