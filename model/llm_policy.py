"""Loads the pretrained model + tokenizer, attaches LoRA adapters and a
small value head (predicts win likelihood from the final hidden state),
and builds prompts.

Implements move selection both ways: masked (score only legal moves, pick
from them) and unmasked (free generation, then parse; illegal output is
logged and handled per config).

TODO: implement (Step 4).
"""
