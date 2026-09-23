"""Canonical move vocabulary (all ~1.9k possible UCI moves): move<->index
mapping, plus a function returning a legal-move mask for a position.

Shared by the AlphaZero policy head, MCTS, and the LLM's masked decoding,
so all agents speak the same move language.

TODO: implement (Step 1).
"""
