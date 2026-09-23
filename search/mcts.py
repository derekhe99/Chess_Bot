"""PUCT Monte Carlo tree search that works with any model exposing
(policy, value) = evaluate(positions). Batches leaf evaluations into
single GPU calls.

Written model-agnostic so an optional LLM + MCTS arm can reuse it later.

TODO: implement (Step 9; shared interface with model/llm_policy.py is a
plan Sec 5 "next alignment point" -- not yet locked).
"""
