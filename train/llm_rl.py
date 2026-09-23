"""LLM self-play RL: play a round of games, assign the final win/loss/draw
result as the reward to every move in that game, update with a
policy-gradient loss (baseline + KL penalty toward the starting model).

Can start from the base model (pure RL) or an SFT checkpoint (warm-start).

TODO: implement (Step 7).
"""
