# Chess AI — Framing (research intent)

Converted from `Framing_Docs_updated.docx` (Sep 23, 2026). This is Derek's framing of
*why* the project exists and what it should teach; the implementation plan
(`docs/implementation_plan_v2.md`) is *what* gets built and *how*. The `> Review note`
lines are comments from the Sep 23 review, kept because they explain decisions the plan
later locked in. **Where this doc and the plan differ, the plan wins** — it was written
afterwards and incorporates these notes.

The Word document is the original; if it changes, re-convert this file.

---

**Q: What is the core question I am trying to answer?**

**A:** I want to understand how different model architectures and different training
regimes and different scaffolds impacts the ability for a model to "hill-climb" elo in
chess, which has its own set of unique constraints. We are assuming a fixed budget.

More concretely, the learning outcomes are:

## Research

- **Different base model architectures (i.e., CNN-based vs LLM-based):** AlphaZero uses a
  CNN-based architecture to evaluate 2 outputs (next best move, likelihood of winning)
  combined with a Monte Carlo tree search algorithm on top of it to search for candidate
  moves, whereas an LLM uses a transformer-based architecture to evaluate 2 outputs (next
  best move, likelihood of winning) which is all done in a single forward-pass through the
  transformer.
  - I wonder if I add Monte Carlo tree search on LLM how that would impact performance?
    > Review note: Good instinct, but this isn't in the plan's 2x2 LLM design or the
    > hypotheses — needs to be added as an explicit arm if you want data on it, otherwise
    > it's just an idea worth flagging, not yet a scoped experiment.
    > *(Plan outcome: listed as the optional LLM + MCTS stretch arm.)*

- **How scaffolding an LLM makes a big difference:**
  - Guarantee legal moves (hard masking / constrained decoding, not just injecting the
    legal-move list) vs not guarantee
    > Review note: Worth distinguishing 'guaranteed' (constrained decoding/masking —
    > actually restricts output) from 'legal moves listed in context' (a hint the model can
    > still ignore). These test different things; the plan treats them as separate arms.
  - Visualize the board and piece positions in different ways — FEN vs structured board
    format
    > Review note: This only resolves once you pick pretrained vs. from-scratch (see your
    > 'pretrained Open Model' choice below — good call). With a pretrained text LLM, richer
    > visualization realistically means structured text (piece-per-square tokens), not
    > true image input.

- **How different training techniques for LLMs can look (SFT + RL is a common theme in
  modern LLM):**
  - Self-play (sparse rewards)
  - Supervised fine tuning (distilling from an "oracle")
  - SFT + Self-play (second boost from self-play potentially)

- **Evaluating 4 critical metrics:**
  > Review note: You're missing sample efficiency (data/games needed to reach a given
  > Elo) — that's the 4th metric, and the one SFT vs. pure self-play is expected to differ
  > on most.
  - Sample efficiency - # of games/positions needed to reach a given Elo
  - ELO - use a "global" source of truth being Stockfish elo
    > Review note: Gap: Stockfish's weakest setting is still ~1300+ Elo. Under-trained
    > early agents will likely lose 100% of games against it — no signal, no confidence
    > interval. Add fixed anchors below that floor: random mover, material-greedy player,
    > Stockfish capped at depth 1.
    - Also add weak fixed anchors below Stockfish's floor (~1300+ elo): Stockfish capped
      at depth 1
  - Training cost - I am not sure how to factor in the MCTS compute cost here? Framework
    so far
    > Review note: Two separate leaks here: (1) Stockfish-labeling compute for
    > SFT/warm-start should count against their budget too, or SFT looks artificially
    > cheap vs pure self-play. (2) MCTS at inference is real per-move compute a single LLM
    > forward pass doesn't spend — fix by capping inference compute per move (time or node
    > budget) in the eval, not just logging training cost.
    - One metric: measured by hours of GPUs during training (SFT + RL both) and inference
    - Another metric (compare against LLMs): # of tokens consumed during training and
      inference
    - Another metric (compare LLMs to alphazero): # of computations performed
    - Stockfish-labeling compute (for SFT/warm-start data) counts against that regime's
      budget too, or it looks artificially cheap
    - Cap inference-time compute per move (time or node budget) in the eval harness, so
      AlphaZero's MCTS and the LLM's single forward pass are comparable
  - Inference time - how quickly to produce the output move

## Engineering

- **Implementation architecture** - what are the core parts of the architecture on a high
  level
  - "The chess backbone" - i.e., the board, game rules, game states, win conditions
  - Model layer
    - Alphazero
      - CNN base layer
      - MCTS implementation
    - LLM
      - Transformer base layer - utilize a pretrained Open Model (selected for cost and
        size constraints)
        > Review note: Good — this resolves the single biggest open question (pretrained
        > vs. from-scratch). Right call for the stated research question. Everything
        > downstream (board repr, masking difficulty, overlap with AlphaZero) follows from
        > this choice, so worth stating it explicitly as a load-bearing assumption.
      - MCTS implementation maybe?
  - **Training layer**
    - Implement functions for allowing forward pass and backward pass each base model (CNN
      and transformer)
      - Inputs: board representation
        - CNN: XX
        - Transformer: FEN vs other one
      - Outputs (for both):
        - Next move prediction
        - Likelihood of winning
    - **Implement training rounds**
      > Review note: Right instinct — self-play means the same network plays both colors
      > against itself, no separate mover needed. No need to bring in Stockfish here; that
      > would turn this into the SFT/warm-start regime, not fix self-play.
      - AlphaZero:
        - Forward pass: (generate this number of unique data points which essentially
          consists of a tuple: (board position, next move prediction, prediction of
          likelihood of winning)
          - Round = a batch of self-play games (e.g. 50 games/round); 50 rounds total
            > Review note: Careful with terminology — elsewhere 'round' means a batch of
            > many games (e.g. 50 games/round), not one game. Worth standardizing this
            > before scaffolding, since '50 rounds' means ~50x more games under one
            > definition than the other.
          - Game = ~50 moves
          - Move = expanded via N MCTS simulations (e.g. ~10-15 sims/move)
        - Backward pass: (calculate loss and update the weights to specifically improve
          the model outputs i.e., next move prediction, prediction of likelihood of
          winning)
      - LLM (assumes legal moves + "correct" board representation but WLOG) - self-play
        only
        - Forward pass:
          - Sequential game play
          - Receives reward signal at the end of the game only based on win-loss
          - The terminal win/loss/draw outcome is backpropagated as the reward for every
            move made in that game.
        - Backward pass
          - Update the transformer based on the reward
      - LLM - SFT
        - Forward pass:
          - Total position count: e.g. tens of thousands of Stockfish-annotated positions
            (not rounds/games - SFT trains on a static dataset)
            > Review note: SFT doesn't naturally need 'rounds' of self-played games at
            > training time — it's supervised learning over a static set of
            > Stockfish-annotated positions, no game sequencing required. Games/rounds only
            > re-enter at eval time.
          - Source: sampled from real game archives (e.g. Lichess/PGN databases)
          - Labeling: each position annotated by Stockfish with its best move + eval score
          - Distribution: sample across openings/midgame/endgame so no single phase
            dominates
        - Backward pass
          - Stockfish is an oracle that says next move and outputs likelihood of winning
      - LLM - SFT + RL
        - Combine the prev two
  - **Evaluation harness**
    > Review note: Two adds: (1) run enough games per matchup for real confidence
    > intervals (hundreds, not tens) and 2-3 seeds per RL arm, or RL variance swallows the
    > differences you're trying to detect; (2) eval against the weak anchors above too, not
    > just Stockfish. Also: get the SFT-on-Stockfish path working end-to-end first —
    > fastest way to validate the whole stack before sinking time into parallel MCTS.
    - get the LLM to run against various stockfish levels to determine an ELO and also use
      logs to track things like training cost and inference time
    - Run 2-3 seeds per RL arm; start with a small number of eval games per matchup and
      scale up later, rather than hundreds all at once
    - Sequencing: build the SFT-on-Stockfish baseline end-to-end first (data pipeline,
      training, eval producing an Elo number) to validate the whole stack before parallel
      MCTS/self-play
