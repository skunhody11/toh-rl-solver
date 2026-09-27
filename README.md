# Tabular Reinforcement Learning on a Towers-of-Hanoi MDP

Implementation of **Value Iteration** and **Q-Learning** for a stochastic,
reward-shaped Towers-of-Hanoi environment, formulated as a Markov Decision
Process. Built as part of an AI coursework project; the code in this repo
is my own solver logic, written against a provided MDP environment
interface.

## What's here

`solver_utils.py` contains the core algorithms:

- **`value_iteration`** — Bellman-backup value iteration. For every
  non-terminal state, computes Q-values for all actions via
  `Q(s,a) = Σ_s' T(s,a,s') [R(s,a,s') + γ·V(s')]`, then updates `V(s)` to
  the max Q-value, tracking the largest change (`max_delta`) for
  convergence checking.
- **`q_update`** — the tabular Q-learning TD update:
  `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]`.
- **`choose_next_action`** — epsilon-greedy action selection among
  current best actions.
- **`custom_epsilon`** / **`custom_alpha`** — decay schedules for the
  exploration rate and learning rate. Epsilon decays from ~1.0 toward a
  floor of 0.01 (satisfying GLIE — greedy in the limit with infinite
  exploration); alpha decays from 1.0 toward 0 so early noisy estimates
  get overwritten quickly while later updates refine convergence.
- **`exploration_fn`** / **`choose_next_action_with_exploration_fn`** —
  a UCB-style exploration bonus, `u + c/√(n+1)`, that prioritizes
  under-visited state-action pairs instead of relying on randomness.

## Environment (not included)

The MDP itself — states as disk configurations across three pegs,
stochastic transitions with configurable noise, and rewards for reaching
the goal — was provided as course infrastructure and isn't reproduced
here, since it isn't my own work to redistribute. The functions above
are written against that environment's interface (`TohMdp`, `TohState`,
`VTable`, `QTable`, etc.) and expect the same contract described in the
docstrings and type hints.

## Notes on results

Across three exploration strategies (constant ε=0.2, a decaying custom
ε schedule, and the UCB-style exploration function) trained for 10,000
steps on a 3-disk, 20%-noise version of the MDP, the exploration-function
approach converged to the highest Q-values along the solution path,
consistent with its bias toward resolving uncertainty in
under-visited states rather than exploring at random. All three
strategies converged to similar policies given enough steps, differing
mainly in how efficiently they got there.
