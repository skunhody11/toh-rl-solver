#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Value Iteration and Q-Learning solver logic for a Towers-of-Hanoi MDP.

Implements the core update rules from Sutton & Barto's tabular RL methods:
Bellman backups for Value Iteration, TD-style Q-updates for Q-Learning,
epsilon-greedy and UCB-style exploration strategies, and custom decay
schedules for the learning rate (alpha) and exploration rate (epsilon).

Written as coursework built on a provided MDP framework (`toh_mdp.py`,
not included here); this file contains only my own implementation.
"""
import math
import random
from typing import Callable, Dict, List, Tuple

import toh_mdp as tm


def get_actions(mdp: tm.TohMdp, state: tm.TohState) -> List[tm.TohAction]:
    if state == mdp.terminal:
        return []
    if mdp.is_goal(state):
        return ['Exit']
    return [op.name for op in mdp.operators if op.pre_condition(state)]


def value_iteration(
        mdp: tm.TohMdp, v_table: tm.VTable
) -> Tuple[tm.VTable, tm.QTable, float]:
    new_v_table: tm.VTable = v_table.copy()
    q_table: tm.QTable = {}
    max_delta = 0.0
    for state in mdp.nonterminal_states:
        for action in mdp.actions:
            q_table[(state, action)] = sum(
                mdp.transition(state, action, next_state) * (
                    mdp.reward(state, action, next_state) +
                    mdp.config.gamma * v_table.get(next_state, 0.0)
                )
                for next_state in mdp.all_states
            )
        best_q = max(q_table[(state, action)] for action in mdp.actions)
        max_delta = max(max_delta, abs(best_q - v_table[state]))
        new_v_table[state] = best_q
    return new_v_table, q_table, max_delta


def extract_policy(
        mdp: tm.TohMdp, q_table: tm.QTable
) -> tm.Policy:
    return {state: max(mdp.actions, key=lambda a: q_table[(state, a)])
            for state in mdp.nonterminal_states}


def q_update(
        mdp: tm.TohMdp, q_table: tm.QTable,
        transition: Tuple[tm.TohState, tm.TohAction, float, tm.TohState],
        alpha: float) -> None:
    state, action, reward, next_state = transition
    max_q_next = max((q_table.get((next_state, a), 0.0) for a in mdp.actions), default=0.0)
    q_table[(state, action)] = q_table.get((state, action), 0.0) + alpha * (
        reward + mdp.config.gamma * max_q_next - q_table.get((state, action), 0.0)
    )


def extract_v_table(mdp: tm.TohMdp, q_table: tm.QTable) -> tm.VTable:
    return {s: max(q_table.get((s, a), 0.0) for a in mdp.actions)
            for s in mdp.nonterminal_states}


def choose_next_action(
        mdp: tm.TohMdp, state: tm.TohState, epsilon: float, q_table: tm.QTable,
        epsilon_greedy: Callable[[List[tm.TohAction], float], tm.TohAction]
) -> tm.TohAction:
    best_q = max(q_table.get((state, a), 0.0) for a in mdp.actions)
    best_actions = [a for a in mdp.actions if math.isclose(q_table.get((state, a), 0.0), best_q)]
    return epsilon_greedy(best_actions, epsilon)


def custom_epsilon(n_step: int) -> float:
    return max(0.01, 1.0 / (1.0 + 0.005 * n_step))


def custom_alpha(n_step: int) -> float:
    return 1.0 / (1.0 + 0.001 * n_step)


def exploration_fn(u: float, n: int) -> float:
    return u + 100.0 / math.sqrt(n + 1)


def choose_next_action_with_exploration_fn(
        mdp: tm.TohMdp, state: tm.TohState, q_table: tm.QTable,
        visit_counts: Dict[Tuple[tm.TohState, tm.TohAction], int],
        exploration_fn: Callable[[float, int], float]
) -> tm.TohAction:
    actions = mdp.actions
    scores = {a: exploration_fn(q_table.get((state, a), 0.0), visit_counts.get((state, a), 0))
              for a in actions}
    best_score = max(scores.values())
    best_actions = [a for a in actions if math.isclose(scores[a], best_score)]
    chosen = random.choice(best_actions)
    visit_counts[(state, chosen)] = visit_counts.get((state, chosen), 0) + 1
    return chosen
