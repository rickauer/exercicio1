"""Model-based methods (dynamic programming)."""

from __future__ import annotations
import time
from random import random

import numpy as np

from .env import State
from .model import TabularModel, all_states
from .policy import PolicyMixin


class TabularPolicy(PolicyMixin):
    """Deterministic policy stored as one action per state index."""

    def __init__(self, model: TabularModel, actions: np.ndarray):
        self.model = model
        self.actions = actions

    def act(self, state: State, seed=None) -> int:
        return int(self.actions[self.model.state_index(state)])


def tabulate(model: TabularModel, policy: PolicyMixin) -> np.ndarray:
    """Action of a deterministic policy in every state, in index order."""
    return np.fromiter(
        (policy.act(s) for s in all_states(model.env)), np.int64, count=model.n
    )


def policy_evaluation(
    model: TabularModel, policy: np.ndarray, gamma: float
) -> np.ndarray:
    """Value V^pi of a policy, computed with the model.

    `policy` is either one action per state index (deterministic, shape (n,)) or the
    probability of each action in each state (shape (n, n_actions)).
    Returns V with V[i] the value of the state of index i.

    The model is model.P (one sparse n x n matrix per action, model.P[a][i, j] = P(j | i, a))
    and model.R (reward of each state).
    """
    V = np.zeros(model.n)
    R = model.R
    is_deterministic = policy.ndim == 1
    while True:
        expected_future = np.zeros(model.n)
        if is_deterministic:
            for a in range(5):
                mask = policy == a
                if np.any(mask):
                    expected_future[mask] = (model.P[a] @ V)[
                        mask
                    ]  # aplica apenas onde a ação "a" é escolhida
        else:
            for a in range(5):
                expected_future += policy[:, a] * (model.P[a] @ V)

        V_new = R + gamma * expected_future  # recompensa não depende da ação

        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < 1e-7:
            break
    return V


class ValueIteration(TabularPolicy):
    """Value iteration.

    After construction, self.actions must hold the greedy action of every state index:
    call super().__init__(model, actions) at the end, and act() comes for free.
    """

    def __init__(self, model: TabularModel, gamma: float):
        t0 = time.perf_counter()
        V = np.zeros(model.n)
        iterations = 0

        while True:
            iterations += 1
            V_new = np.full(model.n, -np.inf)
            for a in range(5):
                val_a = model.R + gamma * (model.P[a] @ V)
                V_new = np.maximum(V_new, val_a)

            if np.max(np.abs(V_new - V)) < 1e-7:
                break
            V = V_new

        actions = np.zeros(model.n, dtype=np.int64)
        best_val = np.full(model.n, -np.inf)
        for a in range(5):
            val_a = model.R + gamma * (model.P[a] @ V)
            melhorou = val_a > best_val
            actions[melhorou] = a
            best_val[melhorou] = val_a[melhorou]

        self.n_iterations = iterations
        self.convergence_time = time.perf_counter() - t0
        self.V = V
        super().__init__(model, actions)


class PolicyIteration(TabularPolicy):
    """Policy iteration.

    After construction, self.actions must hold the action of every state index:
    call super().__init__(model, actions) at the end, and act() comes for free.
    """

    def __init__(self, model: TabularModel, gamma: float):
        t0 = time.perf_counter()
        policy = np.random.randint(
            0, 5, size=model.n, dtype=np.int64
        )  # inicializa com ações aleatórias
        iterations = 0

        while True:
            iterations += 1
            V_pi_s = policy_evaluation(model, policy, gamma)

            Q = np.zeros((model.n, 5))
            for a in range(5):
                Q[:, a] = model.R + gamma * (model.P[a] @ V_pi_s)
            new_policy = np.argmax(Q, axis=1)

            if np.array_equal(policy, new_policy):
                break
            policy = new_policy

        actions = np.argmax(Q, axis=1)
        self.n_iterations = iterations
        self.convergence_time = time.perf_counter() - t0
        self.V = V_pi_s
        super().__init__(model, actions)
