"""Model-based methods (dynamic programming)."""

from __future__ import annotations

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
    return np.fromiter((policy.act(s) for s in all_states(model.env)), np.int64, count=model.n)


def policy_evaluation(model: TabularModel, policy: np.ndarray, gamma: float) -> np.ndarray:
    """Value V^pi of a policy, computed with the model.

    `policy` is either one action per state index (deterministic, shape (n,)) or the
    probability of each action in each state (shape (n, n_actions)).
    Returns V with V[i] the value of the state of index i.

    The model is model.P (one sparse n x n matrix per action, model.P[a][i, j] = P(j | i, a))
    and model.R (reward of each state).
    """
    raise NotImplementedError


class ValueIteration(TabularPolicy):
    """Value iteration.

    After construction, self.actions must hold the greedy action of every state index:
    call super().__init__(model, actions) at the end, and act() comes for free.
    """

    def __init__(self, model: TabularModel, gamma: float):
        raise NotImplementedError


class PolicyIteration(TabularPolicy):
    """Policy iteration.

    After construction, self.actions must hold the action of every state index:
    call super().__init__(model, actions) at the end, and act() comes for free.
    """

    def __init__(self, model: TabularModel, gamma: float):
        raise NotImplementedError
