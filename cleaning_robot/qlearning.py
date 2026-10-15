"""Tabular Q-learning: model-free control from sampled transitions only."""

from __future__ import annotations

from .env import CleaningEnv, State
from .policy import PolicyMixin


class QLearning(PolicyMixin):
    """Tabular Q-learning.

    Learns only from interaction with the environment: env.random_init, env.next_state_sample
    and env.reward. Do not use the model (P and R). state_index(env, state) from
    cleaning_robot.model gives a row of the Q table for each state.

    act() must return the greedy action of the learned Q.
    """

    def __init__(self, env: CleaningEnv, gamma: float, n_steps: int, seed=None):
        raise NotImplementedError

    def act(self, state: State, seed=None) -> int:
        raise NotImplementedError
