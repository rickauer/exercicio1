"""Tabular Q-learning: model-free control from sampled transitions only."""

from __future__ import annotations

import numpy as np
from .env import CleaningEnv, State
from .policy import PolicyMixin
from .model import state_index, n_states


class QLearning(PolicyMixin):
    """Tabular Q-learning.

    Learns only from interaction with the environment: env.random_init, env.next_state_sample
    and env.reward. Do not use the model (P and R). state_index(env, state) from
    cleaning_robot.model gives a row of the Q table for each state.

    act() must return the greedy action of the learned Q.
    """

    def __init__(
        self,
        env: CleaningEnv,
        gamma: float,
        n_steps: int,
        seed=None,
        alpha: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        init_mode: str = "zeros",  # "zeros" ou "pessimistic"
        episode_len: int = 200,
        callback=None,
        eval_interval: int | None = None,
        verbose: bool = True,
    ):
        self.env = env
        self.gamma = gamma
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.rng = np.random.default_rng(seed)

        n = n_states(env)
        if init_mode == "pessimistic":
            v_min = -1.0 / (1.0 - gamma)
            self.Q = np.full((n, 5), v_min, dtype=np.float64)
        else:
            self.Q = np.zeros((n, 5), dtype=np.float64)

        Q, rng = self.Q, self.rng
        state = env.random_init(seed=rng)
        s_idx = state_index(env, state)
        step_in_ep = 0

        for step in range(1, n_steps + 1):
            # Ação epsilon-greedy
            if rng.random() < self.epsilon:
                action = int(rng.integers(5))
            else:
                action = self._best_action(Q[s_idx])

            reward = env.reward(state)
            next_state = env.next_state_sample(state, action, seed=rng)
            next_idx = state_index(env, next_state)

            td_target = reward + gamma * Q[next_idx].max()
            Q[s_idx, action] += alpha * (td_target - Q[s_idx, action])

            state, s_idx = next_state, next_idx
            step_in_ep += 1

            # Reinício para espalhar a exploração pela sala
            if episode_len > 0 and step_in_ep >= episode_len:
                state = env.random_init(seed=rng)
                s_idx = state_index(env, state)
                step_in_ep = 0

            epsilon_decay_steps = int(0.4 * n_steps)
            # Decaimento linear de epsilon com piso
            self.epsilon = max(
                epsilon_min,
                epsilon_start
                - (epsilon_start - epsilon_min) * min(step / epsilon_decay_steps, 1.0),
            )

            if verbose and (step % 200_000 == 0 or step == n_steps):
                print(f"   Passo {step:,} / {n_steps:,} | Epsilon: {self.epsilon:.3f}")

            # Callback para rastrear evolução ao longo do treino (Item 3b)
            if eval_interval is not None and (
                step % eval_interval == 0 or step == n_steps
            ):
                if callback is not None:
                    callback(step, self)

    def _best_action(self, q_values: np.ndarray) -> int:
        max_v = np.max(q_values)
        best = np.flatnonzero(q_values == max_v)
        if len(best) == 1:
            return int(best[0])
        return int(self.rng.choice(best))

    def act(self, state: State, seed=None) -> int:
        i = state_index(self.env, state)
        return self._best_action(self.Q[i])
