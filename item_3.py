"""Experimento do Item 3: Q-Learning."""

import time
import numpy as np
import matplotlib.pyplot as plt

from cleaning_robot.env import CleaningEnv
from cleaning_robot.model import TabularModel
from cleaning_robot.policy import GreedyPolicy, RandomPolicy
from cleaning_robot.dp import tabulate, policy_evaluation, ValueIteration
from cleaning_robot.qlearning import QLearning
from cleaning_robot.animate import animate


def run_experiment_item3():

    # Definição dos dois cenários escolhidos
    scenarios = [
        {"k": 1, "gamma": 0.7, "desc": "k=1, gamma=0.7"},
        {"k": 10, "gamma": 0.99, "desc": "k=10, gamma=0.99"},
    ]

    total_steps = 50_000_000  # passos de interação
    eval_interval = 10_000_000  # intervalo de avaliação da política

    scen1 = scenarios[0]
    k1, g1 = scen1["k"], scen1["gamma"]
    env1 = CleaningEnv(k=k1)
    model1 = TabularModel(env1, verbose=False)

    vi1 = ValueIteration(model1, gamma=g1)
    v_opt1 = float(vi1.V.mean())
    v_greedy1 = float(
        policy_evaluation(model1, tabulate(model1, GreedyPolicy(env1)), gamma=g1).mean()
    )
    v_random1 = float(
        policy_evaluation(model1, np.full((model1.n, 5), 0.2), gamma=g1).mean()
    )

    init_modes = ["zeros", "pessimistic"]
    init_histories = {}

    for mode in init_modes:
        steps_log = []
        v_eval_log = []

        def eval_callback(current_step, agent):
            # Extrai ações gulosas da tabela Q atual
            actions = np.argmax(agent.Q, axis=1)
            v_val = policy_evaluation(model1, actions, gamma=g1)
            v_mean = float(v_val.mean())
            steps_log.append(current_step)
            v_eval_log.append(v_mean)
            print(
                f"[Passo {current_step:>7,}] V_médio da política aprendida: {v_mean:.3f}"
            )

        t0 = time.perf_counter()
        agent = QLearning(
            env=env1,
            gamma=g1,
            n_steps=total_steps,
            seed=42,
            alpha=0.1,
            init_mode=mode,
            eval_interval=eval_interval,
            callback=eval_callback,
            verbose=False,
        )
        t_total = time.perf_counter() - t0
        init_histories[mode] = (steps_log, v_eval_log, t_total)

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(
        init_histories["zeros"][0],
        init_histories["zeros"][1],
        "o-",
        label="Inicialização Zeros (Otimista)",
        color="#1f77b4",
        lw=2,
    )
    plt.plot(
        init_histories["pessimistic"][0],
        init_histories["pessimistic"][1],
        "s--",
        label="Inicialização Pessimista ($-1/(1-\gamma)$)",
        color="#d62728",
        lw=2,
    )
    plt.axhline(
        v_opt1,
        color="#2ca02c",
        linestyle=":",
        label=f"Ótima DP ($V^*={v_opt1:.2f}$)",
        lw=2,
    )
    plt.axhline(
        v_greedy1,
        color="#ff7f0e",
        linestyle="-.",
        label=f"Gulosa ($V={v_greedy1:.2f}$)",
        lw=1.5,
    )
    plt.axhline(
        v_random1,
        color="#7f7f7f",
        linestyle="--",
        label=f"Aleatória ($V={v_random1:.2f}$)",
        lw=1.5,
    )
    plt.title(
        f"Efeito da Inicialização de Q no Aprendizado ({scen1['desc']})",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Passos de Interação", fontsize=11)
    plt.ylabel("Valor Médio da Política $\\bar{V}$", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()

    fname_init = "grafico_item3_inicializacao_q.png"
    plt.savefig(fname_init)
    plt.close()

    for idx, sc in enumerate(scenarios, 1):
        k, g = sc["k"], sc["gamma"]
        env = CleaningEnv(k=k)
        print(f"\n--- Cenário {idx}: {sc['desc']} ---")
        print("Construindo TabularModel...")
        t_model_start = time.perf_counter()
        model = TabularModel(env, verbose=False)
        t_model = time.perf_counter() - t_model_start

        # Executa Value Iteration como baseline
        print("Executando Value Iteration...")
        vi = ValueIteration(model, gamma=g)
        v_opt = float(vi.V.mean())
        v_greedy = float(
            policy_evaluation(model, tabulate(model, GreedyPolicy(env)), gamma=g).mean()
        )

        steps_curve = []
        v_curve = []

        def track_progress(current_step, agent):
            acts = np.argmax(agent.Q, axis=1)
            v_m = float(policy_evaluation(model, acts, gamma=g).mean())
            steps_curve.append(current_step)
            v_curve.append(v_m)
            print(f"[Passo {current_step:>7,}] Qualidade da política: {v_m:.3f}")

        t_ql_start = time.perf_counter()
        ql_agent = QLearning(
            env=env,
            gamma=g,
            n_steps=total_steps,
            seed=42,
            alpha=0.1,
            init_mode="pessimistic",
            eval_interval=eval_interval,
            callback=track_progress,
            verbose=False,
        )
        t_ql = time.perf_counter() - t_ql_start

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(steps_curve, v_curve, "o-", label="Q-Learning", color="#1f77b4", lw=2)
        plt.axhline(
            v_opt,
            color="#2ca02c",
            linestyle=":",
            label=f"Ótimo DP ($V^*={v_opt:.2f}$)",
            lw=2,
        )
        plt.axhline(
            v_greedy,
            color="#ff7f0e",
            linestyle="-.",
            label=f"Gulosa ($V={v_greedy:.2f}$)",
            lw=1.5,
        )
        plt.title(
            f"Curva de Aprendizado Q-Learning ({sc['desc']})",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Passos de Interação", fontsize=11)
        plt.ylabel("Valor Médio da Política $\\bar{V}$", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10, loc="lower right")
        plt.tight_layout()

        fname_curve = f"grafico_item3_curva_aprendizado_cenario_{idx}.png"
        plt.savefig(fname_curve)
        plt.close()

        gif_name = f"animacao_ql_cenario_{idx}.gif"
        animate(ql_agent, env, steps=80, filename=gif_name)


if __name__ == "__main__":
    run_experiment_item3()
