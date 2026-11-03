"""Experimento do Item 2"""

import time
import numpy as np
import matplotlib.pyplot as plt

from cleaning_robot.env import CleaningEnv
from cleaning_robot.model import TabularModel
from cleaning_robot.policy import GreedyPolicy
from cleaning_robot.dp import (
    tabulate,
    policy_evaluation,
    ValueIteration,
    PolicyIteration,
)


def run_experiment_item2():
    k_values = [0, 1, 2, 4, 6, 10]
    gammas = [0.7, 0.99]

    data = {
        g: {
            "vi_time": [],
            "pi_time": [],
            "vi_iters": [],
            "pi_iters": [],
            "v_opt_mean": [],
            "v_greedy_mean": [],
            "v_random_mean": [],
            "policies": [],  # lista com as ações da política ótima para cada k
        }
        for g in gammas
    }

    for k in k_values:
        t0 = time.perf_counter()
        env = CleaningEnv(k=k)
        model = TabularModel(env, verbose=False)
        print(f"     Modelo pronto em {time.perf_counter() - t0:.2f}s")

        greedy_actions = tabulate(model, GreedyPolicy(env))
        random_probs = np.full((model.n, 5), 0.2)

        for g in gammas:
            print(f"\n     [gamma = {g}] Executando algoritmos...")

            vi = ValueIteration(model, gamma=g)
            t_vi = vi.convergence_time
            iters_vi = vi.n_iterations
            print(f"       -> Value Iteration:  {iters_vi:3d} iterações em {t_vi:.2f}s")

            pi = PolicyIteration(model, gamma=g)
            t_pi = pi.convergence_time
            iters_pi = pi.n_iterations
            print(f"       -> Policy Iteration: {iters_pi:3d} iterações em {t_pi:.2f}s")

            v_opt = float(vi.V.mean())
            v_greedy = float(policy_evaluation(model, greedy_actions, gamma=g).mean())
            v_random = float(policy_evaluation(model, random_probs, gamma=g).mean())

            data[g]["vi_time"].append(t_vi)
            data[g]["pi_time"].append(t_pi)
            data[g]["vi_iters"].append(iters_vi)
            data[g]["pi_iters"].append(iters_pi)
            data[g]["v_opt_mean"].append(v_opt)
            data[g]["v_greedy_mean"].append(v_greedy)
            data[g]["v_random_mean"].append(v_random)
            data[g]["policies"].append(vi.actions)

            print(
                f"Valores médios: V*={v_opt:.3f} | V_greedy={v_greedy:.3f} | V_random={v_random:.3f}"
            )

    for g in gammas:
        plt.figure(figsize=(7, 5), dpi=150)
        plt.plot(
            k_values,
            data[g]["vi_time"],
            "o-",
            label="Value Iteration",
            color="#1f77b4",
            lw=2,
        )
        plt.plot(
            k_values,
            data[g]["pi_time"],
            "s--",
            label="Policy Iteration",
            color="#d62728",
            lw=2,
        )
        plt.title(
            f"Tempo de Convergência vs k ($\gamma = {g}$)",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("Parâmetro k (rodadas de reinício)", fontsize=11)
        plt.ylabel("Tempo (segundos)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()

        fname = f"grafico_item2_tempo_gamma_{g}.png"
        plt.savefig(fname)
        plt.close()
        print(f"✓ Gráfico salvo: '{fname}'")

    for g in gammas:
        plt.figure(figsize=(7, 5), dpi=150)
        plt.plot(
            k_values,
            data[g]["vi_iters"],
            "o-",
            label="Value Iteration",
            color="#1f77b4",
            lw=2,
        )
        plt.plot(
            k_values,
            data[g]["pi_iters"],
            "s--",
            label="Policy Iteration",
            color="#d62728",
            lw=2,
        )
        plt.title(
            f"Número de Iterações vs k ($\gamma = {g}$)", fontsize=13, fontweight="bold"
        )
        plt.xlabel("Parâmetro k (rodadas de reinício)", fontsize=11)
        plt.ylabel("Número de Iterações", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()

        fname = f"grafico_item2_iteracoes_gamma_{g}.png"
        plt.savefig(fname)
        plt.close()
        print(f"✓ Gráfico salvo: '{fname}'")

    plt.figure(figsize=(8, 5), dpi=150)
    transitions_labels = [
        f"{k_values[i]} -> {k_values[i+1]}" for i in range(len(k_values) - 1)
    ]

    for g, cor in zip(gammas, ["#2ca02c", "#9467bd"]):
        consecutive_diff = []
        for i in range(len(k_values) - 1):
            pol_prev = data[g]["policies"][i]
            pol_next = data[g]["policies"][i + 1]
            diff_pct = float(np.mean(pol_prev != pol_next) * 100)
            consecutive_diff.append(diff_pct)

        plt.plot(
            transitions_labels,
            consecutive_diff,
            "o-",
            label=f"$\gamma = {g}$",
            color=cor,
            lw=2,
        )

    plt.title(
        "Mudança Consecutiva na Política Ótima ($k_i \\to k_{i+1}$) (Item 2b)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Transição entre valores de k", fontsize=11)
    plt.ylabel("% de Estados que Mudaram de Ação", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    fname_pol = "grafico_item2_mudanca_politica_vs_k.png"
    plt.savefig(fname_pol)
    plt.close()

    for g in gammas:
        plt.figure(figsize=(7, 5), dpi=150)
        plt.plot(
            k_values,
            data[g]["v_opt_mean"],
            "o-",
            label="Ótima ($\pi^*$ - DP)",
            color="#2ca02c",
            lw=2,
        )
        plt.plot(
            k_values,
            data[g]["v_greedy_mean"],
            "s--",
            label="Gulosa (Greedy)",
            color="#1f77b4",
            lw=2,
        )
        plt.plot(
            k_values,
            data[g]["v_random_mean"],
            "^:",
            label="Aleatória (Random)",
            color="#d62728",
            lw=2,
        )
        plt.title(
            f"Comparação de Políticas vs k ($\gamma = {g}$) (Item 2c)",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("Parâmetro k (rodadas de reinício)", fontsize=11)
        plt.ylabel("Valor Médio $\\bar{V}$", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()

        fname_comp = f"grafico_item2_comparacao_politicas_gamma_{g}.png"
        plt.savefig(fname_comp)
        plt.close()


if __name__ == "__main__":
    run_experiment_item2()
