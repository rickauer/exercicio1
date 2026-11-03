"""Experimento do Item 1: Avaliação de Políticas variando k e gamma,
incluindo a comparação entre o método iterativo e a solução direta (1c).
"""

import time
import tracemalloc
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from cleaning_robot.env import CleaningEnv
from cleaning_robot.model import TabularModel
from cleaning_robot.policy import GreedyPolicy
from cleaning_robot.dp import tabulate, policy_evaluation


def policy_evaluation_direct(
    model: TabularModel, policy: np.ndarray, gamma: float
) -> np.ndarray:
    """Resolve diretamente o sistema linear analítico: (I - gamma * P_pi) V = R."""
    if policy.ndim == 1:
        P_pi = sum(sp.diags((policy == a).astype(float)) @ model.P[a] for a in range(5))
    else:
        P_pi = sum(sp.diags(policy[:, a]) @ model.P[a] for a in range(5))

    A = sp.eye(model.n, format="csr") - gamma * P_pi
    return spsolve(A, model.R)


def run_experiment():
    k_values = [0, 1, 2, 4, 6, 10]
    gammas = [0.7, 0.99]

    # Dicionários de resultados para 1(a) e 1(b)
    results = {
        g: {"greedy_mean": [], "random_mean": [], "diff_mean": []} for g in gammas
    }

    # Resultados para 1(c): Comparação Iterativo vs Direto (testado na política gulosa com gamma=0.7)
    comparison_1c = {
        "k": k_values,
        "time_iter": [],
        "time_direct": [],
        "mem_iter": [],
        "mem_direct": [],
    }

    # experimento 1
    for k in k_values:
        print(f"\n---> Construindo ambiente e modelo para k = {k}...")
        t0 = time.perf_counter()
        env = CleaningEnv(k=k)
        model = TabularModel(env, verbose=False)
        print(f"     Modelo pronto em {time.perf_counter() - t0:.2f}s")

        greedy_actions = tabulate(model, GreedyPolicy(env))
        random_probs = np.full((model.n, 5), 0.2)

        for g in gammas:
            t_eval = time.perf_counter()
            V_greedy = policy_evaluation(model, greedy_actions, gamma=g)
            V_random = policy_evaluation(model, random_probs, gamma=g)
            t_total = time.perf_counter() - t_eval

            mean_g = float(V_greedy.mean())
            mean_r = float(V_random.mean())
            diff = mean_g - mean_r

            results[g]["greedy_mean"].append(mean_g)
            results[g]["random_mean"].append(mean_r)
            results[g]["diff_mean"].append(diff)

            print(
                f"     [gamma={g:<4}] Eval em {t_total:.2f}s | V_greedy: {mean_g:7.3f} | V_random: {mean_r:7.3f} | Vantagem: {diff:6.3f}"
            )

        # item 1c
        g_bench = 0.7

        # Medição do Método Iterativo
        tracemalloc.start()
        t_start = time.perf_counter()
        _ = policy_evaluation(model, greedy_actions, gamma=g_bench)
        t_iter = time.perf_counter() - t_start
        _, peak_iter = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_iter_mb = peak_iter / (1024 * 1024)

        # Medição do Método Direto
        tracemalloc.start()
        t_start = time.perf_counter()
        _ = policy_evaluation_direct(model, greedy_actions, gamma=g_bench)
        t_direct = time.perf_counter() - t_start
        _, peak_direct = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_direct_mb = peak_direct / (1024 * 1024)

        comparison_1c["time_iter"].append(t_iter)
        comparison_1c["time_direct"].append(t_direct)
        comparison_1c["mem_iter"].append(mem_iter_mb)
        comparison_1c["mem_direct"].append(mem_direct_mb)

        print(
            f"       -> Iterativo: {t_iter:.2f}s | Pico Memória: {mem_iter_mb:.1f} MB"
        )
        print(
            f"       -> Direto:    {t_direct:.2f}s | Pico Memória: {mem_direct_mb:.1f} MB"
        )

    # grafico 1b
    for g in gammas:
        plt.figure(figsize=(7, 5), dpi=150)
        plt.plot(
            k_values,
            results[g]["greedy_mean"],
            "o-",
            label="Greedy (Gulosa)",
            color="#1f77b4",
            lw=2,
        )
        plt.plot(
            k_values,
            results[g]["random_mean"],
            "s--",
            label="Random (Aleatória)",
            color="#d62728",
            lw=2,
        )
        plt.title(f"Valor Médio vs k ($\gamma = {g}$)", fontsize=13, fontweight="bold")
        plt.xlabel("Parâmetro k (rodadas de reinício)", fontsize=11)
        plt.ylabel("Valor Médio $\\bar{V}$", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()

        filename = f"grafico_item1_gamma_{g}.png"
        plt.savefig(filename)
        plt.close()
        print(f"✓ Gráfico salvo em '{filename}'!")

    # grafico 1c
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    axes[0].plot(
        k_values,
        comparison_1c["time_iter"],
        "o-",
        label="Método Iterativo (Bellman)",
        color="#2ca02c",
        lw=2,
    )
    axes[0].plot(
        k_values,
        comparison_1c["time_direct"],
        "s-",
        label="Método Direto (spsolve)",
        color="#d62728",
        lw=2,
    )
    axes[0].set_title("Tempo de Execução vs k", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Parâmetro k", fontsize=11)
    axes[0].set_ylabel("Tempo (segundos)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    axes[1].plot(
        k_values,
        comparison_1c["mem_iter"],
        "o-",
        label="Método Iterativo (Bellman)",
        color="#2ca02c",
        lw=2,
    )
    axes[1].plot(
        k_values,
        comparison_1c["mem_direct"],
        "s-",
        label="Método Direto (spsolve)",
        color="#d62728",
        lw=2,
    )
    axes[1].set_title("Pico de Memória Adicional vs k", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Parâmetro k", fontsize=11)
    axes[1].set_ylabel("Memória (MB)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    mem_time_filename = "grafico_item1_tempo_memoria.png"
    plt.savefig(mem_time_filename)
    plt.close()


if __name__ == "__main__":
    run_experiment()
