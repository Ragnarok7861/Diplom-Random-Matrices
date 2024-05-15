import numpy as np
import matplotlib.pyplot as plt
from scipy.special import loggamma, logsumexp
from scipy.linalg import eigh

def generate_goe(N):
    A = np.random.normal(0, 1, (N, N))
    return (A + A.T) / np.sqrt(2)

def selberg_integral(N):
    log_C = N / 2 * np.log(2 * np.pi)
    for j in range(1, N + 1):
        log_C += loggamma(j / 2 + 1) - (j / 2) * np.log(2)
    return log_C  # Возвращаем логарифм C

def joint_distribution_goe(eigenvalues):
    N = len(eigenvalues)
    log_C = selberg_integral(N)
    log_term1 = -0.5 * np.sum(eigenvalues**2)
    pairwise_log_diffs = [np.log(np.abs(eigenvalues[i] - eigenvalues[j])) for i in range(N) for j in range(i + 1, N)]
    log_term2 = logsumexp(pairwise_log_diffs)  # Стабилизация суммы логарифмов
    log_probability = log_term1 + log_term2 - log_C
    return np.exp(log_probability - np.max(log_probability))  # Преобразование обратно в обычную шкалу с предотвращением переполнения

def plot_eigenvalues(eigenvalues, weights):
    weights_normalized = weights / np.sum(weights)  # Нормализация весов
    plt.hist(eigenvalues, bins=100, weights=weights_normalized, alpha=0.75, density=True)
    plt.title(" Распределение собственных значений GOE")
    plt.xlabel("Собственные значения")
    plt.ylabel("Плотность вероятности")
    plt.show()

def main(N=50, num_matrices=1000):
    eigenvalues = []
    all_weights = []
    for _ in range(num_matrices):
        goe_matrix = generate_goe(N)
        evs = np.linalg.eigvalsh(goe_matrix)
        weight = joint_distribution_goe(evs)
        eigenvalues.extend(evs)
        all_weights.extend([weight] * N)

    plot_eigenvalues(eigenvalues, all_weights)

if __name__ == "__main__":
    main()
