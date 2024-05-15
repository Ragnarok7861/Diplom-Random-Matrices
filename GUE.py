import numpy as np
import matplotlib.pyplot as plt
from scipy.special import loggamma, logsumexp

def generate_gue(N):
    """ Генерация Гауссова Унитарного Ансамбля размерности N. """
    A = np.random.normal(0, 1/np.sqrt(2), (N, N)) + 1j * np.random.normal(0, 1/np.sqrt(2), (N, N))
    return (A + A.conj().T) / 2

def log_selberg_integral(N, beta):
    """ Вычисление логарифма нормализационной константы по формуле Сельберга. """
    return np.sum([loggamma(1 + beta * j / 2) - loggamma(1 + beta / 2) for j in range(1, N + 1)])

def joint_distribution_gue(lambdas, beta):
    """ Расчет совместного распределения собственных значений для GUE с учетом переполнений. """
    N = len(lambdas)
    log_norm_const = log_selberg_integral(N, beta)
    term1 = -np.sum(lambdas**2) / 2
    pairwise_log_diffs = [np.log(np.abs(lambdas[i] - lambdas[j])) for i in range(N) for j in range(i+1, N)]
    term2 = beta * logsumexp(pairwise_log_diffs)  # Используем logsumexp для суммирования логарифмов
    log_probability = term1 + term2 - log_norm_const
    # Используем минимальный порог для вероятности
    return max(np.exp(log_probability), 1e-300)

def plot_eigenvalues(eigenvalues, weights):
    """ Функция для визуализации собственных значений. """
    plt.hist(eigenvalues, bins=100, weights=weights, alpha=0.75, density=True)
    plt.title(" Распределение собственных значений GUE")
    plt.xlabel("Собственные значения")
    plt.ylabel("Плотность вероятности")
    plt.show()

def main(N=10, num_matrices=1000, beta=2):
    """ Основная функция для генерации GUE и визуализации результатов. """
    eigenvalues = []
    weights = []
    for _ in range(num_matrices):
        gue_matrix = generate_gue(N)
        evs = np.linalg.eigvalsh(gue_matrix)
        weight = joint_distribution_gue(evs, beta)
        eigenvalues.extend(evs)
        weights.extend([weight] * N)  # Предполагается, что все собственные значения в одной матрице имеют одинаковый вес
    plot_eigenvalues(np.array(eigenvalues), np.array(weights))

if __name__ == "__main__":
    main(N=50, num_matrices=1000, beta=2)
