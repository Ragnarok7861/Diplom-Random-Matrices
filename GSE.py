import numpy as np
import matplotlib.pyplot as plt
from scipy.special import loggamma, logsumexp
from scipy.linalg import eigh

def generate_gse(N):
    """ Генерация Гауссовского Симплектического Ансамбля размерности N. """
    A = np.random.normal(0, 1/np.sqrt(2), (2*N, 2*N)) + 1j * np.random.normal(0, 1/np.sqrt(2), (2*N, 2*N))
    return (A + A.conj().T) / 2

def selberg_integral_gse(N):
    """ Вычисление логарифма нормализационной константы по формуле Сельберга для GSE. """
    log_C = N * (2*N + 1) * np.log(2) + N * np.log(np.pi)
    for j in range(1, 2*N + 1):
        log_C += loggamma(1 + j / 2) - loggamma(1 + (2*N + j) / 2)
    return log_C

def joint_distribution_gse(eigenvalues):
    """ Расчет совместного распределения собственных значений для GSE. """
    N = len(eigenvalues)
    log_C = selberg_integral_gse(N // 2)
    log_term1 = -0.5 * np.sum(eigenvalues**2)
    pairwise_log_diffs = [np.log(np.abs(eigenvalues[i] - eigenvalues[j])) for i in range(N) for j in range(i + 1, N)]
    log_term2 = 4 * logsumexp(pairwise_log_diffs)  # Стабилизация суммы логарифмов с учетом β=4
    log_probability = log_term1 + log_term2 - log_C
    return np.exp(log_probability - np.max(log_probability))  # Преобразование обратно в обычную шкалу с предотвращением переполнения

def plot_eigenvalues(eigenvalues, weights):
    """ Функция для визуализации собственных значений. """
    weights_normalized = weights / np.sum(weights)  # Нормализация весов
    plt.hist(eigenvalues, bins=100, weights=weights_normalized, alpha=0.75, density=True)
    plt.title(" Распределение собственных значений GSE")
    plt.xlabel("Собственные значения")
    plt.ylabel("Плотность вероятности")
    plt.show()

def main(N=50, num_matrices=1000):
    """ Основная функция для генерации GSE и визуализации результатов. """
    eigenvalues = []
    all_weights = []
    for _ in range(num_matrices):
        gse_matrix = generate_gse(N)
        evs = np.linalg.eigvalsh(gse_matrix)
        weight = joint_distribution_gse(evs)
        eigenvalues.extend(evs)
        all_weights.extend([weight] * len(evs))

    plot_eigenvalues(np.array(eigenvalues), np.array(all_weights))

if __name__ == "__main__":
    main()
