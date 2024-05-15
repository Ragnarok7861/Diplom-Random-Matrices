import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, loggamma, logsumexp

def generate_ginibre(N):
    """ Генерация матрицы из ансамбля Жинибра размерности N. """
    real_part = np.random.normal(0, 1, (N, N))
    imag_part = np.random.normal(0, 1, (N, N))
    return real_part + 1j * imag_part

def selberg_integral_ginibre(N):
    """ Вычисление логарифма нормировочной константы для ансамбля Жинибра. """
    log_C = N * np.log(np.pi)
    log_C += np.sum(loggamma(np.arange(1, N + 1)))
    return log_C

def joint_distribution_ginibre(eigenvalues):
    """ Расчет совместного распределения собственных значений для ансамбля Жинибра. """
    N = len(eigenvalues)
    log_C = selberg_integral_ginibre(N)
    log_term1 = -np.sum(np.abs(eigenvalues)**2)
    pairwise_log_diffs = [np.log(np.abs(eigenvalues[i] - eigenvalues[j])) for i in range(N) for j in range(i + 1, N)]
    log_term2 = 2 * logsumexp(pairwise_log_diffs)  # Учитываем β=2
    log_probability = log_term1 + log_term2 - log_C
    return np.exp(log_probability - np.max(log_probability))  # Преобразование обратно в обычную шкалу с предотвращением переполнения

def plot_eigenvalues(eigenvalues, weights):
    """ Функция для визуализации собственных значений. """
    weights_normalized = weights / np.sum(weights)  # Нормализация весов
    plt.hist(eigenvalues.real, bins=100, weights=weights_normalized, alpha=0.75, density=True, label='Real part')
    plt.hist(eigenvalues.imag, bins=100, weights=weights_normalized, alpha=0.75, density=True, label='Imaginary part', color='orange')
    plt.title("Взвешенное распределение собственных значений ансамбля Жинибра")
    plt.xlabel("Собственные значения")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    plt.show()

def main(N=50, num_matrices=1000):
    """ Основная функция для генерации ансамбля Жинибра и визуализации результатов. """
    eigenvalues = []
    all_weights = []
    for _ in range(num_matrices):
        ginibre_matrix = generate_ginibre(N)
        evs = np.linalg.eigvals(ginibre_matrix)
        weight = joint_distribution_ginibre(evs)
        eigenvalues.extend(evs)
        all_weights.extend([weight] * len(evs))

    plot_eigenvalues(np.array(eigenvalues), np.array(all_weights))

if __name__ == "__main__":
    main()
