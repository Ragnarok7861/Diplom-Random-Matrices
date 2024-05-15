import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cProfile
import pstats

# Определения и инициализация параметров
N = 50  # Размер матрицы
num_batches = 10  # Количество пакетов
batch_size = 100  # Количество матриц в пакете
sigma = 1.0  # Стандартное отклонение элементов матрицы
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Массив различных значений alpha
L = 40  # Размер гиперкуба
beta = 0.1  # Степень отталкивания собственных значений
epsilon = 1e-5  # Маленькое число для стабилизации логарифма
num_processes = 4  # Количество процессов для многопроцессорной обработки
triu_indices = np.triu_indices(N, 1)  # Индексы для верхнего треугольника матрицы


def generate_asymmetric_matrix(N, alpha, sigma, triu_indices):
    diagonal = np.random.normal(0, sigma, N)
    off_diagonal = np.random.normal(0, alpha * sigma, len(triu_indices[0]))
    A = np.zeros((N, N))
    np.fill_diagonal(A, diagonal)
    A[triu_indices] = off_diagonal
    A += A.T - np.diag(A.diagonal())
    return A


def V(lambdas, alpha):
    term1 = alpha ** 2 * np.sum(lambdas ** 2)
    term2 = (1 - alpha ** 2) * np.sum(np.subtract.outer(lambdas, lambdas) ** 2)
    return term1 + term2


def potential_log(V_func, eigenvalues, alpha, beta, epsilon, N):
    diffs = np.subtract.outer(eigenvalues, eigenvalues)
    log_vandermonde = beta * np.sum(np.log(np.abs(diffs) + epsilon), where=(diffs != 0))
    log_exp_term = -N * V_func(eigenvalues, alpha)
    return log_vandermonde + log_exp_term

def process_batch(alpha, batch_size, N, sigma, L, beta, epsilon, triu_indices):
    all_eigenvalues = []
    all_weights = []
    for _ in range(batch_size):
        A = generate_asymmetric_matrix(N, alpha, sigma, triu_indices)
        eigenvalues = np.linalg.eigvalsh(A)
        if not all(abs(ev) <= L / 2 for ev in eigenvalues):
            continue
        log_pot = potential_log(V, eigenvalues, alpha, beta, epsilon, N)
        if not np.isnan(log_pot):
            all_eigenvalues.extend(eigenvalues)
            all_weights.extend([log_pot] * len(eigenvalues))
    return all_eigenvalues, all_weights, alpha


def main():
    tasks = [(alpha, batch_size, N, sigma, L, beta, epsilon, triu_indices) for alpha in alphas for _ in
             range(num_batches)]
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_batch, tasks)

    all_eigenvalues = {alpha: [] for alpha in alphas}
    all_weights = {alpha: [] for alpha in alphas}
    for eigenvalues, weights, alpha in results:
        all_eigenvalues[alpha].extend(eigenvalues)
        all_weights[alpha].extend(weights)

    for alpha in alphas:
        if all_weights[alpha]:
            plt.hist(all_eigenvalues[alpha], bins=50, weights=all_weights[alpha], density=True, alpha=0.6,
                     label=f'alpha = {alpha}')

    plt.title("Распределение собственных значений для разных alpha")
    plt.xlabel("Собственные значения")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    results = pstats.Stats(profiler).sort_stats('time')
    # Перенаправление вывода в файл
    with open('profiling_results.txt', 'w') as file:
        results.stream = file
        results.print_stats()