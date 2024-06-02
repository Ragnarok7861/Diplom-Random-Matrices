import numpy as np
import matplotlib.pyplot as plt

def ginibre_matrix(N):
    """Генерирует комплексную матрицу Жинибра размером N x N."""
    real_part = np.random.normal(0, 1 / np.sqrt(N), (N, N))
    imag_part = np.random.normal(0, 1 / np.sqrt(N), (N, N))
    return real_part + 1j * imag_part

def plot_empirical_density(N, num_samples=1000):
    """Строит гистограмму плотности собственных значений для ансамбля Жинибра."""
    eigenvalues = []

    for _ in range(num_samples):
        G = ginibre_matrix(N)
        eigenvalues.extend(np.linalg.eigvals(G))

    eigenvalues = np.array(eigenvalues)

    # Построение гистограммы собственных значений
    fig, ax = plt.subplots(figsize=(10, 8))
    h = ax.hist2d(eigenvalues.real, eigenvalues.imag, bins=100, density=True, cmap='viridis')
    cbar = plt.colorbar(h[3], ax=ax, label='Плотность ', pad=0.1)
    ax.set_xlabel('Вещественная часть')
    ax.set_ylabel('Мнимая часть')
    ax.set_title(f' Распределение собственных значений ансамбля Жинибра ')
    ax.set_aspect('equal', adjustable='box')  # Устанавливаем одинаковый масштаб для осей
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)  # Настраиваем макет
    plt.show()

def plot_theoretical_density(grid_size=100):
    """Строит теоретическую плотность собственных значений для ансамбля Жинибра."""
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    density = np.zeros(Z.shape)

    for i in range(grid_size):
        for j in range(grid_size):
            if np.abs(Z[i, j]) <= 1:
                density[i, j] = 1 / (np.pi)

    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.contourf(X, Y, density, levels=100, cmap='viridis')
    cbar = plt.colorbar(c, ax=ax, label='Плотность (теоретическая)', pad=0.1)
    ax.set_xlabel('Вещественная часть')
    ax.set_ylabel('Мнимая часть')
    ax.set_title('Теоретическая плотность собственных значений ансамбля Жинибра')
    ax.set_aspect('equal', adjustable='box')  # Устанавливаем одинаковый масштаб для осей
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)  # Настраиваем макет
    plt.show()

if __name__ == '__main__':
    N = 100  # Размер матрицы Жинибра
    num_samples_empirical = 1000  # Количество выборок для усреднения эмпирической плотности

    plot_empirical_density(N, num_samples=num_samples_empirical)
    plot_theoretical_density(grid_size=100)
