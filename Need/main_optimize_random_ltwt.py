import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.constants import c

from tools import (
    plot_array,
    plot_surf_uv,
    plot_surf_theta_phi,
    get_side_lobes,
    calc_F_theta_phi,
    get_directivity_uv,
)
from arraygrid import (
    rect_random_symmentry,
    random_symmetry_size,
    rect_random_symmentry_with_gap,
    rect_random_symmentry_with_gap_with_initial,
)
from tsv import writeToTSV

# Целевая функция по УБЛ
# Функция вычисляет максимальный УБЛ для заданных координат элементов
# и направлений фазирования. Чем меньше значение, тем лучше
def goal_func_side_lobe(
    coord: npt.NDArray,
    freq: float,
    angles: list[tuple[float, float]],
    theta_mesh: npt.NDArray,
    phi_mesh: npt.NDArray,
) -> float:
    max_val = -np.inf
    for theta_0, phi_0 in angles:
        F = calc_F_theta_phi(coord, freq, theta_0, phi_0, theta_mesh, phi_mesh)
        side_lobes = get_side_lobes(F)
        first_lobe = side_lobes[1][2]
        if first_lobe > max_val:
            max_val = first_lobe
    return max_val

# Целевая функция по КНД
# Функция вычисляет минимальный КНД для заданных координат
# элементов и направлений фазирования. Возвращает отрицательное значение, т.к.
# мы ищем минимум, а чем больше КНД, тем лучше
def goal_func_directivity(
    coord: npt.NDArray,
    freq: float,
    angles: list[tuple[float, float]],
    theta_mesh: npt.NDArray,
    phi_mesh: npt.NDArray,
) -> float:
    max_val = np.inf
    for theta_0, phi_0 in angles:
        F = calc_F_theta_phi(coord, freq, theta_0, phi_0, theta_mesh, phi_mesh)
        directivity = get_directivity_uv(F, u_mesh, v_mesh, step_u, step_v)
        if directivity < max_val:
            max_val = directivity
    return -max_val


if __name__ == "__main__":
    # np.random.seed(100)
    freq = 10e9  # Частота в Гц (10 ГГц)
    wavelength = c / freq  # Длина волны в метрах

    # Список направлений фазирования (theta, phi) в градусах
    angles = [(45.0, 0.0)]

    # Максимальное количество итераций при оптимизации
    max_iter = 200

    # Количество элементов в антенной решетке
    x_count = 14
    y_count = 14
    element_count = x_count * y_count

    # Размеры антенной решетки в метрах
    size_x = 300e-3
    size_y = 300e-3

    # Минимальный зазор между элементами (в долях длины волны)
    min_gap_x = 0.35 * wavelength
    min_gap_y = 0.35 * wavelength

    # Начальный шаг случайного смещения элементов
    max_shift_x = 2 * wavelength
    max_shift_y = 2 * wavelength

    # Минимальный шаг случайного смещения элементов (критерий останова оптимизации)
    max_shift_stop = 0.165e-3

    # Коэффициент сжатия максимального смещения (уменьшает шаг на каждой итерации)
    max_shift_change = 0.965

    # Имя файла с координатами лучшего решения
    coord_fname_out = "example_main_optimize_random_out_14x14.tsv"

    # Шаги для равномерной сетки (расстояние между элементами)
    dx = size_x / x_count
    dy = size_y / y_count

    # Вывод параметров оптимизации
    print(f"Частота: {freq / 1e9} ГГц")
    print(f"Длина волны: {wavelength / 1e-3} мм")
    print(f"dx: {dx / 1e-3}, dy: {dy / 1e-3} мм")
    print(f"Размеры: {size_x / 1e-3} x {size_y / 1e-3} мм")
    print(f"Минимальный зазор: {min_gap_x / 1e-3} x {min_gap_y / 1e-3} мм")
    print(f"Число элементов: {element_count}")
    print()

    # Параметры сетки для расчета диаграммы направленности в координатах (u,v)
    u_min = -1
    u_max = 1
    v_min = -1
    v_max = 1
    step_u = 0.02
    step_v = 0.02

    u = np.arange(u_min, u_max + step_u, step_u)
    v = np.arange(v_min, v_max + step_v, step_v)

    # Создание сетки координат (u,v)
    u_mesh, v_mesh = np.meshgrid(u, v)
    uv2 = u_mesh**2 + v_mesh**2
    # Ограничение области видимости (u^2 + v^2 <= 1)
    uv2[uv2 > 1] = np.nan

    # Преобразование координат (u,v) в (theta, phi)
    theta_mesh = np.rad2deg(np.arcsin(np.sqrt(uv2)))
    phi_mesh = np.rad2deg(np.arctan2(v_mesh, u_mesh))

    # Оптимизация методом случайного поиска
    best_goal = np.inf
    best_coord = None
    n = 0
    # Цикл оптимизации: продолжается до достижения max_iter или пока шаг не станет меньше max_shift_stop
    while (
        n < max_iter and max_shift_x > max_shift_stop and max_shift_y > max_shift_stop
    ):
        # Генерация новой конфигурации антенной решетки
        if best_coord is None:
            # Первая итерация: создаем случайную АР с симметрией и минимальным зазором
            coord = rect_random_symmentry_with_gap(
                x_count, y_count, dx, dy, dx / 2, dy / 2, min_gap_x, min_gap_y
            )
        else:
            # Последующие итерации: используем лучшее решение как основу и вносим случайные изменения
            coord = rect_random_symmentry_with_gap_with_initial(
                best_coord[: best_coord.shape[0] // 4, :].copy(),
                x_count,
                y_count,
                dx,
                dy,
                max_shift_x,
                max_shift_y,
                min_gap_x,
                min_gap_y,
            )

        # Вычисление ЦФ (УБЛ)
        # goal = goal_func_side_lobe(coord, freq, angles, theta_mesh, phi_mesh)
        # Альтернативная ЦФ (КНД)
        goal = goal_func_directivity(coord, freq, angles, theta_mesh, phi_mesh)
        
        print(
            f"Итерация {n:03g}:    {best_goal:.5f}    ({goal:.5f})    max_shift = {max_shift_x / 1e-3:.5f} мм"
        )
        
        # Если текущее решение лучше предыдущего, сохраняем его
        if goal < best_goal:
            best_goal = goal
            best_coord = coord

        # Уменьшаем шаг случайного смещения для следующей итерации
        max_shift_x *= max_shift_change
        max_shift_y *= max_shift_change
        n += 1

    # Сохранение лучшего решения в TSV-файл
    x = best_coord[:, 0]
    y = best_coord[:, 1]
    z = np.zeros_like(x)
    mag = np.ones_like(x)  # Амплитуды элементов (одинаковые)
    phase = np.zeros_like(x)  # Фазы элементов (нулевые, т.к. фазирование учитывается в calc_F_theta_phi)
    writeToTSV(coord_fname_out, x, y, z, mag, phase, freq)

    # Визуализация результатов для лучшего решения
    print(f"Число элементов: {best_coord.shape[0]}")
    for theta_0, phi_0 in angles:
        print(f"theta_0={theta_0}, phi_0={phi_0}")
        # Расчет диаграммы направленности
        F = calc_F_theta_phi(best_coord, freq, theta_0, phi_0, theta_mesh, phi_mesh)
        F_dB = 10 * np.log10(F)

        # Расчет КНД
        directivity = get_directivity_uv(F, u_mesh, v_mesh, step_u, step_v)
        directivity_dB = 10 * np.log10(directivity)

        print(f"    КНД: {directivity_dB} дБ")

        # Расчет УБЛ
        side_lobes = get_side_lobes(F_dB)
        first_lobe = side_lobes[1][2]
        print(f"    Первый боковой лепесток: {first_lobe} дБ")

        F_dB[F_dB < -50] = -50

        fig = plt.figure()
        axes_uv = fig.add_subplot(1, 1, 1, projection="3d")
        plot_surf_uv(u_mesh, v_mesh, F_dB, axes_uv)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        # fig = plt.figure()
        # axes_theta_phi = fig.add_subplot(1, 1, 1, projection="3d")
        # plot_surf_theta_phi(theta_mesh, phi_mesh, F_dB, axes_theta_phi)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)

    plot_array(best_coord)
    plt.show()
