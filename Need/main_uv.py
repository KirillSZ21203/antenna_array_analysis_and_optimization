import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from tools import (
    plot_array,
    plot_surf_uv,
    get_side_lobes,
    calc_F_theta_phi,
    get_directivity_uv,
)
from arraygrid import (
    rect_random,
    rect_regular,
    rect_random_symmentry,
    random_symmetry_size,
    rect_random_symmentry_with_gap,
    load_from_tsv,
)
from tsv import writeToTSV


if __name__ == "__main__":
    # np.random.seed(100)
    freq = 10e9  # Частота
    wavelength = c / freq  # Длина волны в метрах
    size_x = 300e-3  
    size_y = 300e-3  # Размеры АР
    # dx = 24.0e-3
    # dy = 24.0e-3
    dx = 17.57e-3  
    dy = 17.57e-3  # Шаги сетки
    theta_0 = 45.0  # Угол места направления фазирования 
    phi_0 = 0.0    # Азимутальный угол направления фазирования

    # Расчет количества элементов по осям X и Y
    x_count = int(size_x / dx)
    y_count = int(size_y / dy)

    coord = rect_regular(14, 14, 0.7 * wavelength, 0.7 * wavelength)
    # coord = rect_random(8, 8, 1.5 * wavelength, 1.5 * wavelength, 0.3 * wavelength, 0.3 * wavelength)
    # coord = rect_random_symmentry(
    #     10, 10, 0.5 * wavelength, 0.5 * wavelength, 0.0 * wavelength, 0.0 * wavelength
    # )
    # coord = np.array([0.0, 0.0], ndmin=2)
    # coord = random_symmetry_size(100, 0.3, 0.3, 0.01, 0.01)

    # Создание АР со случайным расположением элементов и симметрией
    # coord = rect_random_symmentry(
    #    x_count, y_count, dx, dy, 0 * wavelength, 0 * wavelength
    #)

    coord = rect_random_symmentry_with_gap(
        8, 8, 1.5 * wavelength, 1.5 * wavelength, 
        0.3 * wavelength, 0.3 * wavelength,
        0.2 * wavelength, 0.2 * wavelength
    )
    # coord = load_from_tsv("main_optimize_random_out.tsv")

    # Сохранение координат элементов в TSV-файл
    writeToTSV(
        "main_uv_out2.tsv",
        coord[:, 0],  # Координаты X
        coord[:, 1],  # Координаты Y
        np.zeros(coord.shape[0]),  # Координаты Z (все нули для плоской ФАР)
        np.ones(coord.shape[0]),   # Амплитудное распределение
        np.zeros(coord.shape[0]),  # Фазовое распределение
        freq,  # Частота
    )

    # Вывод параметров антенной решетки
    print(f"Частота: {freq / 1e9} ГГц")
    print(f"Длина волны: {wavelength / 1e-3} мм")
    print(f"Число элементов: {coord.shape[0]}")
    print(f"Направление фазирования: theta={theta_0} phi={phi_0} град.")

    # Параметры сетки для расчета диаграммы направленности в координатах (u,v)
    u_min = -1
    u_max = 1
    v_min = -1
    v_max = 1
    step_u = 0.01
    step_v = 0.01

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

    # Расчет ДН
    F = calc_F_theta_phi(coord, freq, theta_0, phi_0, theta_mesh, phi_mesh)
    F_dB = 10 * np.log10(F)

    # Расчет КНД
    directivity = get_directivity_uv(F, u_mesh, v_mesh, step_u, step_v)
    directivity_dB = 10 * np.log10(directivity)

    # Расчет УБЛ
    side_lobes = get_side_lobes(F_dB)
    first_lobe = side_lobes[1][2]  # Уровень первого БЛ
    print(f"КНД: {directivity_dB} дБ")
    print(f"УБЛ: {first_lobe} дБ")

    # Визуализация расположения элементов антенной решетки
    plot_array(coord)
    
    # Ограничение минимального значения для визуализации
    F_dB[F_dB < -50] = -50


    # Построение графика ДН
    fig = plt.figure()
    axes_uv = fig.add_subplot(1, 1, 1, projection="3d")
    plot_surf_uv(u_mesh, v_mesh, F_dB, axes_uv)

    plt.show()
