import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from tools import (
    plot_array,
    get_side_lobes,
    calc_F_theta_phi,
    get_directivity_uv,
)
from arraygrid import load_from_tsv

def load_cst_pattern(filename):
    """Загрузка диаграммы направленности из файла CST
    Для файлов patch_for_array_rect_1... (АР 18х18)
    """
    data = np.loadtxt(filename, skiprows=2)
    theta = data[:, 0]  # Первый столбец — углы theta
    abs_dir = data[:, 2]  # Третий столбец — Abs(Dir.) [dBi]
    return theta, abs_dir

def load_cst_pattern_v2(filename):
    """Загрузка диаграммы направленности из файла CST (версия 2)
    Для файлов patch_for_array_rect_2... (АР 14х14) 
    """
    data = np.loadtxt(filename, skiprows=3)  # Пропускаем 3 строки заголовка
    theta = data[:, 0]  # Первый столбец — углы theta
    abs_dir = data[:, 1]  # Второй столбец — значения в dBi
    return theta, abs_dir

def calculate_array_pattern(coord_file, freq, theta_0, phi_0, phi_slice, theta_range, normalize=False):
    """Расчет диаграммы направленности антенной решетки"""
    coord = load_from_tsv(coord_file)
    wavelength = c / freq
    
    theta_min, theta_max, step_theta = theta_range
    theta = np.arange(theta_min, theta_max + step_theta, step_theta)
    
    # Создание сетки для расчета ДН
    theta_mesh = np.tile(theta, (1, 1)).T
    phi_mesh = np.ones_like(theta_mesh) * phi_slice
    
    # Вывод параметров антенной решетки
    print(f"Частота: {freq / 1e9} ГГц")
    print(f"Длина волны: {wavelength / 1e-3} мм")
    print(f"Число элементов: {coord.shape[0]}")
    print(f"Направление фазирования: theta={theta_0} phi={phi_0} град.")
    print(f"Срез ДН при phi={phi_slice} град.")
    
    # Расчет ДН из координат элементов
    F = calc_F_theta_phi(coord, freq, theta_0, phi_0, theta_mesh, phi_mesh)
    
    # Поскольку calc_F_theta_phi уже не нормирует результат (строка закомментирована),
    # мы можем применить масштабирование, если нужно
    if not normalize:
        # Для ненормированной ДН умножаем на количество элементов в решетке
        # Это приблизительная оценка, может потребоваться корректировка
        F *= coord.shape[0]
    
    F_dB = 10 * np.log10(F)
    
    # Ограничение минимального значения для визуализации
    F_dB[F_dB < -50] = -50
    
    # Расчет УБЛ для расчетной ДН
    side_lobes = get_side_lobes(F_dB)
    if len(side_lobes) > 1:
        first_lobe = side_lobes[1][2] - side_lobes[0][2]  # Уровень первого БЛ относительно главного
        print(f"УБЛ расчетной ДН: {first_lobe} дБ")
    
    # Визуализация расположения элементов антенной решетки
    plt.figure(figsize=(10, 6))
    plot_array(coord)
    # plt.title(f"Расположение элементов антенной решетки, содержащей ({coord.shape[0]} элементов)")
    
    return theta, F_dB, coord

if __name__ == "__main__":
    # Параметры для расчетной ДН
    coord_file = "main_optimizeDir_random_out_14х14.tsv"
    # coord_file = "main_uv_out_18x18_rect.tsv"
    freq = 10e9  # Частота в Гц (10 ГГц)
    
    # Направление фазирования
    theta_0 = 45.0
    phi_0 = 0.0     
    
    # Параметры для построения одномерной ДН
    phi_slice = 0.0  # Срез ДН
    theta_range = (-90.0, 90.0, 0.5)  # (min, max, step)
    
    # Расчет ДН антенной решетки (ненормированной)
    theta_calc, F_dB_calc, coord = calculate_array_pattern(
        coord_file, freq, theta_0, phi_0, phi_slice, theta_range, normalize=False
    )
    
    # Загрузка ДН из CST (используем новый метод для нового файла)
    cst_file = 'patch_for_array_rect_2_theta_45_phi_0.txt'
    theta_cst, abs_dir_cst = load_cst_pattern_v2(cst_file)
    # theta_cst, abs_dir_cst = load_cst_pattern(cst_file)
    
    # Создаем фигуру для сравнительного графика
    plt.figure(figsize=(12, 8))
    
    # Построение ненормированных диаграмм
    # plt.plot(theta_calc, F_dB_calc, 'b-', linewidth=2, label='ДН из Python')
    plt.plot(theta_cst, abs_dir_cst, '--r', linewidth=2, label='Метод конечных элементов')
    
    # Смещение графика из python
    offset = 6  # Смещение в дБ
    plt.plot(theta_calc, F_dB_calc + offset, '-b', linewidth=2, label=f'Аналитический расчет')
    
    plt.xticks(np.arange(-90, 90 + 1, 10.0))
    plt.grid(True)
    plt.xlabel('θ, град.', fontsize=18)
    plt.ylabel('КНД, дБ', fontsize=18)
    # plt.title(f'Сравнение диаграмм направленности при φ={phi_slice}°\n'
    #           f'(направление фазирования: θ₀={theta_0}°, φ₀={phi_0}°)', fontsize=14)
    plt.xlim([theta_range[0], theta_range[1]])
    plt.ylim([-35, 35])  # если нужно ограничить диапазон по оси Y
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=18, handlelength=5, markerscale=3, 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    
    plt.tight_layout()
    plt.show() 
