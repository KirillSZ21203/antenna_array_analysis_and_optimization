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
    """Загрузка диаграммы направленности из файла CST"""
    data = np.loadtxt(filename, skiprows=2)
    theta = data[:, 0]  # Первый столбец — углы theta
    abs_dir = data[:, 2]  # Третий столбец — Abs(Dir.) [dBi]
    return theta, abs_dir

def calculate_array_pattern(coord_file, freq, theta_0, phi_0, phi_slice, theta_range):
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
    plt.title(f"Расположение элементов антенной решетки, содержащей ({coord.shape[0]} элементов)")
    
    return theta, F_dB, coord

if __name__ == "__main__":
    # Параметры для расчетной ДН
    coord_file = "main_uv_out_18x18_rect.tsv"
    freq = 10e9  # Частота в Гц (10 ГГц)
    
    # Направление фазирования
    theta_0 = 0.0
    phi_0 = 0.0     
    
    # Параметры для построения одномерной ДН
    phi_slice = 0.0  # Срез ДН
    theta_range = (-90.0, 90.0, 0.5)  # (min, max, step)
    
    # Расчет ДН антенной решетки
    theta_calc, F_dB_calc, coord = calculate_array_pattern(
        coord_file, freq, theta_0, phi_0, phi_slice, theta_range
    )
    
    # Загрузка ДН из CST
    cst_file = 'patch_for_array_rect_1_theta_0_phi_0.txt'
    theta_cst, abs_dir_cst = load_cst_pattern(cst_file)
    
    # Создаем фигуру для сравнительного графика
    plt.figure(figsize=(12, 8))
    
    
    # Вариант 1 - нормировка обеих диаграмм к их максимумам
    F_dB_calc_norm = F_dB_calc - np.max(F_dB_calc)
    abs_dir_cst_norm = abs_dir_cst - np.max(abs_dir_cst)
    
    # Построение нормированных диаграмм
    plt.plot(theta_calc, F_dB_calc_norm, 'k-', linewidth=2, label='ДН, полученная в ходе аналитического расчета')
    plt.plot(theta_cst, abs_dir_cst_norm, 'k-.', linewidth=2, label='ДН, полученная в программе электродинамического моделирования')
    

    # Вариант 2 - ненормирвоанные диаграммы
    # plt.plot(theta_calc, F_dB_calc, 'b-', linewidth=2, label='ДН из Python')
    # plt.plot(theta_cst, abs_dir_cst, 'r-', linewidth=2, label='ДН из CST')
    
    plt.xticks(np.arange(-90, 90 + 1, 10.0))
    plt.grid(True)
    plt.xlabel('Угол θ, град.', fontsize=12)
    plt.ylabel('Уровень, дБ', fontsize=12)
    plt.title(f'Сравнение диаграмм направленности при φ={phi_slice}°\n'
              f'(направление фазирования: θ₀={theta_0}°, φ₀={phi_0}°)', fontsize=14)
    plt.xlim([theta_range[0], theta_range[1]])
    plt.ylim([-50, 0])
    plt.legend()
    
    '''
    # Линия, указывающая направление фазирования
    if theta_range[0] <= theta_0 <= theta_range[1]:
        plt.axvline(x=theta_0, color='g', linestyle='--', 
                   label=f'Направление фазирования θ₀={theta_0}°')
    '''
                   
    plt.tight_layout()
    plt.show() 
