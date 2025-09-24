import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from tools import (
    plot_surf_theta_phi,
    plot_slice,
    get_side_lobes,
    calc_F_theta_phi,
    plot_array,
    get_directivity_theta_phi,
)
from arraygrid import (
    rect_random,
    rect_regular,
    rect_random_symmentry,
    load_from_tsv
)


if __name__ == "__main__":
    np.random.seed(100)
    freq = 10e9
    wavelength = c / freq

    coord = rect_regular(14, 14, 0.7 * wavelength, 0.7 * wavelength)
    # coord = rect_random(20, 10, 0.7 * wavelength, 0.7 * wavelength, 0.3 * wavelength, 0.3 * wavelength)
    # coord = rect_random_symmentry(
    #    20, 10, 0.7 * wavelength, 0.7 * wavelength, 0.3 * wavelength, 0.3 * wavelength
    # )
    # coord = np.array([0.0, 0.0], ndmin=2)
    # coord = load_from_tsv("main_uv_out_18x18_rect.tsv")
    # print(coord.shape)

    theta_0 = 45.0
    phi_0 = 0.0

    theta_min = 0
    theta_max = 90
    phi_min = 0
    phi_max = 360
    step_theta = 0.5
    step_phi = 0.5

    # theta = np.arange(0, 90 + step_theta, step_theta)
    # phi = np.arange(0, 360, step_phi)
    theta = np.arange(theta_min, theta_max + step_theta, step_theta)
    phi = np.arange(phi_min, phi_max, step_phi)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)

    F = calc_F_theta_phi(coord, freq, theta_0, phi_0, theta_mesh, phi_mesh)
    F_dB = 10 * np.log10(F)
    F_dB[F_dB < -50] = -50

    directivity = get_directivity_theta_phi(F, theta_mesh, step_theta, step_phi)
    directivity_dB = 10 * np.log10(directivity)

    print(f"КНД: {directivity}")
    print(f"КНД: {directivity_dB} дБ")

    phi_slice = phi_0
    F_slice = F_dB[int(phi_slice / step_phi), :]
    angle_slice = theta_mesh[0, :]

    side_lobes = get_side_lobes(F_dB)
    main_lobe = side_lobes[0][2]
    for phi_val, theta_val, val in side_lobes[:10]:
        print(
            f"theta={theta_min + theta_val * step_theta}, phi={phi_min + phi_val * step_phi }, val={val - main_lobe} дБ"
        )

    plot_array(coord)

    fig = plt.figure()
    axes_theta_phi = fig.add_subplot(1, 1, 1, projection="3d")
    plot_surf_theta_phi(theta_mesh, phi_mesh, F_dB, axes_theta_phi)

    plot_slice(angle_slice, F_slice)

    plt.show()
