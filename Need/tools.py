import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from arrayfactor import ArrayFactor


def calc_F_theta_phi(coord, freq, theta_0, phi_0, theta_mesh, phi_mesh):
    coord_x = coord[:, 0]
    coord_y = coord[:, 1]
    A0 = np.ones_like(coord_x)

    F = np.zeros_like(theta_mesh)
    phased_array = ArrayFactor(coord_x, coord_y, A0, phi_0, theta_0)
    theta_count = theta_mesh.shape[0]
    phi_count = theta_mesh.shape[1]

    for n_theta in range(theta_count):
        for n_phi in range(phi_count):
            current_theta = theta_mesh[n_theta, n_phi]
            current_phi = phi_mesh[n_theta, n_phi]
            if np.isnan(current_theta) or np.isnan(current_phi):
                F[n_theta, n_phi] = np.nan
                continue
            current_F = phased_array.getFactor(freq, current_phi, current_theta)
            F[n_theta, n_phi] = current_F

    F *= np.cos(np.deg2rad(theta_mesh))
    # F /= np.nanmax(F)
    return F


def get_side_lobes(F):
    result: list[tuple[int, int, np.float64]] = []
    for row in range(1, F.shape[0] - 1):
        for col in range(1, F.shape[1] - 1):
            mask = F[row - 1: row + 2, col - 1: col + 2]
            if not np.isnan(mask).any() and mask[1, 1] == np.max(mask):
                result.append((row, col, mask[1, 1]))

    result.sort(key=lambda x: x[2], reverse=True)
    return result


def plot_surf_theta_phi(theta_mesh, phi_mesh, F, axes):
    axes.plot_surface(theta_mesh, phi_mesh, F, 
                      cmap='jet', linewidth=0.5, edgecolors='k')
    axes.set_xlabel("Theta")
    axes.set_ylabel("Phi")
    axes.view_init(90, -90)


def plot_surf_uv(u_mesh, v_mesh, F, axes):
    axes.plot_surface(u_mesh, v_mesh, F, 
                      cmap='jet', linewidth=0.5, edgecolors='k', 
                      rcount=100, ccount=100)
    axes.set_xlabel("u")
    axes.set_ylabel("v")
    axes.set_zlabel("ДН (Дб)")
    axes.view_init(90, -90)


def plot_slice(angle, F):
    fig = plt.figure()
    axes = fig.add_subplot()
    axes.plot(angle, F)
    axes.set_ylim(bottom=-50)
    axes.grid()


def plot_array(array: npt.NDArray):
    fig = plt.figure()
    axes = fig.add_subplot()
    axes.scatter(array[:, 0], array[:, 1], s=42, c="k")
    axes.set_aspect('equal')
    axes.set_xlabel("X, м", fontsize=19)
    axes.set_ylabel("Y, м", fontsize=19)
    axes.tick_params(axis='both', which='major', labelsize=19)
    # Установка большего количества меток на оси X
    x_ticks = np.linspace(np.min(array[:, 0]), np.max(array[:, 0]), num=5)
    x_ticks = np.unique(np.concatenate(([0], x_ticks)))  # Добавление 0 и удаление дубликатов
    x_ticks = np.round(x_ticks, 2)
    axes.set_xticks(x_ticks)
    axes.grid(color='gray', alpha=0.42)


def get_directivity_uv(F_norm: npt.NDArray,
                       u_mesh: npt.NDArray,
                       v_mesh: npt.NDArray,
                       du: float,
                       dv: float) -> float:
    cos2_theta = 1 - u_mesh**2 - v_mesh**2
    cos2_theta[cos2_theta <= 0] = np.nan
    tmp = F_norm / np.sqrt(cos2_theta)
    return 4 * np.pi / (np.nansum(tmp) * du * dv)



def get_directivity_theta_phi(F_norm: npt.NDArray,
                              theta_mesh_deg: npt.NDArray,
                              step_theta_deg: float,
                              step_phi_deg: float) -> float:
    theta_mesh_rad = np.deg2rad(theta_mesh_deg)
    sin_theta = np.sin(theta_mesh_rad)
    tmp = np.sum(F_norm * sin_theta) * (np.deg2rad(step_theta_deg) * np.deg2rad(step_phi_deg))
    return 4 * np.pi / tmp
