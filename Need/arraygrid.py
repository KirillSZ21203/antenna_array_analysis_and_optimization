import numpy as np
import numpy.typing as npt


def rect_regular(nx: int, ny: int, dx: float, dy: float) -> npt.NDArray:
    """Создает прямоугольную регулярную сетку.

    nx - количество элементов по оси X
    ny - количество элементов по оси Y
    dx - шаг по оси X
    dy - шаг по оси Y
    """
    x = np.arange(nx) * dx - (nx - 1) * dx / 2
    y = np.arange(ny) * dy - (ny - 1) * dy / 2
    return np.array([(x_i, y_i) for x_i in x for y_i in y])


def rect_random(
    nx: int, ny: int, dx: float, dy: float, max_shift_x: float, max_shift_y: float
) -> npt.NDArray:
    """Создает прямоугольную сетку на основе прямоугольной, добавляя случайные сдвиги.
    Не используется симметрия."""
    array = rect_regular(nx, ny, dx, dy)
    shift_x = np.random.uniform(-max_shift_x, max_shift_x, array.shape[0])
    shift_y = np.random.uniform(-max_shift_y, max_shift_y, array.shape[0])
    array[:, 0] += shift_x
    array[:, 1] += shift_y
    return array


def create_symmetry(q1: npt.NDArray) -> npt.NDArray:
    """Создает полную антенную решетку по координатам излучателей одной четверти."""
    q2 = np.array([(-x_i, y_i) for x_i, y_i in q1])
    q3 = np.array([(-x_i, -y_i) for x_i, y_i in q1])
    q4 = np.array([(x_i, -y_i) for x_i, y_i in q1])
    return np.concatenate((q1, q2, q3, q4))


def rect_random_symmentry(
    nx: int, ny: int, dx: float, dy: float, max_shift_x: float, max_shift_y: float
) -> npt.NDArray:
    """Создает прямоугольную сетку на основе прямоугольной, добавляя случайные сдвиги.
    Используется симметрия."""
    x = np.arange(nx // 2) * dx - (nx - 1) * dx / 2
    y = np.arange(ny // 2) * dy - (ny - 1) * dy / 2

    q1 = np.array([(x_i, y_i) for x_i in x for y_i in y])
    shift_x = np.random.uniform(-max_shift_x, max_shift_x, q1.shape[0])
    shift_y = np.random.uniform(-max_shift_y, max_shift_y, q1.shape[0])

    q1[:, 0] += shift_x
    q1[:, 1] += shift_y

    return create_symmetry(q1)


def rect_random_symmentry_with_gap(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    max_shift_x: float,
    max_shift_y: float,
    min_gap_x: float,
    min_gap_y: float,
) -> npt.NDArray:
    """Создает прямоугольную сетку на основе прямоугольной, добавляя случайные сдвиги.
    Используется симметрия. Используется коррекция координат слишком близко расположенных элементов.
    """
    x = np.arange(nx // 2) * dx - (nx - 1) * dx / 2
    y = np.arange(ny // 2) * dy - (ny - 1) * dy / 2

    q1 = np.array([(x_i, y_i) for x_i in x for y_i in y])
    shift_x = np.random.uniform(-max_shift_x, max_shift_x, q1.shape[0])
    shift_y = np.random.uniform(-max_shift_y, max_shift_y, q1.shape[0])

    q1[:, 0] += shift_x
    q1[:, 1] += shift_y

    _correct_min_dist(q1, min_gap_x, min_gap_y, x[0], x[-1], y[0], y[-1])

    return create_symmetry(q1)


def rect_random_symmentry_with_gap_with_initial(
    q1: npt.NDArray,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    max_shift_x: float,
    max_shift_y: float,
    min_gap_x: float,
    min_gap_y: float,
) -> npt.NDArray:
    """Создает прямоугольную сетку на основе заданного расположения излучателей в одной четверти, добавляя случайные сдвиги.
    Используется симметрия. Используется коррекция координат слишком близко расположенных элементов.
    """
    x = np.arange(nx // 2) * dx - (nx - 1) * dx / 2
    y = np.arange(ny // 2) * dy - (ny - 1) * dy / 2

    shift_x = np.random.uniform(-max_shift_x, max_shift_x, q1.shape[0])
    shift_y = np.random.uniform(-max_shift_y, max_shift_y, q1.shape[0])

    q1[:, 0] += shift_x
    q1[:, 1] += shift_y

    _correct_min_dist(q1, min_gap_x, min_gap_y, x[0], x[-1], y[0], y[-1])

    return create_symmetry(q1)


def random_symmetry_size(
    count: int, size_x: float, size_y: float, gap_x: float = 0, gap_y: float = 0
) -> npt.NDArray:
    """Создает случайную симметричную сетку."""
    x_min = -size_x / 2
    x_max = -gap_x / 2
    y_min = gap_y / 2
    y_max = size_y / 2

    q_count = count // 4

    x = np.random.uniform(x_min, x_max, q_count)
    y = np.random.uniform(y_min, y_max, q_count)
    q1 = np.column_stack((x, y))
    _correct_min_dist(q1, gap_x, gap_y, x_min, x_max, y_min, y_max)
    return create_symmetry(q1)


def load_from_tsv(filename: str) -> npt.NDArray:
    x, y = np.loadtxt(filename, unpack=True, usecols=(1, 2))
    return np.column_stack((x, y))


def _correct_min_dist(
    coord: npt.NDArray,
    min_gap_x: float,
    min_gap_y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    """Коррекция координат слишком близко расположенных элементов.
    Такие элементы передвигаются в случайное место внутри заданной области."""
    finished = False
    while not finished:
        finished = True
        for n in range(coord.shape[0]):
            if not finished:
                break

            x1, y1 = coord[n]
            if abs(x1) < min_gap_x / 2:
                coord[n][0] = np.random.uniform(x_min, x_max)
                finished = False
                break

            if abs(y1) < min_gap_y / 2:
                coord[n][1] = np.random.uniform(y_min, y_max)
                finished = False
                break

            if x1 < x_min:
                coord[n][0] = np.random.uniform(x_min, x_max)
                finished = False
                break

            if y1 < y_min:
                coord[n][1] = np.random.uniform(y_min, y_max)
                finished = False
                break

            for m in range(n + 1, coord.shape[0]):
                x2, y2 = coord[m]
                if abs(x1 - x2) < min_gap_x and abs(y1 - y2) < min_gap_y:
                    # print(f"Координаты элементов {n} и {m} слишком близки. Коррекция.")
                    coord[n] = (
                        np.random.uniform(x_min, x_max),
                        np.random.uniform(y_min, y_max),
                    )
                    finished = False
                    break
