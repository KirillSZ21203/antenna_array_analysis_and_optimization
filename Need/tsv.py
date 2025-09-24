# -*- coding: utf-8 -*-
"""
Модуль для работы с файлами в формате TSV из Antenna Magus
"""

from datetime import datetime

import numpy as np


def writeToTSV(fname_out, x, y, z, mag, phase, freq):
    """Записать данные в формате tsv.

    fname_out - имя выходного файла.
    x, y, z - списки координат излучателей.
    mag - список амплитуд.
    phase - список фаз.
    freq - частота.
    """
    date_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p")

    header = """# Created by Antenna Magus 5.4.0.1919
# On {date}
# unit: meters
# design frequency: {freq:.0f} Hz
# Element	X 	Y 	Z 	Magnitude	Phase	Phi	Theta	Gamma""".format(
        date=date_str, freq=freq
    )

    numbers = np.arange(1, len(x) + 1)
    zeros = np.zeros(len(x))

    data = np.array([numbers, x, y, z, mag, phase, zeros, zeros, zeros]).T
    np.savetxt(fname_out, data, delimiter="\t", fmt="%g", header=header, comments="")
