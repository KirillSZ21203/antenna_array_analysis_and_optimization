#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Версия 1.2.2 (28.10.2024)
    * Переписывание устаревшего кода

Версия 1.2.1 (08.02.2019)
    * Форматирование кода

Версия 1.2 (22.01.2016)
    * Исправление ошибок

Версия 1.1 (13.01.2016)
    * Добавлен метод getPhase

Версия 1.2 (14.01.2016)
    * ArrayFactor разделен на несколько классов
"""

__version__ = "1.2.2"


from abc import ABCMeta, abstractmethod

import numpy as np


class BaseArrayCalculator:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.c0 = 299792458.0


class PhaseCalculator(BaseArrayCalculator):
    """Класс для расчета фаз в излучателях при сканировании."""

    def __init__(self, x, y):
        super().__init__(x, y)

    def getPhase(self, freq, phi, theta):
        """
        Возвращает фазу элементов в радианах.
        phi, theta - направление фазирования решетки (в градусах)
        """
        k = 2.0 * np.pi * freq / self.c0
        phi_rad = np.radians(phi)
        theta_rad = np.radians(theta)

        phase = (
            -k
            * (self.x * np.cos(phi_rad) + self.y * np.sin(phi_rad))
            * np.sin(theta_rad)
        )
        return phase


class BaseArrayFactor(BaseArrayCalculator, metaclass=ABCMeta):
    """Базовый класс для расчета ДН."""

    def __init__(self, x, y, A0):
        super().__init__(x, y)
        self.A0 = np.array(A0)

    @abstractmethod
    def getPhase(self, freq):
        """Возвращает фазу элементов в радианах."""
        pass

    def getNormFactor(self):
        """Возвращает число, на которое будет нормироваться поле в
        дальней зоне."""
        return 1.0

    def _get_a(self, phi, theta):
        """
        Расчет проекции расстояния на ось, задаваемую (phi, theta).
        Углы задаются в градусах.
        """
        R = np.sqrt(self.x**2 + self.y**2)
        alpha = np.radians(phi) - np.arctan2(self.y, self.x)
        R1 = R * np.cos(alpha)
        a = R1 * np.sin(np.radians(theta))
        return a

    def getFactor(self, freq, phi, theta):
        """
        Возвращает значение множителя решетки в направлении
        phi, theta (углы в градусах)
        freq - частота в Гц
        """
        k = 2.0 * np.pi * freq / self.c0

        # Для расчета направления phi, theta
        a = self._get_a(phi, theta)

        phase = self.getPhase(freq)

        # АФР
        A = self.A0 * np.exp(1j * phase)

        # На что нормировать поле в дальней зоне
        norm = self.getNormFactor()

        f = np.sum(A * np.exp(1j * k * a)) / norm
        F0 = np.abs(f) ** 2

        return F0


class ArrayFactor(BaseArrayFactor):
    """Класс для расчета множителя решетки при фазировании при заданном
    угле отклонения"""

    def __init__(self, x, y, A0, phi0, theta0):
        """
        x - список координат X излучателей
        y - список координат Y излучателей
        A0 - амплитуды в излучателях
        phi0, theta0 - направление фазирования решетки (в градусах)
        """
        super().__init__(x, y, A0)
        self.phi0 = phi0
        self.theta0 = theta0

        self._phaseCalculator = PhaseCalculator(x, y)

    def getPhase(self, freq):
        """Возвращает фазу элементов в радианах."""
        return self._phaseCalculator.getPhase(freq, self.phi0, self.theta0)

    def getNormFactor(self):
        """Возвращает число, на которое будет нормироваться поле
        в дальней зоне."""
        return np.sum(self.A0)


class ArrayFactorDirect(BaseArrayFactor):
    """Класс для расчета множителя решетки при фазировании при заданном АФР."""

    def __init__(self, x, y, A0, phase_list):
        """
        x - список координат X излучателей
        y - список координат Y излучателей
        A0 - амплитуды в излучателях
        phase_list - фазы в излучателях (в градусах)
        """
        super().__init__(x, y, A0)
        self._phase = np.radians(phase_list)

    def getPhase(self, freq):
        """Возвращает фазу элементов в радианах."""
        return self._phase

    def getNormFactor(self):
        """Возвращает число, на которое будет нормироваться поле
        в дальней зоне."""
        return np.sum(self.A0)
