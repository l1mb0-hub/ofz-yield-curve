"""Кривая бескупонной доходности: модель Нельсона-Сигеля, её подгонка и КБД Мосбиржи.

Все ставки — эффективные годовые в долях (0.15 = 15 %), как котирует рынок ОФЗ.
Дисконт-фактор для срока t лет: DF(t) = (1 + z(t))^(−t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .bonds import risk_measures, yield_to_maturity

# Стартовые значения τ для мультистарта: по τ целевая функция невыпуклая.
TAU_GRID = (0.25, 0.5, 1.0, 2.0, 3.0, 4.5)
# Границы параметров. τ ≤ 5 лет — не косметика: при большом τ модель вырождается
# (β0 уходит в минус, β1 и β2 гасят друг друга), внутри диапазона данных кривая
# почти та же, но параметры теряют смысл, а экстраполяция на длинный конец ломается.
#                 β0    β1    β2    τ
LOWER = np.array([0.0, -0.5, -0.5, 0.1])
UPPER = np.array([0.5, 0.5, 0.5, 5.0])


def ns_loadings(t, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Нагрузки факторов НС: наклон (1−e^−x)/x и кривизна (1−e^−x)/x − e^−x, x = t/τ."""
    x = np.maximum(np.asarray(t, float), 1e-10) / tau
    slope = -np.expm1(-x) / x
    return slope, slope - np.exp(-x)


def ns_rate(t, beta0: float, beta1: float, beta2: float, tau: float) -> np.ndarray:
    """z(t) = β0 + β1·(1−e^−x)/x + β2·[(1−e^−x)/x − e^−x].

    β0 — уровень длинных ставок, β0+β1 — ставка на сверхкоротком конце,
    β2 — «горб» (кривизна), τ — срок, около которого горб находится.
    """
    slope, curv = ns_loadings(t, tau)
    return beta0 + beta1 * slope + beta2 * curv


class ZeroCurve:
    """Общий интерфейс: rate(t) — эффективная бескупонная ставка, discount(t) — DF."""

    def rate(self, t) -> np.ndarray:
        raise NotImplementedError

    def discount(self, t, shift=0.0) -> np.ndarray:
        t = np.asarray(t, float)
        return (1.0 + self.rate(t) + shift) ** (-t)


@dataclass(frozen=True)
class NelsonSiegel(ZeroCurve):
    beta0: float
    beta1: float
    beta2: float
    tau: float

    @property
    def params(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2, self.tau])

    def rate(self, t) -> np.ndarray:
        return ns_rate(t, *self.params)


class KBDCurve(ZeroCurve):
    """КБД Мосбиржи по опубликованным параметрам.

    Методика биржи: G(t) = НС(t) + Σ g_i·exp(−(t−a_i)²/b_i²) в базисных пунктах
    при непрерывном начислении, где a_1=0, a_2=0.6, a_{i+1}=a_i+a_2·k^(i−1),
    b_1=a_2, b_{i+1}=b_i·k, k=1.6. Эффективная ставка: Y(t) = exp(G(t)/10000) − 1.
    """

    K = 1.6
    A = np.cumsum([0.0, 0.6] + [0.6 * 1.6 ** (i - 1) for i in range(2, 9)])
    B = 0.6 * 1.6 ** np.arange(9)

    def __init__(self, params: pd.Series):
        self.params = params
        self.ns = np.array([params["B1"], params["B2"], params["B3"], params["T1"]], float)
        self.g = np.array([params[f"G{i}"] for i in range(1, 10)], float)

    def g_bp(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        bumps = np.exp(-((t[..., None] - self.A) ** 2) / self.B**2) @ self.g
        return ns_rate(t, *self.ns) + bumps

    def rate(self, t) -> np.ndarray:
        return np.expm1(self.g_bp(t) / 1e4)


# --- Подгонка -------------------------------------------------------------------------

def _ols_betas(t, y, tau, w) -> np.ndarray:
    """При фиксированном τ модель линейна по β — решаем МНК точно."""
    slope, curv = ns_loadings(t, tau)
    X = np.column_stack([np.ones_like(slope), slope, curv])
    return np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]


def _start(beta: np.ndarray, tau: float) -> np.ndarray:
    eps = 1e-6
    return np.clip(np.r_[beta, tau], LOWER + eps, UPPER - eps)


def fit_ns_yields(t, y, weights=None) -> NelsonSiegel:
    """Шаг «в лоб»: НС прямо по точкам (срок до погашения, доходность к погашению).

    Получается гладкая кривая YTM. Это не бескупонная кривая: доходность купонной
    облигации — это «среднее» бескупонных ставок по всем её платежам.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    w = np.ones_like(t) if weights is None else np.asarray(weights, float)

    def residuals(p):
        return w * (ns_rate(t, *p) - y)

    fits = [
        least_squares(residuals, _start(_ols_betas(t, y, tau, w), tau), bounds=(LOWER, UPPER))
        for tau in TAU_GRID
    ]
    return NelsonSiegel(*min(fits, key=lambda r: r.cost).x)


@dataclass
class PriceFit:
    curve: NelsonSiegel
    model_price: np.ndarray   # грязные цены по кривой, руб.
    model_ytm: np.ndarray     # доходность, соответствующая модельной цене
    error_bp: np.ndarray      # модельная YTM − рыночная YTM, б.п.


def fit_ns_prices(flows, dirty, ytm, maturity, robust: bool = False) -> PriceFit:
    """Правильный способ: НС задаёт бескупонную кривую, каждую ОФЗ дисконтируем по ней.

    flows    — список (t, cf) будущих платежей каждой облигации;
    dirty    — рыночные грязные цены, руб.;
    ytm      — рыночные доходности (для весов и стартовой точки);
    maturity — сроки до погашения, лет (для стартовой точки).

    Невязка i-й бумаги: (P_модель − P_рынок) / (P_рынок · D_mod) — это ошибка
    в доходности, поэтому длинные бумаги не «перевешивают» короткие.
    soft_l1 (robust=True) ограничивает влияние отдельных выбросов.
    """
    dirty, ytm = np.asarray(dirty, float), np.asarray(ytm, float)
    dmod = np.array([risk_measures(t, cf, y)["modified"] for (t, cf), y in zip(flows, ytm)])

    def model_prices(p) -> np.ndarray:
        curve = NelsonSiegel(*p)
        return np.array([cf @ curve.discount(t) for t, cf in flows])

    def residuals(p):
        return (model_prices(p) - dirty) / (dirty * dmod)

    ones = np.ones_like(ytm)
    starts = [fit_ns_yields(maturity, ytm).params]
    starts += [_start(_ols_betas(np.asarray(maturity, float), ytm, tau, ones), tau) for tau in TAU_GRID]
    kwargs = {"loss": "soft_l1", "f_scale": 0.001} if robust else {}
    fits = [least_squares(residuals, _start(x0[:3], x0[3]), bounds=(LOWER, UPPER), **kwargs) for x0 in starts]
    best = min(fits, key=lambda r: r.cost)

    curve = NelsonSiegel(*best.x)
    prices = model_prices(best.x)
    model_ytm = np.array([yield_to_maturity(t, cf, p) for (t, cf), p in zip(flows, prices)])
    return PriceFit(curve, prices, model_ytm, (model_ytm - ytm) * 1e4)
