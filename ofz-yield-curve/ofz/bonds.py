"""Облигация с фиксированным купоном: денежные потоки, цена, доходность, риск-метрики.

Конвенции — как у Мосбиржи для ОФЗ:

* срок до платежа считается как ACT/365: t = (дата платежа − дата расчётов) / 365;
* доходность эффективная годовая: P = Σ CF_i / (1 + y)^t_i, где P — «грязная» цена
  (чистая цена + НКД) в рублях;
* расчёты по сделкам T+1, т.е. на следующий рабочий день после даты торгов.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

DAYS_IN_YEAR = 365.0
BP = 1e-4  # один базисный пункт


def settlement_date(trade_date) -> np.datetime64:
    """Дата расчётов T+1. Праздники РФ не учитываются — только выходные."""
    return np.busday_offset(np.datetime64(pd.Timestamp(trade_date).date(), "D"), 1, roll="forward")


@dataclass(frozen=True)
class Bond:
    secid: str
    dates: np.ndarray      # даты платежей, datetime64[D]
    starts: np.ndarray     # начало купонного периода для каждого платежа
    coupons: np.ndarray    # купон, руб. на одну облигацию
    principal: np.ndarray  # погашение номинала, руб.

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Bond":
        df = df.sort_values("date")
        return cls(
            secid=df["secid"].iloc[0],
            dates=pd.to_datetime(df["date"]).values.astype("datetime64[D]"),
            starts=pd.to_datetime(df["start"]).values.astype("datetime64[D]"),
            coupons=df["coupon"].to_numpy(float),
            principal=df["principal"].to_numpy(float),
        )

    @property
    def name(self) -> str:
        return f"ОФЗ {self.secid[2:7]}"

    @property
    def maturity(self) -> np.datetime64:
        return self.dates[-1]

    @property
    def face(self) -> float:
        return float(self.principal.sum())

    @property
    def coupon_rate(self) -> float:
        """Годовая ставка купона по последнему периоду, доля номинала."""
        days = (self.dates[-1] - self.starts[-1]).astype(int)
        return self.coupons[-1] / self.face * DAYS_IN_YEAR / days

    def cashflows(self, settle) -> tuple[np.ndarray, np.ndarray]:
        """Будущие платежи (строго после даты расчётов): сроки в годах и суммы в рублях.

        Купон, дата выплаты которого совпадает с датой расчётов, получает продавец.
        """
        settle = np.datetime64(settle, "D")
        mask = self.dates > settle
        t = (self.dates[mask] - settle).astype(int) / DAYS_IN_YEAR
        return t, (self.coupons + self.principal)[mask]

    def accrued(self, settle) -> float:
        """НКД на дату расчётов, руб. (линейно по дням, округление до копеек как у Мосбиржи)."""
        settle = np.datetime64(settle, "D")
        i = int(np.searchsorted(self.dates, settle, side="right"))
        if i >= len(self.dates):
            return 0.0
        period = (self.dates[i] - self.starts[i]).astype(int)
        elapsed = (settle - self.starts[i]).astype(int)
        return round(self.coupons[i] * max(elapsed, 0) / period, 2)


# --- Функции от потоков (t, cf): работают для любой облигации без опционов ------------

def price_from_yield(t: np.ndarray, cf: np.ndarray, y: float) -> float:
    """Грязная цена по эффективной годовой доходности y."""
    return float(np.sum(cf * (1.0 + y) ** (-t)))


def yield_to_maturity(t: np.ndarray, cf: np.ndarray, dirty_price: float) -> float:
    """Доходность к погашению: корень уравнения P(y) = рыночная грязная цена."""
    return brentq(lambda y: price_from_yield(t, cf, y) - dirty_price, -0.9, 10.0, xtol=1e-12)


def risk_measures(t: np.ndarray, cf: np.ndarray, y: float) -> dict:
    """Аналитические риск-метрики при доходности y (эффективная годовая).

    * дюрация Маколея  D = Σ t·PV_i / P;
    * модифицированная дюрация  D_mod = D / (1 + y) = −(1/P)·dP/dy;
    * выпуклость  C = (1/P)·d²P/dy² = Σ t(t+1)·CF·(1+y)^(−t−2) / P;
    * DV01 = D_mod·P·0.0001 — сколько рублей теряет одна бумага при росте доходности на 1 б.п.
    """
    disc = (1.0 + y) ** (-t)
    price = float(np.sum(cf * disc))
    macaulay = float(np.sum(t * cf * disc)) / price
    modified = macaulay / (1.0 + y)
    convexity = float(np.sum(t * (t + 1) * cf * disc)) / ((1.0 + y) ** 2 * price)
    return {
        "price": price,
        "macaulay": macaulay,
        "modified": modified,
        "convexity": convexity,
        "dv01": modified * price * BP,
    }


def dv01_numeric(t: np.ndarray, cf: np.ndarray, y: float, bump: float = BP) -> float:
    """DV01 центральной разностью: (P(y−1бп) − P(y+1бп)) / 2. Проверка аналитической формулы."""
    return (price_from_yield(t, cf, y - bump) - price_from_yield(t, cf, y + bump)) / 2.0
