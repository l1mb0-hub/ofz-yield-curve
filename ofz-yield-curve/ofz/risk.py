"""Риск портфеля ОФЗ: DV01, сценарии сдвига кривой, разложение по ключевым срокам."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .bonds import BP, yield_to_maturity
from .curve import ZeroCurve
from .market import MarketSnapshot

Shift = Callable[[np.ndarray], np.ndarray]  # срок (лет) → сдвиг ставки (доли)


def parallel(bp: float) -> Shift:
    return lambda t: np.full_like(np.asarray(t, float), bp * BP)


def twist(short_bp: float, long_bp: float, t_short: float = 1.0, t_long: float = 10.0) -> Shift:
    """Изменение наклона: short_bp до t_short лет, long_bp после t_long, между — линейно."""
    def shift(t):
        w = np.clip((np.asarray(t, float) - t_short) / (t_long - t_short), 0.0, 1.0)
        return (short_bp + (long_bp - short_bp) * w) * BP
    return shift


SCENARIOS: dict[str, Shift] = {
    "+100 б.п. параллельно": parallel(+100),
    "−100 б.п. параллельно": parallel(-100),
    "Рост наклона": twist(-50, +50),         # 1 г.: −50 б.п., 10 л. и дальше: +50 б.п.
    "Снижение наклона": twist(+50, -50),     # 1 г.: +50, 10 л.: −50
    "ЦБ ужесточает": twist(+200, +50),       # короткие +200, длинные +50 (как осенью 2024)
    "ЦБ смягчает": twist(-200, -50),         # короткие −200, длинные −50
}

KEY_TENORS = (0.5, 1, 2, 3, 5, 7, 10, 15)


def z_spread(t, cf, dirty: float, curve: ZeroCurve) -> float:
    """Постоянная добавка s к бескупонной кривой, при которой модель даёт рыночную цену.

    Кривая НС не проходит точно через каждую бумагу. Чтобы при нулевом сдвиге
    переоценка давала ровно рыночную цену, держим z-спред бумаги постоянным.
    """
    return brentq(lambda s: cf @ curve.discount(t, s) - dirty, -0.5, 0.5, xtol=1e-12)


def pick_benchmarks(table: pd.DataFrame, targets=(1.5, 3, 5, 10, 15)) -> list[str]:
    """Для каждого целевого срока — самый ликвидный выпуск рядом с ним (±25 %, минимум ±9 мес.)."""
    chosen: list[str] = []
    for target in targets:
        pool = table[~table["secid"].isin(chosen)]
        near = pool[(pool["years"] - target).abs() <= max(0.75, 0.25 * target)]
        if near.empty:
            near = pool.loc[[(pool["years"] - target).abs().idxmin()]]
        chosen.append(near.loc[near["value_mln"].idxmax(), "secid"])
    return chosen


def make_portfolio(snap: MarketSnapshot, curve: ZeroCurve, secids, budget_rub: float = 100e6) -> pd.DataFrame:
    """Покупаем каждую бумагу примерно на budget_rub по грязной цене."""
    tb = snap.table.set_index("secid").loc[list(secids)].reset_index()
    pf = tb[["secid", "name", "maturity", "years", "ytm", "dirty", "modified", "convexity", "dv01"]].copy()
    pf["quantity"] = np.round(budget_rub / pf["dirty"])
    pf["mv"] = pf["quantity"] * pf["dirty"]
    pf["dv01_rub"] = pf["quantity"] * pf["dv01"]
    pf["z_spread_bp"] = [
        z_spread(t, cf, p, curve) / BP for (t, cf), p in zip(snap.flows(pf["secid"]), pf["dirty"])
    ]
    return pf


def tenor_label(k: float) -> str:
    """0,5 г. / 1 г. / 3 г. / 5 л. / 10 л. — по правилам русского языка."""
    return f"{k:g}".replace(".", ",") + (" л." if k >= 5 else " г.")


def portfolio_irr(pf: pd.DataFrame, snap: MarketSnapshot) -> float:
    """Доходность портфеля как одного потока платежей: Σ qty·CF·(1+y)^−t = стоимость портфеля.
    Это не средневзвешенная YTM бумаг — та занижает доходность при растущей кривой."""
    t = np.concatenate([t for t, _ in snap.flows(pf["secid"])])
    cf = np.concatenate([q * cf for (_, cf), q in zip(snap.flows(pf["secid"]), pf["quantity"])])
    return yield_to_maturity(t, cf, pf["mv"].sum())


def summary(pf: pd.DataFrame, snap: MarketSnapshot) -> pd.Series:
    mv = pf["mv"].sum()
    w = pf["mv"] / mv
    return pd.Series({
        "Стоимость, млн руб.": mv / 1e6,
        "DV01, тыс. руб. за 1 б.п.": pf["dv01_rub"].sum() / 1e3,
        "Модифицированная дюрация, лет": pf["dv01_rub"].sum() / mv / BP,
        "Выпуклость": (w * pf["convexity"]).sum(),
        "Доходность портфеля (IRR), %": portfolio_irr(pf, snap) * 100,
    })


def revalue(pf: pd.DataFrame, snap: MarketSnapshot, curve: ZeroCurve, shift: Shift) -> np.ndarray:
    """Полная переоценка: каждый платёж дисконтируем по сдвинутой кривой + z-спред бумаги."""
    out = []
    for (t, cf), qty, z in zip(snap.flows(pf["secid"]), pf["quantity"], pf["z_spread_bp"]):
        out.append(qty * (cf @ curve.discount(t, z * BP + shift(t))))
    return np.array(out)


def scenario_pnl(pf, snap, curve, scenarios: dict[str, Shift] = SCENARIOS) -> pd.DataFrame:
    """Изменение стоимости каждой позиции и портфеля по сценариям, млн руб."""
    base = revalue(pf, snap, curve, parallel(0))
    rows = {name: (revalue(pf, snap, curve, sh) - base) / 1e6 for name, sh in scenarios.items()}
    out = pd.DataFrame(rows, index=pf["name"]).T
    out["Портфель"] = out.sum(axis=1)
    return out


def parallel_profile(pf, snap, curve, shifts_bp=np.arange(-300, 301, 25)) -> pd.DataFrame:
    """Точная переоценка против приближений: дюрация и дюрация + выпуклость.

    Чтобы сравнение было честным, DV01 и выпуклость берём той же переоценкой по кривой
    (сдвиг ±1 б.п.), а не из доходности к погашению: иначе к ошибке приближения
    добавится разница между сдвигом кривой и сдвигом YTM."""
    value = lambda bp: revalue(pf, snap, curve, parallel(bp)).sum()  # noqa: E731
    base, up, down = value(0), value(1), value(-1)
    dv01 = (down - up) / 2
    conv = (up + down - 2 * base) / BP**2      # = выпуклость × стоимость
    rows = []
    for bp in shifts_bp:
        dy = bp * BP
        rows.append({
            "shift_bp": bp,
            "full": revalue(pf, snap, curve, parallel(bp)).sum() - base,
            "duration": -dv01 * bp,
            "duration_convexity": -dv01 * bp + 0.5 * conv * dy**2,
        })
    return pd.DataFrame(rows).set_index("shift_bp") / 1e6


def key_rate_bump(tenor: float, tenors=KEY_TENORS, bp: float = 1.0) -> Shift:
    """Треугольный сдвиг на bp в точке tenor, линейно до нуля к соседним ключевым срокам.
    Сумма всех треугольников равна параллельному сдвигу."""
    grid = np.asarray(tenors, float)
    unit = (grid == tenor).astype(float)
    return lambda t: np.interp(t, grid, unit) * bp * BP


def key_rate_dv01(pf, snap, curve, tenors=KEY_TENORS) -> pd.DataFrame:
    """KR01: сколько рублей теряет каждая позиция при росте ставки на 1 б.п. в одной точке кривой."""
    base = revalue(pf, snap, curve, parallel(0))
    cols = {tenor_label(k): base - revalue(pf, snap, curve, key_rate_bump(k, tenors)) for k in tenors}
    return pd.DataFrame(cols, index=pf["name"])
