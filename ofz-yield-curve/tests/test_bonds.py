"""Проверяем расчёты облигаций против чисел, которые публикует Мосбиржа."""
from functools import lru_cache

import numpy as np
import pandas as pd

from ofz.bonds import (Bond, dv01_numeric, price_from_yield, risk_measures, settlement_date,
                       yield_to_maturity)
from ofz.moex import MoexData

DATA = MoexData(offline=True)

# Единственное известное расхождение: в июле 2025 для ОФЗ 26229 (последний купонный период,
# ~0,3 года до погашения) биржа показывает простую доходность, а не эффективную.
KNOWN_SIMPLE_YIELD = {("SU26229RMFS3", "2025-07-24"), ("SU26229RMFS3", "2025-07-28")}


@lru_cache
def all_quotes() -> pd.DataFrame:
    return DATA._read("quotes").dropna(subset=["close"]).reset_index(drop=True)


@lru_cache
def bonds() -> dict:
    flows = DATA._read("cashflows")
    return {s: Bond.from_frame(g) for s, g in flows.groupby("secid")}


def test_settlement_is_next_business_day():
    assert str(settlement_date("2024-12-20")) == "2024-12-23"   # пятница → понедельник
    assert str(settlement_date("2026-10-07")) == "2026-10-08"


def test_accrued_interest_matches_moex():
    """В истории ISS НКД дан на дату торгов — наш расчёт на ту же дату совпадает до копейки."""
    for q in all_quotes().itertuples():
        assert abs(bonds()[q.secid].accrued(q.tradedate) - q.accint) <= 0.011, (q.secid, q.tradedate)


def test_ytm_matches_moex():
    """Доходность по цене закрытия с расчётами T+1 совпадает с YIELDCLOSE Мосбиржи.
    Биржа публикует доходность с точностью 0,01 % = 1 б.п., поэтому допуск 0,6 б.п."""
    diffs = []
    for q in all_quotes().itertuples():
        bond = bonds()[q.secid]
        settle = settlement_date(q.tradedate)
        t, cf = bond.cashflows(settle)
        if t[-1] < 0.25 or (q.secid, q.tradedate) in KNOWN_SIMPLE_YIELD:
            continue
        y = yield_to_maturity(t, cf, q.close / 100 * bond.face + bond.accrued(settle))
        diffs.append((y * 100 - q.yield_close) * 100)
    diffs = np.abs(diffs)
    assert len(diffs) > 300
    assert diffs.max() < 0.6 and np.median(diffs) < 0.35


def test_known_moex_simple_yield_case():
    """Документируем исключение: 15,02 % у биржи — это простая доходность (1035,65/P − 1)/t."""
    q = all_quotes().query("secid == 'SU26229RMFS3' and tradedate == '2025-07-24'").iloc[0]
    bond = bonds()[q.secid]
    settle = settlement_date(q.tradedate)
    t, cf = bond.cashflows(settle)
    dirty = q.close / 100 * bond.face + bond.accrued(settle)
    simple = (cf.sum() / dirty - 1) / t[-1]
    assert abs(simple * 100 - q.yield_close) < 0.01
    assert abs(yield_to_maturity(t, cf, dirty) * 100 - 15.82) < 0.01


def test_macaulay_duration_matches_moex():
    """Дюрацию (в днях) биржа считает по средневзвешенной цене WAPRICE, а не по цене закрытия."""
    for q in all_quotes().dropna(subset=["waprice"]).itertuples():
        bond = bonds()[q.secid]
        settle = settlement_date(q.tradedate)
        t, cf = bond.cashflows(settle)
        if t[-1] < 0.25:
            continue
        y = yield_to_maturity(t, cf, q.waprice / 100 * bond.face + bond.accrued(settle))
        assert abs(risk_measures(t, cf, y)["macaulay"] * 365 - q.duration_days) <= 1.0, (q.secid, q.tradedate)


def test_zero_coupon_bond():
    t, cf = np.array([2.0]), np.array([1000.0])
    assert np.isclose(price_from_yield(t, cf, 0.10), 1000 / 1.1**2)
    m = risk_measures(t, cf, 0.10)
    assert np.isclose(m["macaulay"], 2.0) and np.isclose(m["modified"], 2.0 / 1.1)


def test_dv01_analytic_equals_numeric():
    bond = bonds()["SU26238RMFS4"]
    t, cf = bond.cashflows(np.datetime64("2026-10-08"))
    for y in (0.05, 0.12, 0.165, 0.25):
        assert np.isclose(risk_measures(t, cf, y)["dv01"], dv01_numeric(t, cf, y), rtol=1e-6)


def test_convexity_matches_second_derivative():
    bond = bonds()["SU26254RMFS1"]
    t, cf = bond.cashflows(np.datetime64("2026-10-08"))
    y, h = 0.168, 1e-4
    p0, up, dn = (price_from_yield(t, cf, y + d) for d in (0, h, -h))
    assert np.isclose((up + dn - 2 * p0) / h**2 / p0, risk_measures(t, cf, y)["convexity"], rtol=1e-5)


def test_coupon_on_settlement_day_goes_to_seller():
    bond = bonds()["SU26245RMFS9"]          # купон 07.10.2026
    t, _ = bond.cashflows(np.datetime64("2026-10-07"))
    assert t[0] > 0.4                        # ближайший платёж — через полгода
    assert bond.accrued(np.datetime64("2026-10-07")) == 0.0
