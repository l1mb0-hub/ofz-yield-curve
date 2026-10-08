"""Кривая, КБД, портфель и история."""
from functools import lru_cache

import numpy as np

from ofz import history, risk
from ofz.bonds import yield_to_maturity
from ofz.curve import KBDCurve, NelsonSiegel, fit_ns_prices, fit_ns_yields
from ofz.market import MarketSnapshot
from ofz.moex import MoexData

DATA = MoexData(offline=True)

# Значения КБД (% годовых), опубликованные Мосбиржей (блок yearyields эндпоинта zcyc).
KBD_PUBLISHED = {
    "2026-10-07": [(0.25, 11.4607), (0.5, 12.3098), (1, 13.5831), (2, 15.0607), (5, 16.3723),
                   (10, 16.7364), (20, 16.8928)],
    "2024-12-19": [(0.25, 22.4146), (0.5, 22.3453), (1, 22.0973), (2, 21.3515), (5, 18.8649),
                   (10, 16.2314), (20, 14.2403)],
}


@lru_cache
def snapshot() -> MarketSnapshot:
    return MarketSnapshot.load("2026-10-07", DATA)


def test_kbd_formula_reproduces_moex():
    for date, points in KBD_PUBLISHED.items():
        curve = KBDCurve(DATA.kbd_params(date))
        for t, y in points:
            assert abs(curve.rate(t) * 100 - y) < 1e-4, (date, t)


def test_nelson_siegel_limits():
    ns = NelsonSiegel(0.17, -0.05, 0.02, 1.5)
    assert np.isclose(ns.rate(1e-9), 0.17 - 0.05)     # короткий конец: β0 + β1
    assert abs(ns.rate(500.0) - 0.17) < 1e-3          # длинный конец: β0


def test_fit_recovers_known_curve():
    """Генерируем цены реальных ОФЗ по известной кривой и проверяем, что подгонка её находит."""
    snap = snapshot()
    true = NelsonSiegel(0.165, -0.045, 0.02, 1.8)
    flows = snap.flows()
    dirty = np.array([cf @ true.discount(t) for t, cf in flows])
    ytm = np.array([yield_to_maturity(t, cf, p) for (t, cf), p in zip(flows, dirty)])
    fit = fit_ns_prices(flows, dirty, ytm, snap.table["years"])
    assert np.allclose(fit.curve.params, true.params, atol=1e-4)
    assert np.abs(fit.error_bp).max() < 0.01


def test_fit_on_yields_is_exact_for_ns_points():
    t = np.linspace(0.5, 15, 25)
    true = NelsonSiegel(0.16, 0.04, -0.03, 1.2)
    assert np.allclose(fit_ns_yields(t, true.rate(t)).params, true.params, atol=1e-6)


def test_real_fit_beats_kbd_on_close_prices():
    snap = snapshot()
    fit = snap.fit_prices()
    kbd = snap.kbd(DATA)
    kbd_err = [(yield_to_maturity(t, cf, cf @ kbd.discount(t)) - y) * 1e4
               for (t, cf), y in zip(snap.flows(), snap.table["ytm"])]
    assert np.sqrt(np.mean(fit.error_bp**2)) < np.sqrt(np.mean(np.square(kbd_err)))


def test_portfolio_revaluation_and_key_rates():
    snap = snapshot()
    curve = snap.fit_prices().curve
    pf = risk.make_portfolio(snap, curve, risk.pick_benchmarks(snap.table))
    assert len(pf) == 5 and pf["secid"].is_unique
    # z-спред: без сдвига переоценка даёт ровно рыночную стоимость
    assert np.allclose(risk.revalue(pf, snap, curve, risk.parallel(0)), pf["mv"], rtol=1e-10)
    # сумма KR01 равна потере от параллельного +1 б.п. (треугольники в сумме дают единицу)
    kr = risk.key_rate_dv01(pf, snap, curve).values.sum()
    base = risk.revalue(pf, snap, curve, risk.parallel(0)).sum()
    assert np.isclose(kr, base - risk.revalue(pf, snap, curve, risk.parallel(1)).sum(), rtol=1e-3)
    # и близка к DV01 по доходности к погашению (отличие — лишь в способе сдвига)
    assert abs(kr / pf["dv01_rub"].sum() - 1) < 0.03
    # выпуклость: при −100 б.п. заработаем больше, чем потеряем при +100
    pnl = risk.scenario_pnl(pf, snap, curve)["Портфель"]
    assert pnl["−100 б.п. параллельно"] > -pnl["+100 б.п. параллельно"] > 0
    # IRR портфеля: дисконтирование всех платежей по ней даёт стоимость портфеля
    irr = risk.portfolio_irr(pf, snap)
    pv = sum(q * cf @ (1 + irr) ** -t for (t, cf), q in zip(snap.flows(pf["secid"]), pf["quantity"]))
    assert np.isclose(pv, pf["mv"].sum(), rtol=1e-10) and irr > (pf["ytm"] * pf["mv"]).sum() / pf["mv"].sum()


def test_parallel_profile_convexity_correction_helps():
    snap = snapshot()
    curve = snap.fit_prices().curve
    pf = risk.make_portfolio(snap, curve, risk.pick_benchmarks(snap.table))
    prof = risk.parallel_profile(pf, snap, curve, shifts_bp=np.array([-200, 200]))
    err_dur = prof["full"] - prof["duration"]
    err_conv = (prof["duration_convexity"] - prof["full"]).abs()
    assert (err_dur > 0).all()                       # выпуклость всегда в пользу владельца
    assert err_dur[-200] > err_dur[200]              # и сильнее при снижении ставок
    assert (err_conv < err_dur / 10).all()


def test_key_rate_on_dates():
    assert history.key_rate_on("2024-12-19") == 21.0
    assert history.key_rate_on("2025-06-06") == 21.0     # решение 06.06 действует с 09.06
    assert history.key_rate_on("2025-06-09") == 20.0
    assert history.key_rate_on("2026-10-07") == 14.0


def test_december_2024_rally():
    """20.12.2024 ЦБ оставил 21 % вместо ожидаемого повышения — кривая резко упала."""
    r = history.meeting_reaction("2024-12-20", DATA)
    assert (r["delta_ns_bp"] < -100).all() and (r["delta_kbd_bp"] < -100).all()
