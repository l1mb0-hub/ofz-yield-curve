"""Строит все графики для README и печатает ключевые цифры.

    python scripts/make_figures.py                 # дата среза по умолчанию
    python scripts/make_figures.py --date 2026-10-08 --online

Без --online работает только на данных из data/ (они лежат в репозитории).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ofz import history, plots, risk  # noqa: E402
from ofz.market import MarketSnapshot  # noqa: E402
from ofz.moex import MoexData  # noqa: E402

FIG = ROOT / "figures"
# Даты «накануне» разобранных заседаний; последней добавляется дата среза.
EVOLUTION_DATES = ["2024-07-25", "2024-12-19", "2025-06-05", "2025-07-24", "2026-09-10"]


def main(date: str, online: bool) -> None:
    FIG.mkdir(exist_ok=True)
    data = MoexData(offline=not online)
    pd.set_option("display.width", 160)

    # 1–2. Срез рынка и кривая
    snap = MarketSnapshot.load(date, data)
    fit, ytm_curve, kbd = snap.fit_prices(), snap.fit_yields(), snap.kbd(data)
    b0, b1, b2, tau = fit.curve.params
    print(f"Срез {date}: {len(snap.table)} выпусков ОФЗ-ПД, расчёты {snap.settle}")
    print(f"НС по ценам: β0={b0:.4f} β1={b1:.4f} β2={b2:.4f} τ={tau:.2f}; "
          f"СКО ошибки {np.sqrt(np.mean(fit.error_bp**2)):.1f} б.п.")
    print("Доходность: наша − Мосбиржа, б.п.: медиана "
          f"{((snap.table.ytm - snap.table.ytm_moex).abs() * 1e4).median():.2f}")
    tenors = np.array([0.5, 1, 2, 3, 5, 7, 10, 15])
    print(pd.DataFrame({"НС, %": fit.curve.rate(tenors) * 100, "КБД, %": kbd.rate(tenors) * 100,
                        "Разница, б.п.": (fit.curve.rate(tenors) - kbd.rate(tenors)) * 1e4},
                       index=tenors).round(2))
    plots.plot_curve(snap, fit, ytm_curve, kbd).savefig(FIG / "curve.png", bbox_inches="tight")

    # 3. Портфель и риск
    secids = risk.pick_benchmarks(snap.table)
    pf = risk.make_portfolio(snap, fit.curve, secids)
    scen = risk.scenario_pnl(pf, snap, fit.curve)
    profile = risk.parallel_profile(pf, snap, fit.curve)
    kr = risk.key_rate_dv01(pf, snap, fit.curve)
    print(pf[["name", "years", "ytm", "modified", "convexity", "dv01", "quantity", "dv01_rub", "z_spread_bp"]].round(3))
    print(risk.summary(pf, snap).round(2))
    print(scen.round(2))
    plots.plot_risk(pf, scen, profile, kr).savefig(FIG / "risk.png", bbox_inches="tight")

    # 4. История: реакция на решения ЦБ и эволюция кривой
    reactions = [history.meeting_reaction(m, data) for m in history.EVENTS]
    print(history.reaction_table(reactions))
    plots.plot_meetings(reactions).savefig(FIG / "cbr_meetings.png", bbox_inches="tight")

    curves = []
    for d in sorted(set(EVOLUTION_DATES + [date])):
        s = MarketSnapshot.load(d, data)
        label = f"{pd.Timestamp(d):%d.%m.%Y} · {plots.num(history.key_rate_on(d), 2)} %"
        curves.append((label, s.fit_prices().curve, s.table["years"].min(), s.table["years"].max()))
    plots.plot_evolution(curves).savefig(FIG / "evolution.png", bbox_inches="tight")
    print(f"Графики сохранены в {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-07")
    ap.add_argument("--online", action="store_true", help="докачивать недостающие данные из ISS")
    args = ap.parse_args()
    main(args.date, args.online)
