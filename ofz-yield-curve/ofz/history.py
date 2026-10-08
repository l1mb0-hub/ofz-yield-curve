"""Кривая ОФЗ вокруг заседаний Банка России по ключевой ставке."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .curve import KBDCurve
from .market import MarketSnapshot
from .moex import MoexData
from .risk import tenor_label

# Решения совета директоров Банка России (источник: cbr.ru, «Ключевая ставка»).
CBR_DECISIONS = pd.DataFrame(
    [
        ("2024-02-16", 16.00, 16.00), ("2024-03-22", 16.00, 16.00), ("2024-04-26", 16.00, 16.00),
        ("2024-06-07", 16.00, 16.00), ("2024-07-26", 16.00, 18.00), ("2024-09-13", 18.00, 19.00),
        ("2024-10-25", 19.00, 21.00), ("2024-12-20", 21.00, 21.00), ("2025-02-14", 21.00, 21.00),
        ("2025-03-21", 21.00, 21.00), ("2025-04-25", 21.00, 21.00), ("2025-06-06", 21.00, 20.00),
        ("2025-07-25", 20.00, 18.00), ("2025-09-12", 18.00, 17.00), ("2025-10-24", 17.00, 16.50),
        ("2025-12-19", 16.50, 16.00), ("2026-02-13", 16.00, 15.50), ("2026-03-20", 15.50, 15.00),
        ("2026-04-24", 15.00, 14.50), ("2026-06-19", 14.50, 14.25), ("2026-07-24", 14.25, 14.00),
        ("2026-09-11", 14.00, 14.00),
    ],
    columns=["date", "rate_before", "rate_after"],
)

# Заседания, которые разбираем подробно: от пика ужесточения до текущего цикла снижения.
EVENTS = ["2024-07-26", "2024-10-25", "2024-12-20", "2025-06-06", "2025-07-25", "2026-09-11"]

TENORS = (2, 5, 10)
BUCKETS = ((0, 3, "ОФЗ до 3 лет"), (3, 7, "ОФЗ 3–7 лет"), (7, 50, "ОФЗ от 7 лет"))


def decision_label(date: str) -> str:
    row = CBR_DECISIONS.set_index("date").loc[date]
    before, after = row["rate_before"], row["rate_after"]
    change = round((after - before) * 100)
    verb = "сохранил" if change == 0 else ("повысил" if change > 0 else "снизил")
    tail = f"{after:g} %" if change == 0 else f"{before:g} → {after:g} % ({change:+d} б.п.)"
    return f"{pd.Timestamp(date):%d.%m.%Y}: ЦБ {verb} ставку, {tail}"


def key_rate_on(date: str) -> float:
    """Ключевая ставка, действовавшая в дату (решение вступает в силу со следующего рабочего дня)."""
    effective = [str(np.busday_offset(np.datetime64(d, "D"), 1, roll="forward")) for d in CBR_DECISIONS["date"]]
    past = CBR_DECISIONS[[e <= date for e in effective]]
    return float(past["rate_after"].iloc[-1] if len(past) else CBR_DECISIONS["rate_before"].iloc[0])


def event_window(meeting: str) -> tuple[str, str]:
    """Закрытие накануне решения и закрытие первого торгового дня после него.

    Решение объявляют в 13:30 по пятницам, торги идут до вечера. Окно «четверг → понедельник»
    захватывает и само решение, и реакцию на пресс-конференцию и выходные."""
    day = np.datetime64(meeting, "D")
    before = np.busday_offset(day, -1, roll="backward")
    after = np.busday_offset(day, 1, roll="forward")
    return str(before), str(after)


def meeting_reaction(meeting: str, data: MoexData | None = None, tenors=TENORS) -> dict:
    """Кривые до и после решения и изменение бескупонных ставок на ключевых сроках, б.п."""
    data = data or MoexData()
    out = {"meeting": meeting, "label": decision_label(meeting)}
    for side, date in zip(("before", "after"), event_window(meeting)):
        snap = MarketSnapshot.load(date, data)
        out[side] = {"date": date, "snap": snap, "fit": snap.fit_prices(),
                     "kbd": KBDCurve(data.kbd_params(date))}
    t = np.asarray(tenors, float)
    for key, kind in (("delta_ns_bp", "fit"), ("delta_kbd_bp", "kbd")):
        before, after = out["before"][kind], out["after"][kind]
        before, after = getattr(before, "curve", before), getattr(after, "curve", after)
        out[key] = pd.Series((after.rate(t) - before.rate(t)) * 1e4, index=list(tenors))
    out["bonds"] = bond_moves(out)
    return out


def bond_moves(reaction: dict) -> pd.DataFrame:
    """Проверка без модели: изменение доходности одних и тех же бумаг, б.п."""
    cols = ["secid", "name", "years", "ytm"]
    b = reaction["before"]["snap"].table[cols]
    a = reaction["after"]["snap"].table[cols]
    m = b.merge(a, on=["secid", "name"], suffixes=("_before", "_after"))
    m["delta_bp"] = (m["ytm_after"] - m["ytm_before"]) * 1e4
    return m[["secid", "name", "years_after", "delta_bp"]].rename(columns={"years_after": "years"})


def reaction_table(reactions: list[dict], tenors=TENORS) -> pd.DataFrame:
    """Изменение бескупонных ставок (наша кривая и КБД) и медианное изменение YTM бумаг, б.п."""
    rows = []
    for r in reactions:
        row = CBR_DECISIONS.set_index("date").loc[r["meeting"]]
        rec = {"Заседание": r["meeting"], "Ставка": f"{row['rate_before']:g} → {row['rate_after']:g}"}
        rec.update({f"НС {tenor_label(k)}": round(r["delta_ns_bp"][k]) for k in tenors})
        rec.update({f"КБД {tenor_label(k)}": round(r["delta_kbd_bp"][k]) for k in tenors})
        moves = r["bonds"]
        for lo, hi, name in BUCKETS:
            rec[name] = round(moves.loc[(moves["years"] >= lo) & (moves["years"] < hi), "delta_bp"].median())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("Заседание")
