"""Срез рынка ОФЗ-ПД на дату: котировки + купонные графики → таблица метрик по бумагам."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bonds import Bond, risk_measures, settlement_date, yield_to_maturity
from .curve import KBDCurve, PriceFit, fit_ns_prices, fit_ns_yields
from .moex import MoexData, day


@dataclass
class MarketSnapshot:
    date: str
    settle: np.datetime64
    table: pd.DataFrame          # одна строка на выпуск, отсортировано по сроку
    bonds: dict[str, Bond]

    @classmethod
    def load(cls, date: str, data: MoexData | None = None,
             min_years: float = 0.25, min_value_mln: float = 1.0) -> "MarketSnapshot":
        """Бумаги короче min_years и с оборотом ниже min_value_mln (млн руб.) отбрасываем:
        у самых коротких доходность скачет от каждого тика цены, а у неликвидных цена устаревшая."""
        data = data or MoexData()
        date = day(date)
        quotes = data.quotes(date)
        settle = settlement_date(date)

        # Цена закрытия; если сделок в конце дня не было — средневзвешенная.
        use_close = quotes["close"].notna()
        quotes = quotes.assign(
            price=quotes["close"].where(use_close, quotes["waprice"]),
            ytm_moex=quotes["yield_close"].where(use_close, quotes["yield_wap"]) / 100,
        ).dropna(subset=["price"])

        flows = data.cashflows(quotes["secid"])
        bonds = {s: Bond.from_frame(g) for s, g in flows.groupby("secid")}

        rows = []
        for q in quotes[quotes["value_mln"] >= min_value_mln].itertuples():
            bond = bonds[q.secid]
            t, cf = bond.cashflows(settle)
            if len(t) == 0 or t[-1] < min_years:
                continue
            # В истории ISS поле ACCINT — НКД на дату торгов, а деньги переходят в дату
            # расчётов T+1. Поэтому НКД считаем сами на дату расчётов: только так наша
            # доходность совпадает с биржевой (в пределах 0.5 б.п.).
            accrued = bond.accrued(settle)
            dirty = q.price / 100 * bond.face + accrued
            try:
                ytm = yield_to_maturity(t, cf, dirty)
            except ValueError:
                warnings.warn(f"{date} {q.secid}: доходность не находится, бумага пропущена")
                continue
            risk = risk_measures(t, cf, ytm)
            rows.append({
                "secid": q.secid, "name": bond.name,
                "maturity": pd.Timestamp(bond.maturity), "years": t[-1],
                "coupon": bond.coupon_rate * 100, "price": q.price, "accint": accrued,
                "dirty": dirty, "ytm": ytm, "ytm_moex": q.ytm_moex,
                "duration_moex": q.duration_days / 365, "value_mln": q.value_mln,
                **{k: risk[k] for k in ("macaulay", "modified", "convexity", "dv01")},
            })
        if not rows:
            raise ValueError(f"{date}: после фильтров не осталось ни одной бумаги")
        table = pd.DataFrame(rows).sort_values("years").reset_index(drop=True)
        return cls(date, settle, table, bonds)

    def flows(self, secids=None) -> list[tuple[np.ndarray, np.ndarray]]:
        secids = self.table["secid"] if secids is None else secids
        return [self.bonds[s].cashflows(self.settle) for s in secids]

    def fit_yields(self):
        """НС по точкам (срок до погашения, YTM) — шаг 2 «в лоб»."""
        return fit_ns_yields(self.table["years"], self.table["ytm"])

    def fit_prices(self, robust: bool = False) -> PriceFit:
        """НС как бескупонная кривая, подогнанная к ценам всех бумаг."""
        tb = self.table
        return fit_ns_prices(self.flows(), tb["dirty"], tb["ytm"], tb["years"], robust=robust)

    def kbd(self, data: MoexData | None = None) -> KBDCurve:
        return KBDCurve((data or MoexData()).kbd_params(self.date))
