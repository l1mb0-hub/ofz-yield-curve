"""Загрузка данных из ISS API Мосбиржи с локальным кэшем в CSV.

Используются три эндпоинта ISS (бесплатные, без авторизации):

* history/engines/stock/markets/bonds/boards/TQOB/securities — итоги торгов ОФЗ за день;
* securities/{SECID}/bondization — купонный график и погашение выпуска;
* engines/stock/zcyc — параметры кривой бескупонной доходности (КБД).

Всё скачанное дописывается в ``data/*.csv``. Повторный запуск берёт данные из кэша,
поэтому проект воспроизводится без интернета на тех датах, что уже лежат в репозитории.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

ISS_URL = "https://iss.moex.com/iss"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ОФЗ-ПД (постоянный купон): выпуски серий 25xxx и 26xxx.
# ОФЗ-ПК (29xxx, плавающий купон), ОФЗ-ИН (52xxx) и ОФЗ-АД (46xxx) не берём.
OFZ_PD = re.compile(r"SU2[56]\d{3}RMFS\d")

QUOTE_COLUMNS = {
    "TRADEDATE": "tradedate",
    "SECID": "secid",
    "CLOSE": "close",            # цена последней сделки, % от номинала
    "WAPRICE": "waprice",        # средневзвешенная цена, % от номинала
    "ACCINT": "accint",          # НКД, руб. (в истории — на дату торгов, не на T+1!)
    "YIELDCLOSE": "yield_close", # доходность Мосбиржи по цене закрытия, %
    "YIELDATWAP": "yield_wap",   # доходность по средневзвешенной цене, %
    "DURATION": "duration_days", # дюрация Маколея Мосбиржи, дни
    "VALUE": "value_mln",        # оборот, руб. (переводим в млн)
}
KBD_PARAMS = ["B1", "B2", "B3", "T1"] + [f"G{i}" for i in range(1, 10)]


def day(date) -> str:
    """Любую дату ('2026-10-07', Timestamp, date) приводим к строке YYYY-MM-DD — ключу кэша."""
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def _iss(path: str, **params) -> dict:
    url = f"{ISS_URL}/{path}.json"
    resp = requests.get(url, params={"iss.meta": "off", **params}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _table(js: dict, block: str) -> pd.DataFrame:
    return pd.DataFrame(js[block]["data"], columns=js[block]["columns"])


def download_quotes(date: str) -> pd.DataFrame:
    """Итоги торгов всеми ОФЗ-ПД в режиме TQOB за дату (с постраничной загрузкой)."""
    frames, start = [], 0
    while True:
        js = _iss(
            "history/engines/stock/markets/bonds/boards/TQOB/securities",
            date=date, start=start, limit=100,
        )
        frames.append(_table(js, "history"))
        _, total, page = js["history.cursor"]["data"][0]
        start += page
        if start >= total:
            break
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["SECID"].str.fullmatch(OFZ_PD)]
    out = raw[list(QUOTE_COLUMNS)].rename(columns=QUOTE_COLUMNS)
    out["value_mln"] = (out["value_mln"] / 1e6).round(1)
    return out.reset_index(drop=True)


def download_cashflows(secid: str) -> pd.DataFrame:
    """Полный график платежей выпуска: купоны и погашение номинала."""
    js = _iss(
        f"securities/{secid}/bondization",
        limit="unlimited", **{"iss.only": "coupons,amortizations"},
    )
    coupons = _table(js, "coupons")[["coupondate", "startdate", "value"]]
    coupons.columns = ["date", "start", "coupon"]
    if coupons["coupon"].isna().any():  # так выглядят будущие купоны ОФЗ-ПК
        raise ValueError(f"{secid}: в графике есть купоны с неизвестным размером")
    principal = _table(js, "amortizations")[["amortdate", "value"]]
    principal.columns = ["date", "principal"]
    out = coupons.merge(principal, on="date", how="outer")
    out[["coupon", "principal"]] = out[["coupon", "principal"]].fillna(0.0)
    out.insert(0, "secid", secid)
    return out.sort_values("date").reset_index(drop=True)


def download_kbd(date: str) -> pd.Series:
    """Параметры КБД Мосбиржи на конец дня (B1..B3, T1, G1..G9)."""
    js = _iss("engines/stock/zcyc", date=date, **{"iss.only": "params"})
    params = _table(js, "params")
    if params.empty:
        raise ValueError(f"КБД на {date} не опубликована")
    return params.iloc[-1][KBD_PARAMS].astype(float).rename(date)


class MoexData:
    """Доступ к данным: сначала кэш ``data/*.csv``, при отсутствии — ISS API.

    ``offline=True`` запрещает обращения к сети (удобно в тестах и CI).
    """

    def __init__(self, data_dir: Path | str = DATA_DIR, offline: bool = False):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline

    # --- кэш ---------------------------------------------------------------
    def _read(self, name: str) -> pd.DataFrame:
        path = self.dir / f"{name}.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, dtype={"secid": str, "tradedate": str, "date": str, "start": str})

    def _append(self, name: str, new: pd.DataFrame) -> None:
        path = self.dir / f"{name}.csv"
        new.to_csv(path, mode="a", header=not path.exists(), index=False)
        self._read(name).drop_duplicates().to_csv(path, index=False)  # повторная загрузка не задвоит строки

    def _online(self, what: str) -> None:
        if self.offline:
            raise LookupError(f"{what}: нет в кэше {self.dir}, а режим offline=True")

    # --- публичные методы ----------------------------------------------------
    def quotes(self, date) -> pd.DataFrame:
        date = day(date)
        cache = self._read("quotes")
        if not cache.empty and (cache["tradedate"] == date).any():
            return cache[cache["tradedate"] == date].reset_index(drop=True)
        self._online(f"котировки на {date}")
        fresh = download_quotes(date)
        if fresh.empty:
            raise ValueError(f"{date}: итогов торгов нет — выходной, праздник или день ещё не закрыт")
        self._append("quotes", fresh)
        return self.quotes(date)

    def cashflows(self, secids) -> pd.DataFrame:
        cache = self._read("cashflows")
        known = set(cache["secid"]) if not cache.empty else set()
        missing = [s for s in dict.fromkeys(secids) if s not in known]
        if missing:
            self._online(f"купонные графики {missing}")
            fresh = pd.concat([download_cashflows(s) for s in missing], ignore_index=True)
            self._append("cashflows", fresh)
            cache = self._read("cashflows")
        return cache[cache["secid"].isin(list(secids))].reset_index(drop=True)

    def kbd_params(self, date) -> pd.Series:
        date = day(date)
        cache = self._read("kbd")
        if not cache.empty and (cache["date"] == date).any():
            row = cache[cache["date"] == date].iloc[-1]
            return row[KBD_PARAMS].astype(float).rename(date)
        self._online(f"КБД на {date}")
        params = download_kbd(date)
        self._append("kbd", params.to_frame().T.rename_axis("date").reset_index())
        return params
