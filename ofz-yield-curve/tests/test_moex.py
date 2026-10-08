"""Загрузчик ISS: разбор ответов API (без сети, на подменённых ответах) и работа кэша."""
import tempfile

import pandas as pd

from ofz import moex
from ofz.moex import MoexData

HIST_COLS = ["BOARDID", "TRADEDATE", "SECID", "CLOSE", "WAPRICE", "ACCINT", "YIELDCLOSE",
             "YIELDATWAP", "DURATION", "VALUE"]


def fake_iss(path, **params):
    """Ответы в том же формате, что отдаёт iss.moex.com (iss.meta=off)."""
    if path.startswith("history/"):
        page = [
            ["TQOB", "2026-10-07", "SU26238RMFS4", 50.668, 50.593, 24.51, 16.49, 16.52, 2462, 455825249.39],
            ["TQOB", "2026-10-07", "SU29025RMFS2", 99.1, 99.0, 5.0, None, None, None, 1e8],  # флоатер
        ] if params["start"] == 0 else [
            ["TQOB", "2026-10-07", "SU26254RMFS1", 82.704, 82.763, 59.83, 16.8, 16.79, 2093, 1570300000.0],
        ]
        return {"history": {"columns": HIST_COLS, "data": page},
                "history.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[params["start"], 3, 2]]}}
    if path.endswith("/bondization"):
        return {
            "coupons": {"columns": ["coupondate", "startdate", "value", "secid"],
                        "data": [["2041-05-15", "2040-11-14", 35.4, "SU26238RMFS4"],
                                 ["2040-11-14", "2040-05-16", 35.4, "SU26238RMFS4"]]},
            "amortizations": {"columns": ["amortdate", "value", "secid"],
                              "data": [["2041-05-15", 1000.0, "SU26238RMFS4"]]},
        }
    if path == "engines/stock/zcyc":
        cols = ["tradedate", "tradetime"] + moex.KBD_PARAMS
        return {"params": {"columns": cols, "data": [["2026-10-07", "18:54:59"] + list(range(13))]}}
    raise AssertionError(path)


def with_fake_api(fn):
    def run():
        real, moex._iss = moex._iss, fake_iss
        try:
            fn()
        finally:
            moex._iss = real
    run.__name__ = fn.__name__
    return run


@with_fake_api
def test_download_quotes_paginates_and_filters_ofz_pd():
    q = moex.download_quotes("2026-10-07")
    assert list(q["secid"]) == ["SU26238RMFS4", "SU26254RMFS1"]   # флоатер отброшен, 2 страницы
    assert q.loc[0, "value_mln"] == 455.8 and q.loc[0, "yield_close"] == 16.49


@with_fake_api
def test_download_cashflows_merges_coupons_and_principal():
    cf = moex.download_cashflows("SU26238RMFS4")
    assert list(cf["date"]) == ["2040-11-14", "2041-05-15"]
    assert cf["principal"].tolist() == [0.0, 1000.0] and cf["coupon"].tolist() == [35.4, 35.4]


@with_fake_api
def test_cache_roundtrip_and_offline_mode():
    with tempfile.TemporaryDirectory() as tmp:
        online = MoexData(tmp)
        assert len(online.quotes("2026-10-07")) == 2
        assert online.kbd_params("2026-10-07")["T1"] == 3.0
        online.quotes(pd.Timestamp("2026-10-07"))                  # дата другим типом — из кэша
        online._append("quotes", moex.download_quotes("2026-10-07"))  # повторная загрузка
        offline = MoexData(tmp, offline=True)                      # теперь всё из кэша
        assert len(offline.quotes("2026-10-07")) == 2
        assert len(offline._read("quotes")) == 2
        assert offline.kbd_params("2026-10-07")["G9"] == 12.0
        try:
            offline.quotes("2026-10-08")
        except LookupError:
            pass
        else:
            raise AssertionError("offline-режим должен запрещать сеть")


def test_repository_cache_is_complete():
    data = MoexData(offline=True)
    quotes = data._read("quotes")
    assert quotes["tradedate"].nunique() == 13
    assert set(quotes["secid"]) <= set(data._read("cashflows")["secid"])
    assert set(quotes["tradedate"]) == set(data._read("kbd")["date"])
    assert isinstance(data.kbd_params("2026-10-07"), pd.Series)
