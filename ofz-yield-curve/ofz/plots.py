"""Графики для README и ноутбука (matplotlib, единый стиль)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .bonds import yield_to_maturity

# Палитра: категориальные цвета в фиксированном порядке (проверены на различимость
# при цветовой слепоте), полярность прибыль/убыток — синий/красный, время — один тон.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e89", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GAIN, LOSS = "#2a78d6", "#e34948"
BLUES = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "axes.linewidth": 1, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "text.color": INK, "figure.dpi": 110, "savefig.dpi": 160,
    })


def num(v: float, digits: int = 1, sign: bool = False) -> str:
    """Число по-русски: десятичная запятая, типографский минус, без лишних нулей."""
    s = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",").replace("-", "−")


def comma(digits: int = 1) -> FuncFormatter:
    return FuncFormatter(lambda v, _: num(v, digits))


def _yield_axis(ax) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(comma(1))


def _grid(snap, pad: float = 0.4) -> np.ndarray:
    """Кривую рисуем только там, где есть бумаги: экстраполяция НС за край данных ненадёжна."""
    years = snap.table["years"]
    return np.linspace(years.min(), years.max() + pad, 250)


def model_error_bp(snap, curve) -> np.ndarray:
    """Доходность, которую даёт кривая, минус рыночная доходность, б.п."""
    out = []
    for (t, cf), y in zip(snap.flows(), snap.table["ytm"]):
        out.append((yield_to_maturity(t, cf, cf @ curve.discount(t)) - y) * 1e4)
    return np.array(out)


def plot_curve(snap, fit, ytm_curve, kbd, title: str | None = None):
    """Рыночные доходности ОФЗ, кривая НС (по ценам и по YTM), КБД и ошибки моделей."""
    use_style()
    tb = snap.table
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 6.9), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.2})
    grid = _grid(snap)
    lines = [
        (fit.curve.rate(grid), SERIES[0], "Бескупонная кривая: НС по ценам"),
        (kbd.rate(grid), SERIES[1], "КБД Мосбиржи"),
        (ytm_curve.rate(grid), SERIES[2], "НС по точкам «срок — доходность»"),
    ]
    for z, color, label in lines:
        ax.plot(grid, z * 100, color=color, label=label, zorder=2)
    low = tb["coupon"] < 8
    ax.scatter(tb.loc[~low, "years"], tb.loc[~low, "ytm"] * 100, s=34, color=INK, edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="ОФЗ-ПД, купон ≥ 8 %")
    ax.scatter(tb.loc[low, "years"], tb.loc[low, "ytm"] * 100, s=34, facecolor=SURFACE, edgecolor=INK,
               linewidth=1.3, zorder=3, label="ОФЗ-ПД, купон < 8 %")
    for _, r in tb.iterrows():
        if r["secid"] in ("SU26238RMFS4", "SU26233RMFS5", "SU26207RMFS9"):
            ax.annotate(r["name"][4:], (r["years"], r["ytm"] * 100), xytext=(6, -12),
                        textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_ylabel("Доходность, % годовых")
    _yield_axis(ax)
    ax.legend(loc="lower right")
    ax.set_title(title or f"Кривая ОФЗ на {pd.Timestamp(snap.date):%d.%m.%Y}")

    err_ns, err_kbd = fit.error_bp, model_error_bp(snap, kbd)
    axr.axhline(0, color=MUTED, linewidth=1)
    rmse = lambda e: num(np.sqrt(np.mean(e**2)), 0)  # noqa: E731
    axr.scatter(tb["years"], err_kbd, s=26, color=SERIES[1], edgecolor=SURFACE, linewidth=1, zorder=3,
                label=f"КБД: СКО {rmse(err_kbd)} б.п.")
    axr.scatter(tb["years"], err_ns, s=26, color=SERIES[0], edgecolor=SURFACE, linewidth=1, zorder=4,
                label=f"НС по ценам: СКО {rmse(err_ns)} б.п.")
    axr.set_ylabel("Модель − рынок, б.п.")
    axr.set_xlabel("Срок до погашения, лет")
    lim = np.ceil(max(np.abs(err_ns).max(), np.abs(err_kbd).max()) / 25 + 0.2) * 25
    axr.set_ylim(-lim, lim)
    axr.yaxis.set_major_formatter(comma(0))
    axr.legend(loc="lower left", bbox_to_anchor=(0, 0.98), ncol=2, borderaxespad=0)
    return fig


def plot_risk(pf, scen: pd.DataFrame, profile: pd.DataFrame, kr: pd.DataFrame):
    """Сценарии, точная переоценка против приближений и разложение DV01 по срокам."""
    use_style()
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.5, wspace=0.42,
                          left=0.08, right=0.98, top=0.95, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0])
    total = scen["Портфель"][::-1]
    ax.barh(total.index, total.values, height=0.55, color=[GAIN if v > 0 else LOSS for v in total])
    lim = np.abs(total).max() * 1.45
    for i, v in enumerate(total.values):
        ax.text(v + np.sign(v) * lim * 0.03, i, num(v, 1, sign=True), va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(comma(0))
    ax.set_xlabel("Изменение стоимости, млн руб.")
    ax.set_title("Сценарии сдвига кривой")

    ax = fig.add_subplot(gs[0, 1])
    x = profile.index.to_numpy()
    ax.plot(x, profile["full"], color=SERIES[0], label="Полная переоценка")
    ax.plot(x, profile["duration"], color=SERIES[1], label="Только дюрация")
    ax.plot(x, profile["duration_convexity"], color=SERIES[2], linewidth=1.5,
            linestyle=(0, (1, 2.5)), label="Дюрация + выпуклость")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    up, down = x.max(), x.min()
    loss_gap = profile.loc[up, "full"] - profile.loc[up, "duration"]
    gain_gap = profile.loc[down, "full"] - profile.loc[down, "duration"]
    ax.text(0.03, 0.06, f"При {num(up, 0, True)} б.п. дюрация завышает\nпотерю на {num(loss_gap, 1)} млн руб.,\n"
            f"при {num(down, 0)} б.п. занижает\nприбыль на {num(gain_gap, 1)} млн руб.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    ax.set_xlabel("Параллельный сдвиг, б.п.")
    ax.set_ylabel("Изменение стоимости, млн руб.")
    ax.xaxis.set_major_formatter(comma(0))
    ax.yaxis.set_major_formatter(comma(0))
    ax.legend(loc="upper right")
    ax.set_title("Дюрация против точной переоценки")

    ax = fig.add_subplot(gs[1, :])
    kr_k = kr / 1e3
    bottom = np.zeros(kr_k.shape[1])
    xs = np.arange(kr_k.shape[1])
    for color, (name, row) in zip(SERIES, kr_k.iterrows()):
        ax.bar(xs, row.values, bottom=bottom, width=0.55, color=color, edgecolor=SURFACE,
               linewidth=np.where(row.values > 0.4, 1.5, 0.0), label=name)
        bottom += row.values
    for xi, v in zip(xs, bottom):
        ax.text(xi, v + bottom.max() * 0.015, num(v, 0), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(xs, kr_k.columns)
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Ключевой срок кривой")
    ax.set_ylabel("Потери при +1 б.п., тыс. руб.")
    ax.legend(loc="lower left", ncol=5, bbox_to_anchor=(0, 0.995), borderaxespad=0)
    ax.set_title(f"Разложение DV01 по ключевым срокам (KR01): всего {num(kr_k.values.sum(), 0)} тыс. руб. "
                 "за 1 б.п.", pad=24)
    return fig


def plot_meetings(reactions: list[dict], ncols: int = 3):
    """Кривая до и после решения ЦБ: точки — доходности ОФЗ, линии — кривая НС."""
    use_style()
    n = len(reactions)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.7 * nrows), sharex=True)
    for ax, r in zip(np.ravel(axes), reactions):
        for side, color, filled in (("before", MUTED, False), ("after", SERIES[0], True)):
            snap, fit = r[side]["snap"], r[side]["fit"]
            grid = _grid(snap)
            label = f"{pd.Timestamp(r[side]['date']):%d.%m}"
            ax.plot(grid, fit.curve.rate(grid) * 100, color=color, label=label, zorder=2)
            ax.scatter(snap.table["years"], snap.table["ytm"] * 100, s=16, zorder=3,
                       facecolor=color if filled else SURFACE, edgecolor=color, linewidth=1)
        d2, d10 = r["delta_ns_bp"][2], r["delta_ns_bp"][10]
        head = r["label"].split(": ", 1)[1].replace("ЦБ ", "").replace(" ставку", "").replace("-", "−")
        ax.set_title(f"{pd.Timestamp(r['meeting']):%d.%m.%Y}: {head}", fontsize=9.5, pad=16)
        ax.text(0, 1.015, f"Δ 2 г.: {num(d2, 0, True)} б.п.   Δ 10 л.: {num(d10, 0, True)} б.п.",
                transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
        _yield_axis(ax)
        ax.legend(loc="best", fontsize=8, handlelength=1.2)
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    for ax in np.ravel(axes)[-ncols:]:
        ax.set_xlabel("Срок до погашения, лет")
    for ax in np.ravel(axes)[::ncols]:
        ax.set_ylabel("Доходность, %")
    fig.suptitle("Реакция кривой ОФЗ на решения Банка России: закрытие накануне → первый торговый день после",
                 x=0.015, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_evolution(curves: list[tuple[str, object, float, float]]):
    """Бескупонные кривые на разные даты; последняя дата выделена.

    curves — список (подпись, кривая, мин. срок, макс. срок), по времени.
    """
    use_style()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    hist, (last_label, last_curve, lo_last, hi_last) = curves[:-1], curves[-1]
    for (label, curve, lo, hi), color in zip(hist, BLUES):
        grid = np.linspace(lo, hi, 200)
        ax.plot(grid, curve.rate(grid) * 100, color=color, label=label)
    grid = np.linspace(lo_last, hi_last, 200)
    z = last_curve.rate(grid) * 100
    ax.plot(grid, z, color=SERIES[1], linewidth=2.5, label=last_label, zorder=3)
    ax.set_xlabel("Срок, лет")
    ax.set_ylabel("Бескупонная доходность, %")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _yield_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Дата · ключевая ставка",
              title_fontsize=9, alignment="left")
    ax.set_title("Как менялась кривая ОФЗ в 2024–2026 годах")
    fig.tight_layout()
    return fig
