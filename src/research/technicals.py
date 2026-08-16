"""Deterministic technical read: indicators -> verdict -> model alignment.

Pure functions over a daily OHLCV frame. No I/O, no API keys, no LLM — every
number and every verdict here is reproducible from the bars alone. The LLM
layer *narrates* the verdict produced here; it never computes or overrides it.

Why z20 rather than a percentage distance from the moving average: the three
regions quote in USD, GBp pence and CAD, and a 5%-above-SMA20 threshold means
something completely different on a 3000p UK industrial than on a C$4 Canadian
small cap. `z20 = (close - SMA20) / stdev20(close)` is unit-free and
volatility-scaled, so one set of thresholds works across the whole book.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# --- verdict thresholds -------------------------------------------------------
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
Z_STRETCHED = 2.0       # |z20| beyond this = stretched away from the mean
Z_DRIFT = 0.5           # |z20| below this = sitting on the mean
RANGE_HIGH = 0.95       # fraction of the 52w range that counts as "at the high"
RANGE_LOW = 0.05
VOLUME_SPIKE = 2.0      # x the 20d average volume
GAP_VOL_MULT = 1.5      # move since signal, in weekly-vol units, that counts as a gap

TRADING_DAYS_YEAR = 252
TRADING_DAYS_WEEK = 5
TRADING_DAYS_MONTH = 21
TRADING_DAYS_QUARTER = 63

VERDICT_LABELS = {
    "OVERBOUGHT_PULLBACK": "Stretched high — reversal risk",
    "OVERSOLD_BOUNCE": "Stretched low — bounce candidate",
    "UPTREND_PULLBACK": "Dip inside an uptrend",
    "DOWNTREND_RALLY": "Bounce inside a downtrend",
    "UPTREND_INTACT": "Uptrend, no stretch",
    "DOWNTREND_INTACT": "Downtrend, no stretch",
    "RANGEBOUND": "Rangebound / no clear signal",
    "INSUFFICIENT_DATA": "Not enough price history",
}

# This is a *reversion* model, so a long should ideally be a name that has just
# been sold off, not one making new highs. Where the tape disagrees with the
# pick, that is the single most useful thing this page can surface.
_ALIGNMENT = {
    "OVERSOLD_BOUNCE": {"LONG": "aligned", "SHORT": "conflict"},
    "UPTREND_PULLBACK": {"LONG": "aligned", "SHORT": "conflict"},
    "OVERBOUGHT_PULLBACK": {"LONG": "conflict", "SHORT": "aligned"},
    "DOWNTREND_RALLY": {"LONG": "conflict", "SHORT": "aligned"},
}


def _f(x) -> float | None:
    """Coerce to a JSON-safe float, mapping NaN/inf to None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)


def _pct_change_over(close: pd.Series, n: int) -> float | None:
    """Percent change over the last `n` bars, or None if the history is short."""
    if len(close) <= n:
        return None
    prev = close.iloc[-1 - n]
    if prev == 0:
        return None
    return _f((close.iloc[-1] / prev - 1.0) * 100.0)


def normalize_bars(raw: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """FMP technical-indicator rows -> a clean ascending-by-date OHLCV frame.

    FMP returns newest-first with a string `date`; everything downstream assumes
    oldest-first with a DatetimeIndex.
    """
    df = pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw.copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    for col in ("open", "high", "low", "close", "volume", "rsi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"])


def compute_indicators(bars: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict:
    """Indicator snapshot as of the last bar.

    `as_of` is the forecast's signal date. `move_since_signal_pct` measures how
    far the name has already travelled since the model scored it — a pick that
    has run 8% before you could trade it is not the pick the backtest measured.
    """
    if bars is None or bars.empty:
        return {"verdict": "INSUFFICIENT_DATA", "n_bars": 0}

    close, high, low = bars["close"], bars.get("high"), bars.get("low")
    n = len(bars)
    last = close.iloc[-1]

    sma20 = close.rolling(20).mean().iloc[-1] if n >= 20 else None
    sma50 = close.rolling(50).mean().iloc[-1] if n >= 50 else None
    sma200 = close.rolling(200).mean().iloc[-1] if n >= 200 else None
    std20 = close.rolling(20).std().iloc[-1] if n >= 20 else None

    z20 = None
    if sma20 is not None and std20 is not None and std20 > 0:
        z20 = (last - sma20) / std20

    def _vs(ma):
        return _f((last / ma - 1.0) * 100.0) if ma not in (None, 0) and pd.notna(ma) else None

    # 52-week window, or whatever history exists if shorter.
    win = bars.tail(TRADING_DAYS_YEAR)
    hi52 = win["high"].max() if "high" in win else win["close"].max()
    lo52 = win["low"].min() if "low" in win else win["close"].min()
    pct_range = None
    if pd.notna(hi52) and pd.notna(lo52) and hi52 > lo52:
        pct_range = _f((last - lo52) / (hi52 - lo52))

    # True-range ATR, expressed as a percentage of price so it is comparable
    # across currencies.
    atr_pct = None
    if high is not None and low is not None and n >= 15:
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if pd.notna(atr) and last:
            atr_pct = _f(atr / last * 100.0)

    vol_ann = None
    if n >= 21:
        rets = np.log(close / close.shift(1)).tail(20)
        sd = rets.std()
        if pd.notna(sd):
            vol_ann = _f(sd * math.sqrt(TRADING_DAYS_YEAR) * 100.0)

    vol_ratio = None
    if "volume" in bars and n >= 20:
        avg = bars["volume"].rolling(20).mean().iloc[-1]
        if pd.notna(avg) and avg > 0:
            vol_ratio = _f(bars["volume"].iloc[-1] / avg)

    # Move since the model scored the name: last close vs the close on or before
    # the signal date.
    move_since_signal = None
    if as_of is not None:
        prior = bars.loc[bars.index <= pd.Timestamp(as_of).normalize()]
        if not prior.empty and prior["close"].iloc[-1]:
            move_since_signal = _f((last / prior["close"].iloc[-1] - 1.0) * 100.0)

    ind = {
        "asof": bars.index[-1].date().isoformat(),
        "n_bars": int(n),
        "last_close": _f(last),
        "rsi14": _f(bars["rsi"].iloc[-1]) if "rsi" in bars else None,
        "sma20": _f(sma20),
        "sma50": _f(sma50),
        "sma200": _f(sma200),
        "z20": _f(z20),
        "px_vs_sma20_pct": _vs(sma20),
        "px_vs_sma50_pct": _vs(sma50),
        "px_vs_sma200_pct": _vs(sma200),
        "ret_1w_pct": _pct_change_over(close, TRADING_DAYS_WEEK),
        "ret_1m_pct": _pct_change_over(close, TRADING_DAYS_MONTH),
        "ret_3m_pct": _pct_change_over(close, TRADING_DAYS_QUARTER),
        "range_52w_low": _f(lo52),
        "range_52w_high": _f(hi52),
        "pct_of_52w_range": pct_range,
        "atr14_pct": atr_pct,
        "vol_20d_ann_pct": vol_ann,
        "volume_vs_20d_avg": vol_ratio,
        "move_since_signal_pct": move_since_signal,
    }
    ind["trend"] = _trend(ind)
    return ind


def _trend(ind: dict) -> str:
    """UP / DOWN / SIDEWAYS from the SMA20-SMA50 relationship."""
    close, sma20, sma50 = ind.get("last_close"), ind.get("sma20"), ind.get("sma50")
    if close is None or sma20 is None or sma50 is None:
        return "SIDEWAYS"
    if sma20 > sma50 and close > sma50:
        return "UP"
    if sma20 < sma50 and close < sma50:
        return "DOWN"
    return "SIDEWAYS"


def _z_note(z: float | None) -> str:
    if z is None:
        return "distance from SMA20 unavailable"
    return f"{abs(z):.1f}σ {'above' if z >= 0 else 'below'} SMA20"


def classify(ind: dict) -> tuple[str, str, list[str]]:
    """(verdict, human label, reasons) — first matching rule wins.

    Rules are ordered so that "stretched" states beat "trending" states: a name
    can be both in an uptrend and 3 sigma above its mean, and for a one-week
    reversion horizon the stretch is the thing that matters.
    """
    if not ind or ind.get("n_bars", 0) < 20:
        return "INSUFFICIENT_DATA", VERDICT_LABELS["INSUFFICIENT_DATA"], [
            f"only {ind.get('n_bars', 0)} daily bars available (need 20)"
        ]

    rsi, z = ind.get("rsi14"), ind.get("z20")
    r1w, trend = ind.get("ret_1w_pct"), ind.get("trend", "SIDEWAYS")
    reasons: list[str] = []

    hot = (rsi is not None and rsi >= RSI_OVERBOUGHT) or (z is not None and z >= Z_STRETCHED)
    cold = (rsi is not None and rsi <= RSI_OVERSOLD) or (z is not None and z <= -Z_STRETCHED)

    def _stretch_reasons(up: bool) -> list[str]:
        out = []
        if rsi is not None and (rsi >= RSI_OVERBOUGHT if up else rsi <= RSI_OVERSOLD):
            out.append(f"RSI14 {rsi:.1f} {'≥' if up else '≤'} {RSI_OVERBOUGHT if up else RSI_OVERSOLD:.0f}")
        if z is not None and (z >= Z_STRETCHED if up else z <= -Z_STRETCHED):
            out.append(f"close {abs(z):.1f}σ {'above' if up else 'below'} SMA20")
        return out

    if hot and r1w is not None and r1w > 0:
        reasons = _stretch_reasons(True) + [f"up {r1w:.1f}% over the last week"]
        verdict = "OVERBOUGHT_PULLBACK"
    elif cold and r1w is not None and r1w < 0:
        reasons = _stretch_reasons(False) + [f"down {abs(r1w):.1f}% over the last week"]
        verdict = "OVERSOLD_BOUNCE"
    elif trend == "UP" and r1w is not None and r1w < 0 and z is not None and -Z_STRETCHED < z <= -Z_DRIFT:
        reasons = [f"SMA20 above SMA50 (uptrend)", f"pulled back {abs(r1w):.1f}% this week",
                   f"{abs(z):.1f}σ below SMA20"]
        verdict = "UPTREND_PULLBACK"
    elif trend == "DOWN" and r1w is not None and r1w > 0 and z is not None and Z_DRIFT <= z < Z_STRETCHED:
        reasons = [f"SMA20 below SMA50 (downtrend)", f"rallied {r1w:.1f}% this week",
                   f"{z:.1f}σ above SMA20"]
        verdict = "DOWNTREND_RALLY"
    # Anything still trending but not stretched and not pulling back. Note this
    # is a plain trend test, not `abs(z20) < Z_DRIFT`: a name grinding steadily
    # higher sits ~1-1.5 sigma above its own SMA20 the whole way up, and calling
    # that "rangebound" would be wrong. RANGEBOUND now means what it says —
    # no trend at all.
    elif trend == "UP":
        reasons = ["SMA20 above SMA50 (uptrend)", _z_note(z), "no stretch or pullback signal"]
        verdict = "UPTREND_INTACT"
    elif trend == "DOWN":
        reasons = ["SMA20 below SMA50 (downtrend)", _z_note(z), "no stretch or bounce signal"]
        verdict = "DOWNTREND_INTACT"
    else:
        reasons = ["SMA20 and SMA50 not separated (no trend)", _z_note(z),
                   f"RSI14 {rsi:.1f}" if rsi is not None else "RSI unavailable"]
        verdict = "RANGEBOUND"

    pr = ind.get("pct_of_52w_range")
    if pr is not None:
        reasons.append(f"{pr * 100:.0f}% of the 52-week range")
    return verdict, VERDICT_LABELS[verdict], reasons


def flags(ind: dict) -> list[str]:
    """Independent boolean observations, orthogonal to the verdict."""
    out: list[str] = []
    pr = ind.get("pct_of_52w_range")
    if pr is not None:
        if pr >= RANGE_HIGH:
            out.append("at_52w_high")
        elif pr <= RANGE_LOW:
            out.append("at_52w_low")
    vr = ind.get("volume_vs_20d_avg")
    if vr is not None and vr >= VOLUME_SPIKE:
        out.append("volume_spike")
    move, vol = ind.get("move_since_signal_pct"), ind.get("vol_20d_ann_pct")
    if move is not None and vol:
        weekly_vol = vol / math.sqrt(52.0)
        if weekly_vol > 0 and abs(move) >= GAP_VOL_MULT * weekly_vol:
            out.append("gapped_since_signal")
    return out


def alignment(side: str, verdict: str) -> str:
    """Does the tape agree with the model's side? aligned / neutral / conflict."""
    return _ALIGNMENT.get(verdict, {}).get((side or "").upper(), "neutral")


def compact_series(bars: pd.DataFrame, days: int = TRADING_DAYS_YEAR) -> dict:
    """The last ~12 months of closes and RSI, as parallel arrays.

    Stored on the artifact so the Deep dive price chart renders on the Hugging
    Face Space with no API keys at all. Columnar rather than a list of records,
    and rounded, because this is by far the largest field per name — records
    with repeated keys roughly triple it.
    """
    if bars is None or bars.empty:
        return {}
    tail = bars.tail(days)
    out = {
        "dates": [d.date().isoformat() for d in tail.index],
        "close": [round(float(c), 4) for c in tail["close"]],
    }
    if "rsi" in tail:
        out["rsi"] = [None if pd.isna(v) else round(float(v), 1) for v in tail["rsi"]]
    return out


def series_to_frame(series: dict | None) -> pd.DataFrame:
    """Inverse of `compact_series`, for the page."""
    if not series or not series.get("dates"):
        return pd.DataFrame()
    df = pd.DataFrame({k: v for k, v in series.items() if k != "dates"})
    df.index = pd.DatetimeIndex(pd.to_datetime(series["dates"]), name="date")
    return df


def build_technicals(bars: pd.DataFrame, side: str, as_of=None) -> dict:
    """Full technical block for one name: indicators + verdict + flags + alignment."""
    ind = compute_indicators(bars, as_of=as_of)
    verdict, label, reasons = classify(ind)
    ind.update(
        verdict=verdict,
        verdict_label=label,
        verdict_reasons=reasons,
        flags=flags(ind),
        alignment=alignment(side, verdict),
        series=compact_series(bars),
    )
    return ind
