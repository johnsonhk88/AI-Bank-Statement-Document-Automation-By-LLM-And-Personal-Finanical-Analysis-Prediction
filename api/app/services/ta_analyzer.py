"""Technical analysis for personal cashflow — savings signals and spending alerts.

Phase 1: Bollinger Bands + RSI on net cashflow → INVEST/HOLD/ALERT signals.
Phase 2: MACD + SMA crossover on per-category spend → spending trend alerts.

Window sizes are adapted from daily market data to monthly personal finance:
- Bollinger Bands: 6-month window, 2σ (market standard: 20-day)
- RSI: 6-month window (market standard: 14-day)
- MACD: (4, 8, 3) — fast/slow/signal (market standard: 12, 26, 9)
- SMA crossover: 3-month vs 6-month (market standard: 50-day vs 200-day)

All indicators are implemented directly with vectorised pandas — the
upstream `ta` library (bukosabino/ta v0.11.0) has critical correctness bugs
(np.roll wrap-around, off-by-one in ADX, unguarded division-by-zero).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_MONTHS_REQUIRED = 7       # Phase 1: window(6) + 1 for first valid value
MIN_SPENDING_MONTHS_REQUIRED = 9  # Phase 2: slow EMA(8) + signal(3) warm-up needs ~9 months

# RSI thresholds (adapted for personal finance monthly data)
RSI_OVERBOUGHT = 70.0  # savings momentum very high → invest surplus
RSI_OVERSOLD = 30.0    # savings momentum very low → alert


def compute_bollinger_bands(
    values: list[float],
    window: int = 6,
    num_std: int = 2,
) -> dict[str, list[float | None]]:
    """Compute Bollinger Bands on a numeric series.

    Returns dict with keys: middle, upper, lower, pband.
    Uses ddof=1 (sample std) — the correct choice for small monthly windows.
    Values before the window is filled are None.
    """
    s = pd.Series(values, dtype=float)
    middle = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std(ddof=1)

    upper = middle + num_std * std
    lower = middle - num_std * std

    bandwidth = upper - lower
    pband = pd.Series(np.where(
        bandwidth > 0,
        (s - lower) / bandwidth,
        np.nan,
    ), index=s.index)

    return {
        "middle": _series_to_optional_list(middle),
        "upper": _series_to_optional_list(upper),
        "lower": _series_to_optional_list(lower),
        "pband": _series_to_optional_list(pband),
    }


def compute_rsi(
    values: list[float],
    window: int = 6,
) -> list[float | None]:
    """Compute Relative Strength Index using Wilder's smoothing (EMA).

    Guards against division-by-zero:
    - All gains, no losses → RSI = 100
    - All losses, no gains → RSI = 0
    - No movement → RSI = None (indeterminate)
    """
    s = pd.Series(values, dtype=float)
    delta = s.diff()

    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    avg_gain = gains.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    # Vectorised RSI with division-by-zero guards
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.nan)
    rsi_series = pd.Series(100.0 - (100.0 / (1.0 + rs)), index=s.index)

    # All gains, no losses → RSI = 100
    rsi_series = rsi_series.where(avg_loss != 0, other=100.0)
    # All losses, no gains → RSI = 0
    rsi_series = rsi_series.where(avg_gain != 0, other=0.0)
    # No movement at all → indeterminate
    rsi_series = rsi_series.where(~((avg_gain == 0) & (avg_loss == 0)), other=np.nan)
    # Warm-up period → None
    rsi_series.iloc[:window] = np.nan

    return _series_to_optional_list(rsi_series)


def analyze_savings_capacity(
    monthly_data: list[dict[str, Any]],
    bb_window: int = 6,
    bb_std: int = 2,
    rsi_window: int = 6,
) -> dict[str, Any]:
    """Analyze monthly cashflow for savings capacity signals.

    Args:
        monthly_data: List of {"month", "income", "expenses", "net"} dicts,
                      ordered chronologically.
        bb_window: Bollinger Bands rolling window (months).
        bb_std: Number of standard deviations for bands.
        rsi_window: RSI calculation window (months).

    Returns:
        {
            "current_signal": "INVEST" | "HOLD" | "ALERT",
            "rsi": float | None,
            "bollinger": {"pband": float | None, "position": str},
            "recommendation": str,
            "months": [{"month", "net", "bb_upper", "bb_middle", "bb_lower", "rsi"}, ...],
        }

    Raises:
        ValueError: If fewer than MIN_MONTHS_REQUIRED months provided.
    """
    if len(monthly_data) < MIN_MONTHS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_MONTHS_REQUIRED} months of data, "
            f"got {len(monthly_data)}"
        )

    nets = [d["net"] for d in monthly_data]

    bb = compute_bollinger_bands(nets, window=bb_window, num_std=bb_std)
    rsi_values = compute_rsi(nets, window=rsi_window)

    # Current values (last month)
    current_net = nets[-1]
    current_rsi = rsi_values[-1]
    current_pband = bb["pband"][-1]
    current_bb_upper = bb["upper"][-1]
    current_bb_lower = bb["lower"][-1]

    # Determine signal
    signal = _determine_signal(
        net=current_net,
        rsi=current_rsi,
        pband=current_pband,
        bb_upper=current_bb_upper,
        bb_lower=current_bb_lower,
    )

    # Build per-month detail
    months_detail = [
        {
            "month": monthly_data[i]["month"],
            "net": nets[i],
            "bb_upper": bb["upper"][i],
            "bb_middle": bb["middle"][i],
            "bb_lower": bb["lower"][i],
            "rsi": rsi_values[i],
        }
        for i in range(len(monthly_data))
    ]

    recommendation = _build_recommendation(signal, current_rsi, current_pband, current_net)

    return {
        "current_signal": signal,
        "rsi": current_rsi,
        "bollinger": {
            "pband": current_pband,
            "position": _bb_position(current_pband),
        },
        "recommendation": recommendation,
        "months": months_detail,
    }


def _determine_signal(
    net: float,
    rsi: float | None,
    pband: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
) -> str:
    """Combine BB + RSI into a single savings signal.

    Logic:
    - INVEST: net above upper BB OR RSI > 70 (strong surplus momentum)
    - ALERT:  net below lower BB OR RSI < 30 (spending outpacing income)
    - HOLD:   everything else (normal range)
    """
    # If indicators aren't available yet, default to HOLD
    if rsi is None and pband is None:
        return "HOLD"

    invest_score = 0
    alert_score = 0

    if rsi is not None:
        if rsi >= RSI_OVERBOUGHT:
            invest_score += 1
        elif rsi <= RSI_OVERSOLD:
            alert_score += 1

    if pband is not None:
        if pband > 1.0:  # above upper band
            invest_score += 1
        elif pband < 0.0:  # below lower band
            alert_score += 1

    if invest_score > alert_score:
        return "INVEST"
    elif alert_score > invest_score:
        return "ALERT"
    # Tie (e.g. RSI says invest, BB says alert) → conservative default
    return "HOLD"


def _bb_position(pband: float | None) -> str:
    """Human-readable Bollinger Band position."""
    if pband is None:
        return "insufficient_data"
    if pband > 1.0:
        return "above_upper"
    if pband < 0.0:
        return "below_lower"
    if pband > 0.5:
        return "upper_half"
    return "lower_half"


def _build_recommendation(
    signal: str,
    rsi: float | None,
    pband: float | None,
    net: float,
) -> str:
    """Generate a plain-language investment recommendation."""
    rsi_str = f"{rsi:.0f}" if rsi is not None else "N/A"

    if signal == "INVEST":
        return (
            f"Your net savings ({net:,.0f}) show strong surplus momentum "
            f"(RSI: {rsi_str}). Consider allocating excess to investments "
            f"or increasing retirement contributions."
        )
    elif signal == "ALERT":
        return (
            f"Your net savings ({net:,.0f}) are trending below normal "
            f"(RSI: {rsi_str}). Review discretionary spending and pause "
            f"non-essential investment contributions until cash flow stabilises."
        )
    else:
        return (
            f"Your savings ({net:,.0f}) are within normal range "
            f"(RSI: {rsi_str}). Maintain current allocation strategy."
        )


def _series_to_optional_list(s: pd.Series) -> list[float | None]:
    """Convert a pandas Series to a list, replacing NaN with None."""
    return [None if pd.isna(v) else round(float(v), 2) for v in s]


# ===========================================================================
# Phase 2 — Spending trend alerts (MACD + SMA crossover)
# ===========================================================================

# MACD thresholds: histogram magnitude relative to spend level
MACD_SIGNIFICANCE_THRESHOLD = 0.005  # 0.5% of mean spend = meaningful signal for monthly data


def compute_macd(
    values: list[float],
    fast: int = 4,
    slow: int = 8,
    signal: int = 3,
) -> dict[str, list[float | None]]:
    """Compute MACD (Moving Average Convergence Divergence).

    Adapted for monthly personal finance: (4, 8, 3) instead of market (12, 26, 9).

    Returns dict with keys: macd_line, signal_line, histogram.
    Values before the slow EMA is warmed up are None.
    """
    s = pd.Series(values, dtype=float)

    ema_fast = s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, min_periods=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    # Mask warm-up period where slow EMA isn't ready
    macd_line.iloc[:slow - 1] = np.nan

    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    # Signal line warm-up: needs slow warm-up + signal warm-up
    signal_line.iloc[:slow + signal - 2] = np.nan

    histogram = macd_line - signal_line

    return {
        "macd_line": _series_to_optional_list(macd_line),
        "signal_line": _series_to_optional_list(signal_line),
        "histogram": _series_to_optional_list(histogram),
    }


def compute_sma_crossover(
    values: list[float],
    short_window: int = 3,
    long_window: int = 6,
) -> dict[str, Any]:
    """Compute short vs long SMA and detect crossover.

    Returns:
        short_sma, long_sma: lists with None for warm-up.
        crossover: "golden_cross" (short crossed above long recently),
                   "death_cross" (short crossed below long recently),
                   or "none".
    """
    s = pd.Series(values, dtype=float)
    short_sma = s.rolling(window=short_window, min_periods=short_window).mean()
    long_sma = s.rolling(window=long_window, min_periods=long_window).mean()

    # Detect crossover in the last 2 periods
    crossover = "none"
    if len(values) >= long_window + 1:
        prev_diff = short_sma.iloc[-2] - long_sma.iloc[-2]
        curr_diff = short_sma.iloc[-1] - long_sma.iloc[-1]

        if not (pd.isna(prev_diff) or pd.isna(curr_diff)):
            if prev_diff <= 0 < curr_diff:
                crossover = "golden_cross"
            elif prev_diff >= 0 > curr_diff:
                crossover = "death_cross"

    return {
        "short_sma": _series_to_optional_list(short_sma),
        "long_sma": _series_to_optional_list(long_sma),
        "crossover": crossover,
    }


def analyze_spending_trends(
    category_data: dict[str, list[dict[str, Any]]],
    macd_fast: int = 4,
    macd_slow: int = 8,
    macd_signal: int = 3,
    sma_short: int = 3,
    sma_long: int = 6,
) -> dict[str, Any]:
    """Analyze per-category spending trends using MACD + SMA crossover.

    Args:
        category_data: Dict of {category_name: [{"month", "amount"}, ...]},
                       each list ordered chronologically.

    Returns:
        {
            "categories": [{
                "category": str,
                "trend": "accelerating" | "decelerating" | "stable",
                "alert": bool,
                "macd_histogram": float | None,
                "sma_crossover": str,
                "recommendation": str,
                "months": [{"month", "amount", "macd", "signal", "short_sma", "long_sma"}, ...],
            }, ...],
            "alert_count": int,
            "total_categories": int,
        }

    Raises:
        ValueError: If no category has enough months.
    """
    has_any_valid = any(
        len(months) >= MIN_SPENDING_MONTHS_REQUIRED
        for months in category_data.values()
    )
    if not has_any_valid:
        raise ValueError(
            f"Need at least {MIN_SPENDING_MONTHS_REQUIRED} months of data "
            f"in at least one category"
        )

    categories_result = []

    for category, months_data in category_data.items():
        if len(months_data) < MIN_SPENDING_MONTHS_REQUIRED:
            logger.info("Skipping %s — only %d months", category, len(months_data))
            continue

        amounts = [d["amount"] for d in months_data]
        mean_spend = sum(amounts) / len(amounts)

        macd = compute_macd(amounts, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        sma = compute_sma_crossover(amounts, short_window=sma_short, long_window=sma_long)

        trend = _determine_spending_trend(macd, sma, mean_spend)

        # Build per-month detail
        months_detail = [
            {
                "month": months_data[i]["month"],
                "amount": amounts[i],
                "macd": macd["macd_line"][i],
                "signal": macd["signal_line"][i],
                "short_sma": sma["short_sma"][i],
                "long_sma": sma["long_sma"][i],
            }
            for i in range(len(months_data))
        ]

        alert = trend == "accelerating"
        recommendation = _build_spending_recommendation(
            category, trend, macd["histogram"][-1], sma["crossover"], mean_spend
        )

        categories_result.append({
            "category": category,
            "trend": trend,
            "alert": alert,
            "macd_histogram": macd["histogram"][-1],
            "sma_crossover": sma["crossover"],
            "recommendation": recommendation,
            "months": months_detail,
        })

    alert_count = sum(1 for c in categories_result if c["alert"])

    return {
        "categories": categories_result,
        "alert_count": alert_count,
        "total_categories": len(categories_result),
    }


def _determine_spending_trend(
    macd: dict[str, list[float | None]],
    sma: dict[str, Any],
    mean_spend: float,
) -> str:
    """Combine MACD histogram + SMA crossover + SMA direction into a spending trend.

    Logic:
    - accelerating: MACD histogram positive AND significant,
                    OR golden cross, OR short SMA consistently above long SMA
    - decelerating: MACD histogram negative AND significant,
                    OR death cross, OR short SMA consistently below long SMA
    - stable: MACD near zero, SMAs converged, no crossover
    """
    last_hist = macd["histogram"][-1]
    crossover = sma["crossover"]

    threshold = max(mean_spend * MACD_SIGNIFICANCE_THRESHOLD, 1.0)

    macd_accelerating = last_hist is not None and last_hist > threshold
    macd_decelerating = last_hist is not None and last_hist < -threshold

    # Also check persistent SMA direction (short vs long)
    short_last = sma["short_sma"][-1]
    long_last = sma["long_sma"][-1]
    sma_above = (
        short_last is not None
        and long_last is not None
        and short_last > long_last
    )
    sma_below = (
        short_last is not None
        and long_last is not None
        and short_last < long_last
    )

    # MACD confirms direction, OR SMA position + any non-zero MACD signal
    if macd_accelerating or crossover == "golden_cross":
        return "accelerating"
    elif macd_decelerating or crossover == "death_cross":
        return "decelerating"
    elif sma_above and last_hist is not None and last_hist > 0:
        return "accelerating"
    elif sma_below and last_hist is not None and last_hist < 0:
        return "decelerating"
    return "stable"


def _build_spending_recommendation(
    category: str,
    trend: str,
    histogram: float | None,
    crossover: str,
    mean_spend: float,
) -> str:
    """Generate a plain-language spending trend recommendation."""
    hist_str = f"{histogram:,.0f}" if histogram is not None else "N/A"

    if trend == "accelerating":
        parts = [f"Your {category} spending is accelerating"]
        if crossover == "golden_cross":
            parts.append("with a recent 3-month/6-month crossover")
        parts.append(
            f"(MACD: {hist_str}, avg: {mean_spend:,.0f}/month). "
            f"Review recent {category.lower()} expenses for cuts."
        )
        return " ".join(parts)
    elif trend == "decelerating":
        return (
            f"Your {category} spending is trending down "
            f"(MACD: {hist_str}, avg: {mean_spend:,.0f}/month). "
            f"Good progress — savings from this category can be redirected."
        )
    else:
        return (
            f"Your {category} spending is stable "
            f"(avg: {mean_spend:,.0f}/month). No action needed."
        )
