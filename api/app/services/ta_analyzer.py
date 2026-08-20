"""Technical analysis for personal cashflow — savings signals and spending alerts.

Phase 1: Bollinger Bands + RSI on net cashflow → INVEST/HOLD/ALERT signals.
Phase 2: MACD + SMA crossover on per-category spend → spending trend alerts.
Phase 2.7: Weekly signals — same indicators with adapted windows.
Phase 3: ATR on income — income stability / risk profile.

Window sizes are adapted from daily market data to monthly/weekly personal finance:
  Monthly: BB(6), RSI(6), MACD(4,8,3), SMA(3/6), ATR(6)
  Weekly:  BB(4), RSI(4), MACD(2,4,2)

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
MIN_WEEKS_REQUIRED = 8        # Phase 2.7: BB(4)+RSI(4) need ≥5, MACD(2,4,2) needs ≥5, use 8 for safety

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


# ===========================================================================
# Phase 2.7 — Weekly signals (BB-4w + RSI-4w + MACD-2,4,2)
# ===========================================================================

WEEKLY_BB_WINDOW = 4
WEEKLY_RSI_WINDOW = 4
WEEKLY_MACD_FAST = 2
WEEKLY_MACD_SLOW = 4
WEEKLY_MACD_SIGNAL = 2


def analyze_weekly_signals(
    weekly_data: list[dict[str, Any]],
    bb_window: int = WEEKLY_BB_WINDOW,
    bb_std: int = 2,
    rsi_window: int = WEEKLY_RSI_WINDOW,
) -> dict[str, Any]:
    """Analyze weekly cashflow for savings capacity signals.

    Same logic as monthly analyze_savings_capacity but with weekly windows.
    Adds week-over-week comparison.

    Args:
        weekly_data: List of {"week", "week_start", "income", "expenses", "net"},
                     ordered chronologically.

    Returns:
        {
            "current_signal": "INVEST" | "HOLD" | "ALERT",
            "rsi": float | None,
            "bollinger": {"pband": float | None, "position": str},
            "recommendation": str,
            "weeks": [{"week", "net", "bb_upper", "bb_middle", "bb_lower", "rsi"}, ...],
            "comparison": {
                "this_week_net": float,
                "last_week_net": float,
                "week_over_week_change": float,
                "vs_4_weeks_ago": float | None,
            },
        }

    Raises:
        ValueError: If fewer than MIN_WEEKS_REQUIRED weeks provided.
    """
    if len(weekly_data) < MIN_WEEKS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_WEEKS_REQUIRED} weeks of data, "
            f"got {len(weekly_data)}"
        )

    nets = [d["net"] for d in weekly_data]

    bb = compute_bollinger_bands(nets, window=bb_window, num_std=bb_std)
    rsi_values = compute_rsi(nets, window=rsi_window)

    current_rsi = rsi_values[-1]
    current_pband = bb["pband"][-1]
    current_bb_upper = bb["upper"][-1]
    current_bb_lower = bb["lower"][-1]

    signal = _determine_signal(
        net=nets[-1],
        rsi=current_rsi,
        pband=current_pband,
        bb_upper=current_bb_upper,
        bb_lower=current_bb_lower,
    )

    weeks_detail = [
        {
            "week": weekly_data[i]["week"],
            "net": nets[i],
            "bb_upper": bb["upper"][i],
            "bb_middle": bb["middle"][i],
            "bb_lower": bb["lower"][i],
            "rsi": rsi_values[i],
        }
        for i in range(len(weekly_data))
    ]

    # Week-over-week comparison
    this_week = nets[-1]
    last_week = nets[-2]
    vs_4w = nets[-5] if len(nets) >= 5 else None

    comparison = {
        "this_week_net": this_week,
        "last_week_net": last_week,
        "week_over_week_change": round(this_week - last_week, 2),
        "vs_4_weeks_ago": round(this_week - vs_4w, 2) if vs_4w is not None else None,
    }

    recommendation = _build_weekly_recommendation(signal, current_rsi, this_week, last_week)

    return {
        "current_signal": signal,
        "rsi": current_rsi,
        "bollinger": {
            "pband": current_pband,
            "position": _bb_position(current_pband),
        },
        "recommendation": recommendation,
        "weeks": weeks_detail,
        "comparison": comparison,
    }


def _build_weekly_recommendation(
    signal: str,
    rsi: float | None,
    this_week: float,
    last_week: float,
) -> str:
    """Generate a plain-language weekly recommendation."""
    rsi_str = f"{rsi:.0f}" if rsi is not None else "N/A"
    change = this_week - last_week
    direction = "up" if change > 0 else "down" if change < 0 else "flat"

    if signal == "INVEST":
        return (
            f"Weekly net ({this_week:,.0f}) is {direction} from last week "
            f"({last_week:,.0f}), RSI: {rsi_str}. Strong surplus — consider "
            f"investing excess savings."
        )
    elif signal == "ALERT":
        return (
            f"Weekly net ({this_week:,.0f}) is {direction} from last week "
            f"({last_week:,.0f}), RSI: {rsi_str}. Spending is elevated — "
            f"review this week's discretionary expenses."
        )
    else:
        return (
            f"Weekly net ({this_week:,.0f}) is {direction} from last week "
            f"({last_week:,.0f}), RSI: {rsi_str}. Cash flow is within "
            f"normal range."
        )


# ===========================================================================
# Phase 3 — Income stability (ATR on income)
# ===========================================================================

# ATR% thresholds for income risk profile
ATR_LOW_THRESHOLD = 10.0    # <10% = stable salaried income
ATR_HIGH_THRESHOLD = 25.0   # >25% = volatile freelancer/commission income

INCOME_ATR_WINDOW = 6  # 6-month ATR window


def compute_income_atr(
    incomes: list[float],
    window: int = INCOME_ATR_WINDOW,
) -> list[float | None]:
    """Compute Average True Range adapted for monthly income.

    In markets, ATR uses High-Low-Close candles. For personal income we have
    one value per month, so True Range = |income[t] - income[t-1]|.
    ATR is the EMA-smoothed TR (Wilder smoothing, same as RSI).

    Returns list of ATR values with None for warm-up period.
    """
    s = pd.Series(incomes, dtype=float)
    # True Range = absolute month-over-month change
    tr = s.diff().abs()

    # EMA smoothing (Wilder's method: alpha = 1/window)
    atr = tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    # Warm-up: first value has no diff, then need window periods for EMA
    atr.iloc[:window] = np.nan

    return _series_to_optional_list(atr)


def analyze_income_stability(
    monthly_data: list[dict[str, Any]],
    atr_window: int = INCOME_ATR_WINDOW,
) -> dict[str, Any]:
    """Analyze monthly income for stability and risk profile.

    Args:
        monthly_data: List of {"month", "income", "expenses", "net"} dicts,
                      ordered chronologically.

    Returns:
        {
            "risk_profile": "LOW" | "MEDIUM" | "HIGH",
            "atr": float,
            "atr_percent": float,
            "mean_income": float,
            "income_trend": "growing" | "declining" | "stable",
            "recommendation": str,
            "months": [{"month", "income", "atr"}, ...],
        }

    Raises:
        ValueError: If fewer than MIN_MONTHS_REQUIRED months provided.
    """
    if len(monthly_data) < MIN_MONTHS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_MONTHS_REQUIRED} months of data, "
            f"got {len(monthly_data)}"
        )

    incomes = [d["income"] for d in monthly_data]
    atr_values = compute_income_atr(incomes, window=atr_window)

    # Current ATR (last valid value)
    current_atr = atr_values[-1] if atr_values[-1] is not None else 0.0
    mean_income = sum(incomes) / len(incomes)

    # ATR as percentage of mean income
    atr_pct = (current_atr / mean_income * 100) if mean_income > 0 else 0.0
    atr_pct = round(atr_pct, 1)

    # Risk profile
    risk_profile = _classify_risk(atr_pct)

    # Income trend (simple: compare first-half average to second-half average)
    half = len(incomes) // 2
    first_half_avg = sum(incomes[:half]) / half if half > 0 else 0
    second_half_avg = sum(incomes[half:]) / (len(incomes) - half) if len(incomes) > half else 0

    trend_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
    if trend_pct > 5:
        income_trend = "growing"
    elif trend_pct < -5:
        income_trend = "declining"
    else:
        income_trend = "stable"

    # Build per-month detail
    months_detail = [
        {
            "month": monthly_data[i]["month"],
            "income": incomes[i],
            "atr": atr_values[i],
        }
        for i in range(len(monthly_data))
    ]

    recommendation = _build_stability_recommendation(
        risk_profile, atr_pct, mean_income, income_trend
    )

    return {
        "risk_profile": risk_profile,
        "atr": round(current_atr, 2),
        "atr_percent": atr_pct,
        "mean_income": round(mean_income, 2),
        "income_trend": income_trend,
        "recommendation": recommendation,
        "months": months_detail,
    }


def _classify_risk(atr_pct: float) -> str:
    """Classify income volatility into a risk profile."""
    if atr_pct < ATR_LOW_THRESHOLD:
        return "LOW"
    elif atr_pct > ATR_HIGH_THRESHOLD:
        return "HIGH"
    return "MEDIUM"


def _build_stability_recommendation(
    risk: str,
    atr_pct: float,
    mean_income: float,
    trend: str,
) -> str:
    """Generate plain-language income stability recommendation."""
    trend_note = ""
    if trend == "growing":
        trend_note = " Income is trending upward — a good sign for increasing contributions."
    elif trend == "declining":
        trend_note = " Income is trending downward — consider building a larger buffer."

    if risk == "LOW":
        return (
            f"Your income is very stable (volatility: {atr_pct:.1f}%, "
            f"avg: {mean_income:,.0f}/month). You can invest more "
            f"aggressively — consider allocating 20-30% of surplus to "
            f"growth assets.{trend_note}"
        )
    elif risk == "HIGH":
        return (
            f"Your income is highly variable (volatility: {atr_pct:.1f}%, "
            f"avg: {mean_income:,.0f}/month). Maintain a 6-month emergency "
            f"fund before investing. Prefer liquid, low-risk assets.{trend_note}"
        )
    else:
        return (
            f"Your income has moderate variability (volatility: {atr_pct:.1f}%, "
            f"avg: {mean_income:,.0f}/month). Keep a 3-month emergency "
            f"fund and invest surplus in a balanced portfolio.{trend_note}"
        )
