"""Technical analysis savings-capacity analyzer.

Applies Bollinger Bands and RSI to monthly net cashflow to generate
investment signals: INVEST (surplus detected), HOLD (normal), ALERT (deficit).

Window sizes are adapted from daily market data to monthly personal finance:
- Bollinger Bands: 6-month window, 2σ (market standard: 20-day)
- RSI: 6-month window (market standard: 14-day)

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

MIN_MONTHS_REQUIRED = 7  # window(6) + 1 for first valid value

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
