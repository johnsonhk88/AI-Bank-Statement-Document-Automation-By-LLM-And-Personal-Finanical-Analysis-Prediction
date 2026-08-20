"""Cashflow forecaster — Prophet-based or linear fallback.

Takes monthly income/expense history and projects future months
with confidence intervals.
"""
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False
    logger.warning(
        "Prophet not installed — using linear trend fallback. "
        "Install with: pip install prophet"
    )


def _forecast_with_prophet(
    months: list[str],
    values: list[float],
    horizon: int,
) -> list[dict]:
    """Forecast using Prophet. Returns list of {month, predicted, lower, upper}."""
    import pandas as pd

    df = pd.DataFrame({
        "ds": pd.to_datetime([f"{m}-01" for m in months]),
        "y": values,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)

    # Only return the forecast periods (not historical)
    forecast_rows = forecast.tail(horizon)
    results = []
    for _, row in forecast_rows.iterrows():
        results.append({
            "month": row["ds"].strftime("%Y-%m"),
            "predicted": Decimal(str(round(max(row["yhat"], 0), 2))),
            "lower": Decimal(str(round(max(row["yhat_lower"], 0), 2))),
            "upper": Decimal(str(round(max(row["yhat_upper"], 0), 2))),
        })
    return results


def _forecast_linear(
    months: list[str],
    values: list[float],
    horizon: int,
) -> list[dict]:
    """Simple linear trend fallback when Prophet is not available."""
    n = len(values)
    if n == 0:
        return []

    # Linear regression: y = a + b*x
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean

    # Compute residual std for confidence bands
    predictions = [intercept + slope * i for i in range(n)]
    residuals = [v - p for v, p in zip(values, predictions)]
    std = (sum(r ** 2 for r in residuals) / max(n - 2, 1)) ** 0.5

    # Parse last month to generate future months
    last_year, last_month_num = int(months[-1][:4]), int(months[-1][5:7])

    results = []
    for h in range(1, horizon + 1):
        x = n - 1 + h
        predicted = max(intercept + slope * x, 0)
        lower = max(predicted - 1.96 * std, 0)
        upper = predicted + 1.96 * std

        # Advance month
        new_month = last_month_num + h
        new_year = last_year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1

        results.append({
            "month": f"{new_year:04d}-{new_month:02d}",
            "predicted": Decimal(str(round(predicted, 2))),
            "lower": Decimal(str(round(lower, 2))),
            "upper": Decimal(str(round(upper, 2))),
        })
    return results


def forecast_cashflow(
    monthly_data: list[dict],
    horizon_months: int,
) -> dict[str, list[dict]]:
    """Forecast income, expenses, and net for the given horizon.

    Args:
        monthly_data: List of {"month": "YYYY-MM", "income": float, "expenses": float, "net": float}
        horizon_months: Number of months to forecast

    Returns:
        {"income": [...], "expenses": [...], "net": [...]} where each is a list of
        {"month", "predicted", "lower", "upper"} dicts.
    """
    months = [d["month"] for d in monthly_data]
    forecast_fn = _forecast_with_prophet if _PROPHET_AVAILABLE else _forecast_linear

    result = {}
    for series_key in ("income", "expenses", "net"):
        values = [d[series_key] for d in monthly_data]
        result[series_key] = forecast_fn(months, values, horizon_months)

    return result
