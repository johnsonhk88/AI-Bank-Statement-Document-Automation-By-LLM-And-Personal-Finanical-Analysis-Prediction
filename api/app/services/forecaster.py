"""Cashflow forecaster — Prophet-based or linear fallback.

Takes monthly or weekly income/expense history and projects future
periods with confidence intervals.
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


# ---------------------------------------------------------------------------
# Weekly forecasting (Phase 2.7)
# ---------------------------------------------------------------------------


def _forecast_weekly_linear(
    week_labels: list[str],
    values: list[float],
    horizon: int,
) -> list[dict]:
    """Linear trend forecast for weekly data.

    Args:
        week_labels: ISO week labels like "2026-W23".
        values: Numeric values for each week.
        horizon: Number of future weeks to predict.

    Returns:
        List of {"week", "predicted", "lower", "upper"} dicts.
    """
    n = len(values)
    if n == 0:
        return []

    # Linear regression
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean

    # Residual std for confidence bands
    predictions = [intercept + slope * i for i in range(n)]
    residuals = [v - p for v, p in zip(values, predictions)]
    std = (sum(r ** 2 for r in residuals) / max(n - 2, 1)) ** 0.5

    # Parse last ISO week to generate future weeks
    last_label = week_labels[-1]
    last_year = int(last_label[:4])
    last_week_num = int(last_label.split("W")[1])

    from datetime import date, timedelta

    # Find the Monday of the last ISO week
    jan4 = date(last_year, 1, 4)
    start_of_w1 = jan4 - timedelta(days=jan4.weekday())
    last_monday = start_of_w1 + timedelta(weeks=last_week_num - 1)

    results = []
    for h in range(1, horizon + 1):
        x = n - 1 + h
        predicted = max(intercept + slope * x, 0)
        lower = max(predicted - 1.96 * std, 0)
        upper = predicted + 1.96 * std

        future_monday = last_monday + timedelta(weeks=h)
        iso = future_monday.isocalendar()
        week_label = f"{iso[0]}-W{iso[1]:02d}"

        results.append({
            "week": week_label,
            "predicted": Decimal(str(round(predicted, 2))),
            "lower": Decimal(str(round(lower, 2))),
            "upper": Decimal(str(round(upper, 2))),
        })
    return results


def _forecast_weekly_prophet(
    week_labels: list[str],
    values: list[float],
    horizon: int,
) -> list[dict]:
    """Prophet-based weekly forecast."""
    import pandas as pd
    from datetime import date, timedelta

    # Convert ISO week labels to dates (Monday of each week)
    dates = []
    for label in week_labels:
        year = int(label[:4])
        week_num = int(label.split("W")[1])
        jan4 = date(year, 1, 4)
        start_of_w1 = jan4 - timedelta(days=jan4.weekday())
        monday = start_of_w1 + timedelta(weeks=week_num - 1)
        dates.append(pd.Timestamp(monday))

    df = pd.DataFrame({"ds": dates, "y": values})

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="W-MON")
    forecast = model.predict(future)

    forecast_rows = forecast.tail(horizon)
    results = []
    for _, row in forecast_rows.iterrows():
        d = row["ds"].date()
        iso = d.isocalendar()
        week_label = f"{iso[0]}-W{iso[1]:02d}"
        results.append({
            "week": week_label,
            "predicted": Decimal(str(round(max(row["yhat"], 0), 2))),
            "lower": Decimal(str(round(max(row["yhat_lower"], 0), 2))),
            "upper": Decimal(str(round(max(row["yhat_upper"], 0), 2))),
        })
    return results


def forecast_weekly(
    weekly_data: list[dict],
    horizon_weeks: int,
) -> dict[str, list[dict]]:
    """Forecast income, expenses, and net for the given weekly horizon.

    Args:
        weekly_data: List of {"week": "YYYY-Www", "income": float,
                     "expenses": float, "net": float}
        horizon_weeks: Number of weeks to forecast.

    Returns:
        {"income": [...], "expenses": [...], "net": [...]} where each is a list
        of {"week", "predicted", "lower", "upper"} dicts.
    """
    if not weekly_data:
        return {"income": [], "expenses": [], "net": []}

    weeks = [d["week"] for d in weekly_data]
    forecast_fn = _forecast_weekly_prophet if _PROPHET_AVAILABLE else _forecast_weekly_linear

    result = {}
    for series_key in ("income", "expenses", "net"):
        values = [d[series_key] for d in weekly_data]
        result[series_key] = forecast_fn(weeks, values, horizon_weeks)

    return result
