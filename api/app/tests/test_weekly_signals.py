"""Tests for weekly TA signals — Phase 2.7.

Covers weekly aggregation, weekly TA indicators (RSI-4w, BB-4w, MACD-2,4,2),
weekly forecast (linear fallback), and week-over-week comparison.
"""
import pytest
from datetime import date, timedelta

from app.services.ta_analyzer import (
    compute_bollinger_bands,
    compute_rsi,
    compute_macd,
    analyze_weekly_signals,
    MIN_WEEKS_REQUIRED,
)
from app.services.forecaster import forecast_weekly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _iso_week_label(d: date) -> str:
    """Return 'YYYY-Www' ISO week label for a date."""
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


@pytest.fixture
def normal_weekly_data() -> list[dict]:
    """12 weeks of realistic weekly cashflow data."""
    base = date(2026, 6, 1)  # a Monday
    nets = [3500, 4200, 2800, 5100, 3800, 3200, 4500, 3900, 2500, 5000, 3600, 6000]
    result = []
    for i, net in enumerate(nets):
        d = base + timedelta(weeks=i)
        income = net + 7000.0
        result.append({
            "week": _iso_week_label(d),
            "week_start": d.isoformat(),
            "income": income,
            "expenses": 7000.0,
            "net": float(net),
        })
    return result


@pytest.fixture
def flat_weekly_data() -> list[dict]:
    """12 weeks of identical cashflow — tests division-by-zero guards."""
    base = date(2026, 6, 1)
    return [
        {
            "week": _iso_week_label(base + timedelta(weeks=i)),
            "week_start": (base + timedelta(weeks=i)).isoformat(),
            "income": 10000.0,
            "expenses": 7000.0,
            "net": 3000.0,
        }
        for i in range(12)
    ]


@pytest.fixture
def declining_weekly_data() -> list[dict]:
    """12 weeks of steadily worsening net — RSI should go low."""
    base = date(2026, 6, 1)
    nets = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 800, 500, 200]
    result = []
    for i, net in enumerate(nets):
        d = base + timedelta(weeks=i)
        result.append({
            "week": _iso_week_label(d),
            "week_start": d.isoformat(),
            "income": 10000.0,
            "expenses": 10000.0 - net,
            "net": float(net),
        })
    return result


@pytest.fixture
def surging_weekly_data() -> list[dict]:
    """12 weeks of rapidly increasing net — RSI should go high."""
    base = date(2026, 6, 1)
    nets = [1000, 1500, 2200, 3000, 3800, 4800, 5800, 7000, 8500, 10000, 11500, 13000]
    result = []
    for i, net in enumerate(nets):
        d = base + timedelta(weeks=i)
        result.append({
            "week": _iso_week_label(d),
            "week_start": d.isoformat(),
            "income": float(net + 5000),
            "expenses": 5000.0,
            "net": float(net),
        })
    return result


@pytest.fixture
def short_weekly_data() -> list[dict]:
    """Only 4 weeks — below MIN_WEEKS_REQUIRED."""
    base = date(2026, 6, 1)
    return [
        {
            "week": _iso_week_label(base + timedelta(weeks=i)),
            "week_start": (base + timedelta(weeks=i)).isoformat(),
            "income": 10000.0,
            "expenses": 7000.0,
            "net": 3000.0,
        }
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# Weekly TA indicators (reusing compute_* with weekly window params)
# ---------------------------------------------------------------------------

class TestWeeklyBollingerBands:
    """BB with window=4 for weekly data."""

    def test_returns_correct_length(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_bollinger_bands(nets, window=4, num_std=2)
        assert len(result["middle"]) == len(nets)

    def test_warmup_none(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_bollinger_bands(nets, window=4, num_std=2)
        for i in range(3):
            assert result["middle"][i] is None

    def test_valid_after_warmup(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_bollinger_bands(nets, window=4, num_std=2)
        for i in range(3, len(nets)):
            assert result["middle"][i] is not None
            assert result["upper"][i] >= result["lower"][i]


class TestWeeklyRSI:
    """RSI with window=4 for weekly data."""

    def test_returns_correct_length(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_rsi(nets, window=4)
        assert len(result) == len(nets)

    def test_warmup_none(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_rsi(nets, window=4)
        for i in range(4):
            assert result[i] is None

    def test_bounded_0_100(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_rsi(nets, window=4)
        for v in result:
            if v is not None:
                assert 0.0 <= v <= 100.0

    def test_declining_low_rsi(self, declining_weekly_data: list[dict]):
        nets = [d["net"] for d in declining_weekly_data]
        result = compute_rsi(nets, window=4)
        last = result[-1]
        assert last is not None
        assert last < 30.0

    def test_surging_high_rsi(self, surging_weekly_data: list[dict]):
        nets = [d["net"] for d in surging_weekly_data]
        result = compute_rsi(nets, window=4)
        last = result[-1]
        assert last is not None
        assert last > 70.0


class TestWeeklyMACD:
    """MACD with (2, 4, 2) for weekly data."""

    def test_returns_correct_length(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_macd(nets, fast=2, slow=4, signal=2)
        assert len(result["macd_line"]) == len(nets)

    def test_warmup_none(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_macd(nets, fast=2, slow=4, signal=2)
        # First slow-1 = 3 values should be None
        for i in range(3):
            assert result["macd_line"][i] is None

    def test_signal_line_warmup(self, normal_weekly_data: list[dict]):
        nets = [d["net"] for d in normal_weekly_data]
        result = compute_macd(nets, fast=2, slow=4, signal=2)
        # slow + signal - 2 = 4 values should be None for signal
        for i in range(4):
            assert result["signal_line"][i] is None


# ---------------------------------------------------------------------------
# analyze_weekly_signals integration tests
# ---------------------------------------------------------------------------

class TestAnalyzeWeeklySignals:
    def test_insufficient_data_raises(self, short_weekly_data: list[dict]):
        with pytest.raises(ValueError, match="at least"):
            analyze_weekly_signals(short_weekly_data)

    def test_returns_expected_structure(self, normal_weekly_data: list[dict]):
        result = analyze_weekly_signals(normal_weekly_data)
        assert "current_signal" in result
        assert "rsi" in result
        assert "bollinger" in result
        assert "recommendation" in result
        assert "weeks" in result
        assert "comparison" in result

    def test_signal_is_valid_enum(self, normal_weekly_data: list[dict]):
        result = analyze_weekly_signals(normal_weekly_data)
        assert result["current_signal"] in ("INVEST", "HOLD", "ALERT")

    def test_surging_invest(self, surging_weekly_data: list[dict]):
        result = analyze_weekly_signals(surging_weekly_data)
        assert result["current_signal"] == "INVEST"

    def test_declining_alert(self, declining_weekly_data: list[dict]):
        result = analyze_weekly_signals(declining_weekly_data)
        assert result["current_signal"] == "ALERT"

    def test_flat_hold(self, flat_weekly_data: list[dict]):
        result = analyze_weekly_signals(flat_weekly_data)
        assert result["current_signal"] == "HOLD"

    def test_weeks_output_matches_input(self, normal_weekly_data: list[dict]):
        result = analyze_weekly_signals(normal_weekly_data)
        assert len(result["weeks"]) == len(normal_weekly_data)

    def test_each_week_has_fields(self, normal_weekly_data: list[dict]):
        result = analyze_weekly_signals(normal_weekly_data)
        for w in result["weeks"]:
            assert "week" in w
            assert "net" in w
            assert "bb_upper" in w
            assert "rsi" in w

    def test_comparison_structure(self, normal_weekly_data: list[dict]):
        result = analyze_weekly_signals(normal_weekly_data)
        comp = result["comparison"]
        assert "this_week_net" in comp
        assert "last_week_net" in comp
        assert "week_over_week_change" in comp
        assert "vs_4_weeks_ago" in comp

    def test_week_over_week_change_correct(self, normal_weekly_data: list[dict]):
        result = analyze_weekly_signals(normal_weekly_data)
        comp = result["comparison"]
        expected = comp["this_week_net"] - comp["last_week_net"]
        assert abs(comp["week_over_week_change"] - expected) < 0.01


# ---------------------------------------------------------------------------
# Weekly forecast tests
# ---------------------------------------------------------------------------

class TestWeeklyForecast:
    def test_returns_correct_horizon(self, normal_weekly_data: list[dict]):
        result = forecast_weekly(normal_weekly_data, horizon_weeks=4)
        assert len(result["net"]) == 4
        assert len(result["income"]) == 4
        assert len(result["expenses"]) == 4

    def test_forecast_points_have_required_fields(self, normal_weekly_data: list[dict]):
        result = forecast_weekly(normal_weekly_data, horizon_weeks=4)
        for pt in result["net"]:
            assert "week" in pt
            assert "predicted" in pt
            assert "lower" in pt
            assert "upper" in pt

    def test_lower_le_predicted_le_upper(self, normal_weekly_data: list[dict]):
        result = forecast_weekly(normal_weekly_data, horizon_weeks=4)
        for pt in result["net"]:
            assert pt["lower"] <= pt["predicted"] <= pt["upper"]

    def test_forecast_weeks_are_future(self, normal_weekly_data: list[dict]):
        last_week = normal_weekly_data[-1]["week"]
        result = forecast_weekly(normal_weekly_data, horizon_weeks=4)
        for pt in result["net"]:
            assert pt["week"] > last_week

    def test_empty_data_returns_empty(self):
        result = forecast_weekly([], horizon_weeks=4)
        assert result["net"] == []
        assert result["income"] == []
        assert result["expenses"] == []
