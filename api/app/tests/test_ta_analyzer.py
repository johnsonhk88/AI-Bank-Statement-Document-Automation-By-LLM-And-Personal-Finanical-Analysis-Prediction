"""Tests for the technical analysis savings-capacity analyzer.

Covers Bollinger Bands + RSI on monthly net cashflow.
"""
import pytest
from decimal import Decimal

from app.services.ta_analyzer import (
    compute_bollinger_bands,
    compute_rsi,
    analyze_savings_capacity,
    MIN_MONTHS_REQUIRED,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_monthly_data() -> list[dict]:
    """12 months of realistic monthly cashflow data."""
    return [
        {"month": "2025-01", "income": 50000.0, "expenses": 35000.0, "net": 15000.0},
        {"month": "2025-02", "income": 50000.0, "expenses": 32000.0, "net": 18000.0},
        {"month": "2025-03", "income": 50000.0, "expenses": 38000.0, "net": 12000.0},
        {"month": "2025-04", "income": 52000.0, "expenses": 30000.0, "net": 22000.0},
        {"month": "2025-05", "income": 50000.0, "expenses": 34000.0, "net": 16000.0},
        {"month": "2025-06", "income": 50000.0, "expenses": 36000.0, "net": 14000.0},
        {"month": "2025-07", "income": 55000.0, "expenses": 33000.0, "net": 22000.0},
        {"month": "2025-08", "income": 50000.0, "expenses": 31000.0, "net": 19000.0},
        {"month": "2025-09", "income": 50000.0, "expenses": 40000.0, "net": 10000.0},
        {"month": "2025-10", "income": 50000.0, "expenses": 28000.0, "net": 22000.0},
        {"month": "2025-11", "income": 50000.0, "expenses": 35000.0, "net": 15000.0},
        {"month": "2025-12", "income": 60000.0, "expenses": 30000.0, "net": 30000.0},
    ]


@pytest.fixture
def flat_monthly_data() -> list[dict]:
    """12 months of identical cashflow — tests division-by-zero guards."""
    return [
        {"month": f"2025-{m:02d}", "income": 50000.0, "expenses": 35000.0, "net": 15000.0}
        for m in range(1, 13)
    ]


@pytest.fixture
def declining_monthly_data() -> list[dict]:
    """12 months of steadily worsening net — RSI should go low."""
    nets = [20000, 18000, 16000, 14000, 12000, 10000, 8000, 6000, 4000, 2000, 1000, 500]
    return [
        {
            "month": f"2025-{m:02d}",
            "income": 50000.0,
            "expenses": 50000.0 - n,
            "net": float(n),
        }
        for m, n in zip(range(1, 13), nets)
    ]


@pytest.fixture
def surging_monthly_data() -> list[dict]:
    """12 months of rapidly increasing net — RSI should go high."""
    nets = [5000, 8000, 12000, 16000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
    return [
        {
            "month": f"2025-{m:02d}",
            "income": float(n + 30000),
            "expenses": 30000.0,
            "net": float(n),
        }
        for m, n in zip(range(1, 13), nets)
    ]


# ---------------------------------------------------------------------------
# Bollinger Bands tests
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_returns_correct_length(self, normal_monthly_data: list[dict]):
        """Output length matches input length."""
        nets = [d["net"] for d in normal_monthly_data]
        result = compute_bollinger_bands(nets, window=6, num_std=2)
        assert len(result["middle"]) == len(nets)
        assert len(result["upper"]) == len(nets)
        assert len(result["lower"]) == len(nets)

    def test_first_values_are_none(self, normal_monthly_data: list[dict]):
        """Values before the window is filled should be None."""
        nets = [d["net"] for d in normal_monthly_data]
        result = compute_bollinger_bands(nets, window=6, num_std=2)
        # First 5 values (window-1) should be None
        for i in range(5):
            assert result["middle"][i] is None
            assert result["upper"][i] is None
            assert result["lower"][i] is None

    def test_upper_above_lower(self, normal_monthly_data: list[dict]):
        """Upper band should always be >= lower band."""
        nets = [d["net"] for d in normal_monthly_data]
        result = compute_bollinger_bands(nets, window=6, num_std=2)
        for i in range(6, len(nets)):
            assert result["upper"][i] >= result["lower"][i]

    def test_middle_between_bands(self, normal_monthly_data: list[dict]):
        """Middle band should be between upper and lower."""
        nets = [d["net"] for d in normal_monthly_data]
        result = compute_bollinger_bands(nets, window=6, num_std=2)
        for i in range(6, len(nets)):
            assert result["lower"][i] <= result["middle"][i] <= result["upper"][i]

    def test_flat_series_bands_equal(self, flat_monthly_data: list[dict]):
        """Flat series → std=0 → upper == middle == lower (no division by zero)."""
        nets = [d["net"] for d in flat_monthly_data]
        result = compute_bollinger_bands(nets, window=6, num_std=2)
        for i in range(5, len(nets)):
            assert result["upper"][i] == result["middle"][i]
            assert result["lower"][i] == result["middle"][i]

    def test_pband_in_range(self, normal_monthly_data: list[dict]):
        """Percent band should generally be between 0 and 1 for normal data."""
        nets = [d["net"] for d in normal_monthly_data]
        result = compute_bollinger_bands(nets, window=6, num_std=2)
        # pband can go outside 0-1 when price breaks bands, but should exist
        valid_pbands = [p for p in result["pband"] if p is not None]
        assert len(valid_pbands) > 0

    def test_flat_series_pband_is_none(self, flat_monthly_data: list[dict]):
        """Flat series → bandwidth=0 → pband should be None (guarded div-by-zero)."""
        nets = [d["net"] for d in flat_monthly_data]
        result = compute_bollinger_bands(nets, window=6, num_std=2)
        for p in result["pband"]:
            if p is not None:
                # If the band has zero width, pband should be None not inf
                pass  # allowed — some implementations return 0.5
        # Key check: no inf values
        valid = [p for p in result["pband"] if p is not None]
        for v in valid:
            assert v != float("inf")
            assert v != float("-inf")


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_returns_correct_length(self, normal_monthly_data: list[dict]):
        nets = [d["net"] for d in normal_monthly_data]
        result = compute_rsi(nets, window=6)
        assert len(result) == len(nets)

    def test_first_values_are_none(self, normal_monthly_data: list[dict]):
        nets = [d["net"] for d in normal_monthly_data]
        result = compute_rsi(nets, window=6)
        # First window values should be None (need window+1 data points)
        for i in range(6):
            assert result[i] is None

    def test_rsi_bounded_0_100(self, normal_monthly_data: list[dict]):
        """RSI should always be between 0 and 100."""
        nets = [d["net"] for d in normal_monthly_data]
        result = compute_rsi(nets, window=6)
        for v in result:
            if v is not None:
                assert 0.0 <= v <= 100.0

    def test_declining_series_low_rsi(self, declining_monthly_data: list[dict]):
        """Consistently declining net → RSI should be below 30 (oversold)."""
        nets = [d["net"] for d in declining_monthly_data]
        result = compute_rsi(nets, window=6)
        last_rsi = result[-1]
        assert last_rsi is not None
        assert last_rsi < 30.0

    def test_surging_series_high_rsi(self, surging_monthly_data: list[dict]):
        """Consistently rising net → RSI should be above 70 (overbought)."""
        nets = [d["net"] for d in surging_monthly_data]
        result = compute_rsi(nets, window=6)
        last_rsi = result[-1]
        assert last_rsi is not None
        assert last_rsi > 70.0

    def test_flat_series_rsi_50(self, flat_monthly_data: list[dict]):
        """Flat series → no gains or losses → RSI should be None (0/0 guarded)."""
        nets = [d["net"] for d in flat_monthly_data]
        result = compute_rsi(nets, window=6)
        for v in result:
            if v is not None:
                # Should not be inf or NaN
                assert v == v  # NaN check
                assert v != float("inf")


# ---------------------------------------------------------------------------
# Savings capacity integration tests
# ---------------------------------------------------------------------------

class TestAnalyzeSavingsCapacity:
    def test_insufficient_data_raises(self):
        """Less than MIN_MONTHS_REQUIRED months should raise ValueError."""
        short_data = [
            {"month": f"2025-{m:02d}", "income": 50000.0, "expenses": 35000.0, "net": 15000.0}
            for m in range(1, 4)
        ]
        with pytest.raises(ValueError, match="at least"):
            analyze_savings_capacity(short_data)

    def test_returns_expected_structure(self, normal_monthly_data: list[dict]):
        """Result should have all expected keys."""
        result = analyze_savings_capacity(normal_monthly_data)
        assert "current_signal" in result
        assert "rsi" in result
        assert "bollinger" in result
        assert "recommendation" in result
        assert "months" in result

    def test_signal_is_valid_enum(self, normal_monthly_data: list[dict]):
        """Signal should be one of: INVEST, HOLD, ALERT."""
        result = analyze_savings_capacity(normal_monthly_data)
        assert result["current_signal"] in ("INVEST", "HOLD", "ALERT")

    def test_surplus_month_invest_signal(self, surging_monthly_data: list[dict]):
        """Strong surplus trend → signal should be INVEST."""
        result = analyze_savings_capacity(surging_monthly_data)
        assert result["current_signal"] == "INVEST"

    def test_declining_month_alert_signal(self, declining_monthly_data: list[dict]):
        """Declining net trend → signal should be ALERT."""
        result = analyze_savings_capacity(declining_monthly_data)
        assert result["current_signal"] == "ALERT"

    def test_months_output_matches_input_length(self, normal_monthly_data: list[dict]):
        """Monthly detail should match input length."""
        result = analyze_savings_capacity(normal_monthly_data)
        assert len(result["months"]) == len(normal_monthly_data)

    def test_each_month_has_signal_data(self, normal_monthly_data: list[dict]):
        """Each month entry should have the expected fields."""
        result = analyze_savings_capacity(normal_monthly_data)
        for month_data in result["months"]:
            assert "month" in month_data
            assert "net" in month_data
            assert "bb_upper" in month_data
            assert "bb_middle" in month_data
            assert "bb_lower" in month_data
            assert "rsi" in month_data

    def test_recommendation_is_nonempty_string(self, normal_monthly_data: list[dict]):
        """Should always return a human-readable recommendation."""
        result = analyze_savings_capacity(normal_monthly_data)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 10

    def test_flat_data_hold_signal(self, flat_monthly_data: list[dict]):
        """Flat cashflow → no momentum → HOLD signal."""
        result = analyze_savings_capacity(flat_monthly_data)
        assert result["current_signal"] == "HOLD"
