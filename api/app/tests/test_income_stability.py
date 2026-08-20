"""Tests for income stability analysis — Phase 3.

Covers ATR on monthly income, volatility classification, risk profile,
and investment strategy recommendation.
"""
import pytest

from app.services.ta_analyzer import (
    compute_income_atr,
    analyze_income_stability,
    MIN_MONTHS_REQUIRED,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stable_income_data() -> list[dict]:
    """12 months of very stable income — ATR% should be low."""
    return [
        {"month": f"2025-{m:02d}", "income": 50000.0, "expenses": 35000.0, "net": 15000.0}
        for m in range(1, 13)
    ]


@pytest.fixture
def slightly_variable_income() -> list[dict]:
    """12 months with small income fluctuations — LOW volatility."""
    incomes = [50000, 51000, 49500, 50500, 50200, 49800, 50300, 50100, 49700, 50400, 50600, 49900]
    return [
        {
            "month": f"2025-{m:02d}",
            "income": float(inc),
            "expenses": 35000.0,
            "net": float(inc - 35000),
        }
        for m, inc in zip(range(1, 13), incomes)
    ]


@pytest.fixture
def moderate_variable_income() -> list[dict]:
    """12 months with moderate income swings — MEDIUM volatility."""
    incomes = [50000, 55000, 45000, 60000, 42000, 58000, 47000, 53000, 44000, 56000, 48000, 52000]
    return [
        {
            "month": f"2025-{m:02d}",
            "income": float(inc),
            "expenses": 35000.0,
            "net": float(inc - 35000),
        }
        for m, inc in zip(range(1, 13), incomes)
    ]


@pytest.fixture
def highly_variable_income() -> list[dict]:
    """12 months of wildly swinging income — HIGH volatility (freelancer pattern)."""
    incomes = [80000, 20000, 90000, 15000, 75000, 25000, 85000, 18000, 70000, 30000, 95000, 10000]
    return [
        {
            "month": f"2025-{m:02d}",
            "income": float(inc),
            "expenses": 35000.0,
            "net": float(inc - 35000),
        }
        for m, inc in zip(range(1, 13), incomes)
    ]


@pytest.fixture
def growing_income_data() -> list[dict]:
    """12 months of steadily growing income — moderate ATR but upward trend."""
    incomes = [40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000, 56000, 58000, 60000, 62000]
    return [
        {
            "month": f"2025-{m:02d}",
            "income": float(inc),
            "expenses": 35000.0,
            "net": float(inc - 35000),
        }
        for m, inc in zip(range(1, 13), incomes)
    ]


# ---------------------------------------------------------------------------
# compute_income_atr tests
# ---------------------------------------------------------------------------

class TestComputeIncomeATR:
    def test_returns_correct_length(self, slightly_variable_income: list[dict]):
        incomes = [d["income"] for d in slightly_variable_income]
        result = compute_income_atr(incomes, window=6)
        assert len(result) == len(incomes)

    def test_first_value_is_none(self, slightly_variable_income: list[dict]):
        """First value has no prior month to diff against → None."""
        incomes = [d["income"] for d in slightly_variable_income]
        result = compute_income_atr(incomes, window=6)
        assert result[0] is None

    def test_warmup_values_are_none(self, slightly_variable_income: list[dict]):
        """Values before window is filled should be None."""
        incomes = [d["income"] for d in slightly_variable_income]
        result = compute_income_atr(incomes, window=6)
        # First window values should be None (need diff + EMA warm-up)
        for i in range(6):
            assert result[i] is None

    def test_atr_non_negative(self, moderate_variable_income: list[dict]):
        """ATR should always be >= 0."""
        incomes = [d["income"] for d in moderate_variable_income]
        result = compute_income_atr(incomes, window=6)
        for v in result:
            if v is not None:
                assert v >= 0.0

    def test_flat_income_atr_zero(self, stable_income_data: list[dict]):
        """Completely flat income → ATR should be 0."""
        incomes = [d["income"] for d in stable_income_data]
        result = compute_income_atr(incomes, window=6)
        for v in result:
            if v is not None:
                assert v == 0.0

    def test_high_volatility_larger_atr(
        self,
        slightly_variable_income: list[dict],
        highly_variable_income: list[dict],
    ):
        """Highly variable income should produce larger ATR than stable income."""
        low_atr = compute_income_atr(
            [d["income"] for d in slightly_variable_income], window=6
        )
        high_atr = compute_income_atr(
            [d["income"] for d in highly_variable_income], window=6
        )
        # Compare last valid ATR values
        low_last = [v for v in low_atr if v is not None][-1]
        high_last = [v for v in high_atr if v is not None][-1]
        assert high_last > low_last


# ---------------------------------------------------------------------------
# analyze_income_stability integration tests
# ---------------------------------------------------------------------------

class TestAnalyzeIncomeStability:
    def test_insufficient_data_raises(self):
        short_data = [
            {"month": f"2025-{m:02d}", "income": 50000.0, "expenses": 35000.0, "net": 15000.0}
            for m in range(1, 4)
        ]
        with pytest.raises(ValueError, match="at least"):
            analyze_income_stability(short_data)

    def test_returns_expected_structure(self, slightly_variable_income: list[dict]):
        result = analyze_income_stability(slightly_variable_income)
        assert "risk_profile" in result
        assert "atr" in result
        assert "atr_percent" in result
        assert "mean_income" in result
        assert "income_trend" in result
        assert "recommendation" in result
        assert "months" in result

    def test_risk_profile_valid_enum(self, slightly_variable_income: list[dict]):
        result = analyze_income_stability(slightly_variable_income)
        assert result["risk_profile"] in ("LOW", "MEDIUM", "HIGH")

    def test_stable_income_low_risk(self, stable_income_data: list[dict]):
        result = analyze_income_stability(stable_income_data)
        assert result["risk_profile"] == "LOW"

    def test_stable_income_low_atr_percent(self, stable_income_data: list[dict]):
        result = analyze_income_stability(stable_income_data)
        assert result["atr_percent"] == 0.0

    def test_slightly_variable_low_risk(self, slightly_variable_income: list[dict]):
        result = analyze_income_stability(slightly_variable_income)
        assert result["risk_profile"] == "LOW"
        assert result["atr_percent"] < 10.0

    def test_moderate_variable_medium_risk(self, moderate_variable_income: list[dict]):
        result = analyze_income_stability(moderate_variable_income)
        assert result["risk_profile"] == "MEDIUM"
        assert 10.0 <= result["atr_percent"] <= 25.0

    def test_highly_variable_high_risk(self, highly_variable_income: list[dict]):
        result = analyze_income_stability(highly_variable_income)
        assert result["risk_profile"] == "HIGH"
        assert result["atr_percent"] > 25.0

    def test_growing_income_trend_positive(self, growing_income_data: list[dict]):
        result = analyze_income_stability(growing_income_data)
        assert result["income_trend"] == "growing"

    def test_mean_income_positive(self, slightly_variable_income: list[dict]):
        result = analyze_income_stability(slightly_variable_income)
        assert result["mean_income"] > 0

    def test_months_output_matches_input(self, slightly_variable_income: list[dict]):
        result = analyze_income_stability(slightly_variable_income)
        assert len(result["months"]) == len(slightly_variable_income)

    def test_each_month_has_fields(self, slightly_variable_income: list[dict]):
        result = analyze_income_stability(slightly_variable_income)
        for m in result["months"]:
            assert "month" in m
            assert "income" in m
            assert "atr" in m

    def test_recommendation_is_nonempty(self, slightly_variable_income: list[dict]):
        result = analyze_income_stability(slightly_variable_income)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 10

    def test_atr_matches_last_computed(self, moderate_variable_income: list[dict]):
        """The returned atr should match the last valid computed ATR value."""
        result = analyze_income_stability(moderate_variable_income)
        incomes = [d["income"] for d in moderate_variable_income]
        raw_atr = compute_income_atr(incomes, window=6)
        last_atr = [v for v in raw_atr if v is not None][-1]
        assert abs(result["atr"] - last_atr) < 0.01
