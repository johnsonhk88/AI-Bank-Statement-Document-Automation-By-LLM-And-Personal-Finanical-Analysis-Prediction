"""Tests for spending-trend alerts — MACD + SMA crossover on per-category spend.

Phase 2 of the TA strategy integration.
"""
import pytest

from app.services.ta_analyzer import (
    compute_macd,
    compute_sma_crossover,
    analyze_spending_trends,
    MIN_SPENDING_MONTHS_REQUIRED,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def category_monthly_spend() -> dict[str, list[dict]]:
    """12 months of per-category spending data with realistic patterns."""
    return {
        "Dining": [
            {"month": f"2025-{m:02d}", "amount": amt}
            for m, amt in zip(
                range(1, 13),
                [3000, 3200, 3100, 3500, 3800, 4000, 4200, 4500, 4800, 5000, 5200, 5500],
            )
        ],
        "Transport": [
            {"month": f"2025-{m:02d}", "amount": amt}
            for m, amt in zip(
                range(1, 13),
                [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],
            )
        ],
        "Groceries": [
            {"month": f"2025-{m:02d}", "amount": amt}
            for m, amt in zip(
                range(1, 13),
                [5000, 4800, 4600, 4400, 4200, 4000, 3800, 3600, 3400, 3200, 3000, 2800],
            )
        ],
    }


@pytest.fixture
def accelerating_spend() -> list[dict]:
    """12 months of sharply accelerating spend — MACD should go positive."""
    return [
        {"month": f"2025-{m:02d}", "amount": 1000 + (i * 500)}
        for i, m in enumerate(range(1, 13))
    ]


@pytest.fixture
def decelerating_spend() -> list[dict]:
    """12 months of sharply decelerating spend — MACD should go negative."""
    return [
        {"month": f"2025-{m:02d}", "amount": 6500 - (i * 500)}
        for i, m in enumerate(range(1, 13))
    ]


@pytest.fixture
def flat_spend() -> list[dict]:
    """12 months of constant spend."""
    return [
        {"month": f"2025-{m:02d}", "amount": 3000.0}
        for m in range(1, 13)
    ]


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------

class TestMACD:
    def test_returns_correct_length(self, accelerating_spend: list[dict]):
        values = [d["amount"] for d in accelerating_spend]
        result = compute_macd(values, fast=4, slow=8, signal=3)
        assert len(result["macd_line"]) == len(values)
        assert len(result["signal_line"]) == len(values)
        assert len(result["histogram"]) == len(values)

    def test_warm_up_values_are_none(self, accelerating_spend: list[dict]):
        """First slow-1 values should be None (EMA needs warm-up)."""
        values = [d["amount"] for d in accelerating_spend]
        result = compute_macd(values, fast=4, slow=8, signal=3)
        # slow window = 8, so first 7 MACD values should be None
        for i in range(7):
            assert result["macd_line"][i] is None

    def test_accelerating_positive_histogram(self, accelerating_spend: list[dict]):
        """Accelerating spend → MACD histogram should be positive at end."""
        values = [d["amount"] for d in accelerating_spend]
        result = compute_macd(values, fast=4, slow=8, signal=3)
        last_hist = result["histogram"][-1]
        assert last_hist is not None
        assert last_hist > 0

    def test_decelerating_negative_histogram(self, decelerating_spend: list[dict]):
        """Decelerating spend → MACD histogram should be negative at end."""
        values = [d["amount"] for d in decelerating_spend]
        result = compute_macd(values, fast=4, slow=8, signal=3)
        last_hist = result["histogram"][-1]
        assert last_hist is not None
        assert last_hist < 0

    def test_flat_series_zero_macd(self, flat_spend: list[dict]):
        """Flat spend → MACD line should be ~0 (no divergence)."""
        values = [d["amount"] for d in flat_spend]
        result = compute_macd(values, fast=4, slow=8, signal=3)
        valid_macd = [v for v in result["macd_line"] if v is not None]
        for v in valid_macd:
            assert abs(v) < 1.0  # near zero


# ---------------------------------------------------------------------------
# SMA crossover tests
# ---------------------------------------------------------------------------

class TestSMACrossover:
    def test_returns_correct_length(self, accelerating_spend: list[dict]):
        values = [d["amount"] for d in accelerating_spend]
        result = compute_sma_crossover(values, short_window=3, long_window=6)
        assert len(result["short_sma"]) == len(values)
        assert len(result["long_sma"]) == len(values)

    def test_warm_up_none(self, accelerating_spend: list[dict]):
        """First long_window-1 values should be None for long SMA."""
        values = [d["amount"] for d in accelerating_spend]
        result = compute_sma_crossover(values, short_window=3, long_window=6)
        for i in range(5):
            assert result["long_sma"][i] is None

    def test_accelerating_golden_cross(self, accelerating_spend: list[dict]):
        """Accelerating spend → short SMA above long SMA (golden cross)."""
        values = [d["amount"] for d in accelerating_spend]
        result = compute_sma_crossover(values, short_window=3, long_window=6)
        assert result["crossover"] in ("golden_cross", "none")
        # Short should be above long at end
        short_last = result["short_sma"][-1]
        long_last = result["long_sma"][-1]
        assert short_last is not None and long_last is not None
        assert short_last > long_last

    def test_decelerating_death_cross(self, decelerating_spend: list[dict]):
        """Decelerating spend → short SMA below long SMA (death cross)."""
        values = [d["amount"] for d in decelerating_spend]
        result = compute_sma_crossover(values, short_window=3, long_window=6)
        short_last = result["short_sma"][-1]
        long_last = result["long_sma"][-1]
        assert short_last is not None and long_last is not None
        assert short_last < long_last

    def test_flat_series_no_crossover(self, flat_spend: list[dict]):
        """Flat spend → SMAs converge, no meaningful crossover."""
        values = [d["amount"] for d in flat_spend]
        result = compute_sma_crossover(values, short_window=3, long_window=6)
        assert result["crossover"] == "none"


# ---------------------------------------------------------------------------
# analyze_spending_trends integration tests
# ---------------------------------------------------------------------------

class TestAnalyzeSpendingTrends:
    def test_insufficient_data_raises(self):
        data = {
            "Dining": [
                {"month": f"2025-{m:02d}", "amount": 3000.0}
                for m in range(1, 5)
            ]
        }
        with pytest.raises(ValueError, match="at least"):
            analyze_spending_trends(data)

    def test_returns_per_category_alerts(self, category_monthly_spend: dict):
        result = analyze_spending_trends(category_monthly_spend)
        assert "categories" in result
        assert len(result["categories"]) == 3
        for cat_result in result["categories"]:
            assert "category" in cat_result
            assert "trend" in cat_result
            assert "alert" in cat_result
            assert "recommendation" in cat_result
            assert "months" in cat_result

    def test_trend_is_valid_enum(self, category_monthly_spend: dict):
        result = analyze_spending_trends(category_monthly_spend)
        valid_trends = ("accelerating", "decelerating", "stable")
        for cat_result in result["categories"]:
            assert cat_result["trend"] in valid_trends

    def test_accelerating_category_flagged(self, category_monthly_spend: dict):
        """Dining has steadily increasing spend → should be flagged as accelerating."""
        result = analyze_spending_trends(category_monthly_spend)
        dining = next(c for c in result["categories"] if c["category"] == "Dining")
        assert dining["trend"] == "accelerating"
        assert dining["alert"] is True

    def test_stable_category_no_alert(self, category_monthly_spend: dict):
        """Transport is flat → should be stable, no alert."""
        result = analyze_spending_trends(category_monthly_spend)
        transport = next(c for c in result["categories"] if c["category"] == "Transport")
        assert transport["trend"] == "stable"
        assert transport["alert"] is False

    def test_decelerating_category_flagged(self, category_monthly_spend: dict):
        """Groceries is declining → should be decelerating."""
        result = analyze_spending_trends(category_monthly_spend)
        groceries = next(c for c in result["categories"] if c["category"] == "Groceries")
        assert groceries["trend"] == "decelerating"

    def test_recommendation_nonempty(self, category_monthly_spend: dict):
        result = analyze_spending_trends(category_monthly_spend)
        for cat_result in result["categories"]:
            assert isinstance(cat_result["recommendation"], str)
            assert len(cat_result["recommendation"]) > 10

    def test_summary_present(self, category_monthly_spend: dict):
        """Result should include a top-level summary with alert count."""
        result = analyze_spending_trends(category_monthly_spend)
        assert "alert_count" in result
        assert "total_categories" in result
        assert result["total_categories"] == 3
        assert result["alert_count"] >= 1  # at least Dining should alert

    def test_skips_categories_with_insufficient_data(self):
        """Categories with < MIN_SPENDING_MONTHS_REQUIRED months are skipped."""
        data = {
            "Dining": [
                {"month": f"2025-{m:02d}", "amount": 3000.0 + m * 100}
                for m in range(1, 13)
            ],
            "Shopping": [
                {"month": f"2025-{m:02d}", "amount": 1000.0}
                for m in range(1, 4)  # only 3 months
            ],
        }
        result = analyze_spending_trends(data)
        categories = [c["category"] for c in result["categories"]]
        assert "Dining" in categories
        assert "Shopping" not in categories  # skipped — too few months
