"""Tests for the financial modeling engine — the zero-AI calculation core."""

from src.financial.engine import build_financial_model, calculate_viability_score


class TestFinancialModel:
    """Verify that the P&L engine produces mathematically correct results."""

    def test_basic_model(self) -> None:
        model = build_financial_model(
            price_per_unit=50.0,
            cost_per_unit=10.0,
            customer_acquisition_cost=30.0,
            initial_customers=10,
            monthly_growth_rate=0.10,
            fixed_monthly_costs=2000.0,
            months=12,
            churn_rate=0.05,
        )
        assert len(model.revenue_monthly) == 12
        assert len(model.costs_monthly) == 12
        assert len(model.profit_monthly) == 12
        # Revenue should be positive (price > 0, customers > 0)
        assert all(r > 0 for r in model.revenue_monthly)
        # First month revenue = 10 customers * $50
        assert model.revenue_monthly[0] == 500.0

    def test_annual_totals(self) -> None:
        model = build_financial_model(
            price_per_unit=100.0,
            cost_per_unit=20.0,
            customer_acquisition_cost=50.0,
            initial_customers=100,
            monthly_growth_rate=0.05,
            fixed_monthly_costs=5000.0,
            months=12,
        )
        assert model.annual_revenue == sum(model.revenue_monthly)
        assert model.annual_cost == sum(model.costs_monthly)
        assert model.annual_profit == sum(model.profit_monthly)

    def test_break_even_detection(self) -> None:
        # With enough growth and margin, should eventually break even
        model = build_financial_model(
            price_per_unit=100.0,
            cost_per_unit=10.0,
            customer_acquisition_cost=20.0,
            initial_customers=50,
            monthly_growth_rate=0.20,
            fixed_monthly_costs=3000.0,
            months=24,
        )
        # Should find a break-even month
        assert model.break_even_month is not None
        assert 1 <= model.break_even_month <= 24

    def test_zero_growth(self) -> None:
        model = build_financial_model(
            price_per_unit=50.0,
            cost_per_unit=10.0,
            customer_acquisition_cost=30.0,
            initial_customers=10,
            monthly_growth_rate=0.0,
            fixed_monthly_costs=100.0,
            months=6,
            churn_rate=0.0,
        )
        # With zero growth and zero churn, customer count should stay at 10
        assert model.revenue_monthly[0] == model.revenue_monthly[-1]

    def test_high_churn(self) -> None:
        model = build_financial_model(
            price_per_unit=50.0,
            cost_per_unit=10.0,
            customer_acquisition_cost=30.0,
            initial_customers=100,
            monthly_growth_rate=0.0,
            fixed_monthly_costs=100.0,
            months=12,
            churn_rate=0.50,
        )
        # Revenue should decrease each month due to churn
        assert model.revenue_monthly[-1] < model.revenue_monthly[0]


class TestViabilityScore:
    """Verify the weighted viability algorithm."""

    def test_high_market_low_competition(self) -> None:
        score = calculate_viability_score(
            market_volume_score=0.9,
            competitor_density_score=0.1,
        )
        assert score > 0.7

    def test_low_market_high_competition(self) -> None:
        score = calculate_viability_score(
            market_volume_score=0.1,
            competitor_density_score=0.9,
        )
        assert score < 0.3

    def test_equal_weights(self) -> None:
        score = calculate_viability_score(
            market_volume_score=0.5,
            competitor_density_score=0.5,
            market_weight=0.5,
            competitor_weight=0.5,
        )
        assert 0.45 <= score <= 0.55

    def test_clamped_to_zero_one(self) -> None:
        score = calculate_viability_score(
            market_volume_score=1.0,
            competitor_density_score=0.0,
        )
        assert 0.0 <= score <= 1.0
