"""Financial Modeling Engine — 100 % Code, Zero AI.

Uses a standard Excel-logic engine backed by NumPy.
The AI only supplies the *assumptions* (price, CAC, etc.);
all calculations are deterministic and mathematically perfect.
"""

from __future__ import annotations

import numpy as np

from src.models.venture import FinancialModel


def build_financial_model(
    price_per_unit: float,
    cost_per_unit: float,
    customer_acquisition_cost: float,
    initial_customers: int,
    monthly_growth_rate: float,
    fixed_monthly_costs: float,
    months: int = 24,
    churn_rate: float = 0.05,
) -> FinancialModel:
    """Generate a P&L statement from hard assumptions.

    Parameters
    ----------
    price_per_unit:
        Revenue per customer per month.
    cost_per_unit:
        Variable cost per customer per month (COGS).
    customer_acquisition_cost:
        One-time cost to acquire a new customer.
    initial_customers:
        Number of customers in month 1.
    monthly_growth_rate:
        Month-over-month customer growth rate (0.10 = 10 %).
    fixed_monthly_costs:
        Fixed operating expenses per month (rent, salaries, etc.).
    months:
        Projection horizon in months.
    churn_rate:
        Monthly customer churn rate (0.05 = 5 %).

    Returns
    -------
    FinancialModel
        A fully computed P&L with monthly breakdowns.
    """
    # Build customer count array (growth minus churn)
    customers = np.zeros(months)
    customers[0] = initial_customers
    for m in range(1, months):
        new_customers = customers[m - 1] * monthly_growth_rate
        lost_customers = customers[m - 1] * churn_rate
        customers[m] = customers[m - 1] + new_customers - lost_customers

    # Revenue = price * customers
    revenue = customers * price_per_unit

    # Variable costs = COGS + acquisition cost for new customers
    new_per_month = np.diff(customers, prepend=0)
    new_per_month = np.maximum(new_per_month, 0)  # Only count net new
    variable_costs = (customers * cost_per_unit) + (new_per_month * customer_acquisition_cost)

    # Total costs = variable + fixed
    total_costs = variable_costs + fixed_monthly_costs

    # Profit
    profit = revenue - total_costs

    # Break-even month (first month where cumulative profit >= 0)
    cumulative_profit = np.cumsum(profit)
    break_even_indices = np.where(cumulative_profit >= 0)[0]
    break_even_month = int(break_even_indices[0]) + 1 if len(break_even_indices) > 0 else None

    return FinancialModel(
        assumptions={
            "price_per_unit": price_per_unit,
            "cost_per_unit": cost_per_unit,
            "customer_acquisition_cost": customer_acquisition_cost,
            "initial_customers": float(initial_customers),
            "monthly_growth_rate": monthly_growth_rate,
            "fixed_monthly_costs": fixed_monthly_costs,
            "churn_rate": churn_rate,
            "projection_months": float(months),
        },
        revenue_monthly=revenue.tolist(),
        costs_monthly=total_costs.tolist(),
        profit_monthly=profit.tolist(),
        break_even_month=break_even_month,
        annual_revenue=float(np.sum(revenue[:12])),
        annual_cost=float(np.sum(total_costs[:12])),
        annual_profit=float(np.sum(profit[:12])),
    )


def calculate_viability_score(
    market_volume_score: float,
    competitor_density_score: float,
    market_weight: float = 0.6,
    competitor_weight: float = 0.4,
) -> float:
    """Step 4 viability calculation — weighted algorithm, code not AI.

    Parameters
    ----------
    market_volume_score:
        Normalized score (0-1) from search volume data.
    competitor_density_score:
        Normalized score (0-1) — higher means MORE competitive (worse).
    market_weight:
        Weight for market volume in the final score.
    competitor_weight:
        Weight for competitor density (inverted) in the final score.

    Returns
    -------
    float
        Weighted viability score between 0 and 1.
    """
    # Invert competitor density: fewer competitors = better opportunity
    inverted_density = 1.0 - competitor_density_score
    score = (market_volume_score * market_weight) + (inverted_density * competitor_weight)
    return round(float(np.clip(score, 0.0, 1.0)), 4)
