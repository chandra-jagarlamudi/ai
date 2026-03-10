"""
SIP Investment Calculator Tool
--------------------------------
Calculates the future value of a Systematic Investment Plan (SIP),
where a fixed amount is invested every month at a compounding return rate.

Formula:
    FV = P × ((1 + r)^n - 1) / r

Where:
    P = Monthly investment amount
    r = Monthly return rate (annual rate / 12 / 100)
    n = Total number of monthly investments (years × 12)
"""

from langchain.tools import tool


@tool
def calculate_sip(
    monthly_investment: float,
    annual_return_rate: float,
    investment_years: int,
) -> str:
    """
    Calculate the future value of a monthly SIP (Systematic Investment Plan).

    Use this tool when the user asks about:
    - How much money they will accumulate by investing a fixed amount monthly
    - SIP returns, mutual fund projections, or wealth accumulation over time
    - How long to invest to reach a financial goal

    Args:
        monthly_investment: Fixed amount invested every month (P). e.g. 5000
        annual_return_rate: Expected annual return as a percentage. e.g. 12 means 12% p.a.
        investment_years: Total investment duration in years. e.g. 10

    Returns:
        A formatted summary with future value, total invested,
        and total wealth gained.
    """
    # Convert annual percentage rate → monthly decimal rate
    monthly_rate = annual_return_rate / 100 / 12

    # Total number of monthly investments
    n_months = investment_years * 12

    # Handle edge case of 0% return (no compounding)
    if monthly_rate == 0:
        future_value = monthly_investment * n_months
    else:
        # SIP Future Value formula: P × ((1+r)^n - 1) / r
        future_value = monthly_investment * ((1 + monthly_rate) ** n_months - 1) / monthly_rate

    total_invested = monthly_investment * n_months
    wealth_gained = future_value - total_invested
    returns_pct = (wealth_gained / total_invested) * 100

    return (
        f"📈 **SIP Investment Calculator Results**\n\n"
        f"**Investment Details**\n"
        f"- Monthly Investment (P): {monthly_investment:,.2f}\n"
        f"- Expected Annual Return: {annual_return_rate}%\n"
        f"- Monthly Return Rate (r): {monthly_rate * 100:.4f}%\n"
        f"- Investment Duration: {investment_years} years ({n_months} months)\n\n"
        f"**Results**\n"
        f"- 💰 Future Value: **{future_value:,.2f}**\n"
        f"- Total Amount Invested: {total_invested:,.2f}\n"
        f"- Wealth Gained (Returns): {wealth_gained:,.2f}\n"
        f"- Return on Investment: {returns_pct:.1f}%\n"
    )
