"""
Budget Planner Tool
--------------------
Suggests a monthly budget allocation based on the popular 50-30-20 rule:

    50% → Needs   (rent, groceries, utilities, transport, insurance)
    30% → Wants   (dining out, entertainment, subscriptions, travel)
    20% → Savings (emergency fund, investments, debt repayment)

This rule, popularised by Senator Elizabeth Warren in "All Your Worth",
provides a simple framework for balanced personal finance.
"""

from langchain.tools import tool


@tool
def plan_budget(monthly_income: float) -> str:
    """
    Create a monthly budget plan using the 50-30-20 rule.

    Use this tool when the user asks about:
    - How to allocate or split their monthly salary/income
    - Budget planning, savings goals, or spending limits
    - The 50-30-20 rule applied to their income

    Args:
        monthly_income: Gross (or net take-home) monthly income. e.g. 80000

    Returns:
        A formatted budget breakdown with category allocations and
        practical spending suggestions for each bucket.
    """
    needs_pct = 0.50
    wants_pct = 0.30
    savings_pct = 0.20

    needs = monthly_income * needs_pct
    wants = monthly_income * wants_pct
    savings = monthly_income * savings_pct

    # Suggested breakdown within each category
    emergency_fund = savings * 0.50       # 10% of income
    investments = savings * 0.30          # 6% of income
    debt_repayment = savings * 0.20       # 4% of income

    return (
        f"💼 **50-30-20 Budget Plan**\n\n"
        f"**Monthly Income: {monthly_income:,.2f}**\n\n"
        f"---\n\n"
        f"### 🏠 Needs — 50% → {needs:,.2f}\n"
        f"Essentials you cannot live without:\n"
        f"- Rent / Home loan EMI\n"
        f"- Groceries & household supplies\n"
        f"- Utilities (electricity, water, internet)\n"
        f"- Transport & commuting\n"
        f"- Health insurance & medical bills\n\n"
        f"### 🎉 Wants — 30% → {wants:,.2f}\n"
        f"Lifestyle upgrades and discretionary spending:\n"
        f"- Dining out & cafes\n"
        f"- Entertainment (streaming, movies, events)\n"
        f"- Shopping (clothing, gadgets, hobbies)\n"
        f"- Vacations & travel\n"
        f"- Gym, subscriptions, personal care\n\n"
        f"### 💰 Savings & Investments — 20% → {savings:,.2f}\n"
        f"Building your financial future:\n"
        f"- Emergency Fund (6-month buffer): {emergency_fund:,.2f}\n"
        f"- Investments (SIP / stocks / mutual funds): {investments:,.2f}\n"
        f"- Debt repayment / additional loan payments: {debt_repayment:,.2f}\n\n"
        f"---\n"
        f"💡 **Tip:** If your Needs exceed 50%, look for ways to reduce fixed costs "
        f"before cutting Wants. Build your emergency fund first before investing.\n"
    )
