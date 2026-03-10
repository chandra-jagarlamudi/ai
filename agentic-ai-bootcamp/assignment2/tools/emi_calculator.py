"""
EMI Calculator Tool
-------------------
Calculates Equated Monthly Installment (EMI) for any loan
using the standard financial formula:

    EMI = P × r × (1 + r)^n / ((1 + r)^n - 1)

Where:
    P = Principal loan amount
    r = Monthly interest rate (annual rate / 12 / 100)
    n = Total number of monthly payments (years × 12)
"""

from langchain.tools import tool


@tool
def calculate_emi(
    loan_amount: float,
    annual_interest_rate: float,
    loan_tenure_years: int,
) -> str:
    """
    Calculate the Equated Monthly Installment (EMI) for a loan.

    Use this tool when the user asks about:
    - Monthly loan payment / EMI for a home loan, car loan, or personal loan
    - How much they need to pay per month for a borrowed amount
    - Total interest payable over the life of a loan

    Args:
        loan_amount: The principal amount borrowed (P). e.g. 500000
        annual_interest_rate: Annual interest rate as a percentage. e.g. 8.5 means 8.5%
        loan_tenure_years: Total loan duration in years. e.g. 20

    Returns:
        A formatted summary with monthly EMI, total payable amount,
        and total interest charged.
    """
    # Convert annual percentage rate → monthly decimal rate
    monthly_rate = annual_interest_rate / 100 / 12

    # Total number of monthly installments
    n_months = loan_tenure_years * 12

    # Handle edge case of 0% interest (simple division)
    if monthly_rate == 0:
        emi = loan_amount / n_months
    else:
        # Standard EMI formula: P × r × (1+r)^n / ((1+r)^n - 1)
        compounded = (1 + monthly_rate) ** n_months
        emi = loan_amount * monthly_rate * compounded / (compounded - 1)

    total_payment = emi * n_months
    total_interest = total_payment - loan_amount
    interest_pct = (total_interest / loan_amount) * 100

    return (
        f"📊 **EMI Calculation Results**\n\n"
        f"**Loan Details**\n"
        f"- Principal (P): {loan_amount:,.2f}\n"
        f"- Annual Interest Rate: {annual_interest_rate}%\n"
        f"- Monthly Interest Rate (r): {monthly_rate * 100:.4f}%\n"
        f"- Tenure: {loan_tenure_years} years ({n_months} months)\n\n"
        f"**Results**\n"
        f"- 💳 Monthly EMI: **{emi:,.2f}**\n"
        f"- Total Amount Payable: {total_payment:,.2f}\n"
        f"- Total Interest Charged: {total_interest:,.2f}\n"
        f"- Interest as % of Principal: {interest_pct:.1f}%\n"
    )
