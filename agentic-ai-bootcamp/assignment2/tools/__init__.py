"""
Tools package for the Personal Financial AI Agent.

Exports all LangChain tools so agent.py can import them in one place.
Each tool is a @tool-decorated function whose docstring is used by the
LLM to decide when to call it.
"""

from .budget_planner import plan_budget
from .calculator import calculate_expression
from .emi_calculator import calculate_emi
from .sip_calculator import calculate_sip
from .stock_price import get_stock_price

# Single list consumed by the LangChain AgentExecutor
ALL_TOOLS = [
    calculate_emi,
    calculate_sip,
    plan_budget,
    get_stock_price,
    calculate_expression,
]

__all__ = ["ALL_TOOLS"]
