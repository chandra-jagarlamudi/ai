"""
Math Calculator Tool (powered by NumPy)
-----------------------------------------
Evaluates mathematical expressions by exposing NumPy functions inside a
sandboxed eval environment.

Security:
    - `__builtins__` is blocked to prevent access to Python builtins
      (no `import`, `open`, `exec`, etc.)
    - Only a curated set of NumPy math functions and constants are exposed
    - Prevents code injection while still supporting a rich set of operations

Supported:
    Operators : +  -  *  /  **  %  //  ( )
    Functions : sqrt, log, log10, log2, exp, sin, cos, tan,
                asin, acos, atan, abs, ceil, floor, round, power
    Constants : pi, e, inf
"""

import logging

import numpy as np
from langchain.tools import tool

logger = logging.getLogger(__name__)

# Safe evaluation namespace — only NumPy math, no Python builtins
_SAFE_NAMESPACE: dict = {
    "__builtins__": {},   # block all built-ins (import, open, exec, etc.)
    # ── Arithmetic & rounding ──────────────────────────────────────
    "abs": np.abs,
    "ceil": np.ceil,
    "floor": np.floor,
    "round": np.round,
    "power": np.power,
    # ── Roots & logarithms ────────────────────────────────────────
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "exp": np.exp,
    "log": np.log,        # natural log (base e)
    "log10": np.log10,
    "log2": np.log2,
    # ── Trigonometry (radians) ────────────────────────────────────
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    # ── Constants ────────────────────────────────────────────────
    "pi": np.pi,
    "e": np.e,
    "inf": np.inf,
}


@tool
def calculate_expression(expression: str) -> str:
    """
    Evaluate a mathematical expression using NumPy.

    Use this tool when the user asks to:
    - Calculate or compute a math expression
    - Evaluate formulas with square roots, logarithms, or trigonometry
    - Do quick arithmetic, percentages, or scientific calculations

    The LLM should convert the user's natural language problem into a
    valid Python/NumPy math expression before calling this tool.

    Supported functions: sqrt, log, log10, log2, exp, sin, cos, tan,
                         asin, acos, atan, abs, ceil, floor, round, power, cbrt
    Supported constants: pi, e, inf
    Supported operators: + - * / ** % // ( )

    Args:
        expression: A Python-compatible math expression string.
                    Examples:
                    - "sqrt(144) + 15"
                    - "50000 * 0.15"
                    - "log10(1000) / pi"
                    - "2**10 + sqrt(625)"
                    - "sin(pi/6)"

    Returns:
        The evaluated result formatted as a readable string.
    """
    cleaned = expression.strip()
    logger.info("Evaluating expression: %s", cleaned)

    try:
        # eval with restricted namespace — no builtins, only safe math functions
        result = eval(cleaned, _SAFE_NAMESPACE)  # noqa: S307

        # Convert numpy scalar to native Python type for clean formatting
        if hasattr(result, "item"):
            result = result.item()

        # Format result: use int display if result is a whole number
        if isinstance(result, float) and result.is_integer():
            formatted = f"{int(result):,}"
        elif isinstance(result, float):
            formatted = f"{result:,.6g}"   # up to 6 significant digits
        else:
            formatted = str(result)

        logger.info("Expression '%s' evaluated to: %s", cleaned, formatted)
        return (
            f"🧮 **Calculator Result**\n\n"
            f"`{expression}` = **{formatted}**\n"
        )

    except ZeroDivisionError:
        logger.warning("Division by zero in expression: %s", cleaned)
        return f"❌ Division by zero in expression: `{expression}`"
    except NameError as exc:
        logger.warning("Unknown name in expression '%s': %s", cleaned, exc)
        return (
            f"❌ Unknown function or variable in `{expression}`: {exc}\n"
            f"Supported: sqrt, log, log10, log2, exp, sin, cos, tan, "
            f"asin, acos, atan, abs, ceil, floor, round, power, pi, e"
        )
    except Exception as exc:
        logger.error("Failed to evaluate expression '%s': %s", cleaned, exc)
        return f"❌ Could not evaluate `{expression}`: {exc}"
