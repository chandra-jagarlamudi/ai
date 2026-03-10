"""
Stock Price Tool
-----------------
Fetches real-time (or latest available) stock price and key market data
using the yfinance library, which queries Yahoo Finance — no API key required.

Ticker symbol conventions:
    US stocks  : AAPL, TSLA, MSFT, GOOGL
    Indian NSE : RELIANCE.NS, TCS.NS, INFY.NS
    Indian BSE : RELIANCE.BO
    Crypto     : BTC-USD, ETH-USD
"""

import logging

import yfinance as yf
from langchain.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_stock_price(ticker_symbol: str) -> str:
    """
    Fetch the current stock price and key market information for a ticker symbol.

    Use this tool when the user asks about:
    - Current or live stock price of any publicly traded company
    - Basic stock metrics (day high/low, volume, market cap, P/E ratio)
    - Comparing stock prices or checking portfolio values

    Args:
        ticker_symbol: The stock ticker symbol. Examples:
                       "AAPL" (Apple), "TSLA" (Tesla), "RELIANCE.NS" (NSE India),
                       "BTC-USD" (Bitcoin), "^NSEI" (Nifty 50 index)

    Returns:
        A formatted summary of current price and available market metrics.
    """
    symbol = ticker_symbol.strip().upper()
    logger.info("Fetching stock data for ticker: %s", symbol)

    try:
        ticker = yf.Ticker(symbol)

        # fast_info is a lightweight cache-friendly attribute
        fast = ticker.fast_info

        # Fetch 1-day history to verify data exists for this symbol
        hist = ticker.history(period="5d")
        if hist.empty:
            return (
                f"⚠️ No price data found for ticker **{symbol}**.\n"
                f"Please verify the symbol. Examples: AAPL, TSLA, RELIANCE.NS, BTC-USD"
            )

        current_price = fast.last_price
        day_high = fast.day_high
        day_low = fast.day_low
        volume = fast.last_volume
        prev_close = fast.previous_close

        # Calculate day change
        day_change = current_price - prev_close
        day_change_pct = (day_change / prev_close) * 100
        trend = "📈" if day_change >= 0 else "📉"

        lines = [
            f"{trend} **Stock: {symbol}**\n",
            f"- **Current Price:** {current_price:,.2f}",
            f"- Previous Close: {prev_close:,.2f}",
            f"- Day Change: {day_change:+,.2f} ({day_change_pct:+.2f}%)",
            f"- Day High: {day_high:,.2f}",
            f"- Day Low: {day_low:,.2f}",
            f"- Volume: {int(volume):,}",
        ]

        # Append optional fields that may not always be available
        market_cap = getattr(fast, "market_cap", None)
        if market_cap:
            lines.append(f"- Market Cap: {market_cap:,.0f}")

        fifty_two_week_high = getattr(fast, "year_high", None)
        fifty_two_week_low = getattr(fast, "year_low", None)
        if fifty_two_week_high and fifty_two_week_low:
            lines.append(f"- 52-Week High: {fifty_two_week_high:,.2f}")
            lines.append(f"- 52-Week Low:  {fifty_two_week_low:,.2f}")

        lines.append(
            "\n_Data sourced from Yahoo Finance via yfinance. "
            "Prices may be delayed 15 minutes._"
        )

        logger.info("Successfully fetched data for %s: price=%.2f", symbol, current_price)
        return "\n".join(lines)

    except Exception as exc:
        logger.error("Failed to fetch stock data for %s: %s", symbol, exc)
        return (
            f"❌ Error fetching data for **{symbol}**: {exc}\n"
            f"Please check the ticker symbol and try again."
        )
