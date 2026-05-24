import os
import re
import json
import asyncio
from typing import Any, AsyncIterable

import yfinance as yf
import fastapi_poe as fp
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(".env", override=True)


def _to_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def _fmt_price(v: float | None) -> str:
    return "N/A" if v is None else f"{v:,.2f}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _extract_tickers(text: str) -> list[str]:
    raw = re.findall(r"\$?[A-Za-z]{1,5}\b", text or "")
    stop = {
        "THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WHAT", "WHEN",
        "WILL", "WOULD", "COULD", "SHOULD", "PRICE", "STOCK", "NEWS", "TODAY",
        "ABOUT", "SHOW", "GIVE", "PLEASE", "ANALYSIS", "RISK", "BUY", "SELL",
        "HOLD", "OUTLOOK", "MARKET",
    }
    tickers = []
    for t in raw:
        s = t.replace("$", "").upper()
        if 1 <= len(s) <= 5 and s.isalpha() and s not in stop:
            tickers.append(s)
    return sorted(set(tickers))[:10]


def _fetch_quotes_yfinance_sync(symbols: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            fi = getattr(t, "fast_info", {}) or {}

            price = _to_float(fi.get("lastPrice") or fi.get("regularMarketPrice"))
            prev_close = _to_float(fi.get("previousClose"))
            day_low = _to_float(fi.get("dayLow"))
            day_high = _to_float(fi.get("dayHigh"))

            # Fallback from recent history if fast_info is sparse
            if price is None or prev_close is None or day_low is None or day_high is None:
                hist = t.history(period="5d", interval="1d", auto_adjust=False)
                if not hist.empty:
                    closes = hist.get("Close")
                    lows = hist.get("Low")
                    highs = hist.get("High")

                    if closes is not None:
                        closes = closes.dropna()
                        if price is None and len(closes) >= 1:
                            price = _to_float(closes.iloc[-1])
                        if prev_close is None and len(closes) >= 2:
                            prev_close = _to_float(closes.iloc[-2])

                    if lows is not None:
                        lows = lows.dropna()
                        if day_low is None and len(lows) >= 1:
                            day_low = _to_float(lows.iloc[-1])

                    if highs is not None:
                        highs = highs.dropna()
                        if day_high is None and len(highs) >= 1:
                            day_high = _to_float(highs.iloc[-1])

            change_pct = None
            if price is not None and prev_close not in (None, 0):
                change_pct = ((price - prev_close) / prev_close) * 100.0

            out[sym] = {
                "price": price,
                "changePct": change_pct,
                "dayLow": day_low,
                "dayHigh": day_high,
                "previousClose": prev_close,
                "marketState": "REGULAR",
            }
        except Exception:
            out[sym] = {
                "price": None,
                "changePct": None,
                "dayLow": None,
                "dayHigh": None,
                "previousClose": None,
                "marketState": None,
            }

    return out


class StockMarketWatchBot(fp.PoeBot):
    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(
            allow_attachments=False,
            introduction_message="Hi! Ask me about stocks like AAPL, TSLA, NVDA.",
        )

    async def _optional_llm_commentary(self, user_text: str, quotes: dict[str, dict[str, Any]]) -> str:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key:
            return ""

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        client = AsyncOpenAI(api_key=openai_key)

        try:
            completion = await client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise stock assistant. Use provided data only. No financial advice.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User question: {user_text}\n\n"
                            f"Quote data: {json.dumps(quotes)}\n\n"
                            "Provide a brief trend/risk summary in 3-5 lines."
                        ),
                    },
                ],
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception:
            return ""

    async def get_response(self, request: fp.QueryRequest) -> AsyncIterable[fp.PartialResponse]:
        try:
            user_text = ""
            for m in reversed(request.query):
                if getattr(m, "role", "") == "user":
                    user_text = str(getattr(m, "content", "") or "")
                    break

            if not user_text.strip():
                yield fp.PartialResponse(text="Please send a stock question, e.g. `TSLA outlook`.")
                return

            tickers = _extract_tickers(user_text)
            if not tickers:
                yield fp.PartialResponse(text="Please include a ticker symbol like AAPL or TSLA.")
                return

            quotes = await asyncio.to_thread(_fetch_quotes_yfinance_sync, tickers)

            lines = []
            for t in tickers:
                q = quotes.get(t, {})
                if q.get("price") is None:
                    lines.append(f"• {t}: No data found")
                    continue

                lines.append(
                    f"• {t}: Price={_fmt_price(q.get('price'))} "
                    f"({_fmt_pct(q.get('changePct'))}), "
                    f"DayRange={_fmt_price(q.get('dayLow'))}-{_fmt_price(q.get('dayHigh'))}, "
                    f"PrevClose={_fmt_price(q.get('previousClose'))}, "
                    f"State={q.get('marketState') or 'N/A'}"
                )

            reply = "Live Quotes:\n" + "\n".join(lines)

            # Optional OpenAI summary
            commentary = await self._optional_llm_commentary(user_text, quotes)
            if commentary:
                reply += "\n\nQuick Analysis:\n" + commentary

            reply += "\n\n_Not financial advice._"
            yield fp.PartialResponse(text=reply)

        except Exception as e:
            yield fp.PartialResponse(text=f"Error: {e}")


# ---- App wiring ----
bot_name = os.getenv("POE_BOT_NAME", "").strip()
access_key = os.getenv("POE_ACCESS_KEY", "").strip()

if not bot_name or not access_key:
    raise RuntimeError("Missing POE_BOT_NAME or POE_ACCESS_KEY environment variables.")

bot = StockMarketWatchBot()
app = fp.make_app(bot, access_key=access_key, bot_name=bot_name)
