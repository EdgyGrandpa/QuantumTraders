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


def _classify_trend(change_pct: float | None) -> str:
    if change_pct is None:
        return "Unknown"
    if change_pct > 0.5:
        return "Bullish"
    if change_pct < -0.5:
        return "Bearish"
    return "Neutral"


def _classify_risk(price: float | None, low: float | None, high: float | None) -> str:
    if price in (None, 0) or low is None or high is None:
        return "Unknown"
    intraday_vol = ((high - low) / price) * 100.0
    if intraday_vol >= 4:
        return "High"
    if intraday_vol >= 2:
        return "Medium"
    return "Low"


def _is_bare_ticker_request(user_text: str, tickers: list[str]) -> bool:
    cleaned = re.sub(r"[^A-Za-z$ ]+", " ", (user_text or "")).strip()
    tokens = [t.replace("$", "").upper() for t in cleaned.split() if t.strip()]
    return len(tokens) == 1 and len(tickers) == 1 and tokens[0] == tickers[0]


def _extract_tickers(text: str) -> list[str]:
    text = text or ""
    text_lower = text.lower()

    # Simple company-name aliases for natural input like "Tesla outlook"
    aliases = {
        "tesla": "TSLA",
        "apple": "AAPL",
        "nvidia": "NVDA",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "netflix": "NFLX",
        "amd": "AMD",
    }

    found = set()
    for name, ticker in aliases.items():
        if re.search(rf"\b{re.escape(name)}\b", text_lower):
            found.add(ticker)

    # Ticker-like tokens (AAPL, $TSLA, NVDA, etc.)
    raw = re.findall(r"\$?[A-Za-z]{1,5}\b", text)
    stop = {
        "THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WHAT", "WHEN",
        "WILL", "WOULD", "COULD", "SHOULD", "PRICE", "STOCK", "NEWS", "TODAY",
        "ABOUT", "SHOW", "GIVE", "PLEASE", "ANALYSIS", "RISK", "BUY", "SELL",
        "HOLD", "OUTLOOK", "MARKET", "TREND", "COMPARE", "VERSUS", "VS",
        "QUICK", "LOOK", "HELP", "TELL", "ME",
    }

    for t in raw:
        s = t.replace("$", "").upper()
        if 1 <= len(s) <= 5 and s.isalpha() and s not in stop:
            found.add(s)

    return sorted(found)[:10]


def _fetch_quotes_yfinance_sync(symbols: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            fi = getattr(ticker, "fast_info", {}) or {}

            price = _to_float(fi.get("lastPrice") or fi.get("regularMarketPrice"))
            prev_close = _to_float(fi.get("previousClose"))
            day_low = _to_float(fi.get("dayLow"))
            day_high = _to_float(fi.get("dayHigh"))

            # Fallback if fast_info is sparse
            if price is None or prev_close is None or day_low is None or day_high is None:
                hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
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
                temperature=0.35,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a warm, human-like stock market assistant. "
                            "Be clear, concise, and practical. "
                            "Use friendly language and short paragraphs. "
                            "Never reveal internal reasoning. "
                            "Do not provide financial advice. "
                            "Base analysis strictly on provided quote data."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User question: {user_text}\n\n"
                            f"Quote data: {json.dumps(quotes)}\n\n"
                            "Write a brief trader-style readout with: "
                            "1) what changed, 2) short-term trend tone, "
                            "3) risk/volatility note, 4) one useful next-step question."
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
                    content = getattr(m, "content", "")
                    user_text = content if isinstance(content, str) else str(content)
                    break

            if not user_text.strip():
                yield fp.PartialResponse(
                    text=(
                        "Hey — I can help with live stock insights.\n\n"
                        "Try:\n"
                        "- `TSLA outlook`\n"
                        "- `Compare NVDA vs AMD`\n"
                        "- `Apple trend today`"
                    )
                )
                return

            tickers = _extract_tickers(user_text)
            if not tickers:
                yield fp.PartialResponse(
                    text=(
                        "Got it — what ticker should we analyze?\n\n"
                        "You can send something like:\n"
                        "- `TSLA`\n"
                        "- `AAPL quick outlook`\n"
                        "- `Compare TSLA vs NVDA`"
                    )
                )
                return

            quotes = await asyncio.to_thread(_fetch_quotes_yfinance_sync, tickers)

            # Human-style clarification flow for bare single ticker input
            if _is_bare_ticker_request(user_text, tickers):
                t = tickers[0]
                q = quotes.get(t, {})
                if q.get("price") is None:
                    yield fp.PartialResponse(
                        text=(
                            f"I can analyze **{t}**, but I couldn’t fetch live data right now.\n\n"
                            "Please try again in a moment."
                        )
                    )
                    return

                trend = _classify_trend(q.get("changePct"))
                risk = _classify_risk(q.get("price"), q.get("dayLow"), q.get("dayHigh"))

                reply = (
                    f"# {t} Snapshot\n\n"
                    f"**Price:** USD {_fmt_price(q.get('price'))}  \n"
                    f"**Today:** {_fmt_pct(q.get('changePct'))}  \n"
                    f"**Day Range:** USD {_fmt_price(q.get('dayLow'))} - USD {_fmt_price(q.get('dayHigh'))}  \n"
                    f"**Trend:** {trend}  \n"
                    f"**Risk:** {risk}\n\n"
                    "Want me to go deeper?\n\n"
                    "| Option | What I can do |\n"
                    "|---|---|\n"
                    "| Quick Outlook | 3–5 line near-term read |\n"
                    "| Key Levels | Support/resistance from recent range |\n"
                    "| Momentum Read | RSI/MACD-style interpretation |\n"
                    "| Compare | Compare with another ticker |\n\n"
                    "Reply with one of these:\n"
                    "- `TSLA quick outlook`\n"
                    "- `TSLA key levels`\n"
                    "- `Compare TSLA vs NVDA`\n\n"
                    "_Not financial advice._"
                )
                yield fp.PartialResponse(text=reply)
                return

            # Standard multi/specific request response
            lines = []
            for t in tickers:
                q = quotes.get(t, {})
                if q.get("price") is None:
                    lines.append(f"• {t}: No data found")
                    continue

                trend = _classify_trend(q.get("changePct"))
                risk = _classify_risk(q.get("price"), q.get("dayLow"), q.get("dayHigh"))

                lines.append(
                    f"• {t}: Price={_fmt_price(q.get('price'))} "
                    f"({_fmt_pct(q.get('changePct'))}), "
                    f"DayRange={_fmt_price(q.get('dayLow'))}-{_fmt_price(q.get('dayHigh'))}, "
                    f"PrevClose={_fmt_price(q.get('previousClose'))}, "
                    f"Trend={trend}, Risk={risk}, "
                    f"State={q.get('marketState') or 'N/A'}"
                )

            reply = "## Live Quotes\n" + "\n".join(lines)

            commentary = await self._optional_llm_commentary(user_text, quotes)
            if commentary:
                reply += "\n\n## Quick Analysis\n" + commentary

            reply += "\n\n_Not financial advice._"
            yield fp.PartialResponse(text=reply)

        except Exception as e:
            yield fp.PartialResponse(text=f"Sorry — I hit an error while analyzing that request: {e}")


# ---- App wiring ----
bot_name = os.getenv("POE_BOT_NAME", "").strip()
access_key = os.getenv("POE_ACCESS_KEY", "").strip()

if not bot_name or not access_key:
    raise RuntimeError("Missing POE_BOT_NAME or POE_ACCESS_KEY environment variables.")

bot = StockMarketWatchBot()
app = fp.make_app(bot, access_key=access_key, bot_name=bot_name)
