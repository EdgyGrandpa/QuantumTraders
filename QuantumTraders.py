cat > QuantumTraders.py << 'PY'
import os
import re
import json
import asyncio
from typing import AsyncIterable, Any
from urllib import request as urlrequest
from urllib.parse import quote

import fastapi_poe as fp
from dotenv import load_dotenv
from openai import AsyncOpenAI

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
        "ABOUT", "SHOW", "GIVE", "PLEASE", "ANALYSIS", "RISK", "BUY", "SELL", "HOLD"
    }
    out = []
    for t in raw:
        s = t.replace("$", "").upper()
        if 1 <= len(s) <= 5 and s.isalpha() and s not in stop:
            out.append(s)
    return sorted(set(out))[:10]


def _fetch_yahoo_quotes_sync(symbols: list[str]) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}

    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={quote(','.join(symbols))}"
    req = urlrequest.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        method="GET",
    )

    with urlrequest.urlopen(req, timeout=12) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    items = payload.get("quoteResponse", {}).get("result", [])
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        sym = str(item.get("symbol", "")).upper()
        if not sym:
            continue
        out[sym] = {
            "price": _to_float(item.get("regularMarketPrice")),
            "changePct": _to_float(item.get("regularMarketChangePercent")),
            "dayLow": _to_float(item.get("regularMarketDayLow")),
            "dayHigh": _to_float(item.get("regularMarketDayHigh")),
            "previousClose": _to_float(item.get("regularMarketPreviousClose")),
            "marketState": item.get("marketState"),
        }
    return out


class QuantumTradersBot(fp.PoeBot):
    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(
            allow_attachments=False,
            introduction_message="Hi! Ask me about stocks like AAPL, TSLA, NVDA."
        )

    async def get_response(self, request: fp.QueryRequest) -> AsyncIterable[fp.PartialResponse]:
        user_text = ""
        for m in reversed(request.query):
            if getattr(m, "role", "") == "user":
                user_text = str(getattr(m, "content", "") or "")
                break

        tickers = _extract_tickers(user_text)
        if not tickers:
            yield fp.PartialResponse(text="Please include a ticker symbol (example: AAPL).")
            return

        try:
            quotes = await asyncio.to_thread(_fetch_yahoo_quotes_sync, tickers)
        except Exception as e:
            yield fp.PartialResponse(text=f"Quote fetch error: {e}")
            return

        lines = []
        for t in tickers:
            q = quotes.get(t)
            if not q or q.get("price") is None:
                lines.append(f"• {t}: No data found")
                continue
            lines.append(
                f"• {t}: Price={_fmt_price(q['price'])} ({_fmt_pct(q['changePct'])}), "
                f"DayRange={_fmt_price(q['dayLow'])}-{_fmt_price(q['dayHigh'])}, "
                f"PrevClose={_fmt_price(q['previousClose'])}, "
                f"State={q.get('marketState') or 'N/A'}"
            )

        reply = "Live Quotes:\n" + "\n".join(lines)

        # Optional OpenAI commentary if key exists
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            try:
                model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
                client = AsyncOpenAI(api_key=openai_key)
                comp = await client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are a concise stock assistant. No financial advice."},
                        {"role": "user", "content": f"User: {user_text}\nData: {json.dumps(quotes)}\nGive brief analysis."},
                    ],
                )
                analysis = (comp.choices[0].message.content or "").strip()
                if analysis:
                    reply += "\n\nQuick Analysis:\n" + analysis
            except Exception:
                pass

        reply += "\n\n_Not financial advice._"
        yield fp.PartialResponse(text=reply)


bot_name = os.getenv("POE_BOT_NAME", "").strip()
access_key = os.getenv("POE_ACCESS_KEY", "").strip()
if not bot_name or not access_key:
    raise RuntimeError("Missing POE_BOT_NAME or POE_ACCESS_KEY environment variables.")

bot = QuantumTradersBot()
app = fp.make_app(bot, access_key=access_key, bot_name=bot_name)
PY
