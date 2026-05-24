import os
import re
import json
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Optional, AsyncIterable

import yfinance as yf
import fastapi_poe as fp
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(".env", override=True)

# =========================
# Session memory (ephemeral)
# =========================
@dataclass
class SessionPrefs:
    timeframe: str = "Daily"          # 1m,5m,15m,30m,1H,2H,4H,Daily,Weekly,Monthly
    mode: str = "outlook"             # outlook, trend, levels, patterns, indicators, compare
    style: str = "swing"              # scalp, day, swing
    language: str = "en"              # en, es
    teach_mode: bool = False
    updated_at: float = 0.0


SESSION_STORE: dict[str, SessionPrefs] = {}
SESSION_TTL_SECONDS = 60 * 60 * 12  # 12 hours


# =========================
# Utility helpers
# =========================
def _to_float(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def _fmt_price(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:,.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _classify_trend(change_pct: Optional[float]) -> str:
    if change_pct is None:
        return "Unknown"
    if change_pct > 0.5:
        return "Bullish"
    if change_pct < -0.5:
        return "Bearish"
    return "Neutral"


def _classify_risk(price: Optional[float], low: Optional[float], high: Optional[float]) -> str:
    if price in (None, 0) or low is None or high is None:
        return "Unknown"
    intraday_vol = ((high - low) / price) * 100.0
    if intraday_vol >= 4:
        return "High"
    if intraday_vol >= 2:
        return "Medium"
    return "Low"


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_affirmative(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "do it", "go ahead", "continue"}


def _contains_sp500_intent(text: str) -> bool:
    t = (text or "").lower()
    triggers = ["s&p 500", "sp500", "spx", "s and p", "s&p", "spy index", "spdr s&p 500"]
    return any(k in t for k in triggers)


def _display_symbol(sym: str) -> str:
    return "SPX" if sym == "^GSPC" else sym


def _mode_title(mode: str) -> str:
    return {
        "outlook": "Quick Outlook",
        "trend": "Trend Analysis",
        "levels": "Support & Resistance",
        "patterns": "Pattern Analysis",
        "indicators": "Indicator Analysis",
        "compare": "Comparison",
    }.get(mode, "Quick Outlook")


def _extract_timeframe(text: str) -> Optional[str]:
    t = (text or "").lower()

    patterns = [
        (r"\b(1m|one minute)\b", "1m"),
        (r"\b(5m|5 min|5 minute|5 minutes)\b", "5m"),
        (r"\b(15m|15 min|15 minute|15 minutes)\b", "15m"),
        (r"\b(30m|30 min|30 minute|30 minutes)\b", "30m"),
        (r"\b(1h|1 hr|1 hour|hourly)\b", "1H"),
        (r"\b(2h|2 hr|2 hour)\b", "2H"),
        (r"\b(4h|4 hr|4 hour)\b", "4H"),
        (r"\b(daily|1d|day)\b", "Daily"),
        (r"\b(weekly|1w|week)\b", "Weekly"),
        (r"\b(monthly|1mo|month)\b", "Monthly"),
    ]
    for pattern, tf in patterns:
        if re.search(pattern, t):
            return tf
    return None


def _extract_mode(text: str) -> Optional[str]:
    t = (text or "").lower()

    if any(k in t for k in ["compare", " vs ", " versus ", "against"]):
        return "compare"
    if any(k in t for k in ["support", "resistance", "key levels", "levels"]):
        return "levels"
    if any(k in t for k in ["pattern", "head and shoulders", "triangle", "flag", "wedge", "double top", "double bottom"]):
        return "patterns"
    if any(k in t for k in ["indicator", "rsi", "macd", "moving average", "bollinger", "volume"]):
        return "indicators"
    if any(k in t for k in ["trend", "bullish", "bearish", "momentum"]):
        return "trend"
    if any(k in t for k in ["outlook", "future", "what next", "quick analysis"]):
        return "outlook"

    return None


def _extract_style(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in ["scalp", "scalping"]):
        return "scalp"
    if any(k in t for k in ["day trade", "intraday", "day trading", "daytrading"]):
        return "day"
    if "swing" in t:
        return "swing"
    return None


def _extract_language(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in ["spanish", "español", "en español"]):
        return "es"
    if "english" in t:
        return "en"
    return None


def _extract_teach_mode(text: str) -> Optional[bool]:
    t = (text or "").lower()
    if any(k in t for k in ["teach me", "step by step", "explain why"]):
        return True
    if any(k in t for k in ["no teaching", "just answer", "concise only"]):
        return False
    return None


def _extract_tickers(text: str) -> list[str]:
    text = text or ""
    text_lower = text.lower()
    found: set[str] = set()

    # Name aliases
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
        "intel": "INTC",
        "palantir": "PLTR",
        "spy": "SPY",
        "qqq": "QQQ",
        "berkshire": "BRK.B",
        "tsla": "TSLA",
        "aapl": "AAPL",
        "nvda": "NVDA",
        "msft": "MSFT",
        "amzn": "AMZN",
        "googl": "GOOGL",
    }

    if _contains_sp500_intent(text):
        found.add("^GSPC")

    for name, ticker in aliases.items():
        if re.search(rf"\b{re.escape(name)}\b", text_lower):
            found.add(ticker)

    # explicit $TICKER
    for m in re.findall(r"\$([A-Za-z]{1,5})\b", text):
        found.add(m.upper())

    # dotted ticker like BRK.B
    for m in re.findall(r"\b([A-Z]{1,4}\.[A-Z])\b", text):
        found.add(m.upper())

    # uppercase-only tokens (prevents word slicing bug)
    for m in re.finditer(r"\b([A-Za-z]{1,5})\b", text):
        tok = m.group(1)
        if tok.isupper():
            found.add(tok)

    stop = {
        "THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WHAT", "WHEN", "WILL",
        "WOULD", "COULD", "SHOULD", "PRICE", "STOCK", "NEWS", "TODAY", "ABOUT", "SHOW",
        "GIVE", "PLEASE", "ANALYSIS", "RISK", "BUY", "SELL", "HOLD", "OUTLOOK", "MARKET",
        "TREND", "COMPARE", "VERSUS", "VS", "QUICK", "LOOK", "HELP", "TELL", "ME", "KEY",
        "LEVELS", "SUPPORT", "RESISTANCE", "INDICATOR", "INDICATORS", "PATTERN", "PATTERNS",
        "HOW", "IS", "IN", "DAYS", "COMING", "FUTURE", "WHATS", "YES", "NO"
    }

    cleaned = [s for s in found if s not in stop]
    return sorted(cleaned)[:10]


def _is_bare_ticker_request(user_text: str, tickers: list[str]) -> bool:
    if len(tickers) != 1:
        return False
    cleaned = re.sub(r"[^A-Za-z$.\s]+", " ", (user_text or "")).strip()
    tokens = [x for x in cleaned.split() if x]
    if len(tokens) != 1:
        return False
    extracted = _extract_tickers(tokens[0])
    return len(extracted) == 1 and extracted[0] == tickers[0]


def _confidence_label(q: dict[str, Any]) -> str:
    needed = ["price", "changePct", "dayLow", "dayHigh", "previousClose"]
    present = sum(1 for k in needed if q.get(k) is not None)
    if present == 5:
        return "High"
    if present >= 3:
        return "Medium"
    return "Low"


def _scenario_block(q: dict[str, Any]) -> str:
    p = q.get("price")
    lo = q.get("dayLow")
    hi = q.get("dayHigh")

    if p is None or lo is None or hi is None:
        return (
            "**Scenarios**\n"
            "- Bull trigger: reclaim and hold recent intraday high\n"
            "- Bear trigger: break and fail to reclaim recent intraday low\n"
            "- Invalidation: unclear structure; wait for confirmation\n"
        )

    return (
        "**Scenarios**\n"
        f"- Bull trigger: break/hold above **${_fmt_price(hi)}**\n"
        f"- Bear trigger: lose **${_fmt_price(lo)}**\n"
        f"- Invalidation: chop between **${_fmt_price(lo)} - ${_fmt_price(hi)}**\n"
    )


def _suggested_prompts(symbol: str, timeframe: str) -> str:
    return (
        "**Try next:**\n"
        f"- `{symbol} key levels on {timeframe}`\n"
        f"- `{symbol} trend + indicators on {timeframe}`\n"
        f"- `Compare {symbol} vs NVDA on {timeframe}`"
    )


def _get_session_key(request: fp.QueryRequest) -> str:
    cid = getattr(request, "conversation_id", None) or getattr(request, "conversationId", None)
    uid = getattr(request, "user_id", None) or getattr(request, "userId", None)
    qid = getattr(request, "message_id", None) or getattr(request, "bot_query_id", None)
    return str(cid or uid or qid or "default")


def _get_prefs(request: fp.QueryRequest) -> SessionPrefs:
    key = _get_session_key(request)
    now = time.time()

    prefs = SESSION_STORE.get(key)
    if prefs and now - prefs.updated_at <= SESSION_TTL_SECONDS:
        return prefs

    prefs = SessionPrefs(updated_at=now)
    SESSION_STORE[key] = prefs
    return prefs


def _save_prefs(request: fp.QueryRequest, prefs: SessionPrefs) -> None:
    prefs.updated_at = time.time()
    SESSION_STORE[_get_session_key(request)] = prefs


def _extract_image_urls_from_request(request: fp.QueryRequest) -> list[str]:
    urls: list[str] = []

    for m in request.query:
        if getattr(m, "role", "") != "user":
            continue

        attachments = getattr(m, "attachments", None) or []
        for a in attachments:
            if isinstance(a, dict):
                url = a.get("url")
                ctype = a.get("content_type") or a.get("contentType")
            else:
                url = getattr(a, "url", None)
                ctype = getattr(a, "content_type", None) or getattr(a, "contentType", None)

            if not url:
                continue

            is_image = False
            if ctype and str(ctype).lower().startswith("image/"):
                is_image = True
            if re.search(r"\.(png|jpg|jpeg|webp|gif)(\?|$)", str(url), re.IGNORECASE):
                is_image = True

            if is_image:
                urls.append(str(url))

    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)

    return out


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

            # Fallback via history
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
            allow_attachments=True,
            introduction_message=(
                "Hi! Send a ticker (TSLA), choose a mode (trend/levels/patterns/indicators/compare), "
                "or upload a chart screenshot."
            ),
        )

    def _find_last_context_tickers(self, request: fp.QueryRequest) -> list[str]:
        for m in reversed(request.query[:-1]):
            if getattr(m, "role", "") != "user":
                continue
            c = getattr(m, "content", "")
            txt = c if isinstance(c, str) else str(c)
            if not txt.strip() or _is_affirmative(txt):
                continue
            tks = _extract_tickers(txt)
            if tks:
                return tks
        return []

    async def _optional_llm_commentary(
        self,
        user_text: str,
        quotes: dict[str, dict[str, Any]],
        timeframe: str,
        mode: str,
        style: str,
        language: str,
        teach_mode: bool,
    ) -> str:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key:
            return ""

        client = AsyncOpenAI(api_key=openai_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

        lang_instruction = "Respond in Spanish." if language == "es" else "Respond in English."
        teach_instruction = (
            "Use step-by-step format: What we see / Why it matters / What to watch next."
            if teach_mode else
            "Keep it concise and practical."
        )

        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0.35,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a trader-style assistant. "
                            "Professional, friendly, concise. "
                            "Do not reveal chain-of-thought. "
                            "Do not give guarantees or financial advice. "
                            f"{lang_instruction} {teach_instruction}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Mode: {mode}\n"
                            f"Timeframe: {timeframe}\n"
                            f"Trading style: {style}\n"
                            f"User request: {user_text}\n"
                            f"Quote data: {json.dumps(quotes)}\n\n"
                            "Write a short trader-style analysis and end with one useful follow-up question."
                        ),
                    },
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    async def _chart_image_analysis(
        self,
        user_text: str,
        image_urls: list[str],
        tickers: list[str],
        quotes: dict[str, dict[str, Any]],
        timeframe: str,
        mode: str,
        style: str,
        language: str,
        teach_mode: bool,
    ) -> str:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key:
            return "I can analyze chart uploads after you set `OPENAI_API_KEY`.\n\n_Not financial advice._"

        client = AsyncOpenAI(api_key=openai_key)
        model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")).strip()

        focus = ", ".join(_display_symbol(t) for t in tickers) if tickers else "the charted asset"
        lang_instruction = "Respond in Spanish." if language == "es" else "Respond in English."
        teach_instruction = (
            "Use step-by-step format: What we see / Why it matters / What to watch next."
            if teach_mode else
            "Keep it concise and practical."
        )

        content = [
            {
                "type": "text",
                "text": (
                    "Analyze this trading chart.\n"
                    f"User request: {user_text}\n"
                    f"Focus asset: {focus}\n"
                    f"Mode: {mode}\n"
                    f"Timeframe: {timeframe}\n"
                    f"Style: {style}\n"
                    f"Quote context: {json.dumps(quotes)}\n\n"
                    "Output markdown with sections:\n"
                    "1) Trend\n2) Support/Resistance\n3) Patterns\n4) Indicator-style read (if visible)\n"
                    "5) Bull/Bear scenarios with invalidation.\n"
                    f"{lang_instruction} {teach_instruction} "
                    "End with: Not financial advice."
                ),
            }
        ]

        for u in image_urls[:3]:
            content.append({"type": "image_url", "image_url": {"url": u}})

        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0.25,
                messages=[
                    {"role": "system", "content": "You are a precise technical chart analyst."},
                    {"role": "user", "content": content},
                ],
            )
            return (resp.choices[0].message.content or "").strip() or "I couldn’t extract a confident chart read."
        except Exception as e:
            return f"I couldn’t analyze the chart right now: {e}"

    async def get_response(self, request: fp.QueryRequest) -> AsyncIterable[fp.PartialResponse]:
        try:
            # Latest user message
            user_text = ""
            for m in reversed(request.query):
                if getattr(m, "role", "") == "user":
                    c = getattr(m, "content", "")
                    user_text = c if isinstance(c, str) else str(c)
                    break
            user_text = _normalize_text(user_text)

            # Session prefs
            prefs = _get_prefs(request)

            # Update prefs from explicit user input
            tf = _extract_timeframe(user_text)
            md = _extract_mode(user_text)
            st = _extract_style(user_text)
            lg = _extract_language(user_text)
            tm = _extract_teach_mode(user_text)

            if tf:
                prefs.timeframe = tf
            if md:
                prefs.mode = md
            if st:
                prefs.style = st
            if lg:
                prefs.language = lg
            if tm is not None:
                prefs.teach_mode = tm

            _save_prefs(request, prefs)

            current_tf = prefs.timeframe
            mode = prefs.mode
            style = prefs.style
            language = prefs.language
            teach_mode = prefs.teach_mode

            if not user_text:
                yield fp.PartialResponse(
                    text=(
                        "## Welcome 📈\n\n"
                        "Try:\n"
                        "- `TSLA trend on 4H`\n"
                        "- `TSLA key levels`\n"
                        "- `Compare TSLA vs NVDA`\n"
                        "- `S&P 500 trend analysis`\n"
                        "- Upload a chart screenshot\n\n"
                        "_Not financial advice._"
                    )
                )
                return

            tickers = _extract_tickers(user_text)
            image_urls = _extract_image_urls_from_request(request)

            # Follow-up "yes"
            if _is_affirmative(user_text) and not tickers:
                tickers = self._find_last_context_tickers(request)

            if not tickers and _contains_sp500_intent(user_text):
                tickers = ["^GSPC"]

            # Onboarding/clarify flow
            if not tickers and not image_urls:
                yield fp.PartialResponse(
                    text=(
                        "## Let’s analyze a setup 📊\n\n"
                        "| Option | Example |\n"
                        "|---|---|\n"
                        "| 📈 Trend | `TSLA trend on 4H` |\n"
                        "| 📐 Levels | `TSLA support resistance` |\n"
                        "| ⚙️ Indicators | `TSLA RSI MACD` |\n"
                        "| 🔍 Patterns | `TSLA chart patterns` |\n"
                        "| ⚖️ Compare | `Compare TSLA vs NVDA` |\n\n"
                        f"Current defaults: **{current_tf}**, **{_mode_title(mode)}**, **{style}** style.\n\n"
                        "_Not financial advice._"
                    )
                )
                return

            quotes: dict[str, dict[str, Any]] = {}
            if tickers:
                quotes = await asyncio.to_thread(_fetch_quotes_yfinance_sync, tickers)

            # Chart-upload branch
            if image_urls:
                chart_text = await self._chart_image_analysis(
                    user_text=user_text,
                    image_urls=image_urls,
                    tickers=tickers,
                    quotes=quotes,
                    timeframe=current_tf,
                    mode=mode,
                    style=style,
                    language=language,
                    teach_mode=teach_mode,
                )
                yield fp.PartialResponse(
                    text=f"## Chart Analysis ({current_tf} • {_mode_title(mode)}) 📊\n\n{chart_text}"
                )
                return

            # Bare ticker guided flow
            if _is_bare_ticker_request(user_text, tickers):
                t = tickers[0]
                q = quotes.get(t, {})
                disp = _display_symbol(t)

                if q.get("price") is None:
                    yield fp.PartialResponse(
                        text=f"## {disp}\n\nI couldn’t fetch live data right now. Please try again.\n\n_Not financial advice._"
                    )
                    return

                trend = _classify_trend(q.get("changePct"))
                risk = _classify_risk(q.get("price"), q.get("dayLow"), q.get("dayHigh"))
                conf = _confidence_label(q)

                yield fp.PartialResponse(
                    text=(
                        f"## {disp} ({current_tf}) 📊\n\n"
                        f"Snapshot: **${_fmt_price(q.get('price'))}** ({_fmt_pct(q.get('changePct'))})  \n"
                        f"Trend: **{trend}** | Risk: **{risk}** | Confidence: **{conf}**\n\n"
                        f"{_scenario_block(q)}\n"
                        "| Choose next | Prompt |\n"
                        "|---|---|\n"
                        f"| Trend | `{disp} trend on {current_tf}` |\n"
                        f"| Levels | `{disp} key levels on {current_tf}` |\n"
                        f"| Indicators | `{disp} RSI MACD on {current_tf}` |\n"
                        f"| Compare | `Compare {disp} vs NVDA on {current_tf}` |\n\n"
                        "_Not financial advice._"
                    )
                )
                return

            # Compare mode
            if mode == "compare" and len(tickers) >= 2:
                rows = []
                for t in tickers[:5]:
                    q = quotes.get(t, {})
                    if q.get("price") is None:
                        continue
                    rows.append(
                        f"| {_display_symbol(t)} | ${_fmt_price(q.get('price'))} | {_fmt_pct(q.get('changePct'))} | "
                        f"${_fmt_price(q.get('dayLow'))} - ${_fmt_price(q.get('dayHigh'))} | {_classify_trend(q.get('changePct'))} |"
                    )

                if not rows:
                    yield fp.PartialResponse(text="I couldn’t fetch valid comparison data.\n\n_Not financial advice._")
                    return

                yield fp.PartialResponse(
                    text=(
                        f"## Comparison ({current_tf})\n\n"
                        "| Symbol | Price | Change | Day Range | Trend |\n"
                        "|---|---:|---:|---|---|\n"
                        + "\n".join(rows)
                        + "\n\nWant me to rank these by momentum + risk?\n\n_Not financial advice._"
                    )
                )
                return

            # Standard output
            valid = []
            for t in tickers:
                q = quotes.get(t, {})
                if q.get("price") is None:
                    continue
                disp = _display_symbol(t)
                trend = _classify_trend(q.get("changePct"))
                risk = _classify_risk(q.get("price"), q.get("dayLow"), q.get("dayHigh"))
                conf = _confidence_label(q)

                block = (
                    f"### {disp}\n"
                    f"- Price: **${_fmt_price(q.get('price'))}** ({_fmt_pct(q.get('changePct'))})\n"
                    f"- Day range: **${_fmt_price(q.get('dayLow'))} - ${_fmt_price(q.get('dayHigh'))}**\n"
                    f"- Prev close: **${_fmt_price(q.get('previousClose'))}**\n"
                    f"- Trend: **{trend}** | Risk: **{risk}** | Confidence: **{conf}**\n\n"
                    f"{_scenario_block(q)}\n"
                    f"{_suggested_prompts(disp, current_tf)}"
                )
                valid.append(block)

            if not valid:
                yield fp.PartialResponse(
                    text="I couldn’t find valid live market data. Try `TSLA`, `AAPL`, or `S&P 500`.\n\n_Not financial advice._"
                )
                return

            reply = f"## Live Snapshot ({current_tf} • {_mode_title(mode)})\n\n" + "\n\n".join(valid)

            llm = await self._optional_llm_commentary(
                user_text=user_text,
                quotes=quotes,
                timeframe=current_tf,
                mode=mode,
                style=style,
                language=language,
                teach_mode=teach_mode,
            )
            if llm:
                reply += f"\n\n## {_mode_title(mode)}\n{llm}"

            reply += "\n\n_Not financial advice._"
            yield fp.PartialResponse(text=reply)

        except Exception as e:
            yield fp.PartialResponse(text=f"Sorry — I hit an error while processing your request: {e}")


# =========================
# App wiring
# =========================
bot_name = os.getenv("POE_BOT_NAME", "").strip()
access_key = os.getenv("POE_ACCESS_KEY", "").strip()

if not bot_name or not access_key:
    raise RuntimeError("Missing POE_BOT_NAME or POE_ACCESS_KEY environment variables.")

bot = StockMarketWatchBot()
app = fp.make_app(bot, access_key=access_key, bot_name=bot_name)
