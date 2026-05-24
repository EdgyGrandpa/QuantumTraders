# market_analysis_engine_v34.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from openai import AsyncOpenAI


HOUSE_STYLE = """
Write like a professional market analyst:
- clear, concise, probabilistic
- separate observation from interpretation
- explicitly mention uncertainty/invalidation
- no hype, no certainty language
- no financial advice
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def _json_or_none(text: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads((text or "").strip())
    except Exception:
        return None


def _clean_line(v: Any) -> str:
    s = str(v or "").strip()
    return s if s else "N/A"


def _render_style_locked_analysis(data: dict[str, Any], teach_mode: bool) -> str:
    """
    Fixed section order for text analysis.
    """
    sections: list[tuple[str, Any]] = [
        ("TL;DR", data.get("tldr")),
        ("What I’m seeing", data.get("what_im_seeing")),
    ]

    if teach_mode:
        sections.append(("Why it matters", data.get("why_it_matters")))

    sections.extend(
        [
            ("Levels that matter", data.get("levels_that_matter")),
            ("Scenarios", data.get("scenarios")),
        ]
    )

    if teach_mode:
        sections.append(("Risk note", data.get("risk_note")))

    sections.append(("One smart next question", data.get("one_smart_next_question")))

    out: list[str] = []
    for title, body in sections:
        out.append(f"## {title}\n{_clean_line(body)}")
    return "\n\n".join(out)


def _render_style_locked_chart_analysis(data: dict[str, Any], teach_mode: bool) -> str:
    """
    Fixed section order for chart/image analysis.
    """
    sections: list[tuple[str, Any]] = [
        ("TL;DR", data.get("tldr")),
        ("Trend", data.get("trend")),
        ("Support / Resistance", data.get("support_resistance")),
        ("Patterns", data.get("patterns")),
        ("Indicator read", data.get("indicator_read")),
        ("Scenarios", data.get("scenarios")),
    ]

    if teach_mode:
        sections.append(("Risk note", data.get("risk_note")))

    sections.append(("One smart next question", data.get("one_smart_next_question")))

    out: list[str] = []
    for title, body in sections:
        out.append(f"## {title}\n{_clean_line(body)}")
    return "\n\n".join(out)


# -----------------------------
# Config
# -----------------------------
@dataclass
class AnalysisConfig:
    openai_model: str = os.getenv("OPENAI_MODEL", "o3").strip()
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1").strip()
    temperature: float = 0.2


# -----------------------------
# Engine
# -----------------------------
class MarketAnalysisEngine:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    async def analyze(
        self,
        user_text: str,
        quotes: dict[str, dict[str, Any]],
        timeframe: str = "swing",
        mode: str = "market_analysis",
        style: str = "balanced",
        language: str = "en",
        teach_mode: bool = False,
    ) -> str:
        """
        Main entry point.
        """
        commentary = await self._optional_llm_commentary(
            user_text=user_text,
            quotes=quotes,
            timeframe=timeframe,
            mode=mode,
            style=style,
            language=language,
            teach_mode=teach_mode,
        )

        if commentary:
            return commentary

        # Hard fallback if LLM unavailable
        return self._fallback_from_quotes(quotes=quotes, language=language)

    async def analyze_chart_image(
        self,
        user_text: str,
        image_url: str,
        quotes: dict[str, dict[str, Any]],
        timeframe: str = "swing",
        style: str = "balanced",
        language: str = "en",
        teach_mode: bool = False,
    ) -> str:
        """
        Optional chart/image analysis path.
        """
        text = await self._chart_image_analysis(
            user_text=user_text,
            image_url=image_url,
            quotes=quotes,
            timeframe=timeframe,
            style=style,
            language=language,
            teach_mode=teach_mode,
        )

        if text:
            return text

        return self._fallback_from_quotes(quotes=quotes, language=language)

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
        if not self.client:
            return ""

        lang = "Spanish" if language == "es" else "English"
        schema_hint = (
            "{"
            '"tldr":"...",'
            '"what_im_seeing":"...",'
            '"why_it_matters":"...",'
            '"levels_that_matter":"...",'
            '"scenarios":"...",'
            '"risk_note":"...",'
            '"one_smart_next_question":"..."'
            "}"
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.config.openai_model,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional market analyst.\n"
                            f"{HOUSE_STYLE}\n"
                            "Do not provide financial advice. "
                            "Return JSON only, no markdown, no prose outside JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Respond in {lang}. Return valid JSON only with this schema:\n"
                            f"{schema_hint}\n\n"
                            "Rules:\n"
                            "- Keep each field concise and actionable.\n"
                            "- Include uncertainty and invalidation in 'scenarios'.\n"
                            "- If teach_mode is false, still return all fields (renderer decides visibility).\n\n"
                            f"teach_mode={teach_mode}\n"
                            f"Mode: {mode}\n"
                            f"Timeframe: {timeframe}\n"
                            f"Trading style: {style}\n"
                            f"User request: {user_text}\n"
                            f"Quote data: {json.dumps(quotes, ensure_ascii=False)}"
                        ),
                    },
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _json_or_none(raw)
            if not data:
                return raw  # fallback (still return something)
            return _render_style_locked_analysis(data, teach_mode)

        except Exception:
            return ""

    async def _chart_image_analysis(
        self,
        user_text: str,
        image_url: str,
        quotes: dict[str, dict[str, Any]],
        timeframe: str,
        style: str,
        language: str,
        teach_mode: bool,
    ) -> str:
        if not self.client:
            return ""

        lang = "Spanish" if language == "es" else "English"
        schema_hint = (
            "{"
            '"tldr":"...",'
            '"trend":"...",'
            '"support_resistance":"...",'
            '"patterns":"...",'
            '"indicator_read":"...",'
            '"scenarios":"...",'
            '"risk_note":"...",'
            '"one_smart_next_question":"..."'
            "}"
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.config.openai_vision_model,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional technical analyst.\n"
                            f"{HOUSE_STYLE}\n"
                            "Use the chart image + provided quote context. "
                            "Do not provide financial advice. "
                            "Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Respond in {lang}. Return valid JSON only with this schema:\n"
                                    f"{schema_hint}\n\n"
                                    "Rules:\n"
                                    "- Keep concise.\n"
                                    "- Include uncertainty and invalidation in 'scenarios'.\n"
                                    "- If teach_mode is false, still return all fields.\n\n"
                                    f"teach_mode={teach_mode}\n"
                                    f"Timeframe: {timeframe}\n"
                                    f"Trading style: {style}\n"
                                    f"User request: {user_text}\n"
                                    f"Quote data: {json.dumps(quotes, ensure_ascii=False)}"
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _json_or_none(raw)
            if not data:
                return raw
            return _render_style_locked_chart_analysis(data, teach_mode)

        except Exception:
            return ""

    def _fallback_from_quotes(self, quotes: dict[str, dict[str, Any]], language: str) -> str:
        if language == "es":
            return (
                "## TL;DR\n"
                "No pude generar comentario avanzado del modelo en este momento.\n\n"
                "## What I’m seeing\n"
                f"Resumen de datos recibidos para símbolos: {', '.join(quotes.keys()) or 'N/A'}.\n\n"
                "## Levels that matter\n"
                "N/A\n\n"
                "## Scenarios\n"
                "Escenario base: continuidad de la tendencia actual si no hay ruptura de niveles clave. "
                "Invalidación: ruptura clara en dirección opuesta con volumen.\n\n"
                "## One smart next question\n"
                "¿Qué nivel exacto invalidaría tu hipótesis actual?"
            )

        return (
            "## TL;DR\n"
            "I couldn’t generate advanced model commentary right now.\n\n"
            "## What I’m seeing\n"
            f"Received quote snapshot for symbols: {', '.join(quotes.keys()) or 'N/A'}.\n\n"
            "## Levels that matter\n"
            "N/A\n\n"
            "## Scenarios\n"
            "Base case: trend continuation unless key levels break. "
            "Invalidation: decisive break in the opposite direction with confirmation.\n\n"
            "## One smart next question\n"
            "What exact price level would invalidate your current thesis?"
        )
