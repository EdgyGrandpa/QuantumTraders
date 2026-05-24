import os from typing import AsyncIterable import fastapi_poe as fp from openai import AsyncOpenAI from dotenv import 
load_dotenv

load_dotenv(dotenv_path=".env", override=True)

class QuantumTradingBot(fp.PoeBot):
    def __init__(self, bot_name: str, access_key: str, openai_api_key: str, openai_model: str):
        super().__init__(bot_name=bot_name, access_key=access_key)
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = openai_model

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(allow_attachments=True)

    async def get_response(self, request: fp.QueryRequest) -> AsyncIterable[fp.PartialResponse]:
        messages = [{"role": m.role, "content": m.content} for m in request.query]
        if not messages:
            yield fp.PartialResponse(text="Please send a message.")
            return

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield fp.PartialResponse(text=delta)

def build_app():
    bot_name = os.getenv("POE_BOT_NAME", "").strip()
    access_key = os.getenv("POE_ACCESS_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

    missing = [k for k, v in {
        "POE_BOT_NAME": bot_name,
        "POE_ACCESS_KEY": access_key,
        "OPENAI_API_KEY": openai_key
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    bot = QuantumTradingBot(
        bot_name=bot_name,
        access_key=access_key,
        openai_api_key=openai_key,
        openai_model=model,
    )
    return fp.make_app(bot, allow_without_key=True)

app = build_app()
