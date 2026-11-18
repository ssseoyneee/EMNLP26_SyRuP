"""Async OpenAI-compatible client shared by the LLM-judge evaluators.

Models with a "/" in the name (e.g. deepseek/deepseek-chat-v3-0324) go through OpenRouter
(OPENROUTER_API_KEY); others through OpenAI (OPENAI_API_KEY).
"""
import json
import os

import openai
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio


class LLM:
    def __init__(self, model: str, requests_per_minute: int = 100):
        if "/" in model:
            key, base_url, var = os.environ.get("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
        else:
            key, base_url, var = os.environ.get("OPENAI_API_KEY"), None, "OPENAI_API_KEY"
        if not key:
            raise EnvironmentError(f"{var} is not set")
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=key, base_url=base_url)
        self.limiter = AsyncLimiter(requests_per_minute, 60)

    async def one(self, messages, json_mode: bool = False, **params):
        """Text (or parsed dict when json_mode); None on any API / JSON error."""
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        async with self.limiter:
            try:
                r = await self.client.chat.completions.create(model=self.model, messages=messages, **params)
                text = (r.choices[0].message.content or "").strip()
                return json.loads(text) if json_mode else text
            except Exception as exc:
                print(f"[llm] {type(exc).__name__}: {str(exc)[:200]}")
                return None

    async def many(self, requests, batch_size: int = 100, desc: str = "", **params):
        out = []
        for i in range(0, len(requests), batch_size):
            out += await tqdm_asyncio.gather(*[self.one(m, **params) for m in requests[i:i + batch_size]],
                                             desc=f"{desc} [{i}/{len(requests)}]")
        return out
