"""Thin OpenAI-compatible wrapper."""

from __future__ import annotations

from openai import OpenAI
from openai import NotFoundError


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
    ) -> str:
        try:
            response = self.client.responses.create(
                model=model,
                temperature=temperature,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    },
                ],
            )
            return getattr(response, "output_text", "") or str(response)
        except NotFoundError:
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content or ""
            return str(response)
