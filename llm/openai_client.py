import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = os.getenv("BASE_MODEL", "gpt-5.4")
DEFAULT_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")

client = OpenAI(
    api_key=os.getenv("API_KEY", ""),
    base_url=os.getenv("BASE_URL", ""),
    max_retries=4,
)


@dataclass
class LLMResponse:
    text: str
    api_mode: str


class GPTClient:
    def __init__(self, sdk_client: OpenAI | None = None):
        self._client = sdk_client or client

    @property
    def enabled(self) -> bool:
        return bool(os.getenv("API_KEY", "") and os.getenv("BASE_MODEL", ""))

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_output_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        if not self.enabled:
            raise RuntimeError(
                "LLM environment is not configured. Please provide API_KEY and BASE_MODEL."
            )

        selected_model = model or DEFAULT_MODEL
        selected_effort = reasoning_effort or DEFAULT_REASONING_EFFORT

        try:
            response = self._client.responses.create(
                model=selected_model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": selected_effort},
            )
            text = self._extract_responses_text(response)
            if text:
                return LLMResponse(text=text, api_mode="responses")
        except Exception as exc:
            last_error = exc
        else:
            last_error = RuntimeError("Responses API returned empty text.")

        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return LLMResponse(text=text, api_mode="chat.completions")
        except Exception as exc:
            raise RuntimeError(
                "Both Responses API and Chat Completions API failed while calling the model."
            ) from exc

        raise RuntimeError(
            "The OpenAI-compatible endpoint returned an empty response."
        ) from last_error

    @staticmethod
    def _extract_responses_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", "") != "message":
                continue
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
                    continue
                value = getattr(content, "value", None)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
        return "\n".join(chunks).strip()


gpt_client = GPTClient()
