import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


def _clean_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


DEFAULT_MODEL = _clean_env("BASE_MODEL", "gpt-5.4")
DEFAULT_REASONING_EFFORT = _clean_env("OPENAI_REASONING_EFFORT", "medium")

client = OpenAI(
    api_key=_clean_env("API_KEY", ""),
    base_url=_clean_env("BASE_URL", ""),
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
        return bool(_clean_env("API_KEY", "") and _clean_env("BASE_MODEL", ""))

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

        selected_model = (model or _clean_env("BASE_MODEL", DEFAULT_MODEL)).strip()
        selected_effort = (
            reasoning_effort or _clean_env("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        ).strip()
        error_messages: list[str] = []
        last_error: Exception | None = None

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
                if self._looks_like_html(text):
                    error_messages.append(self._html_response_hint())
                else:
                    return LLMResponse(text=text, api_mode="responses")
            error_messages.append(
                f"Responses API returned an empty payload of type {type(response).__name__}."
            )
        except Exception as exc:
            last_error = exc
            error_messages.append(f"Responses API failed: {exc}")

        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = self._extract_chat_text(response)
            if text:
                if self._looks_like_html(text):
                    error_messages.append(self._html_response_hint())
                else:
                    return LLMResponse(text=text, api_mode="chat.completions")
            error_messages.append(
                f"Chat Completions API returned an empty payload of type {type(response).__name__}."
            )
        except Exception as exc:
            error_messages.append(f"Chat Completions API failed: {exc}")
            raise RuntimeError(
                "Both Responses API and Chat Completions API failed while calling the model. "
                + " ".join(error_messages)
            ) from exc

        raise RuntimeError(
            "The OpenAI-compatible endpoint returned an empty response. "
            + " ".join(error_messages)
        ) from last_error

    @staticmethod
    def _extract_responses_text(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, dict):
            output_text = response.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()
            return GPTClient._extract_text_from_output_items(response.get("output", []))

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        output = getattr(response, "output", None)
        if output is not None:
            return GPTClient._extract_text_from_output_items(output)

        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                return GPTClient._extract_responses_text(dumped)

        return ""

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message", {})
                    return GPTClient._coerce_text_content(message.get("content"))
            return GPTClient._extract_responses_text(response)

        choices = getattr(response, "choices", None)
        if choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if message is not None:
                return GPTClient._coerce_text_content(getattr(message, "content", None))

        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                return GPTClient._extract_chat_text(dumped)

        return GPTClient._extract_responses_text(response)

    @staticmethod
    def _extract_text_from_output_items(items: Any) -> str:
        if not isinstance(items, list):
            return ""

        chunks: list[str] = []
        for item in items:
            item_type = ""
            if isinstance(item, dict):
                item_type = str(item.get("type", ""))
                content = item.get("content", [])
            else:
                item_type = str(getattr(item, "type", ""))
                content = getattr(item, "content", [])

            if item_type and item_type != "message":
                continue

            text = GPTClient._coerce_text_content(content)
            if text:
                chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    @staticmethod
    def _coerce_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, dict):
            for key in ("text", "value", "content"):
                value = content.get(key)
                text = GPTClient._coerce_text_content(value)
                if text:
                    return text
            return ""

        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                text = GPTClient._coerce_text_content(item)
                if text:
                    chunks.append(text)
            return "\n".join(chunks).strip()

        text_attr = getattr(content, "text", None)
        if isinstance(text_attr, str) and text_attr.strip():
            return text_attr.strip()

        value_attr = getattr(content, "value", None)
        if isinstance(value_attr, str) and value_attr.strip():
            return value_attr.strip()

        content_attr = getattr(content, "content", None)
        if content_attr is not None and content_attr is not content:
            return GPTClient._coerce_text_content(content_attr)

        if hasattr(content, "model_dump"):
            dumped = content.model_dump()
            if isinstance(dumped, dict):
                return GPTClient._coerce_text_content(dumped)

        return ""

    @staticmethod
    def _looks_like_html(text: str) -> bool:
        stripped = text.lstrip().lower()
        return stripped.startswith("<!doctype html") or stripped.startswith("<html")

    @staticmethod
    def _html_response_hint() -> str:
        return (
            "The endpoint returned an HTML page instead of model text. "
            "BASE_URL likely points to a web UI rather than an OpenAI-compatible API endpoint. "
            "Try using a BASE_URL that ends with `/v1`."
        )


gpt_client = GPTClient()
