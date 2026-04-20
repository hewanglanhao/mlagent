import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


def _clean_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _clean_int_env(name: str, default: int) -> int:
    raw = _clean_env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _clean_float_env(name: str, default: float) -> float:
    raw = _clean_env(name, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


DEFAULT_MODEL = _clean_env("BASE_MODEL", "gpt-5.4")
DEFAULT_REASONING_EFFORT = _clean_env("OPENAI_REASONING_EFFORT", "medium")
DEFAULT_PRIMARY_API = _clean_env("OPENAI_PRIMARY_API", "chat.completions").lower()
DEFAULT_REQUEST_TIMEOUT_S = _clean_float_env("OPENAI_REQUEST_TIMEOUT_S", 200.0)
DEFAULT_REQUEST_RETRY_ATTEMPTS = _clean_int_env("OPENAI_REQUEST_RETRY_ATTEMPTS", 2)
DEFAULT_REQUEST_RETRY_BACKOFF_S = _clean_float_env("OPENAI_REQUEST_RETRY_BACKOFF_S", 2.0)
DEFAULT_SDK_MAX_RETRIES = _clean_int_env("OPENAI_SDK_MAX_RETRIES", 0)

client = OpenAI(
    api_key=_clean_env("API_KEY", ""),
    base_url=_clean_env("BASE_URL", ""),
    timeout=DEFAULT_REQUEST_TIMEOUT_S,
    max_retries=DEFAULT_SDK_MAX_RETRIES,
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
        timeout_s = max(1.0, _clean_float_env("OPENAI_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S))
        retry_attempts = max(1, _clean_int_env("OPENAI_REQUEST_RETRY_ATTEMPTS", DEFAULT_REQUEST_RETRY_ATTEMPTS))
        retry_backoff_s = max(
            0.0,
            _clean_float_env("OPENAI_REQUEST_RETRY_BACKOFF_S", DEFAULT_REQUEST_RETRY_BACKOFF_S),
        )
        error_messages: list[str] = []
        last_error: Exception | None = None

        for api_mode in self._api_order():
            for attempt_index in range(1, retry_attempts + 1):
                try:
                    if api_mode == "chat.completions":
                        response = self._client.chat.completions.create(
                            model=selected_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            timeout=timeout_s,
                        )
                        text = self._extract_chat_text(response)
                        if text:
                            if self._looks_like_html(text):
                                error_messages.append(self._html_response_hint())
                            else:
                                return LLMResponse(text=text, api_mode="chat.completions")
                        error_messages.append(
                            "Chat Completions API returned an empty payload "
                            f"on attempt {attempt_index}/{retry_attempts} of type {type(response).__name__}."
                        )
                    else:
                        response = self._client.responses.create(
                            model=selected_model,
                            instructions=system_prompt,
                            input=user_prompt,
                            max_output_tokens=max_output_tokens,
                            reasoning={"effort": selected_effort},
                            timeout=timeout_s,
                        )
                        text = self._extract_responses_text(response)
                        if text:
                            if self._looks_like_html(text):
                                error_messages.append(self._html_response_hint())
                            else:
                                return LLMResponse(text=text, api_mode="responses")
                        error_messages.append(
                            "Responses API returned an empty payload "
                            f"on attempt {attempt_index}/{retry_attempts} of type {type(response).__name__}."
                        )
                except Exception as exc:
                    last_error = exc
                    prefix = "Chat Completions API" if api_mode == "chat.completions" else "Responses API"
                    error_messages.append(
                        f"{prefix} attempt {attempt_index}/{retry_attempts} failed: {exc}"
                    )
                    if attempt_index < retry_attempts and self._should_retry_exception(exc):
                        time.sleep(retry_backoff_s * attempt_index)
                        continue
                break

        raise RuntimeError(
            "Both Chat Completions API and Responses API failed while calling the model. "
            + " ".join(error_messages)
        ) from last_error

    @staticmethod
    def _api_order() -> list[str]:
        primary = _clean_env("OPENAI_PRIMARY_API", DEFAULT_PRIMARY_API).lower()
        if primary == "responses":
            return ["responses", "chat.completions"]
        return ["chat.completions", "responses"]

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

    @staticmethod
    def _should_retry_exception(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        retry_markers = (
            "timeout",
            "timed out",
            "connection",
            "readerror",
            "reset by peer",
            "temporarily unavailable",
            "server disconnected",
            "502",
            "503",
            "504",
            "rate limit",
        )
        return any(marker in text for marker in retry_markers)


gpt_client = GPTClient()
