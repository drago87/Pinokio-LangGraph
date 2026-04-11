#!/usr/bin/env python3
"""
llm_client.py — Async HTTP clients for RP and Instruct LLMs.

Communicates with local LLM endpoints (Ollama, Koboldcpp, or any
OpenAI-compatible API) using httpx for async streaming.

Supports:
  - /v1/chat/completions (streaming and non-streaming)
  - /v1/models (model listing)
  - Automatic URL normalization
  - Configurable timeouts and retry logic
"""

import httpx
import json
import asyncio
import logging
from typing import AsyncGenerator, Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Default timeout: 5 minutes for generation (models can be slow)
DEFAULT_TIMEOUT = 300.0
CONNECT_TIMEOUT = 15.0


class LLMClient:
    """Async HTTP client for an OpenAI-compatible LLM endpoint."""

    def __init__(
        self,
        base_url: str,
        template: str = "raw",
        model: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.template = template
        self.model = model
        self.timeout = httpx.Timeout(timeout, connect=CONNECT_TIMEOUT)

    async def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 2048,
        stream: bool = True,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build the request payload for /v1/chat/completions."""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if self.model:
            payload["model"] = self.model
        if stop:
            payload["stop"] = stop
        return payload

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completions from the LLM.

        Yields content deltas as they arrive via SSE.

        Args:
            messages: Formatted message array.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop: Optional stop sequences.

        Yields:
            String content chunks from the LLM.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = await self._build_payload(
            messages, temperature, max_tokens, stream=True, stop=stop
        )

        logger.info(f"Streaming from {url} (model={self.model or 'default'})")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        error_msg = body.decode("utf-8", errors="replace")
                        logger.error(
                            f"LLM returned {response.status_code}: {error_msg}"
                        )
                        raise httpx.HTTPStatusError(
                            f"LLM error {response.status_code}: {error_msg[:500]}",
                            request=response.request,
                            response=response,
                        )

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to LLM at {url}: {e}")
            raise ConnectionError(
                f"Cannot connect to LLM at {url}. "
                f"Is it running? Error: {e}"
            ) from e

    async def chat_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Non-streaming chat completion.

        Used for the Instruct LLM (state extraction) which needs the
        full response at once for JSON parsing.

        Returns:
            The assistant's response content as a string.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = await self._build_payload(
            messages, temperature, max_tokens, stream=False, stop=stop
        )

        logger.info(
            f"Completing from {url} (model={self.model or 'default'}, "
            f"temp={temperature})"
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")

                logger.warning(f"LLM returned no choices: {data}")
                return ""

        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to LLM at {url}: {e}")
            raise ConnectionError(
                f"Cannot connect to LLM at {url}. "
                f"Is it running? Error: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            body = e.response.read().decode("utf-8", errors="replace")
            logger.error(f"LLM HTTP error: {body}")
            raise

    async def list_models(self) -> List[str]:
        """List available models from the endpoint.

        Returns a list of model ID strings. Returns empty list
        if the endpoint doesn't support /v1/models.
        """
        url = f"{self.base_url}/v1/models"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0)
            ) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return [m.get("id", "") for m in data.get("data", [])]
        except Exception as e:
            logger.debug(f"Could not list models from {url}: {e}")
        return []

    async def health_check(self) -> bool:
        """Check if the LLM endpoint is reachable."""
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(5.0, connect=3.0)
            ) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                return response.status_code in (200, 401, 403)
        except Exception:
            return False


class LLMClientManager:
    """Manages RP and Instruct LLM clients, rebuilding when config changes."""

    def __init__(self):
        self._rp_client: Optional[LLMClient] = None
        self._instruct_client: Optional[LLMClient] = None
        self._config_snapshot: Optional[dict] = None

    def get_rp_client(
        self,
        rp_url: str,
        rp_template: str,
        rp_model: str = "",
    ) -> LLMClient:
        """Get or create the RP LLM client.

        Rebuilds the client if URL, template, or model changed.
        """
        snapshot = {"url": rp_url, "template": rp_template, "model": rp_model}
        if self._config_snapshot and self._config_snapshot.get("rp") != snapshot:
            self._rp_client = None  # Force rebuild

        if self._rp_client is None:
            self._rp_client = LLMClient(
                base_url=rp_url,
                template=rp_template,
                model=rp_model,
            )
            logger.info(f"Created RP LLM client: {rp_url} (template={rp_template})")

        return self._rp_client

    def get_instruct_client(
        self,
        instruct_url: str,
        instruct_template: str,
        instruct_model: str = "",
    ) -> LLMClient:
        """Get or create the Instruct LLM client."""
        snapshot = {
            "url": instruct_url,
            "template": instruct_template,
            "model": instruct_model,
        }
        if self._config_snapshot and self._config_snapshot.get("instruct") != snapshot:
            self._instruct_client = None

        if self._instruct_client is None:
            self._instruct_client = LLMClient(
                base_url=instruct_url,
                template=instruct_template,
                model=instruct_model,
                timeout=120.0,  # Instruct LLM can be faster
            )
            logger.info(
                f"Created Instruct LLM client: {instruct_url} "
                f"(template={instruct_template})"
            )

        return self._instruct_client

    def update_config(self, config_snapshot: dict):
        """Store the current config snapshot for change detection."""
        self._config_snapshot = config_snapshot