# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in OpenAI Chat Completions proxy that recalls and injects Experience context."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import re
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr, field_validator

from powercontext._logging import log_safely
from powercontext.builtin.runtime import BuiltinRuntime
from powercontext.builtin.runtime.models import (
    ContextAssembly,
    ContextAssemblySection,
    PrepareContextRequest,
    PreparedContext,
)

_LOGGER = logging.getLogger(__name__)
_HTTP_FIELD_NAME_PATTERN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_FAILURE_SIGNAL_PATTERN = re.compile(
    r"(?:"
    r"traceback \(most recent call last\)"
    r"|\b(?:error|exception|failed|failure|fatal|panic|timed?\s*out|timeout|denied|refused|unreachable)\b"
    r"|exit[\s_-]*code\s*[:=]\s*[1-9]\d*"
    r"|returncode\s*[:=]\s*[1-9]\d*"
    r")",
    re.IGNORECASE,
)
_REPLAY_TTL_SECONDS = 300.0
_REPLAY_MAX_ENTRIES = 1024


class GatewayChatProxyConfig(BaseModel):
    """Configuration for the built-in OpenAI Chat Completions Experience proxy."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    upstream_base_url: AnyHttpUrl | None = None
    upstream_api_key: SecretStr | None = Field(default=None, repr=False)
    upstream_headers: dict[str, SecretStr] = Field(default_factory=dict, repr=False)
    api_key: SecretStr | None = Field(default=None, repr=False)
    default_scope_id: str | None = None
    max_context_bytes: int = Field(default=4096, ge=512, le=32768)
    experience_limit: int = Field(default=2, ge=1, le=2)
    recheck_on_tool_error: bool = True
    upstream_timeout_seconds: float = Field(default=180.0, gt=0.0, le=600.0)

    @property
    def enabled(self) -> bool:
        return self.upstream_base_url is not None

    @property
    def upstream_chat_completions_url(self) -> str:
        if self.upstream_base_url is None:
            raise RuntimeError("Gateway upstream_base_url is not configured")  # noqa: TRY003
        raw = str(self.upstream_base_url).rstrip("/")
        if raw.endswith("/chat/completions"):
            return raw
        return f"{raw}/chat/completions"

    @field_validator("default_scope_id")
    @classmethod
    def validate_default_scope_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("upstream_headers")
    @classmethod
    def validate_upstream_headers(cls, value: dict[str, SecretStr]) -> dict[str, SecretStr]:
        seen: set[str] = set()
        for name in value:
            lowered = name.casefold()
            if _HTTP_FIELD_NAME_PATTERN.fullmatch(name) is None:
                raise ValueError("gateway upstream header names must be valid HTTP field names")  # noqa: TRY003
            if lowered in seen:
                raise ValueError("gateway upstream header names must be unique ignoring case")  # noqa: TRY003
            seen.add(lowered)
        return value


@dataclass(slots=True)
class _CacheEntry:
    content: str | None
    expires_at: float


class _BoundedReplayCache:
    """Bounded in-memory cache of prepared Experience blocks keyed by conversation anchor."""

    def __init__(self, *, max_entries: int = _REPLAY_MAX_ENTRIES, ttl_seconds: float = _REPLAY_TTL_SECONDS) -> None:
        self._entries: OrderedDict[tuple[str, str, str], _CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds

    def get(self, scope_id: str, conversation_id: str, anchor: str) -> tuple[bool, str | None]:
        now = monotonic()
        key = (scope_id, conversation_id, anchor)
        entry = self._entries.get(key)
        if entry is None:
            return False, None
        if entry.expires_at <= now:
            self._entries.pop(key, None)
            return False, None
        self._entries.move_to_end(key)
        return True, entry.content

    def put(self, scope_id: str, conversation_id: str, anchor: str, content: str | None) -> None:
        now = monotonic()
        expired = [k for k, v in self._entries.items() if v.expires_at <= now]
        for k in expired:
            self._entries.pop(k, None)
        key = (scope_id, conversation_id, anchor)
        self._entries[key] = _CacheEntry(content=content, expires_at=now + self._ttl_seconds)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


class ExperienceChatGateway:
    """Intercept OpenAI Chat Completions requests to recall Experience before forwarding."""

    def __init__(
        self,
        config: GatewayChatProxyConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._cache = _BoundedReplayCache()
        self._assembly = ContextAssembly(
            sections=(ContextAssemblySection(family="experience", limit=config.experience_limit),)
        )

    async def handle_chat_completions(self, request: Request) -> Response:
        if not self._authorize(request):
            return _openai_error(401, "Invalid gateway API key.", error_type="authentication_error")

        runtime: BuiltinRuntime | None = getattr(request.app.state, "application", None)
        if runtime is None:
            return _openai_error(503, "PowerContext runtime is not ready.", error_type="server_error")

        scope_id = request.headers.get("x-powercontext-scope-id", "").strip() or self._config.default_scope_id
        if not scope_id:
            return _openai_error(
                400,
                "Missing Scope: supply X-PowerContext-Scope-ID header or configure gateway.default_scope_id.",
                error_type="invalid_request_error",
            )

        raw_body = await request.body()
        injected_blocks = 0
        outbound_body = raw_body
        stream_requested = False

        try:
            payload = json.loads(raw_body)
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            stream_requested = payload.get("stream") is True
            messages = payload.get("messages")
            if isinstance(messages, list):
                conversation_id = _conversation_id(request.headers)
                updated_messages, injected_blocks = await self._inject_experience(
                    runtime,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    messages=messages,
                )
                if injected_blocks > 0:
                    patched = dict(payload)
                    patched["messages"] = updated_messages
                    outbound_body = json.dumps(patched, ensure_ascii=False).encode("utf-8")

        return await self._forward(
            request=request,
            body=outbound_body,
            stream=stream_requested,
            injected_blocks=injected_blocks,
        )

    def _authorize(self, request: Request) -> bool:
        if self._config.api_key is None:
            return True
        expected = self._config.api_key.get_secret_value()
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.casefold() != "bearer" or not token:
            return False
        return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))

    async def _inject_experience(
        self,
        runtime: BuiltinRuntime,
        *,
        scope_id: str,
        conversation_id: str | None,
        messages: list[Any],
    ) -> tuple[list[Any], int]:
        last_user_idx = _last_user_index(messages)
        if last_user_idx is None:
            return messages, 0
        user_text = _extract_message_text(messages[last_user_idx])
        if not user_text:
            return messages, 0

        cloned = copy.deepcopy(messages)
        injected = 0
        turn_anchor = _messages_digest(messages[: last_user_idx + 1])
        turn_block = await self._prepare_or_replay(
            runtime,
            scope_id=scope_id,
            conversation_id=conversation_id,
            anchor=turn_anchor,
            query=user_text,
        )
        if turn_block and _prepend_to_message(cloned[last_user_idx], turn_block):
            injected += 1

        if self._config.recheck_on_tool_error and last_user_idx < len(messages) - 1:
            failing_idx, failing_text = _latest_failing_tool_message(messages, after_index=last_user_idx)
            if failing_idx is not None and failing_text:
                tool_anchor = _messages_digest(messages[: failing_idx + 1])
                combined_query = f"Task: {user_text[:1500]}\nTool failure:\n{failing_text[:6000]}"
                tool_block = await self._prepare_or_replay(
                    runtime,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    anchor=tool_anchor,
                    query=combined_query,
                )
                if tool_block and tool_block != turn_block and _append_to_message(cloned[failing_idx], tool_block):
                    injected += 1

        return cloned, injected

    async def _prepare_or_replay(
        self,
        runtime: BuiltinRuntime,
        *,
        scope_id: str,
        conversation_id: str | None,
        anchor: str,
        query: str,
    ) -> str | None:
        if conversation_id is not None:
            hit, cached = self._cache.get(scope_id, conversation_id, anchor)
            if hit:
                return cached
        bounded_query = query.strip()[:8192]
        if not bounded_query:
            return None
        try:
            prepared = await runtime.context.for_scope(scope_id).prepare(
                PrepareContextRequest(
                    query=bounded_query,
                    max_bytes=self._config.max_context_bytes,
                    assembly=self._assembly,
                )
            )
        except Exception as error:
            log_safely(
                _LOGGER,
                logging.WARNING,
                "Gateway Experience prepare failed; forwarding without injection",
                extra={
                    "event": "gateway.experience_prepare.failed",
                    "outcome": "failure",
                    "unit": "context",
                    "exception_type": type(error).__name__,
                },
            )
            return None
        content = (
            prepared.content
            if isinstance(prepared, PreparedContext) and prepared.status == "ready" and prepared.content
            else None
        )
        if conversation_id is not None:
            self._cache.put(scope_id, conversation_id, anchor, content)
        return content

    async def _forward(
        self,
        *,
        request: Request,
        body: bytes,
        stream: bool,
        injected_blocks: int,
    ) -> Response:
        headers = self._upstream_headers(request)
        target_url = self._config.upstream_chat_completions_url
        http_client = self._client or getattr(request.app.state, "gateway_http_client", None)
        if http_client is None:
            return _openai_error(503, "Gateway upstream HTTP client is not initialized.", error_type="server_error")
        if stream:
            upstream_req = http_client.build_request("POST", target_url, content=body, headers=headers)
            try:
                upstream_resp = await http_client.send(upstream_req, stream=True)
            except httpx.HTTPError as error:
                return _upstream_transport_error(error)

            response_headers = {
                "x-powercontext-experience-injected": str(injected_blocks),
            }
            if content_type := upstream_resp.headers.get("content-type"):
                response_headers["content-type"] = content_type

            async def stream_chunks() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                finally:
                    await upstream_resp.aclose()

            return StreamingResponse(
                stream_chunks(),
                status_code=upstream_resp.status_code,
                headers=response_headers,
            )

        try:
            upstream_resp = await http_client.post(target_url, content=body, headers=headers)
        except httpx.HTTPError as error:
            return _upstream_transport_error(error)
        response_headers = {"x-powercontext-experience-injected": str(injected_blocks)}
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
            headers=response_headers,
        )

    def _upstream_headers(self, request: Request) -> dict[str, str]:
        headers: dict[str, str] = {
            "content-type": request.headers.get("content-type", "application/json"),
        }
        if accept := request.headers.get("accept"):
            headers["accept"] = accept
        for name, secret in self._config.upstream_headers.items():
            headers[name] = secret.get_secret_value()
        if self._config.upstream_api_key is not None:
            headers["authorization"] = f"Bearer {self._config.upstream_api_key.get_secret_value()}"
        elif self._config.api_key is None and (auth := request.headers.get("authorization")):
            headers["authorization"] = auth
        return headers


@asynccontextmanager
async def open_gateway_http_client(config: GatewayChatProxyConfig) -> AsyncIterator[httpx.AsyncClient]:
    timeout = httpx.Timeout(config.upstream_timeout_seconds, connect=min(15.0, config.upstream_timeout_seconds))
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        yield client


def mount_experience_chat_gateway(app: FastAPI, gateway: ExperienceChatGateway) -> None:
    for path in ("/gateway/v1/chat/completions", "/gateway/chat/completions"):
        app.add_api_route(
            path,
            gateway.handle_chat_completions,
            methods=["POST"],
            include_in_schema=False,
        )


def _conversation_id(headers: Any) -> str | None:
    if explicit := headers.get("x-powercontext-conversation-id", "").strip():
        return f"pc:{explicit}"
    if opencode := headers.get("x-opencode-session-id", "").strip():
        return f"opencode:{opencode}"
    if claude_session := headers.get("x-claude-code-session-id", "").strip():
        agent = headers.get("x-claude-code-agent-id", "").strip()
        return f"claude:{claude_session}:{agent}"
    return None


def _last_user_index(messages: list[Any]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return idx
    return None


def _latest_failing_tool_message(messages: list[Any], *, after_index: int) -> tuple[int | None, str | None]:
    # Inspect only the latest contiguous block of tool results at the tail of the conversation.
    for idx in range(len(messages) - 1, after_index, -1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            break
        if role != "tool":
            continue
        text = _extract_message_text(msg)
        if text and _FAILURE_SIGNAL_PATTERN.search(text):
            return idx, text
    return None, None


def _extract_message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts)
    return ""


def _prepend_to_message(message: dict[str, Any], block: str) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = f"{block}\n\n{content}"
        return True
    if isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") in {"text", "input_text"}
                and isinstance(part.get("text"), str)
            ):
                part["text"] = f"{block}\n\n{part['text']}"
                return True
    return False


def _append_to_message(message: dict[str, Any], block: str) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = f"{content}\n\n{block}"
        return True
    if isinstance(content, list):
        for idx in range(len(content) - 1, -1, -1):
            part = content[idx]
            if (
                isinstance(part, dict)
                and part.get("type") in {"text", "input_text"}
                and isinstance(part.get("text"), str)
            ):
                part["text"] = f"{part['text']}\n\n{block}"
                return True
    return False


def _messages_digest(messages: list[Any]) -> str:
    serialized = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _openai_error(status_code: int, message: str, *, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


def _upstream_transport_error(error: httpx.HTTPError) -> JSONResponse:
    log_safely(
        _LOGGER,
        logging.WARNING,
        "Gateway upstream request failed",
        extra={
            "event": "gateway.upstream.failed",
            "outcome": "failure",
            "unit": "context",
            "exception_type": type(error).__name__,
        },
    )
    status = 504 if isinstance(error, httpx.TimeoutException) else 502
    return _openai_error(status, f"Upstream model transport failed ({type(error).__name__}).", error_type="bad_gateway")


__all__ = [
    "ExperienceChatGateway",
    "GatewayChatProxyConfig",
    "mount_experience_chat_gateway",
    "open_gateway_http_client",
]
