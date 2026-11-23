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

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import AnyHttpUrl, SecretStr

from powercontext.builtin.artifacts.experience import ExperienceContent
from powercontext.builtin.inference import GenerationResult, InferenceUsage
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    ApproveCandidateRequest,
    CaptureSource,
    ExperienceQueryRewriteInput,
    ExperienceQueryRewriteOutput,
    LLMExperienceQueryRewriter,
    ProposeExperienceRequest,
)
from powercontext.server.factory import create_server_app
from powercontext.server.gateway_proxy import GatewayChatProxyConfig
from powercontext.server.settings import ServerSettings


class _RoutingRewriter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, payload: ExperienceQueryRewriteInput, /) -> GenerationResult[ExperienceQueryRewriteOutput]:
        self.calls.append(payload.context_text)
        if "GLIBC_2.34" in payload.context_text or "pre-push" in payload.context_text:
            query = "glibc pre-push no-verify"
        else:
            query = "openapi contract api-generate"
        return GenerationResult(
            output=ExperienceQueryRewriteOutput(search_query=query),
            usage=InferenceUsage(requests=1, input_tokens=20, output_tokens=4),
        )


def test_gateway_chat_proxy_distills_query_replays_turn_rechecks_tool_error_and_streams(tmp_path: Path) -> None:
    async def scenario() -> None:
        upstream_requests: list[dict[str, Any]] = []
        upstream_auth_headers: list[str] = []

        def upstream_handler(request: httpx.Request) -> httpx.Response:
            upstream_auth_headers.append(request.headers.get("authorization", ""))
            body = json.loads(request.content.decode("utf-8"))
            upstream_requests.append(body)
            if body.get("stream") is True:
                sse = (
                    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                    ],
                },
            )

        rewriter_backend = _RoutingRewriter()
        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler))
        settings = ServerSettings(
            workspace=tmp_path,
            database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'gateway.db'}"),
            gateway=GatewayChatProxyConfig(
                upstream_base_url=AnyHttpUrl("https://upstream.example/v1"),
                upstream_api_key=SecretStr("sk-upstream-provider"),
                api_key=SecretStr("gw-client-key"),
                default_scope_id="eval-scope",
            ),
        )
        app = create_server_app(
            settings=settings,
            experience_query_rewriter=LLMExperienceQueryRewriter(rewriter_backend),
            gateway_http_client=upstream_client,
        )

        async with (
            upstream_client,
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
        ):
            runtime = app.state.application
            created_scope = await client.post(
                "/v1/scopes",
                json={"title": "Eval", "summary": "Eval", "idempotency_key": "eval-scope"},
            )
            assert created_scope.status_code == 201, created_scope.text
            scope_id = created_scope.json()["scope_id"]
            src = await runtime.sources.for_scope(scope_id).capture(
                CaptureSource(source_id="ev-1", content="evidence", metadata={})
            )
            for situation, action, outcome, lesson in (
                (
                    "openapi/powercontext.yaml contract was modified.",
                    "Run make api-generate.",
                    "Generated models matched the contract.",
                    "Run make api-generate whenever openapi/powercontext.yaml changes.",
                ),
                (
                    "Git push failed because the local pre-push binary required GLIBC_2.34 on an older Linux host.",
                    "Verify the diff contains no secrets and push with --no-verify when authorized.",
                    "Branch push succeeded safely.",
                    "When a host lacks GLIBC_2.34 for the pre-push binary, scan the diff first before bypassing.",
                ),
            ):
                cand = await runtime.experience.for_scope(scope_id).propose(
                    ProposeExperienceRequest(
                        proposal=ExperienceContent(
                            situation=situation,
                            action=action,
                            outcome=outcome,
                            lesson=lesson,
                        ),
                        sources=(src.source_ref,),
                    )
                )
                await runtime.review.for_scope(scope_id).approve(
                    ApproveCandidateRequest(candidate_id=cand.candidate_id, expected_version=cand.version)
                )

            # Unauthorized gateway call is rejected with 401.
            unauth = await client.post("/gateway/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
            assert unauth.status_code == 401

            headers = {
                "Authorization": "Bearer gw-client-key",
                "X-PowerContext-Scope-ID": scope_id,
                "X-PowerContext-Conversation-ID": "conv-1",
            }
            user_prompt = (
                "We just updated an endpoint inside openapi/powercontext.yaml across our Python backend repository "
                "and several schema verification tests are now failing in CI; please check what step we missed."
            )
            base_messages: list[dict[str, Any]] = [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": user_prompt},
            ]

            # 1. Turn start: rewrites verbose prompt -> injects Experience into user message.
            resp1 = await client.post(
                "/gateway/v1/chat/completions",
                headers=headers,
                json={"model": "deepseek-chat", "messages": base_messages},
            )
            assert resp1.status_code == 200
            assert resp1.headers["x-powercontext-experience-injected"] == "1"
            assert upstream_auth_headers[-1] == "Bearer sk-upstream-provider"
            assert upstream_requests[-1]["messages"][0] == {"role": "system", "content": "You are a coding agent."}
            injected_user = upstream_requests[-1]["messages"][1]["content"]
            assert "make api-generate" in injected_user
            assert injected_user.endswith(user_prompt)
            assert len(rewriter_backend.calls) == 1

            # 2. Normal tool output continuation: replays exact user prefix with 0 new rewrite calls.
            normal_tool_messages = [
                *base_messages,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "make api-generate completed with exit_code: 0"},
            ]
            resp2 = await client.post(
                "/gateway/v1/chat/completions",
                headers=headers,
                json={"model": "deepseek-chat", "messages": normal_tool_messages},
            )
            assert resp2.status_code == 200
            assert resp2.headers["x-powercontext-experience-injected"] == "1"
            assert upstream_requests[-1]["messages"][1]["content"] == injected_user
            assert upstream_requests[-1]["messages"][3]["content"] == "make api-generate completed with exit_code: 0"
            assert len(rewriter_backend.calls) == 1

            # 3. Failing tool output continuation: triggers targeted failure recall and appends to trailing tool output.
            failing_tool_messages = [
                *normal_tool_messages,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c2",
                    "content": "git-pre-push-hook: version `GLIBC_2.34' not found; exit_code: 1",
                },
            ]
            resp3 = await client.post(
                "/gateway/v1/chat/completions",
                headers=headers,
                json={"model": "deepseek-chat", "messages": failing_tool_messages},
            )
            assert resp3.status_code == 200
            assert resp3.headers["x-powercontext-experience-injected"] == "2"
            assert upstream_requests[-1]["messages"][1]["content"] == injected_user
            assert "GLIBC_2.34" in upstream_requests[-1]["messages"][5]["content"]
            assert "scan the diff first" in upstream_requests[-1]["messages"][5]["content"]
            assert len(rewriter_backend.calls) == 2

            # 4. Streaming SSE passthrough works end-to-end.
            stream_resp = await client.post(
                "/gateway/v1/chat/completions",
                headers=headers,
                json={"model": "deepseek-chat", "messages": base_messages, "stream": True},
            )
            assert stream_resp.status_code == 200
            assert stream_resp.headers["content-type"].startswith("text/event-stream")
            assert b"data: [DONE]" in stream_resp.content

    asyncio.run(scenario())
