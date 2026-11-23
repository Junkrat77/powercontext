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

from powercontext.builtin.artifacts.experience import ExperienceContent
from powercontext.builtin.inference import GenerationResult, InferenceUsage
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    ApproveCandidateRequest,
    BuiltinConfig,
    CaptureSource,
    ContextAssembly,
    ContextAssemblySection,
    ExperienceQueryRewriteInput,
    ExperienceQueryRewriteOutput,
    LLMExperienceQueryRewriter,
    PrepareContextRequest,
    PreparedContext,
    ProposeExperienceRequest,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft


class _StubRewriteGenerator:
    def __init__(self, rewritten: str = "openapi api-generate contract", *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._rewritten = rewritten
        self._fail = fail

    async def generate(self, payload: ExperienceQueryRewriteInput, /) -> GenerationResult[ExperienceQueryRewriteOutput]:
        self.calls.append(payload.context_text)
        if self._fail:
            raise RuntimeError
        return GenerationResult(
            output=ExperienceQueryRewriteOutput(search_query=self._rewritten),
            usage=InferenceUsage(requests=1, input_tokens=20, output_tokens=5),
        )


def test_query_rewriter_recalls_experience_from_verbose_prompt_and_skips_short_queries() -> None:
    async def scenario() -> None:
        verbose_prompt = (
            "Hello assistant, I am currently working on updating a REST endpoint in our backend repository "
            "and after editing openapi/powercontext.yaml the CI pipeline started complaining about stale "
            "client bindings across several packages. Could you inspect the workspace and figure out the "
            "proper regeneration workflow before we open the pull request?"
        )
        generator = _StubRewriteGenerator("openapi contract api-generate")
        rewriter = LLMExperienceQueryRewriter(generator)

        # Without query rewriting, the 35+ word prompt fails the 60% lexical coverage admission floor.
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig()), dispatch_processing=False) as baseline:
            assert baseline.scopes is not None
            scope = (
                await baseline.scopes.create(
                    ScopeDraft(title="Baseline", summary="Baseline", idempotency_key="baseline")
                )
            ).scope_id
            src = await baseline.sources.for_scope(scope).capture(
                CaptureSource(source_id="src-1", content="openapi changed", metadata={})
            )
            cand = await baseline.experience.for_scope(scope).propose(
                ProposeExperienceRequest(
                    proposal=ExperienceContent(
                        situation="openapi/powercontext.yaml contract was edited.",
                        action="Ran make api-generate to refresh generated Python schemas.",
                        outcome="Contract checks passed.",
                        lesson="Always run make api-generate after editing openapi/powercontext.yaml.",
                    ),
                    sources=(src.source_ref,),
                )
            )
            await baseline.review.for_scope(scope).approve(
                ApproveCandidateRequest(candidate_id=cand.candidate_id, expected_version=cand.version)
            )
            missed = await baseline.context.for_scope(scope).prepare(
                PrepareContextRequest(
                    query=verbose_prompt,
                    assembly=ContextAssembly(sections=(ContextAssemblySection(family="experience", limit=2),)),
                )
            )
            assert isinstance(missed, PreparedContext)
            assert missed.status == "empty"

        # With PC-side LLM query distillation, the same verbose prompt hits the Experience.
        async with open_builtin_runtime(
            BuiltinConfig(database=SQLiteConfig()),
            experience_query_rewriter=rewriter,
            dispatch_processing=False,
        ) as runtime:
            assert runtime.scopes is not None
            scope = (
                await runtime.scopes.create(
                    ScopeDraft(title="Rewritten", summary="Rewritten", idempotency_key="rewritten")
                )
            ).scope_id
            src = await runtime.sources.for_scope(scope).capture(
                CaptureSource(source_id="src-1", content="openapi changed", metadata={})
            )
            cand = await runtime.experience.for_scope(scope).propose(
                ProposeExperienceRequest(
                    proposal=ExperienceContent(
                        situation="openapi/powercontext.yaml contract was edited.",
                        action="Ran make api-generate to refresh generated Python schemas.",
                        outcome="Contract checks passed.",
                        lesson="Always run make api-generate after editing openapi/powercontext.yaml.",
                    ),
                    sources=(src.source_ref,),
                )
            )
            await runtime.review.for_scope(scope).approve(
                ApproveCandidateRequest(candidate_id=cand.candidate_id, expected_version=cand.version)
            )

            recalled = await runtime.context.for_scope(scope).prepare(
                PrepareContextRequest(
                    query=verbose_prompt,
                    assembly=ContextAssembly(sections=(ContextAssemblySection(family="experience", limit=2),)),
                )
            )
            assert isinstance(recalled, PreparedContext)
            assert recalled.status == "ready"
            assert recalled.content is not None and "make api-generate" in recalled.content
            assert len(generator.calls) == 1

            # Concise queries (< 4 distinct lexical terms) skip the LLM rewriter call.
            short = await runtime.context.for_scope(scope).prepare(
                PrepareContextRequest(
                    query="openapi api-generate",
                    assembly=ContextAssembly(sections=(ContextAssemblySection(family="experience", limit=2),)),
                )
            )
            assert isinstance(short, PreparedContext) and short.status == "ready"
            assert len(generator.calls) == 1

    asyncio.run(scenario())


def test_query_rewriter_fails_open_when_model_errors() -> None:
    async def scenario() -> None:
        failing_rewriter = LLMExperienceQueryRewriter(_StubRewriteGenerator(fail=True))
        result = await failing_rewriter.rewrite("openapi contract api-generate schema validation failure during build")
        assert result == "openapi contract api-generate schema validation failure during build"

    asyncio.run(scenario())
