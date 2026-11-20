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
from pathlib import Path

import pytest

from powercontext.builtin.artifacts.experience import ExperienceCandidateInput, ExperienceContent
from powercontext.builtin.inference import InferenceUnavailableError
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.records import ArtifactWrite
from powercontext.builtin.runtime import (
    ApproveCandidateRequest,
    BuiltinConfig,
    BuiltinConfigurationError,
    BuiltinRuntime,
    CaptureSource,
    ExperienceIncubationResult,
    GetExperienceRequest,
    ListCandidatesRequest,
    RuntimeCapabilities,
    RuntimeConfig,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource
from powercontext.sources import Source, SourceRef


class _TaskOutcomePipeline:
    def __init__(self, *, fail_once: bool = False) -> None:
        self._fail_once = fail_once
        self.calls: list[tuple[Source, ...]] = []

    async def incubate(self, sources: tuple[Source, ...], /) -> tuple[ExperienceCandidateInput, ...]:
        self.calls.append(sources)
        if self._fail_once:
            self._fail_once = False
            raise InferenceUnavailableError("experience-incubate")
        task_outcomes = tuple(
            source
            for source in sources
            if isinstance(source, ContentSource) and source.metadata.get("kind") == "task-outcome"
        )
        return tuple(
            ExperienceCandidateInput(
                proposal=ExperienceContent(
                    situation="A strict configuration fixture failed.",
                    action="Set the configuration mode to strict.",
                    outcome=source.content,
                    lesson="Run the strict fixture after configuration changes.",
                ),
                sources=(SourceRef(source_type="content", source_id=source.name),),
            )
            for source in task_outcomes
        )


class _PausedTaskOutcomePipeline(_TaskOutcomePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.resume = asyncio.Event()

    async def incubate(self, sources: tuple[Source, ...], /) -> tuple[ExperienceCandidateInput, ...]:
        self.started.set()
        await self.resume.wait()
        return await super().incubate(sources)


async def _create_scope(runtime: BuiltinRuntime, idempotency_key: str) -> str:
    assert runtime.scopes is not None
    scope = await runtime.scopes.create(
        ScopeDraft(title="Experience Test", summary="Experience incubation test", idempotency_key=idempotency_key)
    )
    return scope.scope_id


def test_capture_during_incubation_preserves_the_selected_window(tmp_path: Path) -> None:
    async def scenario() -> None:
        pipeline = _PausedTaskOutcomePipeline()
        commit_started = asyncio.Event()
        resume_commit = asyncio.Event()

        async def pause_commit(_connection, _candidates) -> None:
            commit_started.set()
            await resume_commit.wait()

        database = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'experience.db'}", busy_timeout_ms=50)
        async with open_builtin_runtime(
            BuiltinConfig(database=database),
            experience_pipeline=pipeline,
            dispatch_processing=False,
        ) as runtime:
            scope = await _create_scope(runtime, "concurrent-capture")
            sources = runtime.sources.for_scope(scope)
            experience = runtime.experience.for_scope(scope)
            original = await sources.capture(
                CaptureSource(source_id="original", content="First check passed.", metadata={"kind": "task-outcome"})
            )
            async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
                incubation = tasks.create_task(experience.incubate(on_commit=pause_commit))
                await pipeline.started.wait()
                during_model = await sources.capture(
                    CaptureSource(source_id="model", content="Second check passed.", metadata={"kind": "task-outcome"})
                )
                pipeline.resume.set()

                await commit_started.wait()
                capture = tasks.create_task(
                    sources.capture(
                        CaptureSource(
                            source_id="commit", content="Third check passed.", metadata={"kind": "task-outcome"}
                        )
                    )
                )
                done, _ = await asyncio.wait((capture,), timeout=0.5)
                assert not done
                resume_commit.set()

            first = incubation.result()
            during_commit = capture.result()
            following = await experience.incubate()
            inbox = await runtime.review.for_scope(scope).list(ListCandidatesRequest(family="experience"))

            assert first.previous_cursor == 0
            assert first.current_cursor == first.high_watermark == original.sequence
            assert first.candidate_count == 1
            assert (following.previous_cursor, following.current_cursor) == (original.sequence, during_commit.sequence)
            assert following.candidate_count == 2
            evidence = {candidate.candidate_id: candidate.sources for candidate in inbox.candidates}
            assert evidence[first.candidate_ids[0]] == (original.source_ref,)
            assert {evidence[candidate_id][0].source_id for candidate_id in following.candidate_ids} == {
                during_model.source_ref.source_id,
                during_commit.source_ref.source_id,
            }

    asyncio.run(scenario())


def test_injected_incubator_keeps_same_scope_calls_serialized() -> None:
    async def scenario() -> None:
        position = 0
        first_entered = asyncio.Event()
        resume_first = asyncio.Event()

        async def incubator(_scope: str, limit: int, /) -> ExperienceIncubationResult:
            nonlocal position
            previous = position
            if limit == 1:
                first_entered.set()
                await resume_first.wait()
            position = previous + 1
            return ExperienceIncubationResult(
                previous_cursor=previous,
                high_watermark=position,
                current_cursor=position,
                source_count=1,
                candidate_count=0,
            )

        async with (
            open_builtin_contexts(BuiltinConfig(database=SQLiteConfig())) as contexts,
            BuiltinRuntime(
                provider=contexts,
                scope_application=contexts.scopes,
                capabilities=RuntimeCapabilities(memory_extraction=False, memory_search_modes=()),
                experience_incubator=incubator,
            ) as runtime,
        ):
            experience = runtime.experience.for_scope(await _create_scope(runtime, "custom-incubator"))
            async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
                first = tasks.create_task(experience.incubate(limit=1))
                await first_entered.wait()
                second = tasks.create_task(experience.incubate(limit=2))
                await asyncio.wait((second,), timeout=0.5)
                resume_first.set()
            assert (first.result().previous_cursor, first.result().current_cursor) == (0, 1)
            assert (second.result().previous_cursor, second.result().current_cursor) == (1, 2)

    asyncio.run(scenario())


def test_incubation_uses_an_independent_cursor_and_keeps_candidates_gated() -> None:
    async def scenario() -> None:
        pipeline = _TaskOutcomePipeline()
        async with open_builtin_runtime(
            BuiltinConfig(database=SQLiteConfig()),
            experience_pipeline=pipeline,
        ) as runtime:
            scope = await _create_scope(runtime, "scheduled-experience")
            await runtime.sources.for_scope(scope).capture(
                CaptureSource(
                    source_id="prompt",
                    content="Please fix the configuration.",
                    metadata={"kind": "prompt"},
                )
            )
            outcome = await runtime.sources.for_scope(scope).capture(
                CaptureSource(
                    source_id="task-1",
                    content="python test_config.py passed",
                    metadata={"kind": "task-outcome"},
                )
            )

            ordinary = await runtime.experience.for_scope(scope).incubate(limit=1)
            assert runtime.atomic_memory is not None
            memory_cursor = await runtime.atomic_memory.for_scope(scope).cursor()
            incubated = await runtime.experience.for_scope(scope).incubate(limit=1)
            replay = await runtime.experience.for_scope(scope).incubate(limit=1)
            inbox = await runtime.review.for_scope(scope).list(ListCandidatesRequest(family="experience"))

            assert ordinary.candidate_count == 0
            assert ordinary.current_cursor == 1
            assert memory_cursor.sequence == 0
            assert incubated.previous_cursor == 1
            assert incubated.current_cursor == outcome.sequence
            assert incubated.candidate_count == 1
            assert replay.processed is False
            assert len(inbox.candidates) == 1
            candidate = inbox.candidates[0]
            assert incubated.candidate_ids == (candidate.candidate_id,)
            assert ordinary.candidate_ids == ()
            assert replay.candidate_ids == ()
            assert candidate.sources == (outcome.source_ref,)
            assert candidate.result_artifact is None

            approved = await runtime.review.for_scope(scope).approve(
                ApproveCandidateRequest(
                    candidate_id=candidate.candidate_id,
                    expected_version=candidate.version,
                )
            )
            assert approved.result_artifact is not None
            experience = await runtime.experience.for_scope(scope).get(
                GetExperienceRequest(artifact=approved.result_artifact)
            )
            assert experience.lineage.sources == (outcome.source_ref,)

    asyncio.run(scenario())


def test_incubation_commits_multiple_candidates_from_one_generation_call() -> None:
    async def scenario() -> None:
        pipeline = _TaskOutcomePipeline()
        async with open_builtin_runtime(
            BuiltinConfig(database=SQLiteConfig()),
            experience_pipeline=pipeline,
        ) as runtime:
            scope = await _create_scope(runtime, "multi-output-experience")
            first = await runtime.sources.for_scope(scope).capture(
                CaptureSource(
                    source_id="task-1",
                    content="python test_config.py passed",
                    metadata={"kind": "task-outcome"},
                )
            )
            second = await runtime.sources.for_scope(scope).capture(
                CaptureSource(
                    source_id="task-2",
                    content="python test_upgrade.py passed",
                    metadata={"kind": "task-outcome"},
                )
            )

            incubated = await runtime.experience.for_scope(scope).incubate()
            inbox = await runtime.review.for_scope(scope).list(ListCandidatesRequest(family="experience"))

            assert len(pipeline.calls) == 1
            assert tuple(source.name for source in pipeline.calls[0] if isinstance(source, ContentSource)) == (
                "task-1",
                "task-2",
            )
            assert (incubated.previous_cursor, incubated.current_cursor) == (0, second.sequence)
            assert incubated.source_count == incubated.candidate_count == 2
            assert set(incubated.candidate_ids) == {candidate.candidate_id for candidate in inbox.candidates}
            assert all(candidate.status == "pending" for candidate in inbox.candidates)
            assert {
                (candidate.sources[0].source_type, candidate.sources[0].source_id) for candidate in inbox.candidates
            } == {
                (first.source_ref.source_type, first.source_ref.source_id),
                (second.source_ref.source_type, second.source_ref.source_id),
            }

    asyncio.run(scenario())


def test_incubation_retries_the_same_window_after_generation_failure() -> None:
    async def scenario() -> None:
        pipeline = _TaskOutcomePipeline(fail_once=True)
        async with open_builtin_runtime(
            BuiltinConfig(database=SQLiteConfig()),
            experience_pipeline=pipeline,
        ) as runtime:
            scope = await _create_scope(runtime, "retry-experience")
            await runtime.sources.for_scope(scope).capture(
                CaptureSource(
                    source_id="task-1",
                    content="python test_config.py passed",
                    metadata={"kind": "task-outcome"},
                )
            )

            with pytest.raises(InferenceUnavailableError):
                await runtime.experience.for_scope(scope).incubate()
            retried = await runtime.experience.for_scope(scope).incubate()
            inbox = await runtime.review.for_scope(scope).list(ListCandidatesRequest(family="experience"))

            assert retried.previous_cursor == 0
            assert retried.current_cursor == 1
            assert retried.candidate_count == 1
            assert len(pipeline.calls) == 2
            assert len(inbox.candidates) == 1

    asyncio.run(scenario())


def test_incubation_uses_the_fixed_source_window_budget() -> None:
    async def scenario() -> None:
        pipeline = _TaskOutcomePipeline()
        async with open_builtin_runtime(
            BuiltinConfig(database=SQLiteConfig()),
            experience_pipeline=pipeline,
        ) as runtime:
            scope = await _create_scope(runtime, "bounded-experience")
            for index in range(33):
                await runtime.sources.for_scope(scope).capture(
                    CaptureSource(
                        source_id=f"prompt-{index}",
                        content="An ordinary prompt.",
                        metadata={"kind": "prompt"},
                    )
                )

            first = await runtime.experience.for_scope(scope).incubate()
            second = await runtime.experience.for_scope(scope).incubate()

            assert first.source_count == 32
            assert first.current_cursor == 32
            assert second.source_count == 1
            assert second.current_cursor == 33

    asyncio.run(scenario())


def test_incubation_skips_lineage_only_sources_but_advances_the_full_window() -> None:
    async def scenario() -> None:
        pipeline = _TaskOutcomePipeline()
        async with open_builtin_runtime(
            BuiltinConfig(database=SQLiteConfig()),
            experience_pipeline=pipeline,
        ) as runtime:
            scope = await _create_scope(runtime, "lineage-only")
            created = await runtime.records.for_scope(scope).create_artifact(
                "atomic-memory",
                ArtifactWrite(content={"kind": "working_note", "text": "Do not incubate direct writes"}),
            )
            result = await runtime.experience.for_scope(scope).incubate()

            assert created.revision == 1
            assert result.current_cursor == 1
            assert result.source_count == 0
            assert result.candidate_count == 0
            assert pipeline.calls == []

    asyncio.run(scenario())


def test_scheduled_incubation_requires_a_configured_pipeline(tmp_path) -> None:
    async def scenario() -> None:
        with pytest.raises(BuiltinConfigurationError, match="Experience incubation"):
            async with open_builtin_runtime(
                BuiltinConfig(
                    database=SQLiteConfig(),
                    runtime=RuntimeConfig(experience_schedule_seconds=60),
                ),
            ):
                pass

    asyncio.run(scenario())
