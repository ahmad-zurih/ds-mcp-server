"""
Tests for the multi-agent orchestration layer.

Everything is exercised with a scripted ``FakeLLMClient`` and a fake async tool
runner — no network, no real MCP server, no provider SDKs. This keeps the tests
fast and deterministic while still covering categorisation, message translation,
worker retry behaviour, and the supervisor round loop.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from ds_mcp_server.agents.categories import (
    TOOL_CATEGORIES,
    categorize_tools,
    category_of,
)
from ds_mcp_server.agents.llm import (
    LLMClient,
    _messages_to_anthropic,
    _messages_to_openai,
    _tools_to_openai,
)
from ds_mcp_server.agents.protocol import (
    LLMResponse,
    ToolCall,
    WorkerResult,
    assistant_msg,
    tool_msg,
    user_msg,
)
from ds_mcp_server.agents.runner import AgentConfig, build_team, config_from_env
from ds_mcp_server.agents.supervisor import Supervisor, _extract_json
from ds_mcp_server.agents.worker import Worker, _looks_like_error


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """Returns a pre-scripted list of LLMResponses, one per ``complete`` call."""

    def __init__(self, script: list[LLMResponse], model: str = "fake-model"):
        self.script = list(script)
        self.model = model
        self.calls: list[dict] = []

    def complete(self, messages, *, system="", tools=None, max_tokens=4096):
        self.calls.append(
            {"messages": list(messages), "system": system, "tools": tools}
        )
        if not self.script:
            return LLMResponse(text="(no more scripted responses)")
        return self.script.pop(0)


def make_tool_runner(responses: dict[str, list[str]]):
    """Async tool runner returning queued responses per tool name."""
    state = {k: list(v) for k, v in responses.items()}
    log: list[tuple[str, dict]] = []

    async def runner(name: str, args: dict) -> str:
        log.append((name, args))
        queue = state.get(name)
        if queue:
            return queue.pop(0)
        return f"{name} ok"

    runner.log = log  # type: ignore[attr-defined]
    return runner


def _plan(status, tasks=None, final="", reasoning="r"):
    return LLMResponse(
        text=json.dumps(
            {
                "reasoning": reasoning,
                "status": status,
                "tasks": tasks or [],
                "final_answer": final,
            }
        )
    )


# ---------------------------------------------------------------------------
# categories
# ---------------------------------------------------------------------------


class TestCategories:
    def test_known_tool_names(self):
        assert category_of("plot_static_histogram") == "plot_static"
        assert category_of("plot_interactive_scatterplot") == "plot_interactive"
        assert category_of("run_correlation") == "stats"
        assert category_of("arxiv_search") == "research"
        assert category_of("search_web") == "web"
        assert category_of("run_shell_command") == "system"
        assert category_of("get_column_summary") == "data"

    def test_prefix_fallback_for_unknown_tool(self):
        # A hypothetical future tool not in the table still routes by prefix.
        assert category_of("plot_static_violin") == "plot_static"
        assert category_of("plot_interactive_sunburst") == "plot_interactive"

    def test_unknown_goes_to_misc(self):
        assert category_of("totally_unknown_tool") == "misc"

    def test_categorize_groups_and_shares_data_tools(self):
        tools = [
            {"name": "get_column_summary"},
            {"name": "get_all_columns_summary"},
            {"name": "plot_static_histogram"},
            {"name": "run_correlation"},
            {"name": "search_web"},
        ]
        grouped = categorize_tools(tools, share_data_tools=True)
        assert set(grouped) == {"data", "plot_static", "stats", "web"}
        # data tools shared into plot_static and stats
        static_names = {t["name"] for t in grouped["plot_static"]}
        assert "get_column_summary" in static_names
        assert "plot_static_histogram" in static_names
        stats_names = {t["name"] for t in grouped["stats"]}
        assert "get_all_columns_summary" in stats_names
        # web worker does NOT get data tools
        web_names = {t["name"] for t in grouped["web"]}
        assert "get_column_summary" not in web_names

    def test_categorize_without_sharing(self):
        tools = [
            {"name": "get_column_summary"},
            {"name": "plot_static_histogram"},
        ]
        grouped = categorize_tools(tools, share_data_tools=False)
        static_names = {t["name"] for t in grouped["plot_static"]}
        assert "get_column_summary" not in static_names

    def test_all_registered_tools_have_a_category(self):
        # Every tool in the table maps back to its own category.
        for cat, names in TOOL_CATEGORIES.items():
            for n in names:
                assert category_of(n) == cat


# ---------------------------------------------------------------------------
# message / tool translation
# ---------------------------------------------------------------------------


class TestTranslation:
    def test_openai_tool_spec(self):
        tools = [{"name": "foo", "description": "d", "inputSchema": {"type": "object"}}]
        out = _tools_to_openai(tools)
        assert out[0]["type"] == "function"
        assert out[0]["function"]["name"] == "foo"

    def test_openai_messages_with_tool_calls(self):
        msgs = [
            user_msg("hi"),
            assistant_msg("", [ToolCall(id="c1", name="foo", arguments={"a": 1})]),
            tool_msg("c1", "foo", "result text"),
        ]
        out = _messages_to_openai(msgs)
        assert out[0] == {"role": "user", "content": "hi"}
        assert out[1]["tool_calls"][0]["id"] == "c1"
        assert json.loads(out[1]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
        assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "result text"}

    def test_anthropic_merges_consecutive_tool_results(self):
        msgs = [
            user_msg("hi"),
            assistant_msg(
                "ok",
                [
                    ToolCall(id="c1", name="foo", arguments={}),
                    ToolCall(id="c2", name="bar", arguments={}),
                ],
            ),
            tool_msg("c1", "foo", "r1"),
            tool_msg("c2", "bar", "r2"),
        ]
        out = _messages_to_anthropic(msgs)
        # last message is a single user message holding BOTH tool_result blocks
        assert out[-1]["role"] == "user"
        assert isinstance(out[-1]["content"], list)
        assert len(out[-1]["content"]) == 2
        assert out[-1]["content"][0]["tool_use_id"] == "c1"
        assert out[-1]["content"][1]["tool_use_id"] == "c2"
        # assistant message has a text block + two tool_use blocks
        assistant = out[1]
        types = [b["type"] for b in assistant["content"]]
        assert types == ["text", "tool_use", "tool_use"]


# ---------------------------------------------------------------------------
# error detection helper
# ---------------------------------------------------------------------------


class TestErrorDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Error: something broke",
            "[Error] bad thing",
            "Traceback (most recent call last):",
            "playwright not installed. Run: ...",
        ],
    )
    def test_detects_errors(self, text):
        assert _looks_like_error(text) is True

    @pytest.mark.parametrize(
        "text",
        ["Generated plot: chart.png", "Success! 5 results found", ""],
    )
    def test_ignores_non_errors(self, text):
        assert _looks_like_error(text) is False


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TestWorker:
    def _worker(self, script, runner, **kw):
        return Worker(
            category="stats",
            llm=FakeLLMClient(script),
            tools=[{"name": "run_correlation", "description": "", "inputSchema": {}}],
            tool_runner=runner,
            description="stats",
            **kw,
        )

    def test_successful_single_tool_task(self):
        script = [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="run_correlation", arguments={})],
            ),
            LLMResponse(text="Correlation is 0.9, strong positive."),
        ]
        runner = make_tool_runner({"run_correlation": ["r=0.9"]})
        worker = self._worker(script, runner)
        result = asyncio.run(worker.run("correlate x and y"))
        assert result.success is True
        assert result.tool_calls == ["run_correlation"]
        assert "0.9" in result.summary
        assert result.attempts == 1

    def test_retry_after_tool_error_then_success(self):
        # Attempt 1: tool errors, then LLM gives up (final text) -> failure.
        # Attempt 2: tool ok -> success.
        script = [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="run_correlation", arguments={})],
            ),
            LLMResponse(text="I could not complete it."),  # ends attempt 1 (error round)
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c2", name="run_correlation", arguments={})],
            ),
            LLMResponse(text="Done, correlation computed."),  # ends attempt 2 clean
        ]
        runner = make_tool_runner(
            {"run_correlation": ["Error: bad column", "r=0.5"]}
        )
        worker = self._worker(script, runner, max_attempts=2)
        result = asyncio.run(worker.run("correlate"))
        assert result.success is True
        assert result.attempts == 2

    def test_failure_when_all_attempts_exhausted(self):
        script = [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="run_correlation", arguments={})],
            ),
            LLMResponse(text="failed"),
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c2", name="run_correlation", arguments={})],
            ),
            LLMResponse(text="failed again"),
        ]
        runner = make_tool_runner({"run_correlation": ["Error: x", "Error: y"]})
        worker = self._worker(script, runner, max_attempts=2)
        result = asyncio.run(worker.run("correlate"))
        assert result.success is False
        assert result.attempts == 2

    def test_step_limit_is_enforced(self):
        # LLM keeps calling tools forever; max_steps caps it -> failure.
        loop_call = LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c", name="run_correlation", arguments={})],
        )
        script = [loop_call for _ in range(10)]
        runner = make_tool_runner({"run_correlation": ["ok"] * 10})
        worker = self._worker(script, runner, max_steps=3, max_attempts=1)
        result = asyncio.run(worker.run("loop"))
        assert result.success is False
        assert result.steps == 3
        assert "step" in result.error.lower()

    def test_artifact_extraction(self):
        script = [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="run_correlation", arguments={})],
            ),
            LLMResponse(text="made a plot"),
        ]
        runner = make_tool_runner(
            {"run_correlation": ["/tmp/plots/chart.png|||import plotly"]}
        )
        worker = self._worker(script, runner)
        result = asyncio.run(worker.run("plot it"))
        assert result.success is True
        assert result.artifacts
        assert result.artifacts[0]["path"].endswith("chart.png")
        assert result.artifacts[0]["kind"] == "image"

    def test_tool_runner_exception_is_caught(self):
        async def boom(name, args):
            raise RuntimeError("kaboom")

        script = [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="run_correlation", arguments={})],
            ),
            LLMResponse(text="giving up"),
        ]
        worker = self._worker(script, boom, max_attempts=1)
        result = asyncio.run(worker.run("x"))
        assert result.success is False


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"a": 2}\n```') == {"a": 2}

    def test_embedded_json(self):
        assert _extract_json('sure!\n{"a": 3}\ndone') == {"a": 3}

    def test_garbage_returns_none(self):
        assert _extract_json("no json here") is None


class TestSupervisor:
    def _worker_double(self, category, result):
        class _W:
            def __init__(self):
                self.category = category
                self.description = category
                self.tool_names = [f"{category}_tool"]

            async def run(self, task):
                out = WorkerResult(**{**result, "task": task, "category": category})
                return out

        return _W()

    def test_single_round_delegation_then_done(self):
        # Round 1: delegate one stats task. Round 2: done.
        planner = FakeLLMClient(
            [
                _plan("continue", [{"category": "stats", "task": "correlate"}]),
                _plan("done", [], final="All finished. r=0.9."),
            ]
        )
        worker = self._worker_double(
            "stats",
            {"success": True, "summary": "r=0.9", "error": ""},
        )
        events = []
        sup = Supervisor(
            planner, {"stats": worker}, max_rounds=3, on_event=events.append
        )
        answer = asyncio.run(sup.run("analyse the data"))
        assert "finished" in answer.lower()
        # a plan event and a worker_result event were emitted
        types = [e["type"] for e in events]
        assert "plan" in types
        assert "worker_result" in types

    def test_unknown_category_reported_back(self):
        planner = FakeLLMClient(
            [
                _plan("continue", [{"category": "nonexistent", "task": "do"}]),
                _plan("done", [], final="handled"),
            ]
        )
        worker = self._worker_double(
            "stats", {"success": True, "summary": "ok", "error": ""}
        )
        sup = Supervisor(planner, {"stats": worker}, max_rounds=3)
        answer = asyncio.run(sup.run("x"))
        assert answer == "handled"

    def test_done_on_first_round(self):
        planner = FakeLLMClient([_plan("done", [], final="nothing to do")])
        sup = Supervisor(planner, {}, max_rounds=3)
        answer = asyncio.run(sup.run("hi"))
        assert answer == "nothing to do"

    def test_max_rounds_triggers_synthesis(self):
        # Planner always wants to continue; supervisor must synthesise at the end.
        planner = FakeLLMClient(
            [
                _plan("continue", [{"category": "stats", "task": "t1"}]),
                _plan("continue", [{"category": "stats", "task": "t2"}]),
                # synthesis call (after max_rounds) returns final answer:
                _plan("done", [], final="synthesised summary"),
            ]
        )
        worker = self._worker_double(
            "stats", {"success": True, "summary": "ok", "error": ""}
        )
        sup = Supervisor(planner, {"stats": worker}, max_rounds=2)
        answer = asyncio.run(sup.run("do stuff"))
        assert answer == "synthesised summary"

    def test_unparseable_plan_returns_raw_text(self):
        planner = FakeLLMClient([LLMResponse(text="just a plain answer")])
        sup = Supervisor(planner, {}, max_rounds=3)
        answer = asyncio.run(sup.run("hi"))
        assert answer == "just a plain answer"


# ---------------------------------------------------------------------------
# build_team wiring + config
# ---------------------------------------------------------------------------


class TestBuildTeam:
    def test_builds_one_worker_per_category(self):
        tools = [
            {"name": "get_column_summary", "description": "", "inputSchema": {}},
            {"name": "plot_static_histogram", "description": "", "inputSchema": {}},
            {"name": "run_correlation", "description": "", "inputSchema": {}},
            {"name": "search_web", "description": "", "inputSchema": {}},
        ]

        async def runner(name, args):
            return "ok"

        planner = FakeLLMClient([])
        made_models: list[str] = []

        def worker_factory(category):
            made_models.append(category)
            return FakeLLMClient([], model=f"model-{category}")

        cfg = AgentConfig(max_worker_retries=1, max_worker_steps=4)
        sup = build_team(
            tools,
            runner,
            cfg,
            planner_llm=planner,
            worker_llm_factory=worker_factory,
        )
        assert set(sup.workers) == {"data", "plot_static", "stats", "web"}
        # worker retry/step config propagated
        assert sup.workers["stats"].max_attempts == 2  # retries(1)+1
        assert sup.workers["stats"].max_steps == 4
        # data tools shared into plot_static + stats but not web
        assert "get_column_summary" in sup.workers["plot_static"].tool_names
        assert "get_column_summary" not in sup.workers["web"].tool_names

    def test_config_from_env_precedence(self, monkeypatch):
        monkeypatch.setenv("MODEL", "base-model")
        monkeypatch.delenv("PLANNER_MODEL", raising=False)
        monkeypatch.delenv("WORKER_MODEL", raising=False)
        # CLI args win over env
        cfg = config_from_env(
            provider="openai",
            planner_model="planner-x",
            worker_model=None,
            max_rounds=5,
            max_worker_retries=None,
            max_worker_steps=None,
        )
        assert cfg.planner_model == "planner-x"
        assert cfg.worker_model == "base-model"  # falls back to MODEL
        assert cfg.max_rounds == 5
        assert cfg.max_worker_retries == 2  # default
        assert cfg.max_worker_attempts == 3

    def test_config_from_env_reads_env(self, monkeypatch):
        monkeypatch.setenv("MODEL", "base")
        monkeypatch.setenv("PLANNER_MODEL", "big-model")
        monkeypatch.setenv("WORKER_MODEL", "cheap-model")
        monkeypatch.setenv("MAX_WORKER_RETRIES", "4")
        cfg = config_from_env(
            provider="openai",
            planner_model=None,
            worker_model=None,
            max_rounds=None,
            max_worker_retries=None,
            max_worker_steps=None,
        )
        assert cfg.planner_model == "big-model"
        assert cfg.worker_model == "cheap-model"
        assert cfg.max_worker_retries == 4
