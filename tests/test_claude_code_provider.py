"""The claude_code provider runs Claude on the person's plan through `claude -p`.

Every call here goes to a stand-in for `subprocess.run`, so no Claude Code is
needed and nothing reaches the network.
"""

import json
import logging
import os
from types import SimpleNamespace

import pytest
import typer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents import default_config
from tradingagents.agents.analysts.turn import WRAP_UP, take_turn
from tradingagents.llm_clients import claude_code_client as cc, factory
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.validators import validate_model

# Absolute on every OS, as claude_binary() returns it.
BINARY = os.path.abspath("/usr/local/bin/claude")
PLAN = {"loggedIn": True, "authMethod": "claude.ai", "email": "a@example.com", "subscriptionType": "max"}


def _result(result="", structured=None, **extra):
    event = {
        "type": "result", "subtype": "success", "is_error": False, "result": result,
        "usage": {"input_tokens": 5, "cache_read_input_tokens": 100,
                  "cache_creation_input_tokens": 20, "output_tokens": 40},
        "modelUsage": {"claude-sonnet-5-5": {}}, "total_cost_usd": 0.01,
    }
    if structured is not None:
        event["structured_output"] = structured
        event["result"] = json.dumps(structured)
    event.update(extra)
    return json.dumps(event)


def _error(text, **extra):
    return _result(text, is_error=True, subtype="error_during_execution", **extra)


@pytest.fixture(autouse=True)
def _forget_confirmed_sign_ins():
    cc._CONFIRMED.clear()
    yield
    cc._CONFIRMED.clear()


@pytest.fixture
def claude(monkeypatch):
    """Record each `claude -p` call, with its system prompt, and answer from a queue."""
    calls, answers = [], []

    def run(command, input=None, env=None, cwd=None, encoding=None, **kwargs):
        system = None
        if "--system-prompt-file" in command:
            with open(_arg(command, "--system-prompt-file"), encoding="utf-8") as handle:
                system = handle.read()
        calls.append(SimpleNamespace(command=command, input=input, env=env, cwd=cwd,
                                     encoding=encoding, system=system))
        answer = answers.pop(0)
        if isinstance(answer, SimpleNamespace):
            return answer
        return SimpleNamespace(returncode=0, stdout=answer, stderr="")

    monkeypatch.setattr(cc.subprocess, "run", run)
    monkeypatch.setattr(cc, "claude_binary", lambda configured=None: BINARY)
    monkeypatch.setattr(cc, "sign_in_status", lambda binary=None, timeout=30: dict(PLAN))
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)
    return SimpleNamespace(calls=calls, answers=answers)


def _arg(command, flag):
    return command[command.index(flag) + 1]


def _llm(**config):
    config = {**default_config.DEFAULT_CONFIG, "llm_provider": "claude_code", **config}
    kwargs = factory.build_llm_kwargs(config)
    return factory.create_llm_client("claude_code", config.get("quick_think_llm"), **kwargs).get_llm()


@tool
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Retrieve stock price data for a symbol between two dates."""
    return "rows"


@tool
def get_indicators(symbol: str, indicator: str) -> str:
    """Retrieve a technical indicator for a symbol."""
    return "values"


class Decision(BaseModel):
    rating: str
    reason: str


# -- plain calls --------------------------------------------------------------


@pytest.mark.unit
def test_a_plain_call_runs_claude_with_no_tools_and_the_agents_system_prompt(claude, caplog):
    claude.answers.append(_result("Hold, for now."))
    with caplog.at_level(logging.WARNING):
        llm = _llm(quick_think_llm="sonnet", anthropic_effort="high", temperature=0.2, max_tokens=500)

    reply = llm.invoke([SystemMessage("You are a trader."), HumanMessage("NVDA?")])

    assert reply.content == "Hold, for now."
    assert reply.usage_metadata == {"input_tokens": 125, "output_tokens": 40, "total_tokens": 165}
    call = claude.calls[0]
    assert call.command[:2] == [BINARY, "-p"]
    assert _arg(call.command, "--tools") == ""
    assert {"--safe-mode", "--restricted", "--strict-mcp-config", "--no-session-persistence"} <= set(call.command)
    assert "--system-prompt" not in call.command  # a file, never argv
    assert call.system == f"You are a trader.\n\n{cc._DATE_NOTE}"
    assert _arg(call.command, "--model") == "sonnet"
    assert _arg(call.command, "--effort") == "high"
    assert "--json-schema" not in call.command
    assert call.input == "NVDA?"
    assert call.encoding == "utf-8"
    assert "temperature" in caplog.text and "max_tokens" in caplog.text


@pytest.mark.unit
def test_a_call_with_no_system_prompt_gets_a_neutral_one_and_the_date_note(claude):
    claude.answers.append(_result("ok"))

    _llm().invoke("hi")

    assert claude.calls[0].system == f"{cc._DEFAULT_SYSTEM}\n\n{cc._DATE_NOTE}"


@pytest.mark.unit
def test_a_runs_date_is_stated_as_today_in_every_call(claude):
    """The graph tags a run with its date; nested analyst graphs pass it down to each call."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from tradingagents.graph.propagation import Propagator

    claude.answers.append(_result("ok"))
    llm = _llm()

    class State(TypedDict):
        out: str

    inner = StateGraph(State)
    inner.add_node("agent", lambda state: {"out": llm.invoke("Decide.").content})
    inner.add_edge(START, "agent")
    inner.add_edge("agent", END)
    outer = StateGraph(State)
    outer.add_node("analyst", inner.compile())
    outer.add_edge(START, "analyst")
    outer.add_edge("analyst", END)

    args = Propagator.tag_run_date({"config": {"recursion_limit": 10}}, "2024-05-01")
    assert args["config"] == {"recursion_limit": 10, "metadata": {"trade_date": "2024-05-01"}}
    outer.compile().invoke({"out": ""}, **args)

    system = claude.calls[0].system
    assert "Today's date is 2024-05-01." in system and cc._DATE_NOTE not in system


@pytest.mark.unit
def test_claude_never_gets_api_credentials_or_the_parent_sessions_variables(claude, monkeypatch):
    for key, value in {"ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_BASE_URL": "http://relay",
                       "ANTHROPIC_AUTH_TOKEN": "t", "CLAUDE_CODE_USE_BEDROCK": "1",
                       "CLAUDECODE": "1", "OPENAI_API_KEY": "sk-x", "HOME": "/home/me",
                       "CLAUDE_CODE_CLIENT_CERT": "/certs/me.pem", "SSL_CERT_DIR": "/certs",
                       "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-plan"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(cc._HOST_MANAGED_AUTH, raising=False)
    claude.answers.append(_result("ok"))

    _llm().invoke("hi")

    env = claude.calls[0].env
    assert env["HOME"] == "/home/me"
    assert env["CLAUDE_CODE_CLIENT_CERT"] == "/certs/me.pem" and env["SSL_CERT_DIR"] == "/certs"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-plan"  # the plan's own headless sign-in
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_USE_BEDROCK", "CLAUDECODE", "OPENAI_API_KEY"):
        assert key not in env


@pytest.mark.unit
def test_a_host_sessions_managed_plan_token_stays_out():
    env = cc.claude_environment({"HOME": "/h", "CLAUDE_CODE_OAUTH_TOKEN": "short-lived",
                                 cc._HOST_MANAGED_AUTH: "1"})
    assert env == {"HOME": "/h"}


# -- tools ---------------------------------------------------------------------


@pytest.mark.unit
def test_bound_tools_come_back_as_langchain_tool_calls(claude):
    claude.answers.append(_result(structured={
        "tool_calls": [{"name": "get_stock_data",
                        "arguments": {"symbol": "NVDA", "start_date": "2026-08-01", "end_date": "2026-09-01"}}],
        "content": "",
    }))

    reply = _llm().bind_tools([get_stock_data, get_indicators]).invoke([HumanMessage("Analyse NVDA")])

    assert [c["name"] for c in reply.tool_calls] == ["get_stock_data"]
    assert reply.tool_calls[0]["args"]["symbol"] == "NVDA"
    assert reply.tool_calls[0]["id"].startswith("call_")
    call = claude.calls[0]
    schema = json.loads(_arg(call.command, "--json-schema"))
    names = [option["properties"]["name"]["const"] for option in schema["properties"]["tool_calls"]["items"]["anyOf"]]
    assert names == ["get_stock_data", "get_indicators"]
    assert "minItems" not in schema["properties"]["tool_calls"]
    assert "`get_stock_data`" in call.system and "between two dates" in call.system
    assert "only tool you can call is `StructuredOutput`" in call.system
    assert cc._FINISH_FREE in call.system
    assert call.input == "Analyse NVDA" + cc._TOOL_REMINDER


@pytest.mark.unit
def test_an_answer_with_no_tool_calls_is_the_report(claude):
    claude.answers.append(_result(structured={"tool_calls": [], "content": "## Report\nUptrend."}))

    reply = _llm().bind_tools([get_stock_data]).invoke([HumanMessage("Analyse NVDA")])

    assert reply.tool_calls == []
    assert reply.content == "## Report\nUptrend."


@pytest.mark.unit
def test_tool_calls_and_results_are_written_into_the_next_prompt(claude):
    claude.answers.append(_result(structured={"tool_calls": [], "content": "done"}))
    history = [
        SystemMessage("You are the market analyst."),
        HumanMessage("NVDA"),
        AIMessage(content="", tool_calls=[{"name": "get_stock_data", "id": "call_1",
                                           "args": {"symbol": "NVDA"}}]),
        ToolMessage(content="Close 181.2", tool_call_id="call_1", name="get_stock_data"),
    ]

    _llm().bind_tools([get_stock_data]).invoke(history)

    prompt = claude.calls[0].input
    assert '<tool_call name="get_stock_data" call_id="call_1">{"symbol": "NVDA"}</tool_call>' in prompt
    assert '<tool_result name="get_stock_data" call_id="call_1">\nClose 181.2\n</tool_result>' in prompt
    assert prompt.index("<user>\nNVDA") < prompt.index("<tool_result")
    assert prompt.endswith(cc._TOOL_REMINDER)


@pytest.mark.unit
def test_an_analysts_wrap_up_turn_runs_with_no_tools(claude):
    """take_turn's last turn writes the history out as text and binds no tools."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    claude.answers.append(_result("Final report."))
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Analyst. Tools: {tool_names}."), MessagesPlaceholder(variable_name="messages")])
    history = [
        HumanMessage("NVDA"),
        AIMessage(content="", tool_calls=[{"name": "get_stock_data", "id": "call_1", "args": {}}]),
        ToolMessage(content="rows", tool_call_id="call_1", name="get_stock_data"),
        HumanMessage(WRAP_UP),
    ]

    _, report = take_turn(prompt, _llm(), [get_stock_data], history)

    assert report == "Final report."
    call = claude.calls[0]
    assert "--json-schema" not in call.command
    assert "`StructuredOutput`" not in call.system
    assert "[get_stock_data returned]" in call.input


@pytest.mark.unit
def test_tool_choice_none_offers_no_tools(claude):
    claude.answers.append(_result("plain"))

    reply = _llm().bind_tools([get_stock_data], tool_choice="none").invoke("hi")

    assert reply.content == "plain" and "--json-schema" not in claude.calls[0].command


@pytest.mark.unit
def test_a_required_call_among_several_tools_must_list_one(claude):
    claude.answers.append(_result(structured={
        "tool_calls": [{"name": "get_indicators", "arguments": {"symbol": "NVDA", "indicator": "rsi"}}],
        "content": ""}))

    reply = _llm().bind_tools([get_stock_data, get_indicators], tool_choice="any").invoke("go")

    assert reply.tool_calls[0]["name"] == "get_indicators"
    call = claude.calls[0]
    assert json.loads(_arg(call.command, "--json-schema"))["properties"]["tool_calls"]["minItems"] == 1
    assert cc._FINISH_REQUIRED in call.system and cc._FINISH_FREE not in call.system


@pytest.mark.unit
@pytest.mark.parametrize("choice", ["get_indicators", {"type": "function", "function": {"name": "get_indicators"}}])
def test_a_tool_forced_by_name_is_the_answer_schema(claude, choice):
    claude.answers.append(_result(structured={"symbol": "NVDA", "indicator": "atr"}))

    reply = _llm().bind_tools([get_stock_data, get_indicators], tool_choice=choice).invoke("go")

    assert reply.tool_calls[0]["name"] == "get_indicators"
    assert reply.tool_calls[0]["args"] == {"symbol": "NVDA", "indicator": "atr"}
    assert set(json.loads(_arg(claude.calls[0].command, "--json-schema"))["properties"]) == {"symbol", "indicator"}


# -- structured output ---------------------------------------------------------


@pytest.mark.unit
def test_structured_output_asks_for_the_schema_itself_with_no_call_wrapper(claude):
    claude.answers.append(_result(structured={"rating": "Buy", "reason": "Momentum"}))

    decision = _llm().with_structured_output(Decision).invoke("Decide.")

    assert decision == Decision(rating="Buy", reason="Momentum")
    call = claude.calls[0]
    schema = json.loads(_arg(call.command, "--json-schema"))
    assert set(schema["properties"]) == {"rating", "reason"}
    # Told to fill the fields in directly, never to wrap them in tool_calls.
    assert "tool_calls" not in call.system and "tool_calls" not in call.input
    assert cc._FORCED_NOTE in call.system and cc._DATE_NOTE in call.system


# -- output and errors ---------------------------------------------------------


@pytest.mark.unit
def test_verbose_json_output_is_read_from_its_last_result(claude):
    """With Claude Code's verbose setting on, --output-format json prints every message."""
    events = [{"type": "system", "subtype": "init"}, {"type": "assistant", "message": {"content": []}},
              json.loads(_result(structured={"rating": "Hold", "reason": "Range"}))]
    claude.answers.append(json.dumps(events))

    assert _llm().with_structured_output(Decision).invoke("Decide.").rating == "Hold"
    assert len(claude.calls) == 1


@pytest.mark.unit
def test_a_signed_out_claude_fails_at_once_without_retrying(claude):
    claude.answers.append(_error("Not logged in · Please run /login"))

    with pytest.raises(cc.ClaudeCodeNotSignedIn, match="claude auth login"):
        _llm().invoke("hi")
    assert len(claude.calls) == 1


@pytest.mark.unit
def test_a_reached_usage_limit_is_not_retried(claude):
    claude.answers.append(_error("You've hit your limit · resets 3pm", api_error="usage_limit_reached",
                                 api_error_status=429))

    with pytest.raises(cc.ClaudeCodeUsageLimit):
        _llm().invoke("hi")
    assert len(claude.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "API Error: Server is temporarily limiting requests (not your usage limit) · Rate limited",
    "API Error: Request rejected (429) · this may be a temporary capacity issue.",
])
def test_a_passing_throttle_is_retried_after_a_longer_wait(claude, monkeypatch, text):
    waits = []
    monkeypatch.setattr(cc.time, "sleep", waits.append)
    claude.answers += [_error(text, api_error_status=429, api_error="rate_limit"), _result("ok")]

    assert _llm().invoke("hi").content == "ok"
    assert len(claude.calls) == 2 and waits[0] >= 15


@pytest.mark.unit
def test_a_throttle_that_outlasts_the_retries_says_so(claude):
    claude.answers += [_error("API Error: Server is temporarily limiting requests (not your usage limit)",
                              api_error_status=429)] * 2

    with pytest.raises(cc.ClaudeCodeError, match="stayed busy after 2 attempt") as raised:
        _llm(llm_max_retries=1).invoke("hi")
    assert "retrying" not in str(raised.value)


@pytest.mark.unit
def test_a_passing_failure_is_retried(claude):
    claude.answers.append(SimpleNamespace(returncode=1, stdout="", stderr="socket hang up"))
    claude.answers.append(_result("ok"))

    assert _llm().invoke("hi").content == "ok"
    assert len(claude.calls) == 2


@pytest.mark.unit
def test_the_last_error_is_raised_once_retries_run_out(claude):
    claude.answers += [SimpleNamespace(returncode=1, stdout="", stderr=f"crash {i}") for i in range(3)]

    with pytest.raises(cc.ClaudeCodeError, match="crash 2"):
        _llm(llm_max_retries=2).invoke("hi")
    assert len(claude.calls) == 3


@pytest.mark.unit
def test_a_models_own_words_on_stdout_never_read_as_a_sign_in_failure(claude):
    """Only stderr names a failure when stdout holds no result."""
    claude.answers.append(SimpleNamespace(returncode=1, stdout="the oauth token rate limit was reached", stderr=""))
    claude.answers.append(_result("ok"))

    assert _llm().invoke("hi").content == "ok"


@pytest.mark.unit
def test_a_timeout_is_retried(claude, monkeypatch):
    real = cc.subprocess.run
    state = {"first": True}

    def run(command, **kwargs):
        if state["first"]:
            state["first"] = False
            raise cc.subprocess.TimeoutExpired(command, 1)
        return real(command, **kwargs)

    monkeypatch.setattr(cc.subprocess, "run", run)
    claude.answers.append(_result("ok"))

    assert _llm().invoke("hi").content == "ok"


# -- sign-in -------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("status, refused", [
    ({"loggedIn": False}, "not signed in"),
    ({"loggedIn": True, "authMethod": "api_key", "apiKeySource": "/login managed key"}, "API account"),
    ({"loggedIn": True, "authMethod": "api_key_helper"}, "API account"),
    # A Console profile or an ANTHROPIC_AUTH_TOKEN from settings, with no plan token passed.
    ({"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"}, "not a Claude plan sign-in"),
    ({"loggedIn": True, "authMethod": "third_party", "apiProvider": "bedrock"}, "bedrock"),
    ({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "vertex"}, "vertex"),
    ({"loggedIn": True, "authMethod": "something_new"}, "does not know"),
])
def test_building_the_model_stops_unless_claude_code_is_on_a_plan(monkeypatch, status, refused):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(cc, "claude_binary", lambda configured=None: BINARY)
    monkeypatch.setattr(cc, "sign_in_status", lambda binary=None, timeout=30: status)

    with pytest.raises(cc.ClaudeCodeNotSignedIn, match=refused):
        _llm()


@pytest.mark.unit
@pytest.mark.parametrize("method", ["claude.ai", "oauth_token"])
def test_a_plan_sign_in_is_checked_once(monkeypatch, method):
    """``oauth_token`` is the plan's when its `claude setup-token` token reaches Claude Code."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-plan")
    monkeypatch.delenv(cc._HOST_MANAGED_AUTH, raising=False)
    checks = []
    monkeypatch.setattr(cc, "claude_binary", lambda configured=None: BINARY)
    monkeypatch.setattr(cc, "sign_in_status",
                        lambda binary=None, timeout=30: checks.append(1) or {"loggedIn": True, "authMethod": method})

    _llm(), _llm()

    assert len(checks) == 1


@pytest.mark.unit
def test_a_missing_claude_stops_the_build(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_BIN", "/nonexistent/claude")

    with pytest.raises(cc.ClaudeCodeNotSignedIn, match="/nonexistent/claude"):
        _llm()


@pytest.mark.unit
def test_a_bare_binary_name_is_looked_up_on_path(monkeypatch, tmp_path):
    exe = tmp_path / "claude-beta"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert cc.claude_binary("claude-beta") == str(exe)


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt", reason="HOME and extensionless executables are POSIX")
def test_the_configured_binary_is_made_absolute_and_expands_home(monkeypatch, tmp_path):
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_BIN", "~/claude")

    assert cc.claude_binary() == str(exe)


@pytest.mark.unit
def test_the_sign_in_check_reads_the_same_settings_as_the_calls(monkeypatch):
    seen = []
    monkeypatch.setattr(cc.subprocess, "run", lambda command, **kw: seen.append((command, kw)) or SimpleNamespace(
        returncode=0, stdout=json.dumps(PLAN), stderr=""))

    assert cc.sign_in_status(BINARY)["email"] == "a@example.com"
    command, kwargs = seen[0]
    assert command == [BINARY, "--safe-mode", "--restricted", "auth", "status"]
    assert kwargs["encoding"] == "utf-8"


@pytest.mark.unit
@pytest.mark.parametrize("stderr, message", [
    ("error: unknown option '--safe-mode'", "too old"),
    ("segmentation fault", "gave no sign-in status"),
])
def test_a_claude_code_that_cannot_answer_is_not_called_signed_out(monkeypatch, stderr, message):
    monkeypatch.setattr(cc.subprocess, "run", lambda command, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr=stderr))

    with pytest.raises(cc.ClaudeCodeNotSignedIn, match=message):
        cc.sign_in_status(BINARY)


@pytest.mark.unit
def test_a_signed_out_status_is_read_from_claudes_json(monkeypatch):
    monkeypatch.setattr(cc.subprocess, "run", lambda command, **kw: SimpleNamespace(
        returncode=1, stdout='{"loggedIn": false, "authMethod": "none"}', stderr=""))

    assert cc.sign_in_status(BINARY)["loggedIn"] is False


# -- registry and CLI ------------------------------------------------------------


@pytest.mark.unit
def test_the_provider_needs_no_key_and_takes_any_model():
    assert get_api_key_env("claude_code") is None
    assert validate_model("claude_code", "claude-opus-5-5")
    assert [v for _, v in get_model_options("claude_code", "deep")][0] == "opus"


@pytest.mark.unit
def test_the_cli_stops_a_headless_run_when_claude_is_signed_out(monkeypatch, capsys):
    from cli import prompts

    monkeypatch.setattr(cc, "claude_binary", lambda configured=None: BINARY)
    monkeypatch.setattr(cc, "sign_in_status", lambda binary=None: {"loggedIn": False})
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: False)

    with pytest.raises(typer.Exit):
        prompts.ensure_signed_in("claude_code")
    out = " ".join(capsys.readouterr().out.split())
    assert "claude auth login" in out and "claude setup-token" in out


@pytest.mark.unit
def test_the_cli_refuses_a_claude_code_signed_in_to_an_api_account(monkeypatch, capsys):
    from cli import prompts

    monkeypatch.setattr(cc, "claude_binary", lambda configured=None: BINARY)
    monkeypatch.setattr(cc, "sign_in_status", lambda binary=None: {
        "loggedIn": True, "authMethod": "api_key", "apiKeySource": "/login managed key"})

    with pytest.raises(typer.Exit):
        prompts.ensure_signed_in("claude_code")
    out = " ".join(capsys.readouterr().out.split())
    assert "API account" in out and "Using your Claude plan" not in out


@pytest.mark.unit
def test_the_cli_offers_sign_in_at_a_terminal_and_survives_a_failed_recheck(monkeypatch, capsys):
    from cli import prompts

    statuses = [{"loggedIn": False}]

    def status(binary=None):
        if statuses:
            return statuses.pop(0)
        raise cc.ClaudeCodeNotSignedIn("Could not run `claude auth status`: boom")

    ran = []
    monkeypatch.setattr(cc, "claude_binary", lambda configured=None: BINARY)
    monkeypatch.setattr(cc, "sign_in_status", status)
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(prompts.subprocess, "run", lambda cmd: ran.append(cmd) or SimpleNamespace(returncode=0))

    with pytest.raises(typer.Exit):
        prompts.ensure_signed_in("claude_code")
    assert ran == [[BINARY, "auth", "login", "--claudeai"]]
    assert "boom" in capsys.readouterr().out


@pytest.mark.unit
def test_the_cli_names_the_plan_it_runs_on(monkeypatch, capsys):
    from cli import prompts

    monkeypatch.setattr(cc, "claude_binary", lambda configured=None: BINARY)
    monkeypatch.setattr(cc, "sign_in_status", lambda binary=None: dict(PLAN))

    prompts.ensure_signed_in("claude_code")
    assert "a@example.com (max)" in capsys.readouterr().out


@pytest.mark.unit
def test_a_remembered_claude_plan_is_preselected_in_the_subscription_menu(monkeypatch):
    from cli import prompts

    shown = []

    def select(message, choices, **kwargs):
        shown.append((message, kwargs.get("default")))
        answer = prompts._SUBSCRIPTION if len(shown) == 1 else kwargs.get("default")
        return SimpleNamespace(ask=lambda: answer)

    monkeypatch.setattr(prompts.questionary, "select", select)

    assert prompts.select_llm_provider(default="claude_code") == ("claude_code", None)
    assert shown[0][1] == prompts._SUBSCRIPTION
    assert shown[1][1] == ("claude_code", None)
