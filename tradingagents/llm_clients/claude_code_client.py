"""Claude on the person's own Claude plan, through the Claude Code CLI.

Each model call runs ``claude -p`` (Claude Code's headless mode) signed in with
the person's Claude Pro or Max plan, so the run counts toward that plan's usage
limits instead of billing an API key. Claude Code runs with no tools of its own,
no settings, plugins, hooks or MCP servers, in an empty temporary directory: it
is used only as a model. A Claude Code signed in to an API account rather than
a plan is refused before the first call.

TradingAgents' analysts call Python tools through LangChain, which ``claude -p``
cannot run. Bound tools are therefore offered as a JSON schema: the model lists
the calls it wants in its structured answer, they come back as LangChain tool
calls, and the graph runs them and returns their results in the next message,
exactly as with a native tool-calling API.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Only what Claude Code needs to find itself, its sign-in and the network. API
# credentials, a custom base URL and the Bedrock/Vertex switches are left out,
# so a call can never move silently from the plan onto an API account. USER and
# LOGNAME stay: on macOS Claude Code finds its plan sign-in in the Keychain by
# user name. Variables of a Claude Code session this runs inside stay out too.
_ENV_ALLOWLIST = (
    "HOME", "USER", "LOGNAME", "PATH", "SHELL", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "LC_CTYPE", "CLAUDE_CONFIG_DIR",
    # Network: proxies, CA bundles and the client certificate for mutual TLS.
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
    "CLAUDE_CODE_CLIENT_CERT", "CLAUDE_CODE_CLIENT_KEY", "CLAUDE_CODE_CLIENT_KEY_PASSPHRASE",
    "CLAUDE_CODE_CERT_STORE",
    # Windows
    "SYSTEMROOT", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PATHEXT", "COMSPEC",
)

# The long-lived plan token `claude setup-token` issues, only to Claude plans:
# the sign-in of a server, CI job or container with no browser or Keychain. It
# keeps calls on the plan, so it passes, unless a host Claude Code session
# manages it, whose short-lived copy a child process cannot renew.
_PLAN_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"
_HOST_MANAGED_AUTH = "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH"

# `claude auth status` methods that bill an Anthropic API account. Only
# ``claude.ai`` (a plan's browser sign-in) is a plan on its own: ``oauth_token``
# also covers a Console account profile and an injected ANTHROPIC_AUTH_TOKEN,
# so it counts as the plan only when the plan token reaches Claude Code.
API_AUTH_METHODS = ("api_key", "api_key_helper")

# Settings scope shared by the sign-in check and the model calls, so the check
# sees the sign-in the calls will use.
_SCOPE = ("--safe-mode", "--restricted")

# Claude Code's own coding-agent system prompt is replaced by the agent's; a
# call that brings none still gets a neutral one.
_DEFAULT_SYSTEM = "You are a helpful assistant."

# Claude Code tells the model the machine's real date, and no setting that
# keeps the plan sign-in removes it. The run's own date must win, or a run
# dated in the past could reason from the present.
_DATE_NOTE = (
    "Any environment note that gives today's date gives the machine's clock, not the date of "
    "this task: ignore it. Where these instructions or messages give a date, that date is "
    "today; where they give none, today is the latest date in the material you are given. "
    "Reason from that date, and use nothing that happened after it."
)
# A run tags its model calls with its date (the graph run's ``trade_date``
# metadata), and every call then says so.
_RUN_DATE_NOTE = (
    "Today's date is {date}. Any environment note that gives another date gives the machine's "
    "clock, not the date of this task: ignore it, and use nothing that happened after {date}."
)
RUN_DATE_METADATA = "trade_date"

_TOOL_PROTOCOL = """\
## How to use the tools

The tools named in these instructions are not functions you can call: the \
only tool you can call is `StructuredOutput`, and calling any other directly \
fails with "No such tool available". To use a tool, call `StructuredOutput` and \
list each call you want in its `tool_calls` field, with the tool's `name` and \
its `arguments`. The calls are run for you, and their results come back in the \
next message, so you can call more tools after reading them. While you call \
tools, leave `content` empty. {finish}

The tools:
"""
_FINISH_FREE = (
    "When you need no more data, return an empty `tool_calls` list and put your complete "
    "answer in `content`."
)
_FINISH_REQUIRED = "This answer must list at least one call."

_TOOL_REMINDER = (
    "\n\n(To use a tool, call `StructuredOutput` with the call listed in `tool_calls`; "
    "no other tool can be called directly.)"
)

# A structured answer (``with_structured_output``) is the schema itself, with
# no wrapper around it.
_FORCED_NOTE = (
    "Give your answer by calling `StructuredOutput` once. Its input is the answer itself: "
    "fill in its fields directly."
)

_PLAN_HINT = (
    "Sign in with your plan: `claude auth login --claudeai`, or use the anthropic provider with an API key."
)

_CLI_HINT = (
    "Install Claude Code (https://claude.com/claude-code) and sign in with your plan: "
    "`claude auth login` (or, on a machine with no browser, `claude setup-token` and export "
    "CLAUDE_CODE_OAUTH_TOKEN). Set TRADINGAGENTS_CLAUDE_BIN if `claude` is not on PATH."
)

# Claude Code marks a plan's real usage limit with this ``api_error``; other
# 429s are passing throttles it calls "not your usage limit".
_USAGE_LIMIT_ERROR = "usage_limit_reached"
_THROTTLED = ("not your usage limit", "temporarily limiting", "temporary capacity", "overloaded")
_USAGE_LIMIT_TEXT = ("usage limit reached", "you've hit your", "you've reached your")


class ClaudeCodeError(RuntimeError):
    """A call through Claude Code failed."""


class ClaudeCodeNotSignedIn(ClaudeCodeError):
    """Claude Code is missing, not signed in, or not on a Claude plan; retrying will not help."""


class ClaudeCodeUsageLimit(ClaudeCodeError):
    """The plan's usage limit is reached; retrying now will not help."""


def claude_binary(configured: str | None = None) -> str:
    """The ``claude`` executable, absolute: ``configured``, TRADINGAGENTS_CLAUDE_BIN, else ``claude`` on PATH.

    A bare name is looked up on PATH; a path may start with ``~``.
    """
    configured = configured or os.environ.get("TRADINGAGENTS_CLAUDE_BIN")
    found = shutil.which(os.path.expanduser(configured or "claude"))
    if not found:
        where = f"Claude Code executable {configured!r} was not found. " if configured else (
            "Claude Code's `claude` command was not found. ")
        raise ClaudeCodeNotSignedIn(where + _CLI_HINT)
    return os.path.abspath(found)


def claude_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a ``claude`` subprocess gets: the allowlist, and the plan token."""
    source = os.environ if source is None else source
    env = {key: source[key] for key in _ENV_ALLOWLIST if source.get(key)}
    if source.get(_PLAN_TOKEN) and not source.get(_HOST_MANAGED_AUTH):
        env[_PLAN_TOKEN] = source[_PLAN_TOKEN]
    return env


def _subprocess(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run ``claude`` with UTF-8 text both ways: Claude Code reads and writes UTF-8."""
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=claude_environment(), **kwargs)


def sign_in_status(binary: str | None = None, timeout: float = 30) -> dict[str, Any]:
    """``claude auth status`` as a dict (``loggedIn``, ``authMethod``, ``email``, ...)."""
    binary = binary or claude_binary()
    try:
        done = _subprocess([binary, *_SCOPE, "auth", "status"], timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClaudeCodeNotSignedIn(f"Could not run `claude auth status`: {exc}") from exc
    try:
        status = json.loads(done.stdout or "")
    except json.JSONDecodeError:
        status = None
    if not isinstance(status, dict):
        # Only Claude Code's own JSON says whether it is signed in; anything
        # else is a Claude Code that could not answer.
        err = (done.stderr or "").strip()
        detail = (err or (done.stdout or "").strip())[-800:] or "no output"
        if "unknown option" in err or "unknown command" in err:
            raise ClaudeCodeNotSignedIn(
                f"This Claude Code is too old for TradingAgents ({detail}). Update it with `claude update`.")
        raise ClaudeCodeNotSignedIn(
            f"`claude auth status` exited with code {done.returncode} and gave no sign-in status: {detail}")
    status.setdefault("loggedIn", False)
    return status


def plan_problem(status: dict[str, Any], env: dict[str, str] | None = None) -> str | None:
    """Why Claude Code, signed in as ``status`` says, cannot run on a Claude plan; None when it can.

    ``env`` is the environment Claude Code runs with (``claude_environment()``).
    A status from a Claude Code too old to report ``authMethod`` passes.
    """
    if not status.get("loggedIn"):
        return "Claude Code is not signed in."
    env = claude_environment() if env is None else env
    method = status.get("authMethod")
    provider = status.get("apiProvider")
    sign_in = _PLAN_HINT
    if method in API_AUTH_METHODS:
        source = f", {status['apiKeySource']}" if status.get("apiKeySource") else ""
        return (f"Claude Code is signed in to an Anthropic API account ({method}{source}), not a "
                f"Claude plan, so every call would be billed to that account. {sign_in}")
    if provider not in (None, "firstParty") or method == "third_party":
        return (f"Claude Code is set to use {provider or 'a cloud provider'} (likely through managed "
                f"settings on this machine), not a Claude plan, so every call would be billed there. "
                f"Use the matching TradingAgents provider instead, such as bedrock. {sign_in}")
    if method == "oauth_token" and _PLAN_TOKEN not in env:
        return ("Claude Code is signed in with a token that is not a Claude plan sign-in (an Anthropic "
                "Console account profile, or an ANTHROPIC_AUTH_TOKEN from its settings), so calls would "
                f"be billed to an API account. {sign_in}")
    if method not in (None, "claude.ai", "oauth_token"):
        return f"Claude Code reports a sign-in TradingAgents does not know ({method}). {sign_in}"
    return None


# Binaries whose plan sign-in was confirmed in this process. A failed check is
# not remembered, so signing in while a program runs takes effect.
_CONFIRMED: set[tuple[str, str | None]] = set()
_CONFIRMED_LOCK = threading.Lock()


def require_plan(binary: str | None = None) -> str:
    """The ``claude`` binary, once its sign-in is confirmed to be a Claude plan."""
    binary = binary or claude_binary()
    key = (binary, os.environ.get("CLAUDE_CONFIG_DIR"))
    with _CONFIRMED_LOCK:
        if key in _CONFIRMED:
            return binary
    problem = plan_problem(sign_in_status(binary), claude_environment())
    if problem:
        raise ClaudeCodeNotSignedIn(f"{problem} {_CLI_HINT}" if "not signed in" in problem else problem)
    with _CONFIRMED_LOCK:
        _CONFIRMED.add(key)
    return binary


def _text(content: Any) -> str:
    """A message's content as plain text (text blocks joined)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _tool_lines(tools: Sequence[dict]) -> str:
    lines = []
    for tool in tools:
        fn = tool["function"]
        params = json.dumps(fn.get("parameters") or {"type": "object", "properties": {}})
        lines.append(f"- `{fn['name']}`: {fn.get('description', '').strip()}\n  Arguments schema: {params}")
    return "\n".join(lines)


def render_prompt(
    messages: Sequence[BaseMessage],
    tools: Sequence[dict] | None,
    required: bool = False,
    run_date: str | None = None,
) -> tuple[str, str]:
    """(system prompt, user prompt) for one ``claude -p`` call from a chat history.

    A single user message goes as it is. A longer history, with earlier
    answers, tool calls and tool results, is written out as a transcript the
    model continues.
    """
    system = "\n\n".join(_text(m.content) for m in messages if isinstance(m, SystemMessage)).strip()
    date_note = _RUN_DATE_NOTE.format(date=run_date) if run_date else _DATE_NOTE
    system = f"{system or _DEFAULT_SYSTEM}\n\n{date_note}"
    if tools:
        protocol = _TOOL_PROTOCOL.format(finish=_FINISH_REQUIRED if required else _FINISH_FREE)
        system = f"{system}\n\n{protocol}{_tool_lines(tools)}"
    reminder = _TOOL_REMINDER if tools else ""
    rest = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(rest) == 1 and isinstance(rest[0], HumanMessage):
        return system, _text(rest[0].content) + reminder

    parts = ["The conversation so far is below. Write the assistant's next message.\n"]
    for message in rest:
        if isinstance(message, ToolMessage):
            name = message.name or "tool"
            parts.append(f'<tool_result name="{name}" call_id="{message.tool_call_id}">\n'
                         f"{_text(message.content)}\n</tool_result>")
        elif isinstance(message, AIMessage):
            said = _text(message.content)
            calls = "\n".join(
                f'<tool_call name="{c["name"]}" call_id="{c.get("id", "")}">'
                f"{json.dumps(c['args'], default=str)}</tool_call>"
                for c in message.tool_calls or []
            )
            body = "\n".join(p for p in (said, calls) if p)
            parts.append(f"<assistant>\n{body}\n</assistant>")
        else:
            parts.append(f"<user>\n{_text(message.content)}\n</user>")
    return system, "\n\n".join(parts) + reminder


def _call_schema(tool: dict) -> dict:
    fn = tool["function"]
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    return {
        "type": "object",
        "properties": {"name": {"const": fn["name"]}, "arguments": params},
        "required": ["name", "arguments"],
    }


def answer_schema(tools: Sequence[dict], required: bool) -> dict:
    """The structured answer offering ``tools``: calls to make, and/or the content."""
    calls: dict[str, Any] = {
        "type": "array",
        "description": "The tool calls to run for you; the only way to use a tool. Empty when you are done.",
        "items": {"anyOf": [_call_schema(t) for t in tools]},
    }
    if required:
        calls["minItems"] = 1
    content = {"type": "string", "description": "Empty while calling tools; your complete answer once done."}
    return {
        "type": "object",
        "properties": {"tool_calls": calls, "content": content},
        "required": ["tool_calls", "content"],
    }


def _forced_name(tool_choice: Any, tools: Sequence[dict]) -> str | None:
    """The one tool a ``tool_choice`` forces, if it forces exactly one."""
    names = [t["function"]["name"] for t in tools]
    if isinstance(tool_choice, dict):
        tool_choice = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
    if tool_choice in ("any", "required", True) and len(names) == 1:
        return names[0]
    if isinstance(tool_choice, str) and tool_choice in names:
        return tool_choice
    return None


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


class ChatClaudeCode(BaseChatModel):
    """A LangChain chat model that answers through ``claude -p`` on the person's plan."""

    model: str | None = None
    """Claude Code model alias or ID (``sonnet``, ``opus``, ``claude-opus-5-5``); None = its default."""
    effort: str | None = None
    """Claude Code ``--effort`` (low, medium, high, xhigh, max); None = its default."""
    binary: str | None = None
    timeout: float = 900
    max_retries: int = 2

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any):
        formatted = [convert_to_openai_tool(t) for t in tools]
        return super().bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        tools: Sequence[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = list(tools or [])
        if tool_choice == "none":
            tools = []
        forced = _forced_name(tool_choice, tools) if tools else None
        required = tool_choice in ("any", "required", True)
        metadata = (getattr(run_manager, "metadata", None) or {}) if run_manager else {}
        run_date = str(metadata[RUN_DATE_METADATA]) if metadata.get(RUN_DATE_METADATA) else None

        if forced:
            # The schema is the forced tool's own arguments: no call protocol.
            system, prompt = render_prompt(messages, None, run_date=run_date)
            system = f"{system}\n\n{_FORCED_NOTE}"
            tool = next(t for t in tools if t["function"]["name"] == forced)
            schema = tool["function"].get("parameters") or {"type": "object", "properties": {}}
        elif tools:
            system, prompt = render_prompt(messages, tools, required=required, run_date=run_date)
            schema = answer_schema(tools, required=required)
        else:
            system, prompt = render_prompt(messages, None, run_date=run_date)
            schema = None

        result = self._run(system, prompt, schema)
        message = self._message(result, schema, forced)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- one claude -p call ------------------------------------------------

    def _args(self, system_file: str, schema: dict | None) -> list[str]:
        args = [
            "-p", "--output-format", "json", *_SCOPE,
            "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
            "--tools", "",
            # A file, not argv: an agent's prompt can pass Windows' 32K command-line limit.
            "--system-prompt-file", system_file,
        ]
        if self.model:
            args += ["--model", self.model]
        if self.effort:
            args += ["--effort", self.effort]
        if schema is not None:
            args += ["--json-schema", json.dumps(schema)]
        return args

    def _run(self, system: str, prompt: str, schema: dict | None) -> dict[str, Any]:
        binary = claude_binary(self.binary)
        attempts = max(0, int(self.max_retries)) + 1
        last: ClaudeCodeError | None = None
        for attempt in range(attempts):
            if attempt:
                # A passing throttle hits the analysts running side by side
                # together; wait long enough for it to clear.
                throttled = isinstance(last, _Throttled)
                time.sleep(min((15 if throttled else 1) * 2 ** attempt, 120))
            try:
                return self._once(binary, system, prompt, schema)
            except (ClaudeCodeNotSignedIn, ClaudeCodeUsageLimit):
                raise
            except ClaudeCodeError as exc:
                last = exc
                if attempt + 1 < attempts:
                    logger.warning("Claude Code call failed (%s); retrying.", exc)
        if isinstance(last, _Throttled):
            raise ClaudeCodeError(f"Claude stayed busy after {attempts} attempt(s): {last}") from last
        raise last  # type: ignore[misc]

    def _once(self, binary: str, system: str, prompt: str, schema: dict | None) -> dict[str, Any]:
        # The system prompt sits apart from the empty folder Claude Code runs in.
        with tempfile.TemporaryDirectory(prefix="tradingagents-claude-") as cwd, \
                tempfile.TemporaryDirectory(prefix="tradingagents-claude-sys-") as sysdir:
            system_file = os.path.join(sysdir, "system-prompt.txt")
            with open(system_file, "w", encoding="utf-8") as handle:
                handle.write(system)
            command = [binary, *self._args(system_file, schema)]
            try:
                done = _subprocess(command, input=prompt, timeout=self.timeout, cwd=cwd)
            except FileNotFoundError as exc:
                raise ClaudeCodeNotSignedIn(f"Could not run Claude Code: {exc}. {_CLI_HINT}") from exc
            except subprocess.TimeoutExpired as exc:
                raise ClaudeCodeError(f"Claude Code did not answer within {self.timeout:.0f}s") from exc

        result = _parse_result(done.stdout)
        if result is None:
            # stdout may hold the model's own words; only stderr names the failure.
            _raise_for_text(done.stderr or "")
            detail = (done.stderr or done.stdout or "").strip()[-800:]
            raise ClaudeCodeError(f"Claude Code exited with code {done.returncode}: {detail or 'no output'}")
        if result.get("is_error") or result.get("subtype") not in (None, "success"):
            detail = str(result.get("result") or result.get("subtype") or "unknown error").strip()
            _raise_for_text(detail, result.get("api_error_status"), result.get("api_error"))
            raise ClaudeCodeError(f"Claude Code reported an error: {detail[:800]}")
        if schema is not None and _structured(result) is None:
            raise ClaudeCodeError("Claude Code returned no structured answer")
        return result

    @staticmethod
    def _message(result: dict[str, Any], schema: dict | None, forced: str | None) -> AIMessage:
        usage = result.get("usage") or {}
        input_tokens = sum(int(usage.get(k) or 0) for k in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        output_tokens = int(usage.get("output_tokens") or 0)
        models = list((result.get("modelUsage") or {}).keys())
        meta = {
            "model_name": models[0] if models else None,
            "total_cost_usd": result.get("total_cost_usd"),
            "duration_ms": result.get("duration_ms"),
            "num_turns": result.get("num_turns"),
        }
        common = {
            "response_metadata": meta,
            "usage_metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        if schema is None:
            return AIMessage(content=str(result.get("result") or ""), **common)

        answer = _structured(result) or {}
        if forced:
            return AIMessage(content="", tool_calls=[{"name": forced, "args": answer, "id": _call_id()}], **common)

        calls = []
        for call in answer.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("name"):
                args = call.get("arguments")
                calls.append({"name": call["name"], "args": args if isinstance(args, dict) else {},
                              "id": _call_id()})
        return AIMessage(content=str(answer.get("content") or ""), tool_calls=calls, **common)


def _result_event(parsed: Any) -> dict[str, Any] | None:
    """The result event in a parsed ``--output-format json`` value.

    It is one object, or, when Claude Code's verbose setting is on, an array of
    every message ending with that object.
    """
    if isinstance(parsed, dict) and parsed.get("type") == "result":
        return parsed
    if isinstance(parsed, list):
        for item in reversed(parsed):
            if isinstance(item, dict) and item.get("type") == "result":
                return item
    return None


def _parse_result(stdout: str) -> dict[str, Any] | None:
    """The ``--output-format json`` result object, or None when there is none."""
    text = (stdout or "").strip()
    if not text:
        return None
    for candidate in (text, text.splitlines()[-1]):
        try:
            event = _result_event(json.loads(candidate))
        except json.JSONDecodeError:
            continue
        if event is not None:
            return event
    return None


def _structured(result: dict[str, Any]) -> dict[str, Any] | None:
    answer = result.get("structured_output")
    if isinstance(answer, dict):
        return answer
    try:
        parsed = json.loads(str(result.get("result") or "").strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class _Throttled(ClaudeCodeError):
    """A passing 429 or overload: worth a longer wait, then a retry."""


def _raise_for_text(detail: str, status: Any = None, api_error: Any = None) -> None:
    """Raise the error ``detail`` names, when it names one retrying cannot fix or must wait out."""
    lowered = detail.lower()
    if "not logged in" in lowered or "please run /login" in lowered or "invalid api key" in lowered \
            or "oauth token" in lowered or status == 401:
        raise ClaudeCodeNotSignedIn(f"Claude Code is not signed in: {detail[:400]}. {_CLI_HINT}")
    if api_error == _USAGE_LIMIT_ERROR:
        raise ClaudeCodeUsageLimit(f"Your Claude plan's usage limit is reached: {detail[:400]}")
    if any(text in lowered for text in _THROTTLED):
        raise _Throttled(f"Claude is busy: {detail[:400]}")
    if any(text in lowered for text in _USAGE_LIMIT_TEXT):
        raise ClaudeCodeUsageLimit(f"Your Claude plan's usage limit is reached: {detail[:400]}")
    if status in (429, 529):
        raise _Throttled(f"Claude is busy: {detail[:400]}")


# Settings a Claude plan does not take through Claude Code.
_PLAN_REFUSED = ("temperature", "max_tokens")


class ClaudeCodeClient(BaseLLMClient):
    """Client for Claude on the person's plan, through the Claude Code CLI."""

    provider = "claude_code"

    def get_llm(self) -> Any:
        # Stop before the run starts, not at its first model call, when Claude
        # Code is missing, signed out or signed in to an API account.
        binary = require_plan()
        for name in _PLAN_REFUSED:
            if self.kwargs.get(name) is not None:
                logger.warning("Claude Code on a Claude plan does not take %s; it is not sent.", name)
        kwargs: dict[str, Any] = {"model": self.model or None, "binary": binary}
        for key in ("effort", "timeout", "max_retries", "callbacks"):
            if self.kwargs.get(key) is not None:
                kwargs[key] = self.kwargs[key]
        return ChatClaudeCode(**kwargs)

    def validate_model(self) -> bool:
        # Claude Code takes its own aliases (sonnet, opus, haiku) and full model IDs.
        return True
