"""
The provider seam.

This is the only module in the package that imports `anthropic`. Everything
above it deals in plain messages, tool schemas and a `Reply` — so adding a
second provider later is one file, not a refactor.

Deliberately Anthropic-only for now: the levers that make a coding agent
affordable (prompt-cache breakpoints) and reliable (tool-use blocks, refusal
stop reasons) are the parts that differ most between providers, and an
abstraction that satisfies all of them drops exactly those.
"""

from __future__ import annotations

import difflib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from anthropic import AsyncAnthropic


def load_env(root: Path | str | None = None) -> None:
    """Read `.env` from the repository being worked on, then where you ran the
    command, then `~/.codepilot.env`.

    Explicit paths, because `find_dotenv()` searches upward from the *caller*,
    which for an installed tool is site-packages. This lives beside the client
    rather than in the CLI because the CLI is not the only thing that needs a
    key: the eval harness ran a whole sweep without one and scored every task
    a failure, which is the same mistake as counting an unreachable judge's
    zero as a verdict.

    The working directory is in the list because of the ordinary case that
    used to fail: the key sits in CodePilot's own `.env`, and the repository
    being edited is somewhere else entirely.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    base = Path(root) if root is not None else Path.cwd()
    for candidate in (base / ".env", Path.cwd() / ".env", Path.home() / ".codepilot.env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


# Overridable so a model retirement is a config change, not a code change.
# Note the asymmetry, which cost a live 404 to discover: some models are served
# under a bare alias, others only under a dated id.
STRONG_MODEL = os.getenv("CODEPILOT_STRONG_MODEL", "claude-opus-5")
FAST_MODEL = os.getenv("CODEPILOT_FAST_MODEL", "claude-haiku-4-5-20251001")
#: The middle rung, for asking whether routing work down beats turning the
#: effort dial down on the strong model. Not used by the agent itself.
ROUTED_MODEL = os.getenv("CODEPILOT_ROUTED_MODEL", "claude-sonnet-5")

#: USD per token, (input, output). Keyed by un-dated model id.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00e-6, 25.00e-6),
    "claude-sonnet-5": (2.00e-6, 10.00e-6),
    "claude-haiku-4-5": (1.00e-6, 5.00e-6),
}


def price_of(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost in USD, or None when the model has no published price.

    None rather than 0.0: a cost display must be able to tell "free" apart from
    "unknown", or an unpriced model silently reads as costing nothing.
    """
    key = re.sub(r"-\d{8}$", "", model)
    if key not in PRICING:
        return None
    in_price, out_price = PRICING[key]
    return input_tokens * in_price + output_tokens * out_price


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class Reply:
    """One assistant turn, provider-neutral."""

    text: str
    #: Raw content blocks, echoed back verbatim on the next request. Thinking
    #: blocks in particular must round-trip unmodified on the same model.
    content: list[Any]
    tool_calls: list[ToolCall]
    stop_reason: str | None
    model: str
    usage: Usage
    latency_ms: int
    cost_usd: float | None

    @property
    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use" and bool(self.tool_calls)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised for failures the caller cannot retry its way out of."""


class LLMClient:
    """Messages in, `Reply` out.

    Retries are the SDK's (429/5xx/connection, exponential backoff). What this
    adds is the shape the agent needs: tool calls parsed out, usage and cost
    attached, and a cache breakpoint placed on the stable prefix.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = STRONG_MODEL,
        max_retries: int = 3,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self._client = AsyncAnthropic(
            api_key=api_key, max_retries=max_retries, timeout=timeout
        )

    async def available_models(self) -> set[str]:
        return {m.id async for m in self._client.models.list()}

    async def validate(self, *models: str) -> None:
        """Fail at startup, not on the first live request.

        Hardcoding a model id means a retirement surfaces as a 404 in the middle
        of somebody's task. Ask the provider what it actually serves.
        """
        wanted = {m for m in models if m} or {self.model}
        available = await self.available_models()
        missing = wanted - available
        if not missing:
            return
        hints = []
        for want in sorted(missing):
            close = difflib.get_close_matches(want, sorted(available), n=2, cutoff=0.6)
            hints.append(repr(want) + (f" — did you mean {close}?" if close else ""))
        raise LLMError(
            "Models not available to this API key:\n  "
            + "\n  ".join(hints)
            + "\nSet CODEPILOT_STRONG_MODEL / CODEPILOT_FAST_MODEL to override."
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        system: list[dict[str, Any]] | str | None = None,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        effort: str | None = None,
        thinking: bool = False,
    ) -> Reply:
        model = model or self.model
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            params["system"] = system
        if tools:
            params["tools"] = tools
        if thinking:
            params["thinking"] = {"type": "adaptive"}
        if effort:
            # The cost dial. Lower effort on a strong model is often a better
            # trade than routing the work to a weaker one — which is the whole
            # premise of the routing table, and worth measuring rather than
            # assuming (see the eval's experiment 4).
            params["output_config"] = {"effort": effort}

        start = time.monotonic()
        try:
            response = await self._client.messages.create(**params)
        except anthropic.NotFoundError as exc:
            raise LLMError(
                f"Model {model!r} was not found. It may have been retired; "
                "run validate() at startup to catch this before a request."
            ) from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("The API key was rejected.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"This key may not use {model!r}.") from exc
        # RateLimitError / APIStatusError>=500 / APIConnectionError are retried
        # by the SDK; if they still surface, they are genuinely terminal.
        latency_ms = int((time.monotonic() - start) * 1000)

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0)
            or 0,
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
            for b in response.content
            if getattr(b, "type", "") == "tool_use"
        ]
        return Reply(
            text=text,
            content=list(response.content),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            model=response.model,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=price_of(
                response.model, usage.input_tokens, usage.output_tokens
            ),
        )

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: list[dict[str, Any]] | str | None = None,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> int:
        """Real token count from the provider — never a character-count estimate."""
        params: dict[str, Any] = {"model": model or self.model, "messages": messages}
        if system is not None:
            params["system"] = system
        if tools:
            params["tools"] = tools
        result = await self._client.messages.count_tokens(**params)
        return result.input_tokens
