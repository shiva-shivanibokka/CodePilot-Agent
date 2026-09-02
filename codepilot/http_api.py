"""
The HTTP surface for hosted mode.

Separate from `codepilot.server` for two reasons. The policy — what a
deployment refuses, what a request may not choose — has no web dependency and
is tested without one. And FastAPI resolves a handler's type hints against its
*module* globals, so `Request` and `Header` have to be importable at module
scope; declaring them inside a factory function yields 422 on every route
instead of the handler ever running.

Every route here is thin. If a rule looks like it lives in this file, it is in
`codepilot.server` and this is only calling it.
"""

from __future__ import annotations

import hmac
import json
import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from codepilot.server import ServerConfig, run_stream


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    repo: str = Field(min_length=1, max_length=200)


def build_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="CodePilot", version="0.2.0")
    app.state.config = config

    def authorise(request: Request) -> None:
        # Constant-time, so a rejection does not report how much of the token
        # was right.
        expected = f"Bearer {config.token}"
        if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
            raise HTTPException(status_code=401, detail="bad or missing bearer token")

    def api_key(supplied: str | None) -> str:
        if supplied:
            return supplied
        if config.allow_server_key and os.getenv("ANTHROPIC_API_KEY"):
            return os.environ["ANTHROPIC_API_KEY"]
        raise HTTPException(
            status_code=400,
            detail=(
                "send your own key in X-Anthropic-Key. This deployment does not "
                "lend out its own."
            ),
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "sandbox": config.sandbox}

    @app.get("/v1/config")
    async def describe(request: Request) -> dict:
        """What an operator needs to confirm they deployed what they meant to."""
        authorise(request)
        return config.describe()

    @app.post("/v1/runs")
    async def create_run(
        body: RunRequest,
        request: Request,
        x_anthropic_key: str | None = Header(default=None),
    ) -> StreamingResponse:
        authorise(request)
        key = api_key(x_anthropic_key)
        try:
            config.resolve_repo(body.repo)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def events():
            try:
                async for payload in run_stream(
                    config, prompt=body.prompt, repo=body.repo, api_key=key
                ):
                    yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:  # noqa: BLE001 - the stream is the only channel
                # The type, not the message. An upstream error can quote the
                # request that caused it, and the request carried someone's key.
                yield f"data: {json.dumps({'type': 'error', 'error': type(exc).__name__})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
