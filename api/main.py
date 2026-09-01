"""
FastAPI gateway — WebSocket streaming + REST endpoints.

Endpoints:
  POST /sessions              — start a new agent session, returns session_id + ws_url
  GET  /sessions/{id}         — get current session state (polling fallback)
  WS   /ws/{session_id}       — real-time AgentEvent stream
  GET  /health                — health check (Redis status, Docker status)

Architecture:
  1. Client POSTs a TaskRequest → session created, background task starts the graph
  2. Client connects to WS /ws/{session_id} → receives AgentEvent objects in real time
  3. Graph runs in asyncio background task, publishing events to an asyncio.Queue
  4. WS handler reads from queue and forwards to client
  5. On completion, full state available via GET /sessions/{id}
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import build_graph
from models.schemas import (
    AgentState,
    FinalResponse,
    SessionResponse,
    TaskRequest,
)
from router.model_router import ModelRouter
from sandbox.factory import create_sandbox
from session.manager import SessionManager

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

load_dotenv()

_session_manager = SessionManager()
_event_queues: dict[str, asyncio.Queue] = {}  # session_id → Queue[AgentEvent | None]
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    _graph = build_graph()
    yield


app = FastAPI(
    title="CodePilot-Agent API",
    description="Autonomous coding agent — natural language → working code",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    redis_ok = await _session_manager.ping()
    docker_ok = _check_docker()
    return {
        "status": "ok",
        "redis": "connected" if redis_ok else "in-memory fallback",
        "docker": "available" if docker_ok else "unavailable (subprocess mode)",
    }


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: TaskRequest):
    """Start a new agent session. Returns session_id and WebSocket URL."""
    api_key = request.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY required")

    state = AgentState(
        user_request=request.user_request,
        mode=request.mode,
        github_url=request.github_url,
        sandbox_type=request.sandbox_type,
        workspace_files=request.workspace_files,
        original_files=dict(request.workspace_files),
    )

    # Create event queue for this session
    queue: asyncio.Queue = asyncio.Queue()
    _event_queues[state.session_id] = queue

    # Save initial state
    await _session_manager.save(state)

    # Start background graph execution
    asyncio.create_task(_run_graph(state, api_key, queue))

    ws_url = f"/ws/{state.session_id}"
    return SessionResponse(session_id=state.session_id, ws_url=ws_url)


@app.get("/sessions/{session_id}", response_model=FinalResponse)
async def get_session(session_id: str):
    """Get the current state of a session (polling fallback for non-WS clients)."""
    state = await _session_manager.load(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return FinalResponse(
        session_id=state.session_id,
        is_complete=state.is_complete,
        final_output=state.final_output,
        pr_url=state.pr_url,
        error_message=state.error_message,
        total_cost_usd=state.total_cost_usd,
        total_tokens=state.total_tokens,
        workspace_files=state.workspace_files,
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Stream AgentEvents to the client in real time.
    Sends JSON-serialised AgentEvent objects as text frames.
    Sends {"type": "done"} when the session completes.
    """
    await websocket.accept()

    queue = _event_queues.get(session_id)
    if not queue:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "ping"})
                continue

            if event is None:
                # Sentinel: session complete
                await websocket.send_json({"type": "done"})
                break

            await websocket.send_text(event.model_dump_json())

    except WebSocketDisconnect:
        pass
    finally:
        _event_queues.pop(session_id, None)


# ---------------------------------------------------------------------------
# Background graph runner
# ---------------------------------------------------------------------------


async def _run_graph(state: AgentState, api_key: str, queue: asyncio.Queue) -> None:
    """Run the LangGraph graph and publish events to the session queue."""
    router = ModelRouter(api_key=api_key)

    # Initialise sandbox
    sandbox = create_sandbox(state.sandbox_type)
    sandbox_id = await sandbox.setup(state.workspace_files)
    state.sandbox_id = sandbox_id

    # Build initial state dict for LangGraph
    try:
        final_state_dict = await _graph.ainvoke(
            state.model_dump(mode="json"),
            config={"configurable": {"router": router, "sandbox": sandbox}},
        )

        # Reconstruct final state and save
        final_state = AgentState.model_validate(
            {k: v for k, v in final_state_dict.items() if not k.startswith("_")}
        )
        final_state.is_complete = final_state.reviewer_approved
        await _session_manager.save(final_state)

        # Push all events to queue
        for event in final_state.events:
            await queue.put(event)

    except Exception as exc:
        from models.schemas import AgentEvent, AgentEventType, AgentName

        # Persist the failure too, or GET /sessions/{id} reports the initial state
        # with is_complete=False forever and a polling client hangs (F-26).
        state.error_message = str(exc)
        state.is_complete = True
        await _session_manager.save(state)

        err_event = AgentEvent(
            session_id=state.session_id,
            agent=AgentName.ORCHESTRATOR,
            event_type=AgentEventType.ERROR,
            message=f"Agent error: {exc}",
        )
        await queue.put(err_event)
    finally:
        await sandbox.teardown()
        await queue.put(None)  # sentinel — tells WS handler we're done


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_docker() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False
