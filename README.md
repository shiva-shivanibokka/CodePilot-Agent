# CodePilot-Agent

An autonomous coding agent that takes a natural language task, plans an implementation, writes code across multiple files, runs it in an isolated sandbox, debugs failures, and delivers working results — all streamed to you in real time.

Built to demonstrate the same architecture used by production coding agents (Devin, SWE-agent, OpenDevin), with purpose-built solutions to their known failure modes.

---

## What It Does

Describe what you want to build. The agent handles the rest:

```
You: "Write a FastAPI CRUD API for a todo list with SQLite persistence and full pytest coverage"

Agent:
  🤔 [orchestrator] Received request. Complexity: COMPLEX
  📝 [planner] Execution plan (4 steps):
       Step 1: Create database module with SQLite connection and models
       Step 2: Write FastAPI routes for CRUD operations
       Step 3: Write pytest test suite with fixtures
       Step 4: Verify all tests pass
  ➡️  [coder] Step 1: Create database module...
  ✍️  Written: database.py (47 lines)
  ➡️  [coder] Step 2: Write FastAPI routes...
  ✍️  Written: main.py (89 lines)
  ➡️  [coder] Step 3: Write test suite...
  ✍️  Written: tests/test_api.py (63 lines)
  🧪 [tester] Running: pytest tests/ -v --tb=short
  📊 [tester] ❌ 7 passed, 1 failed — sending to Debugger
  🔍 [debugger] Root cause: Missing Content-Type header in test fixture
  🩹 [debugger] Patched: tests/test_api.py
  🧪 [tester] Running: pytest tests/ -v --tb=short
  📊 [tester] ✅ 8 passed, 0 failed (1.2s)
  ✅ [reviewer] Approved: Implementation complete and correct
  💰 Session cost: $0.0034 | 4,891 tokens
```

---

## Architecture

### Multi-Agent System (LangGraph)

```
START → Orchestrator
          ├── Task mode → Planner → Coder → Tester
          │                              ↑        │
          │                         (iterate)   pass│fail
          │                                     │    ↓
          │                             Reviewer  Debugger
          │                              ↓
          │                           DONE / loop back
          └── Q&A mode → QA Agent → DONE
```

Six specialized agents, each with a focused system prompt and appropriate model:

| Agent | Model | Responsibility |
|---|---|---|
| Orchestrator | Haiku/Sonnet | Intent classification, complexity routing, control flow |
| Planner | Sonnet | AST-indexed codebase analysis, step-by-step ExecutionPlan |
| Coder | Haiku/Sonnet | File writes per plan step, follows existing code style |
| Tester | Haiku | Runs pytest, parses pass/fail, triggers Debugger on failure |
| Debugger | Sonnet | Root cause analysis, targeted patch, max 3 iterations |
| Reviewer | Sonnet | Validates output against original task, approves or loops back |

### Execution Sandbox

Every `run_code` and `run_tests` call executes inside an isolated environment:

```
Subprocess mode (default, no Docker needed):
  Agent writes file → asyncio subprocess → stdout/stderr captured → result returned

Docker mode (production):
  Agent writes file → docker exec into container → 512MB RAM limit, 1 CPU,
  no network, non-root user → result returned → container destroyed on session end
```

The `SandboxBackend` protocol means the agent never touches Docker or subprocess directly — both are swappable at config time.

### Context Window Management

The Planner queries the **AST Indexer** before any LLM call:
- Parses every `.py` file into a structural index: functions, classes, call graph, imports
- Identifies the 3-5 most relevant files for the current task via keyword overlap scoring
- The Coder receives only those files — not the entire codebase
- This prevents the #1 failure mode of coding agents: context window exhaustion

### Model Routing

```python
routing_table = {
    (Planner, any):       Sonnet,   # always plan carefully
    (Coder, simple):      Haiku,    # pattern-based, fast
    (Coder, complex):     Sonnet,   # architectural changes
    (Tester, any):        Haiku,    # pytest output parsing
    (Debugger, any):      Sonnet,   # root cause reasoning
    (Reviewer, any):      Sonnet,   # correctness validation
}
```

Every LLM call is logged with model, input/output tokens, latency, and cost. A live cost ticker shows in the UI.

### Real-Time Streaming

Every agent action emits an `AgentEvent` that streams to the Gradio UI via a generator. The user sees planning, coding, testing, and debugging as it happens — not just a result at the end.

### Session Persistence (Redis)

Full `AgentState` is serialised to Redis after every node with a 24-hour TTL. Sessions survive browser refreshes and can be resumed.

---

## Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph (stateful multi-agent graph) |
| LLM | Claude Sonnet 4.5 + Claude Haiku 3.5 (model routing) |
| Sandboxed execution | Docker SDK / asyncio subprocess |
| AST indexing | Python `ast` stdlib + custom call graph builder |
| API + streaming | FastAPI + WebSocket |
| Session state | Redis (in-memory fallback) |
| UI | Gradio (real-time generator streaming) |
| Containerisation | Docker + docker-compose |

---

## Local Setup

```bash
git clone https://github.com/shiva-shivanibokka/CodePilot-Agent
cd CodePilot-Agent
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
python app.py          # opens at http://localhost:7860
```

No Docker required for local development — the agent runs in subprocess sandbox mode by default.

### Full Stack with Docker + Redis

```bash
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker-compose up --build
```

- Gradio UI: http://localhost:7860
- FastAPI docs: http://localhost:8000/docs
- Redis: localhost:6379

### Enable Docker Sandbox Mode

For production-grade isolation, build the sandbox image first:

```bash
docker build -t codepilot-sandbox:latest sandbox/docker/
```

Then select **Docker** in the Gradio UI sandbox selector (or set `SANDBOX_TYPE=docker` in `.env`).

---

## Project Structure

```
CodePilot-Agent/
├── app.py                    # Gradio UI — entry point for local use
├── models/
│   └── schemas.py            # All Pydantic models: AgentState, AgentEvent,
│                             #   ExecutionPlan, LLMCall, TestResult, etc.
├── agent/
│   ├── graph.py              # LangGraph state graph — all nodes + edges
│   └── prompts.py            # System prompts for each agent
├── sandbox/
│   ├── base.py               # SandboxBackend protocol
│   ├── subprocess_sandbox.py # Local execution (no Docker)
│   ├── docker_sandbox.py     # Isolated Docker container execution
│   ├── factory.py            # create_sandbox() — selects backend by config
│   └── docker/Dockerfile     # Sandbox container image
├── tools/
│   └── registry.py           # All agent tools: read_file, write_file,
│                             #   run_code, run_tests, search_codebase, git_*
├── indexer/
│   └── ast_indexer.py        # Codebase AST parser + relevance scorer
├── router/
│   └── model_router.py       # Claude model selection by agent + complexity
├── session/
│   └── manager.py            # Redis session persistence (in-memory fallback)
├── api/
│   └── main.py               # FastAPI: POST /sessions, WS /ws/{id}, GET /health
├── docker-compose.yml        # App + Redis
├── Dockerfile                # Main app image
├── requirements.txt
└── .env.example
```

---

## API Usage

Start a session:
```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "Write a binary search function with type hints and docstring",
    "mode": "task",
    "sandbox_type": "subprocess",
    "anthropic_api_key": "sk-ant-..."
  }'

# Response: {"session_id": "abc123", "ws_url": "/ws/abc123"}
```

Stream events via WebSocket:
```python
import asyncio, websockets, json

async def watch():
    async with websockets.connect("ws://localhost:8000/ws/abc123") as ws:
        async for msg in ws:
            event = json.loads(msg)
            if event.get("type") == "done":
                break
            print(event.get("message", ""))

asyncio.run(watch())
```

Poll for result:
```bash
curl http://localhost:8000/sessions/abc123
```

---

## What This Demonstrates

| System design concept | Where it appears |
|---|---|
| Sandboxed code execution | `sandbox/docker_sandbox.py` — Docker with memory/CPU/network limits |
| Protocol-based abstraction | `SandboxBackend` — DockerSandbox and SubprocessSandbox are interchangeable |
| Multi-agent state machine | `agent/graph.py` — LangGraph with conditional edges and shared state |
| Context window management | `indexer/ast_indexer.py` — AST-based relevance scoring, not full file loading |
| Intelligent model routing | `router/model_router.py` — cost/quality tradeoff per agent and task |
| Real-time event streaming | `api/main.py` WebSocket + Gradio generator in `app.py` |
| Session persistence | `session/manager.py` — Redis with in-memory fallback |
| Self-healing debug loop | `agent/graph.py` — Tester → Debugger → Tester cycle with iteration cap |
| Structured inter-agent contracts | `models/schemas.py` — Pydantic validation at every node boundary |
