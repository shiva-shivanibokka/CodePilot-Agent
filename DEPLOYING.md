# Deploying CodePilot

Built by Shivani Bokka

CodePilot has two modes, and they are not the same program wearing a different
hat. Which one you want depends on a single question: **whose machine runs the
commands the model chooses?**

| | local | hosted |
|---|---|---|
| `codepilot run` / `chat` / the browser UI | `codepilot serve` |
| commands run | on your machine, in your repository | in a container, on a copy |
| unlisted command | asks you | refused, with the reason |
| your repository | edited in place, behind git checkpoints | untouched; you get a patch back |
| the API key | yours, from `.env` | the caller's, per request |
| who it is for | you | your team, behind your own gateway |

If you are one person working on your own code, use local mode. Everything
below is about the other case.

---

## Local

```bash
pip install -r requirements.txt && pip install -e .
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
codepilot doctor           # 16 checks, no billing
codepilot run "fix the failing test in payments.py"
codepilot undo             # if you disagree with it
```

Commands run on your machine because you asked them to. Isolating your code
from your own computer would be theatre: the agent is running your tests, at
your request, on files you already own. What protects you is the git
checkpoint before every turn and the permission prompt on anything that is not
a read or a test.

---

## Hosted

### What the server refuses

These are not settings with safe defaults. They are refusals, and each one
exists because the local mode's assumption stops being true the moment someone
else is typing the prompt.

**It will not start without a token.** `CODEPILOT_SERVER_TOKEN` must be at
least 16 characters and has no default. A server with a guessable token runs
shell commands for whoever guesses it.

**It will not run commands on the host.** The default sandbox is a container,
started per run with `--network none`, a memory cap and a CPU cap, and removed
afterwards. `CODEPILOT_SANDBOX=local` is available and refuses to load unless
you *also* set `CODEPILOT_ALLOW_HOST_EXECUTION=1` — two variables, so nobody
reaches host execution by omitting one.

**It will not approve anything on your behalf.** Locally, a command outside the
read-only allowlist prompts you. A server has nobody at the keyboard, so the
permission gate is built with no prompt callback and every such command is
denied with its reason. Widen the allowlist deliberately with
`CODEPILOT_EXTRA_ALLOWED="ruff mypy"` if your repository needs it.

**It will not lend out its key.** Callers send `X-Anthropic-Key` per request.
The operator is not paying for other people's tokens and is not storing their
credentials. `CODEPILOT_ALLOW_SERVER_KEY=1` turns this off if you are the only
caller.

**It will not let a request raise its own ceiling.** `CODEPILOT_MAX_USD` and
`CODEPILOT_MAX_TURNS` come from the deployment. Nothing in the request body can
change them.

**It will not write to your repository.** Each run works on a copy and returns
a unified diff in the final event. The server does not own your branch, your
history or your review, and a patch is the smallest thing that hands the work
back without taking any of them.

### Running it

```bash
docker build -f deploy/sandbox.Dockerfile -t codepilot-sandbox:latest .

export CODEPILOT_SERVER_TOKEN=$(openssl rand -hex 24)
export CODEPILOT_WORKSPACE_ROOT=/srv/repos

codepilot serve --check          # prints the effective policy, starts nothing
codepilot serve --host 0.0.0.0 --port 8000
```

`--check` is the one to run first. It prints exactly what you deployed —
sandbox, whether host execution is on, whether BYOK is required, the ceiling,
and which repositories are visible — so you can compare it against what you
meant.

### Using it

```bash
curl -N http://localhost:8000/v1/runs \
  -H "Authorization: Bearer $CODEPILOT_SERVER_TOKEN" \
  -H "X-Anthropic-Key: sk-ant-..." \
  -H "Content-Type: application/json" \
  -d '{"repo": "payments", "prompt": "mean() crashes on an empty list. Raise ValueError instead."}'
```

Server-sent events arrive as the run happens — every model call with its cost,
every tool call, every diff, every test result — and the last one carries the
patch:

```json
{"type": "run_end", "finished": true, "stopped_by": "finished",
 "edited": ["stats.py", "test_stats.py"],
 "patch": "--- a/stats.py\n+++ b/stats.py\n@@ -1,2 +1,4 @@\n def mean(values):\n+    if not values:\n+        raise ValueError(...)\n..."}
```

| endpoint | auth | what it does |
|---|---|---|
| `GET /healthz` | none | liveness, and which sandbox is configured |
| `GET /v1/config` | bearer | the effective policy, for verifying a deployment |
| `POST /v1/runs` | bearer + `X-Anthropic-Key` | one run, streamed as SSE |

### Configuration

| variable | default | notes |
|---|---|---|
| `CODEPILOT_SERVER_TOKEN` | **none** | ≥16 chars. Refuses to start without it. |
| `CODEPILOT_WORKSPACE_ROOT` | **none** | Directory of repositories. Names are resolved inside it and rejected if they escape. |
| `CODEPILOT_SANDBOX` | `docker` | `local` needs `CODEPILOT_ALLOW_HOST_EXECUTION=1` too. |
| `CODEPILOT_SANDBOX_IMAGE` | `codepilot-sandbox:latest` | Build it from `deploy/sandbox.Dockerfile`. |
| `CODEPILOT_ALLOW_SERVER_KEY` | `0` | On: falls back to the server's `ANTHROPIC_API_KEY`. |
| `CODEPILOT_MAX_USD` | `1.00` | Per run. |
| `CODEPILOT_MAX_TURNS` | `40` | Per run. |
| `CODEPILOT_MODEL` | `claude-opus-5` | |
| `CODEPILOT_EFFORT` | `low` | The cost dial. |
| `CODEPILOT_EXTRA_ALLOWED` | empty | Extra executables, e.g. `"ruff mypy npm"`. |

### Running the service in a container

`deploy/Dockerfile` and `deploy/docker-compose.yml` exist, and you should read
this before using them: **a container that starts sandbox containers needs the
Docker socket, and mounting the Docker socket into a container is equivalent to
giving that container root on the host.** It undoes most of what the sandbox is
for.

Running the service directly on a host that has Docker — systemd, or whatever
your platform uses — avoids that completely, and is the recommended shape. The
compose file is there for people who have already decided otherwise.

---

## What this is not

Said plainly, because a deployment guide that overstates its own maturity is
worse than no guide.

- **Not multi-tenant.** One shared bearer token, not accounts. No per-caller
  quota, no rate limit, no audit log. It is a correct single-tenant service —
  the shape you put behind your own gateway, not one you expose to the public.
- **Not persistent.** Runs are not stored. Local sessions resume; hosted ones
  do not. If you want history, keep the event stream your client received.
- **Not concurrency-tested.** Each run copies its repository, so two runs
  cannot interleave edits, but nothing limits how many run at once. Put a
  concurrency limit in front of it before you need one.
- **The container path is not verified end to end here.** The machine this was
  built on has no Docker daemon. Every refusal, the policy, the HTTP surface
  and a full run were tested — the last against the live API, with host
  execution deliberately enabled — and the container path is covered as far as
  "the right arguments are built" and no further. Run `codepilot serve --check`
  and one throwaway request on your own host before you trust it with anything.
