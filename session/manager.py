"""
Session Manager — stores and retrieves AgentState using Redis (with in-memory fallback).

Redis TTL: 24 hours per session.
Fallback: plain dict (for local dev without Redis).
"""

from __future__ import annotations

import json
import os
from typing import Any

from models.schemas import AgentState


class SessionManager:
    """
    Persists AgentState across HTTP requests and WebSocket reconnects.
    Uses Redis if available, falls back to an in-memory dict.
    """

    def __init__(self) -> None:
        self._memory: dict[str, dict] = {}
        self._redis: Any = None
        self._ttl = 86_400  # 24 hours

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url, decode_responses=True)
        except ImportError:
            pass  # redis not installed — use in-memory

    async def save(self, state: AgentState) -> None:
        data = state.model_dump_json()
        if self._redis:
            try:
                await self._redis.setex(f"session:{state.session_id}", self._ttl, data)
                return
            except Exception:
                pass
        self._memory[state.session_id] = json.loads(data)

    async def load(self, session_id: str) -> AgentState | None:
        if self._redis:
            try:
                raw = await self._redis.get(f"session:{session_id}")
                if raw:
                    return AgentState.model_validate_json(raw)
            except Exception:
                pass
        data = self._memory.get(session_id)
        if data:
            return AgentState.model_validate(data)
        return None

    async def delete(self, session_id: str) -> None:
        if self._redis:
            try:
                await self._redis.delete(f"session:{session_id}")
            except Exception:
                pass
        self._memory.pop(session_id, None)

    async def ping(self) -> bool:
        """Returns True if Redis is reachable, False if using in-memory fallback."""
        if self._redis:
            try:
                await self._redis.ping()
                return True
            except Exception:
                pass
        return False
