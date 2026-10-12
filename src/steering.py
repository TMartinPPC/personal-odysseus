"""steering.py

Per-session mid-run steering queue (pi/opencode pattern).

When an agent run is already streaming for a session, a follow-up user
message can be *steered* into the running turn instead of replacing it:
the loop drains the queue at each round boundary and appends the messages
as ordinary user turns after the current round's tool results. The model
sees them exactly as if the user had typed them mid-task — no run is
cancelled, no partial work is lost.

Scope note: the frontend currently queues follow-ups client-side
(`_queuedAgentRequests` in static/js/chat.js) and sends them when idle, so
nothing enqueues here yet. This module provides the backend mechanism and
the loop hook; wiring the chat route to enqueue (instead of
agent_runs.start's cancel-and-replace) is the follow-up step.
"""

import asyncio
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Bound per-session queues so a client bug can't fill memory; beyond the
# cap the oldest pending steer is dropped (FIFO overflow) and the enqueue
# is refused.
MAX_PENDING_STEERS = 8

_QUEUES: Dict[str, "asyncio.Queue[str]"] = {}


def _queue_for(session_id: str) -> "asyncio.Queue[str]":
    q = _QUEUES.get(session_id)
    if q is None:
        q = asyncio.Queue()
        _QUEUES[session_id] = q
    return q


def enqueue(session_id: str, message: str) -> bool:
    """Queue one steering message for a session's active/next run.

    Returns True when accepted, False when the message is empty or the
    queue is full. Older entries are dropped first on overflow.
    """
    msg = (message or "").strip()
    if not msg or not session_id:
        return False
    q = _queue_for(session_id)
    while q.qsize() >= MAX_PENDING_STEERS:
        try:
            dropped = q.get_nowait()
            logger.warning(f"Steering queue overflow for session {session_id}; dropped oldest: {dropped[:60]!r}")
        except asyncio.QueueEmpty:
            break
    q.put_nowait(msg)
    return True


def drain(session_id: str) -> List[str]:
    """Remove and return all pending steering messages, oldest first."""
    q = _QUEUES.get(session_id)
    if q is None:
        return []
    out: List[str] = []
    while True:
        try:
            out.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


def pending(session_id: str) -> int:
    q = _QUEUES.get(session_id)
    return q.qsize() if q else 0


def clear(session_id: str) -> None:
    """Drop any pending steering messages for a session."""
    _QUEUES.pop(session_id, None)
