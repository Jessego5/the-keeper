"""a2a.py — the Agent-to-Agent (A2A) protocol. The Keeper as a peer among agents.

MCP lets the Keeper use TOOLS; A2A lets it interoperate with other AGENTS. It is an
open standard (Google's A2A, now under the Linux Foundation): an agent publishes an
**Agent Card** at /.well-known/agent.json describing who it is and what it can do, and
exposes an endpoint that accepts a **message/send** JSON-RPC call and returns the reply.

This module is both halves of a faithful core subset:
  - build_agent_card()  — the Keeper's card, served so other agents can discover it
  - the request/response envelope helpers for message/send (server + client)
  - consult()           — the CLIENT: discover a remote agent by its card, send it a
                          message, read its reply.

Scope kept honest: the synchronous message/send core, text parts, no streaming or
push-notification task states — the parts that prove interoperability without the
distributed-task machinery a single-user companion doesn't need.
"""

from __future__ import annotations

import uuid
from typing import Optional

import httpx

WELL_KNOWN = "/.well-known/agent.json"
PROTOCOL_VERSION = "0.2.0"


def build_agent_card(base_url: str) -> dict:
    """The Keeper's A2A Agent Card. `base_url` is where it's reachable (no trailing /)."""
    base = base_url.rstrip("/")
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "name": "The Keeper",
        "description": ("A proactive companion agent — keeps long-term memory, pursues "
                        "goals, and can research, compute, and dig through its own "
                        "materials via specialist sub-agents."),
        "url": f"{base}/a2a",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {"id": "companion", "name": "Companion",
             "description": "Answers in the Keeper's voice, grounded in what it keeps.",
             "tags": ["companion", "memory"]},
            {"id": "research", "name": "Research",
             "description": "Delegates to a researcher that searches the web.",
             "tags": ["research", "web"]},
            {"id": "analyze", "name": "Analyze",
             "description": "Works answers out by running code.",
             "tags": ["compute", "code"]},
        ],
    }


# --------------------------------------------------------------------------- #
# message/send envelope — shared by the server (parse in / build out) and client.
# --------------------------------------------------------------------------- #

def text_of(message: dict) -> str:
    """Pull the plain text out of an A2A message's parts."""
    parts = (message or {}).get("parts", []) or []
    return "\n".join(p.get("text", "") for p in parts
                     if p.get("kind", "text") == "text").strip()


def make_message(text: str, role: str = "agent") -> dict:
    return {
        "role": role,
        "messageId": uuid.uuid4().hex,
        "parts": [{"kind": "text", "text": text}],
        "kind": "message",
    }


def rpc_result(req_id, message: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": message}


def rpc_error(req_id, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}


# --------------------------------------------------------------------------- #
# Client — discover a peer, send it a message, read the reply.
# --------------------------------------------------------------------------- #

async def fetch_card(base_url: str, timeout: float = 8.0) -> dict:
    """GET the peer's Agent Card from its well-known URL."""
    url = base_url.rstrip("/") + WELL_KNOWN
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def send_message(endpoint: str, text: str, timeout: float = 60.0) -> str:
    """Send one message/send JSON-RPC call to an A2A endpoint; return the reply text."""
    payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {"message": make_message(text, role="user")},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
    if "error" in data:
        return f"(peer error: {data['error'].get('message', 'unknown')})"
    return text_of(data.get("result", {}))


async def consult(base_url: str, text: str) -> str:
    """The full client flow: discover the peer by its card, then message its endpoint."""
    try:
        card = await fetch_card(base_url)
    except Exception as exc:  # noqa: BLE001
        return f"(could not reach a peer agent at {base_url}: {exc})"
    endpoint = card.get("url") or (base_url.rstrip("/") + "/a2a")
    name = card.get("name", "peer")
    reply = await send_message(endpoint, text)
    return f"[{name}] {reply}" if reply else f"[{name}] (no reply)"
