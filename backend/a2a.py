"""
This is the Agent-to-Agent (A2A) protocol, the Keeper as a peer among agents.

MCP lets the Keeper use TOOLS; A2A lets it interoperate with other AGENTS. It is an
open standard (Google's A2A, now under the Linux Foundation): an agent publishes an
**Agent Card** at /.well-known/agent.json describing who it is and what it can do, and
exposes an endpoint that accepts a **message/send** JSON-RPC call and returns the reply.

This module is both halves of a faithful core subset. build_agent_card() builds
the Keeper's own card, served so other agents can discover it; the envelope
helpers carry a message/send call in either direction, as server and as client;
and consult() is the client end, discovering a remote agent by its card, sending
it a message and reading the reply.

Scope kept honest: the synchronous message/send core, text parts, no streaming or
push-notification task states: the parts that prove interoperability without the
distributed-task machinery a single-user companion doesn't need.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import uuid
from urllib.parse import urlparse

import httpx

WELL_KNOWN = "/.well-known/agent.json"
PROTOCOL_VERSION = "0.2.0"


def build_agent_card(base_url: str) -> dict:
    """The Keeper's A2A Agent Card. `base_url` is where it's reachable (no trailing /)."""
    base = base_url.rstrip("/")
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "name": "The Keeper",
        "description": ("A proactive companion agent, keeps long-term memory, pursues "
                        "goals, and can research, compute, and dig through its own "
                        "materials via specialist sub-agents."),
        "url": f"{base}/a2a",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        # Declared so a peer knows to present a token rather than discovering it
        # from a 401. Omitted when unconfigured, since then the endpoint is closed.
        **({"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
            "security": [{"bearer": []}]} if auth_token() else {}),
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
# message/send envelope: shared by the server (parse in / build out) and client.
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
# Client: discover a peer, send it a message, read the reply.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Outbound safety. consult_peer takes a URL the MODEL chose from the conversation,
# and the Keeper reads web pages, so a prompt injection in a page can name one.
# Unchecked, that is a readable SSRF: the response comes back into the chat. It is
# also a second path from untrusted text to arbitrary network requests, alongside
# the fetch -> run_python chain sandbox.py documents: and unlike run_python this
# one runs in the app process, so containerising does not contain it.
# --------------------------------------------------------------------------- #

ALLOW_ENV = "KEEPER_A2A_ALLOW"
# Shared secret for the SERVER half. /a2a answers as the Keeper, and answering
# means recalling memory, so an open endpoint lets any local process ask what it
# knows about the person. TrustedHostMiddleware checks the Host HEADER, not the
# source, so it stops a browser elsewhere and nothing else.
AUTH_ENV = "KEEPER_A2A_TOKEN"


def auth_token() -> str:
    return os.environ.get(AUTH_ENV, "").strip()


def outbound_headers(url: str) -> dict:
    """Bearer header for a peer we are consulting, when there is one to send.

    Sent ONLY to origins the operator explicitly allowlisted, which in practice
    means the loopback peer. Presenting our own secret to an arbitrary public agent
    would hand it our credential for nothing.
    """
    tok = auth_token()
    if not tok:
        return {}
    parsed = urlparse(url or "")
    origin = f"{parsed.scheme}://{parsed.netloc}".lower().rstrip("/")
    return {"Authorization": f"Bearer {tok}"} if origin in _allowlist() else {}


class PeerBlocked(Exception):
    """An outbound peer URL that is not safe to fetch."""


def _allowlist() -> set:
    """Origins the operator has explicitly permitted, e.g. a loopback peer."""
    raw = os.environ.get(ALLOW_ENV, "")
    return {o.strip().rstrip("/").lower() for o in raw.split(",") if o.strip()}


def check_peer_url(url: str) -> str:
    """Return `url` if it is safe to fetch, else raise PeerBlocked.

    Public addresses are allowed: consulting a real agent on the internet is the
    point of the protocol. Private, loopback, link-local and reserved addresses are
    refused unless their origin is listed in KEEPER_A2A_ALLOW, because those are
    what an injection reaches for: cloud metadata at 169.254.169.254, a database on
    localhost, anything on the LAN.

    Resolution happens here rather than trusting the hostname, so a name that
    resolves to 127.0.0.1 is caught too.
    """
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https"):
        raise PeerBlocked(f"only http and https are allowed, not {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise PeerBlocked("no host in the peer URL")

    origin = f"{parsed.scheme}://{parsed.netloc}".lower().rstrip("/")
    if origin in _allowlist():
        return url

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise PeerBlocked(f"could not resolve {host}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise PeerBlocked(
                f"{host} resolves to the non-public address {ip}. Add "
                f"{origin} to {ALLOW_ENV} if you meant to reach it.")
    return url


async def fetch_card(base_url: str, timeout: float = 8.0) -> dict:
    """GET the peer's Agent Card from its well-known URL."""
    url = await asyncio.to_thread(check_peer_url, base_url.rstrip("/") + WELL_KNOWN)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers=outbound_headers(url))
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
    # Checked again: the endpoint comes from the peer's own card, so a hostile card
    # could otherwise redirect a permitted base URL at an internal address.
    endpoint = await asyncio.to_thread(check_peer_url, endpoint)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, json=payload,
                                 headers=outbound_headers(endpoint))
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
