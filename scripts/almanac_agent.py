"""almanac_agent.py — a small A2A peer for the Keeper to actually talk to.

The Keeper implements both halves of A2A, but until now the only agent it could
consult was itself. A loopback proves the plumbing and nothing else: two halves of
one module agreeing about a format they both define.

This is a genuinely separate agent. It shares no code with backend/a2a.py — the
JSON-RPC envelope here is written from the spec, not imported — so a successful
consult is two independent implementations interoperating, which is the only thing
that actually demonstrates a protocol.

It is deliberately unlike the Keeper:

  * it has no memory of anyone, no voice, no proactivity, and no model behind it
  * it answers from a fixed reference table, so it is deterministic and free
  * it knows things the Keeper does not, which is what makes consulting it
    meaningful rather than decorative
  * it needs no auth, because it holds nothing private. The Keeper's own /a2a
    requires a bearer token precisely because answering means reading someone's
    memory. Two agents, two honest postures.

Run:
    .venv/bin/python -m uvicorn scripts.almanac_agent:app --port 8791

Then, from the Keeper:
    consult the agent at http://localhost:8791 and ask what gouache is

That requires the Keeper to allow the origin, since it is loopback:
    KEEPER_A2A_ALLOW=http://localhost:8790,http://localhost:8791
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request

PORT_HINT = 8791
NAME = "The Almanac"

# A painter's reference. Small on purpose: the point is that it knows things the
# Keeper's memory does not, not that it knows many things.
ENTRIES: dict[str, str] = {
    "gouache": ("Gouache is watercolour made opaque, usually with added chalk. It "
                "dries matte and flat, can be repainted over, and reactivates with "
                "water long after it has dried."),
    "watercolour": ("Watercolour is transparent: the white of the paper does the "
                    "lighting, not the paint. Mistakes are hard to lift, which is "
                    "why it rewards planning the lightest areas first."),
    "gesso": ("Gesso primes a surface so paint sits on it rather than sinking in. "
              "Two thin coats, sanded between, beat one thick one."),
    "ultramarine": ("Ultramarine is a warm, slightly violet blue, historically "
                    "ground from lapis lazuli and once costlier than gold."),
    "sizing": ("Sizing is the gelatine or synthetic layer in watercolour paper that "
               "stops it drinking paint like blotting paper."),
    "impasto": ("Impasto is paint laid thick enough to hold the mark of the brush "
                "or knife, so the surface itself carries the light."),
}


def _answer(question: str) -> str:
    q = (question or "").lower()
    hits = [text for term, text in ENTRIES.items() if term in q]
    if hits:
        return "\n\n".join(hits)
    return (f"{NAME} keeps entries on: " + ", ".join(sorted(ENTRIES)) +
            ". Ask about one of those and I will read it back.")


app = FastAPI(title=NAME)


@app.get("/.well-known/agent.json")
async def agent_card() -> dict:
    """Discovery. A peer reads this to learn who this is and where to send to."""
    return {
        "protocolVersion": "0.2.0",
        "name": NAME,
        "description": ("A painter's reference. Answers from a fixed table of "
                        "materials and techniques. Keeps no memory of anyone."),
        "url": f"http://localhost:{PORT_HINT}/a2a",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [{
            "id": "reference",
            "name": "Materials reference",
            "description": "Explains a pigment, a surface, or a technique.",
            "tags": ["painting", "materials", "reference"],
        }],
    }


@app.post("/a2a")
async def a2a(req: Request) -> dict:
    """One synchronous message/send. No auth: this holds nothing private."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "parse error"}}
    req_id = body.get("id")
    if body.get("method") != "message/send":
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601,
                          "message": f"unsupported method: {body.get('method')}"}}
    parts = ((body.get("params") or {}).get("message") or {}).get("parts") or []
    text = "\n".join(p.get("text", "") for p in parts
                     if p.get("kind", "text") == "text").strip()
    if not text:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": "empty message"}}
    return {"jsonrpc": "2.0", "id": req_id, "result": {
        "role": "agent",
        "messageId": uuid.uuid4().hex,
        "parts": [{"kind": "text", "text": _answer(text)}],
        "kind": "message",
    }}
