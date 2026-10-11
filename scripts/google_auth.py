"""
This connects the Keeper to one Google account, once.

The google server in mcp.json authenticates the first time a tool is called: it
answers with a consent URL instead of a result, and catches the redirect on its own
small callback server (localhost:8000). Left to happen inside a conversation, that
URL arrives in a tool result the Keeper then re-voices, which is no way to hand
someone an OAuth link. So this does it out of band: it launches the same server
with the same configuration the app uses, starts the flow, opens the consent page,
and waits until a real read succeeds.

The token is kept by the server in ~/.google_workspace_mcp/credentials, outside
this repo. Run it again to re-consent (a token from a project still in "Testing"
expires after seven days, which is the price of skipping Google's app review for
an app with one user).

    .venv/bin/python scripts/google_auth.py            # wait up to 5 minutes
    .venv/bin/python scripts/google_auth.py --wait 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import webbrowser
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")
except ImportError:
    pass

import tools  # noqa: E402

SERVER = "google"
_URL = re.compile(r"https://accounts\.google\.com/[^\s)\]>\"']+")


def _spec() -> dict:
    try:
        data = json.loads(tools.CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"cannot read {tools.CONFIG_PATH}: {exc}")
    for spec in data.get("servers", []):
        if spec.get("name") == SERVER:
            return spec
    sys.exit(f'no "{SERVER}" server in {tools.CONFIG_PATH}; copy it from '
             f"mcp.example.json")


def _needs_consent(text: str) -> bool:
    return bool(_URL.search(text or "")) or "authenticat" in (text or "").lower()


async def main(wait_s: float) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    spec = _spec()
    unset: set = set()
    env = tools._expand_env(spec.get("env") or {}, unset)
    fixed = tools._expand_env(spec.get("fixed_args") or {}, unset)
    unset |= {v for v in spec.get("requires_env") or [] if not os.environ.get(v)}
    if unset:
        sys.exit(f"set these in backend/.env first: {', '.join(sorted(unset))}")
    email = fixed.get("user_google_email", "")

    params = StdioServerParameters(command=spec["command"],
                                   args=tools._expand(spec.get("args", [])), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            print("· starting the google server (the first run downloads it)…")
            await asyncio.wait_for(session.initialize(), tools.STARTUP_TIMEOUT_S)

            async def probe() -> str:
                res = await session.call_tool(
                    "search_gmail_messages",
                    {"query": "in:inbox", "page_size": 1, **fixed})
                return "\n".join(getattr(b, "text", "") for b in res.content or [])

            first = await probe()
            if not _needs_consent(first):
                print(f"· already connected as {email}. Nothing to do.")
                return 0

            res = await session.call_tool(
                "start_google_auth",
                {"service_name": "Gmail and Calendar", "user_google_email": email})
            text = "\n".join(getattr(b, "text", "") for b in res.content or [])
            url = _URL.search(text)
            if url is None:
                print(text)
                print("· no consent URL came back; see the message above.")
                return 1
            print(f"· open this to let the Keeper act as {email}:\n\n  "
                  f"{url.group(0)}\n")
            webbrowser.open(url.group(0))

            loop = asyncio.get_running_loop()
            deadline = loop.time() + wait_s
            while loop.time() < deadline:
                await asyncio.sleep(5)
                if not _needs_consent(await probe()):
                    print(f"· connected. The Keeper can now read {email}'s mail and "
                          f"calendar, and propose what you approve.")
                    return 0
            print("· timed out waiting for consent. Run this again when ready.")
            return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--wait", type=float, default=300.0,
                    help="seconds to wait for you to finish consent")
    sys.exit(asyncio.run(main(ap.parse_args().wait)))
