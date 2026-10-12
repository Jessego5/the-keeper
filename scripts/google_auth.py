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

# Where the server keeps the token once consent completes (google_auth.py in
# workspace-mcp). Watching this directory is how the wait below learns you are done.
CREDS_DIR = Path(os.path.expanduser(
    os.environ.get("WORKSPACE_MCP_CREDENTIALS_DIR")
    or os.environ.get("GOOGLE_MCP_CREDENTIALS_DIR")
    or "~/.google_workspace_mcp/credentials"))


def _say(text: str) -> None:
    # Flushed: under a pipe, buffered output hid the link until the script exited.
    print(text, flush=True)


def _newest_token_mtime() -> float:
    return max((f.stat().st_mtime for f in CREDS_DIR.glob("*.json")), default=0.0)


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

    # The server opens a browser window for every sign-in flow it starts. This
    # script prints one link instead, for a private window signed in as the right
    # account: a window that pops up in the default browser may well be signed in
    # as the wrong one.
    env = {**env, "BROWSER": "true"}
    params = StdioServerParameters(command=spec["command"],
                                   args=tools._expand(spec.get("args", [])), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            _say("· starting the google server (the first run downloads it)…")
            await asyncio.wait_for(session.initialize(), tools.STARTUP_TIMEOUT_S)

            async def probe() -> str:
                res = await session.call_tool(
                    "search_gmail_messages",
                    {"query": "in:inbox", "page_size": 1, **fixed})
                return "\n".join(getattr(b, "text", "") for b in res.content or [])

            # Probe only if a token already exists. Without one, a probe does
            # nothing but start a sign-in flow of its own.
            if _newest_token_mtime() and not _needs_consent(await probe()):
                _say(f"· already connected as {email}. Nothing to do.")
                return 0

            res = await session.call_tool(
                "start_google_auth",
                {"service_name": "Gmail and Calendar", "user_google_email": email})
            text = "\n".join(getattr(b, "text", "") for b in res.content or [])
            url = _URL.search(text)
            if url is None:
                _say(text)
                _say("· no consent URL came back; see the message above.")
                return 1
            _say(f"· open this in a PRIVATE window, signed in as {email}:\n\n  "
                 f"{url.group(0)}\n")
            _say("· waiting for you to finish. Nothing will pop up meanwhile; "
                 "keep this running until it says connected.")

            # Wait by watching for the token file, not by calling Gmail: every call
            # made before consent completes starts another flow, and each flow used
            # to open another browser window over the one being signed in to.
            started = _newest_token_mtime()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + wait_s
            while loop.time() < deadline:
                await asyncio.sleep(2)
                if _newest_token_mtime() > started:
                    await asyncio.sleep(1)       # let the server finish writing it
                    if not _needs_consent(await probe()):
                        _say(f"· connected. The Keeper can now read {email}'s mail "
                             f"and calendar, and propose what you approve.")
                        return 0
                    _say("· a token arrived but Gmail still refused it; run this "
                         "again.")
                    return 1
            _say("· timed out waiting for consent. Run this again when ready.")
            return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--wait", type=float, default=300.0,
                    help="seconds to wait for you to finish consent")
    sys.exit(asyncio.run(main(ap.parse_args().wait)))
