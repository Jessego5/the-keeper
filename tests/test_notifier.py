"""
These are the Tier 1 tests for which Keeper the native banner wears (notifier.py).

Pure path selection: no banner is ever fired here, and the bundles are faked on
disk, so this runs anywhere.
"""
import pytest
import notifier

pytestmark = pytest.mark.unit



def test_each_register_picks_its_own_bundle(tmp_path, monkeypatch):
    """The badge is baked into the .app, so wearing a register means choosing a
    bundle. Build them all and each state must select its own."""
    monkeypatch.setattr(notifier, "_NOTIFY_DIR", tmp_path)
    for app in notifier.STATE_APPS.values():
        binary = tmp_path / app / "Contents" / "MacOS" / "terminal-notifier"
        binary.parent.mkdir(parents=True)
        binary.write_text("")
    for state, app in notifier.STATE_APPS.items():
        assert notifier._binary(state) == str(
            tmp_path / app / "Contents" / "MacOS" / "terminal-notifier")


def test_a_register_with_no_bundle_falls_back(tmp_path, monkeypatch):
    """A partial build must not silence the Keeper: any register whose bundle is
    missing falls back to the plain Keeper.app rather than failing."""
    monkeypatch.setattr(notifier, "_NOTIFY_DIR", tmp_path)
    keeper = tmp_path / "Keeper.app" / "Contents" / "MacOS" / "terminal-notifier"
    keeper.parent.mkdir(parents=True)
    keeper.write_text("")
    monkeypatch.setattr(notifier, "KEEPER_APP_BIN", keeper)
    assert notifier._binary("frozen") == str(keeper)
    assert notifier._binary("nonsense-register") == str(keeper)


def test_unknown_register_falls_back_to_the_default_icon():
    assert notifier._icon_for("nonsense-register") == notifier.ICON
    assert notifier._icon_for(None) == notifier.ICON


def _cmd_for(monkeypatch, state, *, bundles=(), keeper=True, tmp_path=None):
    """Capture the argv notify() would run, without firing anything."""
    seen = {}
    monkeypatch.setattr(notifier, "_IS_MAC", True)
    monkeypatch.setattr(notifier, "_NOTIFY_DIR", tmp_path)
    for app in bundles:
        b = tmp_path / app / "Contents" / "MacOS" / "terminal-notifier"
        b.parent.mkdir(parents=True); b.write_text("")
    kb = tmp_path / "Keeper.app" / "Contents" / "MacOS" / "terminal-notifier"
    if keeper:
        kb.parent.mkdir(parents=True); kb.write_text("")
    monkeypatch.setattr(notifier, "KEEPER_APP_BIN", kb)
    monkeypatch.setattr(notifier.subprocess, "run",
                        lambda cmd, **kw: seen.update(cmd=cmd))
    notifier.notify("The Keeper", "a line", state)
    return seen.get("cmd", [])


def test_a_state_bundle_needs_no_thumbnail(tmp_path, monkeypatch):
    """It is already wearing that register as its badge."""
    cmd = _cmd_for(monkeypatch, "turn", bundles=[notifier.STATE_APPS["turn"]],
                   tmp_path=tmp_path)
    assert "-contentImage" not in cmd


def test_default_bundle_carries_the_register_as_a_thumbnail(tmp_path, monkeypatch):
    """No bundle for this register, so the badge shows the default pose: the
    thumbnail is the only place the register can appear."""
    cmd = _cmd_for(monkeypatch, "turn", tmp_path=tmp_path)
    assert "-contentImage" in cmd
    assert cmd[cmd.index("-contentImage") + 1].endswith("keeper_turning.png")


def test_no_register_means_no_duplicate_keeper(tmp_path, monkeypatch):
    """REGRESSION: with no state the thumbnail resolved to the same keeper.png the
    badge already shows, printing the figure twice in one banner."""
    cmd = _cmd_for(monkeypatch, None, tmp_path=tmp_path)
    assert "-contentImage" not in cmd
