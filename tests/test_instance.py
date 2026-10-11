"""
These are the Tier 1 tests for the single-instance lock (instance.py): one Keeper per
memory, and a crash never leaves the next one locked out.
"""
import subprocess
import sys

import pytest

import instance

pytestmark = pytest.mark.unit


def test_a_second_keeper_on_the_same_memory_is_refused(tmp_path):
    """The Night of Thirty-Nine Banners, as a test: three servers, three proactive
    loops, three of every gate."""
    first = instance.InstanceLock(tmp_path / "keeper.lock")
    first.acquire(port="8790")
    try:
        with pytest.raises(instance.AlreadyRunning, match="port 8790"):
            instance.InstanceLock(tmp_path / "keeper.lock").acquire(port="8737")
    finally:
        first.release()


def test_the_lock_is_free_again_once_released(tmp_path):
    a = instance.InstanceLock(tmp_path / "keeper.lock")
    a.acquire()
    a.release()
    b = instance.InstanceLock(tmp_path / "keeper.lock")
    b.acquire()
    b.release()


def test_a_killed_keeper_leaves_no_stale_lock(tmp_path):
    """The OS drops a flock when its process dies, however it dies. That is why
    this is a lock and not a pid file: there is nothing to clean up after kill -9."""
    path = tmp_path / "keeper.lock"
    code = ("import os, sys; sys.path.insert(0, sys.argv[1]); import instance; "
            "instance.InstanceLock(__import__('pathlib').Path(sys.argv[2]))"
            ".acquire(); os._exit(9)")
    backend = str(instance.Path(instance.__file__).parent)
    subprocess.run([sys.executable, "-c", code, backend, str(path)], check=False)
    survivor = instance.InstanceLock(path)
    survivor.acquire()
    survivor.release()
