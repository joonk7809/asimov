"""Spend safety of scripts/runpod_pod.py, with the RunPod API faked."""
import importlib.util
import os
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "runpod_pod", os.path.join(os.path.dirname(__file__), "..", "scripts", "runpod_pod.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def fake_api(monkeypatch, pods, fail=()):
    deleted = []

    def call(method, path, body=None):
        if method == "GET":
            return pods
        pid = path.rsplit("/", 1)[1]
        if pid in fail:
            raise SystemExit(f"RunPod DELETE {path} -> 500")
        deleted.append(pid)

    monkeypatch.setattr(rp, "call", call)
    return deleted


def test_down_all_only_touches_pods_this_script_started(monkeypatch):
    deleted = fake_api(monkeypatch, [{"id": "a", "name": "asiimov-qwen3-4b"}, {"id": "b", "name": "joons-other-job"},
                                     {"id": "c", "name": None}])
    rp.down(SimpleNamespace(all=True, pod=None))
    assert deleted == ["a"]


def test_down_keeps_going_after_a_failed_delete_and_exits_nonzero(monkeypatch):
    deleted = fake_api(monkeypatch, [{"id": "a", "name": "asiimov-x"}, {"id": "b", "name": "asiimov-y"}], fail={"a"})
    with pytest.raises(SystemExit, match="a"):
        rp.down(SimpleNamespace(all=True, pod=None))
    assert deleted == ["b"]


def test_ttl_cmd_is_a_background_prefix_carrying_the_terminate_script():
    import base64
    import subprocess
    cmd = rp.ttl_cmd(1793750000)
    assert cmd.startswith("(sleep $((1793750000 - $(date +%s)));") and cmd.rstrip().endswith("&")
    script = base64.b64decode(cmd.split("echo ")[1].split(" |")[0]).decode()
    assert script == rp.TTL_SCRIPT
    compile(script, "ttl", "exec")
    # Valid shell when followed by the server command it prefixes.
    assert subprocess.run(["bash", "-n", "-c", cmd + "exec vllm serve x"]).returncode == 0
