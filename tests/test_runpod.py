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
        if method == "GET":  # REST v2's page shape
            return {"pods": pods, "pagination": {"hasNextPage": False, "nextCursor": None}}
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


def test_pods_list_follows_the_cursor(monkeypatch):
    pages = {"/pods": {"pods": [{"id": "a"}], "pagination": {"hasNextPage": True, "nextCursor": "x"}},
             "/pods?cursor=x": {"pods": [{"id": "b"}], "pagination": {"hasNextPage": False, "nextCursor": None}}}
    monkeypatch.setattr(rp, "call", lambda method, path, body=None: pages[path])
    assert [p["id"] for p in rp.pods_list()] == ["a", "b"]


def test_up_tries_the_next_gpu_type_when_one_is_refused(monkeypatch, tmp_path):
    seen = []

    def call(method, path, body=None):
        seen.append(body["gpu"]["id"])
        if body["gpu"]["id"] == "G1":
            raise rp.RunPodError("RunPod POST /pods -> 409: no capacity", 409)
        return {"id": "p1", "cost": 1.0, "gpu": {"id": body["gpu"]["id"]}}

    monkeypatch.setattr(rp, "call", call)
    monkeypatch.setattr(rp, "POD_DIR", str(tmp_path))
    rp.up(SimpleNamespace(model="m/x", gpu=["G1", "G2"], cloud="SECURE", max_len=8192, vllm_args="", name="t",
                          port=[], ttl_hours=0.1, stall_s=180))
    assert seen == ["G1", "G2"]


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


@pytest.mark.parametrize("incomplete,trips", [(True, True), (False, False)])
def test_stall_watchdog_fires_only_on_a_stuck_download(monkeypatch, tmp_path, incomplete, trips):
    """2026-11-17: a shard sat .incomplete at 29.7 of ~33 GB and vLLM never served. The watchdog terminates the
    pod when a weight file is still incomplete and the cache stops growing; a finished download never trips it."""
    import subprocess
    cache = tmp_path / "hf"
    cache.mkdir()
    (cache / ("shard.incomplete" if incomplete else "shard.safetensors")).write_bytes(b"x" * 10)
    monkeypatch.setattr(rp, "HF_CACHE", str(cache))
    monkeypatch.setattr(rp, "TTL_SCRIPT", "print('TERMINATE')")
    cmd = rp.stall_cmd(2, every=1).rstrip().rstrip("&")  # the loop in the foreground
    try:
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=6).stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    assert ("TERMINATE" in out and "stalled" in out) == trips, out
