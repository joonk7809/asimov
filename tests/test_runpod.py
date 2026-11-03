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
