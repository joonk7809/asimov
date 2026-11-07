"""The on-pod job runner (scripts/pod_agent.py) and its client (scripts/podjob.py), on localhost."""
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

import pytest

HERE = os.path.dirname(__file__)
AGENT = os.path.join(HERE, "..", "scripts", "pod_agent.py")
spec = importlib.util.spec_from_file_location("podjob", os.path.join(HERE, "..", "scripts", "podjob.py"))
podjob = importlib.util.module_from_spec(spec)
spec.loader.exec_module(podjob)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _agent(root, port):
    p = subprocess.Popen([sys.executable, AGENT, str(port), "k3y", "some/model"],
                         env={**os.environ, "ASIIMOV_JOBS": str(root)})
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return p
        except OSError:
            time.sleep(0.05)
    p.kill()
    raise RuntimeError("agent did not start")


@pytest.fixture
def pod(tmp_path):
    port = _free_port()
    procs = [_agent(tmp_path / "jobs", port)]
    rec = {"jobs_url": f"http://127.0.0.1:{port}", "api_key": "k3y"}

    def restart():
        procs[0].kill()
        procs[0].wait()
        procs[0] = _agent(tmp_path / "jobs", port)

    yield rec, restart, tmp_path / "jobs"
    procs[0].kill()
    procs[0].wait()


def _code(files: dict) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text)
            t.addfile(info, io.BytesIO(text))
    import base64
    return base64.b64encode(buf.getvalue()).decode()


def _until(rec, job, want, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = podjob.request(rec, "GET", f"/jobs/{job}")
        if s["status"] == want:
            return s
        time.sleep(0.05)
    raise AssertionError(f"{job} never reached {want}: {s}")


def test_a_job_runs_its_shipped_code_against_the_pods_model_and_returns_its_outputs(pod, tmp_path):
    rec, _, _ = pod
    podjob.request(rec, "POST", "/jobs", {"name": "j1", "cmd": "mkdir -p out/a+b && python3 run.py > out/a+b/r.txt",
                                          "code": _code({"run.py": b"import os; print(os.environ['LLM_URL'], os.environ['LLM_MODEL'])"})})
    s = _until(rec, "j1", "done")
    assert s["rc"] == 0
    dest = tmp_path / "got"
    sys.argv = ["podjob", "fetch", "x", "j1", "--out", str(dest), "--path", "out/a+b"]
    os.makedirs(podjob.POD_DIR, exist_ok=True)
    with open(os.path.join(podjob.POD_DIR, "x.json"), "w") as f:
        json.dump(rec, f)
    try:
        podjob.main()
    finally:
        os.remove(os.path.join(podjob.POD_DIR, "x.json"))
    assert (dest / "out" / "a+b" / "r.txt").read_text() == "http://127.0.0.1:8000 some/model\n"


def test_the_log_resumes_from_any_offset_and_follow_ends_with_the_job(pod):
    rec, _, _ = pod
    podjob.request(rec, "POST", "/jobs", {"name": "j2", "cmd": "for i in 1 2 3; do echo line$i; sleep 0.2; done; exit 3",
                                          "code": _code({})})
    out = io.BytesIO()
    s = podjob.follow(rec, "j2", 0.05, out)
    assert (s["status"], s["rc"]) == ("done", 3)
    assert out.getvalue() == b"line1\nline2\nline3\n"
    assert podjob.request(rec, "GET", "/jobs/j2/log?offset=6") == b"line2\nline3\n"


def test_a_job_outlives_its_runner_and_is_still_tracked_after_the_runner_restarts(pod):
    rec, restart, root = pod
    # The job waits for the test's go-ahead, not a fixed sleep: on a slow machine the restart alone can
    # outlast a sleep, and then the job is legitimately done before the status check.
    podjob.request(rec, "POST", "/jobs", {"name": "j3", "cmd": "until [ -e go ]; do sleep 0.05; done; echo finished",
                                          "code": _code({})})
    restart()
    assert podjob.request(rec, "GET", "/jobs/j3")["status"] == "running"
    (root / "j3" / "go").touch()
    s = _until(rec, "j3", "done")
    assert s["rc"] == 0 and podjob.request(rec, "GET", "/jobs/j3/log") == b"finished\n"


def test_a_job_killed_without_an_exit_code_is_lost_not_done(pod):
    rec, _, _ = pod
    s = podjob.request(rec, "POST", "/jobs", {"name": "j4", "cmd": "sleep 30", "code": _code({})})
    os.killpg(s["pid"], 9)  # what a container restart does: no shell is left to write the exit code
    _until(rec, "j4", "lost")


def _alive_in_group(pgid):
    return [p for p in os.listdir("/proc") if p.isdigit() and _pgrp(p) == pgid]


def _pgrp(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[2]) if fields[0] != "Z" else None
    except OSError:
        return None


@pytest.mark.parametrize("cmd", ["sleep 30 & sleep 30; wait",
                                 "trap '' TERM; sleep 300",  # ignores TERM: needs the KILL
                                 "python3 -c 'import signal,time; signal.signal(signal.SIGTERM, lambda *a: time.sleep(1.5)); time.sleep(60)'"])
def test_kill_returns_only_once_no_process_of_the_job_is_left(pod, cmd):
    rec, _, _ = pod
    s = podjob.request(rec, "POST", "/jobs", {"name": "j5", "cmd": cmd, "code": _code({})})
    time.sleep(0.3)
    t = podjob.request(rec, "POST", "/jobs/j5/kill", {}, timeout=60)
    assert t["status"] == "done" and t["rc"] != 0
    assert _alive_in_group(s["pid"]) == []


def test_a_job_whose_wrapper_died_but_whose_children_run_on_is_running_not_lost(pod):
    rec, _, _ = pod
    s = podjob.request(rec, "POST", "/jobs", {"name": "j9", "cmd": "sleep 30", "code": _code({})})
    os.kill(s["pid"], 9)  # only the wrapper; its sleep child keeps the group alive
    time.sleep(0.2)
    assert podjob.request(rec, "GET", "/jobs/j9")["status"] == "running"
    podjob.request(rec, "POST", "/jobs/j9/kill", {}, timeout=60)
    assert _alive_in_group(s["pid"]) == []


def test_refuses_bad_keys_reused_names_and_paths_out_of_the_job(pod):
    rec, _, root = pod
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(rec["jobs_url"] + "/jobs", headers={"Authorization": "Bearer no"}))
    assert e.value.code == 401
    podjob.request(rec, "POST", "/jobs", {"name": "j6", "cmd": "true", "code": _code({})})
    for body, path in (({"name": "j6", "cmd": "true", "code": _code({})}, "/jobs"),
                       ({"name": "j7", "cmd": "true", "code": _code({"../escape": b"x"})}, "/jobs"),
                       ({"name": "../j8", "cmd": "true", "code": _code({})}, "/jobs")):
        with pytest.raises(SystemExit):
            podjob.request(rec, "POST", path, body)
    assert not (root.parent / "escape").exists() and not (root / "escape").exists()
    with pytest.raises(SystemExit):
        podjob.request(rec, "GET", "/jobs/j6/tar?path=../../")


def test_a_job_that_ends_while_its_status_is_read_is_done_not_lost(tmp_path, monkeypatch):
    # The CI race on c81971c: exit missing, then the wrapper writes it and exits before the process scan.
    spec = importlib.util.spec_from_file_location("pod_agent", AGENT)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    monkeypatch.setattr(agent, "ROOT", str(tmp_path))
    (tmp_path / "j").mkdir()
    (tmp_path / "j" / "job.json").write_text(json.dumps({"id": "j", "cmd": "x", "pid": 1, "pid_start": "b:1"}))

    def wrapper_finishes(pgid, start):
        (tmp_path / "j" / "exit").write_text("0\n")
        return []

    monkeypatch.setattr(agent, "_members", wrapper_finishes)
    assert (agent.status("j")["status"], agent.status("j")["rc"]) == ("done", 0)
