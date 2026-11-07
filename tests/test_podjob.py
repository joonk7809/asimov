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
    rec, restart, _ = pod
    podjob.request(rec, "POST", "/jobs", {"name": "j3", "cmd": "sleep 1; echo finished", "code": _code({})})
    restart()
    assert podjob.request(rec, "GET", "/jobs/j3")["status"] == "running"
    s = _until(rec, "j3", "done")
    assert s["rc"] == 0 and podjob.request(rec, "GET", "/jobs/j3/log") == b"finished\n"


def test_a_job_killed_without_an_exit_code_is_lost_not_done(pod):
    rec, _, _ = pod
    s = podjob.request(rec, "POST", "/jobs", {"name": "j4", "cmd": "sleep 30", "code": _code({})})
    os.killpg(s["pid"], 9)  # what a container restart does: no shell is left to write the exit code
    _until(rec, "j4", "lost")


def test_kill_stops_the_whole_job(pod):
    rec, _, _ = pod
    podjob.request(rec, "POST", "/jobs", {"name": "j5", "cmd": "sleep 30 & sleep 30; wait", "code": _code({})})
    podjob.request(rec, "POST", "/jobs/j5/kill", {})
    s = _until(rec, "j5", "done")
    assert s["rc"] != 0


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
