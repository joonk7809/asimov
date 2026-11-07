"""Job runner that lives inside an asiimov pod, so a long run doesn't depend on the machine that started it.

scripts/runpod_pod.py up starts it next to vLLM (port 8001); scripts/podjob.py talks to it. A job is a code
tarball plus a shell command. It runs in its own session under ROOT/<id>/, and every bit of its state is a
file there (pid, exit code, log), so the operator can die, reconnect and pick up the log where it stopped,
and this agent can restart without losing track of a running job. A job whose process is gone without an
exit code is reported as "lost" (the pod's container restarted under it), never as finished.

    python3 pod_agent.py PORT KEY MODEL     # KEY guards every request; MODEL is what vLLM serves on :8000
Stdlib only: it runs on the vLLM image's Python.
"""
import base64
import hmac
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.environ.get("ASIIMOV_JOBS", "/workspace/jobs")
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def status(job: str) -> dict:
    d = os.path.join(ROOT, job)
    with open(os.path.join(d, "job.json")) as f:
        out = json.load(f)
    try:
        with open(os.path.join(d, "exit")) as f:
            out.update(status="done", rc=int(f.read().strip()), ended=os.path.getmtime(os.path.join(d, "exit")))
        return out
    except (FileNotFoundError, ValueError):
        pass  # no exit code yet, or a torn write: decide by the process
    # Alive means the same process: after a container restart the pid can belong to someone else.
    out["status"] = "running" if _started(out["pid"]) == out.get("pid_start") else "lost"
    return out


def _started(pid: int) -> str | None:
    """The process's start time in clock ticks since boot, plus the boot id; None if it isn't running."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()  # the name in parentheses may hold spaces
        if fields[0] == "Z":  # exited; after a runner restart nobody may reap it (pid 1 is vLLM)
            return None
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip() + ":" + fields[19]
    except (OSError, IndexError):
        return None


def start(name: str, cmd: str, code: bytes, env: dict) -> dict:
    if not NAME.match(name):
        raise ValueError("job name: lowercase letters, digits, '.', '_' or '-', at most 64")
    d = os.path.join(ROOT, name)
    os.makedirs(ROOT, exist_ok=True)
    os.mkdir(d)  # FileExistsError: a name is used once, so a resubmit can't clobber a run's outputs
    with tarfile.open(fileobj=io.BytesIO(code), mode="r:gz") as t:
        t.extractall(d, filter="data")  # refuses absolute paths, '..' and links out of d
    # The outer shell writes the exit code once the command ends, even if this agent is gone by then.
    wrapped = f"({cmd}) </dev/null >>log 2>&1; echo $? > exit.tmp && mv exit.tmp exit"
    p = subprocess.Popen(["bash", "-c", wrapped], cwd=d, env={**os.environ, **env}, start_new_session=True)
    threading.Thread(target=p.wait, daemon=True).start()  # reap it, so a finished job isn't left a zombie
    rec = {"id": name, "cmd": cmd, "pid": p.pid, "pid_start": _started(p.pid), "started": time.time()}
    with open(os.path.join(d, "job.json"), "w") as f:
        json.dump(rec, f)
    return status(name)


def kill(job: str) -> dict:
    """TERM the job's whole process group, KILL it after 10 s, and record the exit code the shell couldn't."""
    s = status(job)
    if s["status"] != "running":
        return s
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(s["pid"], sig)
        except ProcessLookupError:
            break
        for _ in range(100):
            if _started(s["pid"]) != s["pid_start"]:
                break
            time.sleep(0.1)
        else:
            continue
        break
    path = os.path.join(ROOT, job, "exit")
    if not os.path.exists(path):
        with open(path + ".tmp", "w") as f:
            f.write("143\n")  # 128 + SIGTERM, as a shell would report it
        os.replace(path + ".tmp", path)
    return status(job)


def pack(job: str, sub: str) -> bytes:
    d = os.path.realpath(os.path.join(ROOT, job))
    path = os.path.realpath(os.path.join(d, sub))
    if path != d and not path.startswith(d + os.sep):
        raise ValueError("path leaves the job directory")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(path, arcname=os.path.relpath(path, d))
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    key = ""
    env: dict = {}

    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _route(self, method: str):
        if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {self.key}"):
            return self._send(401, {"error": "unauthorized"})
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [p for p in u.path.split("/") if p]
        try:
            if parts[:1] != ["jobs"] or len(parts) > 3 or (len(parts) > 1 and not NAME.match(parts[1])):
                return self._send(404, {"error": "not found"})
            if method == "GET" and len(parts) == 1:
                jobs = sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []
                return self._send(200, [status(j) for j in jobs if os.path.exists(os.path.join(ROOT, j, "job.json"))])
            if method == "POST" and len(parts) == 1:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                return self._send(201, start(body["name"], body["cmd"], base64.b64decode(body["code"]), self.env))
            job = parts[1]
            if method == "GET" and len(parts) == 2:
                return self._send(200, status(job))
            if method == "GET" and parts[2] == "log":
                with open(os.path.join(ROOT, job, "log"), "rb") as f:
                    f.seek(int(q.get("offset", 0)))
                    return self._send(200, f.read(int(q.get("limit", 1 << 20))), "application/octet-stream")
            if method == "GET" and parts[2] == "tar":
                return self._send(200, pack(job, q.get("path", ".")), "application/gzip")
            if method == "POST" and parts[2] == "kill":
                return self._send(200, kill(job))
            return self._send(404, {"error": "not found"})
        except FileNotFoundError as e:
            return self._send(404, {"error": str(e)})
        except FileExistsError:
            return self._send(409, {"error": "a job with that name exists"})
        except (ValueError, KeyError, TypeError, tarfile.TarError) as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def log_message(self, *a):
        pass


def main():
    port, key, model = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    Handler.key = key
    # What a job needs to reach this pod's model without the public proxy.
    Handler.env = {"LLM_URL": "http://127.0.0.1:8000", "LLM_MODEL": model, "LLM_KEY": key}
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
