"""Run a command on a pod and collect what it writes, without keeping this machine alive for it.

The pod runs the job (scripts/pod_agent.py, started by runpod_pod.py up). This side only submits, reads and
downloads, and holds no state of its own, so after a crash or a restart here you just ask again:

    python scripts/podjob.py run POD t1m-4b-full 'python -m asiimov.scorecard --mind llm \\
        --llm-url $LLM_URL --llm-model $LLM_MODEL --llm-key $LLM_KEY --arms full --out out' 
    python scripts/podjob.py ls POD
    python scripts/podjob.py log POD JOB [--follow]     # --follow returns when the job ends; rc = the job's
    python scripts/podjob.py fetch POD JOB --out runs/t1m [--path out]
    python scripts/podjob.py kill POD JOB

`run` ships `git archive HEAD` (the commit is recorded in the job name's record; a dirty tree is refused
unless --allow-dirty) plus any --add files, and runs the command with bash in the unpacked tree. On the pod
LLM_URL, LLM_MODEL and LLM_KEY point at its own vLLM on localhost, so jobs skip the public proxy.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(__file__))
from runpod_pod import AGENT_PORT, POD_DIR  # noqa: E402

UA = "asiimov-podjob/0.1"  # RunPod's Cloudflare refuses urllib's default User-Agent


def pod_record(pod: str) -> dict:
    with open(os.path.join(POD_DIR, f"{pod}.json")) as f:
        rec = json.load(f)
    rec.setdefault("jobs_url", f"https://{pod}-{AGENT_PORT}.proxy.runpod.net")
    return rec


def request(rec: dict, method: str, path: str, body: dict | None = None, timeout: float = 120, tries: int = 4):
    """One call to the pod's agent. Network errors and 5xx are retried (the proxy drops calls now and then);
    a 4xx is the agent's answer and exits with its message."""
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(tries):
        req = urllib.request.Request(rec["jobs_url"] + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {rec['api_key']}", "User-Agent": UA,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if r.headers.get("Content-Type") == "application/json" else raw
        except urllib.error.HTTPError as e:
            if e.code < 500:
                sys.exit(f"{method} {path} -> {e.code}: {e.read()[:500].decode(errors='replace')}")
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = e
        if attempt + 1 < tries:
            time.sleep(2 ** (attempt + 1))
    sys.exit(f"{method} {path} failed after {tries} tries: {err}")


def code_tarball(add: list[str], allow_dirty: bool) -> tuple[bytes, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not allow_dirty:
        sys.exit("tracked files differ from HEAD; commit them (the job records the commit) or pass --allow-dirty")
    tree = subprocess.run(["git", "archive", "--format=tar", "HEAD"], capture_output=True, check=True).stdout
    buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(tree)) as src, tarfile.open(fileobj=buf, mode="w:gz") as out:
        for m in src.getmembers():
            out.addfile(m, src.extractfile(m) if m.isfile() else None)
        for path in add:
            out.add(path, arcname=os.path.normpath(path))
        info = tarfile.TarInfo("COMMIT")
        stamp = (commit + (" dirty" if dirty else "") + "\n").encode()
        info.size = len(stamp)
        out.addfile(info, io.BytesIO(stamp))
    return buf.getvalue(), commit + ("+dirty" if dirty else "")


def follow(rec: dict, job: str, poll: float, out=sys.stdout.buffer) -> dict:
    """Print the job's log from the start until it ends. Safe to rerun at any time."""
    offset = 0
    while True:
        s = request(rec, "GET", f"/jobs/{job}")
        while True:  # drain everything written so far, then check status again
            chunk = request(rec, "GET", f"/jobs/{job}/log?offset={offset}")
            if not chunk:
                break
            out.write(chunk)
            out.flush()
            offset += len(chunk)
        if s["status"] != "running":
            return s
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("pod")
    p.add_argument("job")
    p.add_argument("--add", nargs="*", default=[], help="extra files to ship, at the same relative paths")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("command", help="one shell string, run with bash on the pod (single-quote it here)")
    p = sub.add_parser("ls")
    p.add_argument("pod")
    for name in ("status", "kill"):
        p = sub.add_parser(name)
        p.add_argument("pod")
        p.add_argument("job")
    p = sub.add_parser("log")
    p.add_argument("pod")
    p.add_argument("job")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--poll", type=float, default=15)
    p = sub.add_parser("fetch")
    p.add_argument("pod")
    p.add_argument("job")
    p.add_argument("--out", required=True)
    p.add_argument("--path", default=".", help="part of the job directory to fetch")
    a = ap.parse_args()
    rec = pod_record(a.pod)
    if a.cmd == "run":
        code, commit = code_tarball(a.add, a.allow_dirty)
        s = request(rec, "POST", "/jobs", {"name": a.job, "cmd": a.command,
                                           "code": base64.b64encode(code).decode()}, tries=1)
        print(json.dumps({**s, "commit": commit}))
    elif a.cmd == "ls":
        for s in request(rec, "GET", "/jobs"):
            print(s["id"], s["status"], s.get("rc", ""), time.strftime("%H:%MZ", time.gmtime(s["started"])), s["cmd"][:80])
    elif a.cmd == "status":
        print(json.dumps(request(rec, "GET", f"/jobs/{a.job}")))
    elif a.cmd == "kill":
        print(json.dumps(request(rec, "POST", f"/jobs/{a.job}/kill", {})))
    elif a.cmd == "log":
        if a.follow:
            s = follow(rec, a.job, a.poll)
            print(f"\n[job {a.job}: {s['status']} rc={s.get('rc')}]", file=sys.stderr)
            sys.exit(0 if s.get("rc") == 0 else 1)
        sys.stdout.buffer.write(request(rec, "GET", f"/jobs/{a.job}/log?limit={1 << 30}", timeout=600))
    elif a.cmd == "fetch":
        raw = request(rec, "GET", f"/jobs/{a.job}/tar?path={quote(a.path)}", timeout=1800)
        os.makedirs(a.out, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
            t.extractall(a.out, filter="data")
        print(f"{len(raw)} bytes into {a.out}")


if __name__ == "__main__":
    main()
