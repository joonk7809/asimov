"""Start, check and terminate a RunPod GPU pod that serves a model with vLLM.

    python scripts/runpod_pod.py up --model Qwen/Qwen3-4B-Instruct-2507
    python scripts/runpod_pod.py wait POD_ID      # blocks until /v1/models answers
    python scripts/runpod_pod.py list
    python scripts/runpod_pod.py down POD_ID      # terminate (stops billing)
    python scripts/runpod_pod.py down --all

Needs RUNPOD_API_KEY. The endpoint is https://POD_ID-8000.proxy.runpod.net and is protected
by a per-pod vLLM API key, written to runs/pods/POD_ID.json (gitignored).
"""
import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
IMAGE = "vllm/vllm-openai:v0.11.0"
NAME_PREFIX = "asiimov-"  # every pod this script starts; `down --all` touches only these
GPU_TYPES = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA A100 80GB PCIe",
             "NVIDIA A100-SXM4-80GB"]
POD_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "pods")


def call(method: str, path: str, body: dict | None = None) -> dict | list | None:
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod {method} {path} -> {e.code}: {e.read()[:800].decode(errors='replace')}")


def up(args):
    key = secrets.token_urlsafe(24)
    cmd = (f"vllm serve {args.model} --host 0.0.0.0 --port 8000 --api-key {key} "
           f"--max-model-len {args.max_len} --gpu-memory-utilization 0.92 --enable-prefix-caching "
           f"--served-model-name {args.model}")
    body = {
        "name": f"{NAME_PREFIX}{args.model.split('/')[-1].lower()}"[:60],
        "imageName": IMAGE,
        "gpuTypeIds": args.gpu or GPU_TYPES,
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "cloudType": args.cloud,
        "containerDiskInGb": 80,
        "volumeInGb": 0,
        "ports": ["8000/http"],
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [cmd],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    pod = call("POST", "/pods", body)
    os.makedirs(POD_DIR, exist_ok=True)
    rec = {"id": pod["id"], "model": args.model, "api_key": key, "created": time.time(),
           "url": f"https://{pod['id']}-8000.proxy.runpod.net", "cost_per_hr": pod.get("costPerHr"),
           "gpu": (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpu")}
    with open(os.path.join(POD_DIR, f"{pod['id']}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != "api_key"}, indent=2))


def wait(args):
    rec = json.load(open(os.path.join(POD_DIR, f"{args.pod}.json")))
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        try:
            req = urllib.request.Request(rec["url"] + "/v1/models",
                                         headers={"Authorization": f"Bearer {rec['api_key']}"})
            with urllib.request.urlopen(req, timeout=20) as r:
                print(f"ready after {time.time() - t0:.0f}s:", r.read()[:300].decode())
                return
        except Exception as e:  # noqa: BLE001 - startup produces 404/502/timeouts until vLLM is up
            print(f"{time.time() - t0:5.0f}s not ready: {str(e)[:80]}", flush=True)
            time.sleep(15)
    sys.exit("timed out waiting for the server; check the pod logs, and terminate it if stuck")


def ls(_args):
    pods = call("GET", "/pods") or []
    for p in pods:
        print(p["id"], p.get("name"), p.get("desiredStatus"), p.get("costPerHr"), p.get("imageName"))
    if not pods:
        print("no pods")


def down(args):
    # --all means every pod this script started (name prefix), never other pods on the account.
    if args.all:
        ids = [p["id"] for p in call("GET", "/pods") or [] if (p.get("name") or "").startswith(NAME_PREFIX)]
    elif args.pod:
        ids = [args.pod]
    else:
        sys.exit("give a POD_ID or --all")
    failed = []
    for pid in ids:
        try:  # keep going: one failed DELETE must not leave the remaining pods billing
            call("DELETE", f"/pods/{pid}")
            print("terminated", pid)
        except SystemExit as e:
            print(f"FAILED to terminate {pid}: {e}", file=sys.stderr)
            failed.append(pid)
    if failed:
        sys.exit(f"still running, terminate by hand: {' '.join(failed)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("up")
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", nargs="*")
    p.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--max-len", type=int, default=8192)
    p.set_defaults(fn=up)
    p = sub.add_parser("wait")
    p.add_argument("pod")
    p.add_argument("--timeout", type=int, default=1500)
    p.set_defaults(fn=wait)
    sub.add_parser("list").set_defaults(fn=ls)
    p = sub.add_parser("down")
    p.add_argument("pod", nargs="?")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=down)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
