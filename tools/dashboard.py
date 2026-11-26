"""Live training page. Run on the Mac, then open http://localhost:8765

    python3 tools/dashboard.py                      # laptops "acer" and "asus" over SSH, plus this Mac
    python3 tools/dashboard.py --hosts mac          # only the runs on this computer

Only the Python standard library is used. The server listens on this computer only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Runs where the training runs are (sent to the laptop on stdin); prints one JSON object.
READER = r'''
import json, os, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1]).expanduser()
out = {"running": False, "run": "", "log": [], "eval": [], "console": [], "gpu": {}, "max_hours": None}
def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
p = sh("tmux", "has-session", "-t", "=nova-train")
q = sh("pgrep", "-f", "training.train --config")
out["running"] = bool((p and p.returncode == 0) or (q and q.returncode == 0))
p = sh("nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits")
if p and p.returncode == 0:
    out["gpu"] = dict(zip(("util", "temp", "mem", "mem_total"), (float(x) for x in p.stdout.split("\n")[0].split(","))))
runs = root / "runs"
consoles = sorted(runs.glob("*.console.log"), key=lambda f: f.stat().st_mtime) if runs.is_dir() else []
if consoles:
    name = consoles[-1].name[: -len(".console.log")]
    text = consoles[-1].read_text(errors="replace").splitlines()
    out["console"] = [l[:200] for l in text[-4:]]
else:
    dirs = sorted((d for d in runs.glob("*") if (d / "log.jsonl").exists()), key=lambda d: (d / "log.jsonl").stat().st_mtime) if runs.is_dir() else []
    name = dirs[-1].name if dirs else ""
out["run"] = name
run = runs / name
def rows(path):
    found = []
    if name and path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                found.append(json.loads(line))
            except ValueError:
                pass
    return found
log = rows(run / "log.jsonl")
keep = ("step", "progress", "hours", "fps", "mean_reward_ext", "odom_turn_error_deg")
parts = ("success", "exit", "keys", "kill_frac", "level_open")
step = max(1, len(log) // 240)
picked = log[::step] + ([log[-1]] if log and (len(log) - 1) % step else [])
out["log"] = [{k: v for k, v in r.items() if k in keep or k.split("/")[-1] in parts} for r in picked]
out["eval"] = rows(run / "eval.jsonl")
if name and (run / "config.json").exists():
    out["max_hours"] = json.loads((run / "config.json").read_text()).get("max_hours")
print(json.dumps(out))
'''


def read_host(host: str) -> dict:
    if host in ("local", "mac"):  # this computer
        command = [sys.executable, "-", str(ROOT)]
    else:
        command = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", host, "python3", "-", "nova-train"]
    try:
        proc = subprocess.run(command, input=READER, capture_output=True, text=True, timeout=40)
        if proc.returncode != 0:
            return {"error": (proc.stderr.strip().splitlines() or ["could not read"])[-1][:200]}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


class Poller(threading.Thread):
    def __init__(self, hosts: list[str], every: float):
        super().__init__(daemon=True)
        self.hosts, self.every = hosts, every
        self.data: dict[str, dict] = {h: {} for h in hosts}
        self.read_at: float | None = None

    def run(self) -> None:
        with ThreadPoolExecutor(len(self.hosts)) as pool:
            while True:
                for host, result in zip(self.hosts, pool.map(read_host, self.hosts)):
                    # Keep the last good history while a laptop is briefly unreachable.
                    if "error" in result:
                        self.data[host] = {**self.data[host], "error": result["error"], "running": False}
                    else:
                        self.data[host] = result
                self.read_at = time.time()
                time.sleep(self.every)


def serve(hosts: list[str], port: int, every: float) -> None:
    poller = Poller(hosts, every)
    poller.start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/data":
                age = None if poller.read_at is None else time.time() - poller.read_at
                body, kind = json.dumps({"hosts": poller.data, "age": age}).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, kind = (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the terminal quiet
            pass

    print(f"open http://localhost:{port}   (reading {', '.join(hosts)} every {every:.0f}s; Ctrl+C stops the page only)", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="+", default=["acer", "asus", "mac"], help="up to 3: SSH aliases, or 'mac' for this computer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--every", type=float, default=20.0, help="seconds between readings")
    args = ap.parse_args(argv)
    serve(args.hosts[:3], args.port, args.every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
