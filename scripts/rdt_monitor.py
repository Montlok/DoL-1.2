# -*- coding: utf-8 -*-

"""Training monitor CLI for RDT runs.

Modes against the ``monitor/`` directory that ``train_rdt.py`` keeps:

    # one-shot status (machine readable: --json)
    python -m scripts.rdt_monitor status --run runs/pretrain

    # HTTP API + web dashboard (pure stdlib server; open http://host:port/)
    python -m scripts.rdt_monitor serve --run runs/pretrain --port 8787

    # interactive dashboard (requires `rich`)
    python -m scripts.rdt_monitor tui --run runs/pretrain

    # queue a control command without UI (also works while serve/tui run)
    python -m scripts.rdt_monitor control save --run runs/pretrain

Runs that never wired a :class:`StatusReporter` (e.g. ``train_vlm_align``
launched with plain stdout logging) can still be monitored from their log
file: ``--log run.log`` makes ``status``/``serve`` parse ``step=N k=v ...``
lines instead of reading ``monitor/``. ``--max-steps`` supplies the target
the log itself does not carry (progress + ETA); the step rate is measured
live by the server between polls. Control commands are unavailable in log
mode (nothing is listening).

    python -m scripts.rdt_monitor serve --run /data/run \
        --log /data/run.log --max-steps 20000

The server binds 127.0.0.1 by default; on a remote box keep it that way and
reach the dashboard through an SSH tunnel (``ssh -N -L 8787:localhost:8787
host``) rather than exposing the control endpoint publicly.

HTTP API:

    GET  /            -> live web dashboard (loss/lr/... charts)
    GET  /api/health  -> {"ok": true, "control": bool}
    GET  /api/status  -> latest snapshot of training (status.json)
    GET  /api/history?n=200 -> recent step metrics
    POST /api/control {"command": "save" | "eval" | "stop"}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.training.status import (  # noqa: E402
    VALID_COMMANDS,
    StatusReporter,
    read_history,
    read_status,
)


# ------------------------------------------------------------- data sources


class MonitorSource:
    """status()/history() backed by the ``monitor/`` dir a StatusReporter keeps."""

    can_control = True

    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    def status(self) -> dict | None:
        return read_status(self.run_dir)

    def history(self, n: int) -> list[dict]:
        return read_history(self.run_dir, last_n=n)


_LOG_KV_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_log_row(line: str) -> dict | None:
    """Parse one ``step=N key=value ...`` logger line into numeric fields."""
    if not line.startswith("step="):
        return None
    row: dict = {}
    for key, val in _LOG_KV_RE.findall(line):
        try:
            row[key] = int(val) if key == "step" else float(val)
        except ValueError:
            continue
    return row if row.get("step") is not None else None


def downsample_history(rows: list[dict], max_points: int) -> list[dict]:
    """Bucket-mean numeric metrics so charts stay light on long runs.

    Each bucket keeps the last step number (monotonic x axis) and averages
    every other numeric key over the bucket; missing keys are skipped.
    """
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    bucket = math.ceil(len(rows) / max_points)
    out: list[dict] = []
    for i in range(0, len(rows), bucket):
        chunk = rows[i : i + bucket]
        agg: dict = {"step": chunk[-1]["step"]}
        keys = {k for row in chunk for k in row if k != "step"}
        for key in keys:
            vals = [
                row[key]
                for row in chunk
                if isinstance(row.get(key), (int, float))
            ]
            if vals:
                agg[key] = sum(vals) / len(vals)
        out.append(agg)
    return out


class LogSource:
    """status()/history() reconstructed from a ``step=N ...`` stdout log.

    For runs that never wired a StatusReporter. State is inferred: reaching
    ``max_steps`` means finished, a log untouched for ``stale_after`` seconds
    means stalled, anything else is running. The step rate (for ETA) is
    measured between successive ``status()`` calls instead of trusting the
    log, which carries no timestamps.
    """

    can_control = False

    def __init__(
        self, log_path: str, max_steps: int | None = None, stale_after: float = 180.0
    ):
        self.path = Path(log_path)
        self.max_steps = max_steps
        self.stale_after = stale_after
        self._samples: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def history(self, n: int) -> list[dict]:
        rows: list[dict] = []
        try:
            with self.path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    row = parse_log_row(line)
                    if row is not None:
                        rows.append(row)
        except OSError:
            return []
        return downsample_history(rows, n)

    def _last_row(self) -> dict | None:
        """Read the trailing ~8KB and parse the final metrics line."""
        try:
            with self.path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 8192))
                tail = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        for line in reversed(tail.splitlines()):
            row = parse_log_row(line)
            if row is not None:
                return row
        return None

    def _rate(self, now: float, step: int) -> float | None:
        with self._lock:
            self._samples.append((now, step))
            self._samples = self._samples[-64:]
            for t0, s0 in self._samples:
                if now - t0 >= 20.0 and step > s0:
                    return (step - s0) / (now - t0)
        return None

    def status(self) -> dict | None:
        row = self._last_row()
        if row is None:
            return None
        mtime = self.path.stat().st_mtime
        now = time.time()
        step = int(row.pop("step"))
        if self.max_steps and step >= self.max_steps:
            state = "finished"
        elif now - mtime > self.stale_after:
            state = "stalled"
        else:
            state = "running"
        return {
            "state": state,
            "step": step,
            "max_steps": self.max_steps,
            "progress": (step / self.max_steps) if self.max_steps else None,
            "updated_at": mtime,
            "metrics": row,
            "rate_steps_per_s": self._rate(now, step),
            "source": "log",
        }


def _make_source(args) -> MonitorSource | LogSource:
    if getattr(args, "log", None):
        return LogSource(args.log, max_steps=getattr(args, "max_steps", None))
    return MonitorSource(args.run)


# ----------------------------------------------------------------- HTTP API


def _make_handler(run_dir: str, source: MonitorSource | LogSource | None = None):
    source = source or MonitorSource(run_dir)
    class Handler(BaseHTTPRequestHandler):
        server_version = "rdt-monitor/1.0"

        def log_message(self, fmt, *args):  # noqa: D102 - silence stderr spam
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                body = _DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/health":
                self._send(
                    200,
                    {"ok": True, "run": run_dir, "control": source.can_control},
                )
            elif url.path == "/api/status":
                status = source.status()
                if status is None:
                    self._send(404, {"error": "no status yet"})
                else:
                    self._send(200, status)
            elif url.path == "/api/history":
                n = 200
                qs = parse_qs(url.query)
                if "n" in qs:
                    try:
                        n = max(1, min(int(qs["n"][0]), 10000))
                    except ValueError:
                        self._send(400, {"error": "n must be an integer"})
                        return
                self._send(200, {"history": source.history(n)})
            else:
                self._send(404, {"error": f"unknown path {url.path}"})

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/control":
                self._send(404, {"error": "unknown path"})
                return
            if not source.can_control:
                self._send(
                    409,
                    {"error": "log-mode run has no control channel"},
                )
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            command = data.get("command")
            if command not in VALID_COMMANDS:
                self._send(
                    400,
                    {"error": f"command must be one of {list(VALID_COMMANDS)}"},
                )
                return
            StatusReporter.request(run_dir, command)
            self._send(200, {"ok": True, "queued": command})

    return Handler


def serve(run_dir: str, host: str, port: int, source=None) -> int:
    source = source or MonitorSource(run_dir)
    httpd = ThreadingHTTPServer((host, port), _make_handler(run_dir, source))
    print(
        f"[rdt-monitor] dashboard on http://{host}:{httpd.server_address[1]}/ "
        f"run={run_dir}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop the server; shut down quietly.
        pass
    finally:
        httpd.server_close()
    return 0


# ---------------------------------------------------------------- dashboard
# Single-file web UI served at GET /. Pure static HTML + Chart.js from a CDN;
# it only talks to the JSON API above, so it works for both monitor-dir and
# log-mode sources (control buttons appear only when the API allows them).

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RDT monitor</title>
<style>
:root { --bg:#fafaf8; --fg:#1a1a18; --muted:#6b6a64; --card:#ffffff;
        --border:#e3e2dc; --accent:#1d9e75; --warn:#ba7517; --bad:#a32d2d; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141413; --fg:#e8e7e2; --muted:#94938c; --card:#1e1e1c;
          --border:#33322f; }
}
* { box-sizing: border-box; }
body { margin:0; padding:20px; background:var(--bg); color:var(--fg);
       font:14px/1.5 system-ui, sans-serif; }
.row { display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
h1 { font-size:16px; font-weight:600; margin:0; }
.pill { padding:2px 10px; border-radius:999px; font-size:12px;
        border:1px solid var(--border); }
.pill.running { color:var(--accent); border-color:var(--accent); }
.pill.finished { color:#378add; border-color:#378add; }
.pill.stalled, .pill.crashed { color:var(--bad); border-color:var(--bad); }
.muted { color:var(--muted); font-size:12px; }
#bar { height:6px; background:var(--border); border-radius:3px;
       margin:12px 0 4px; overflow:hidden; }
#bar > div { height:100%; width:0; background:var(--accent);
             transition:width .5s; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
         gap:10px; margin:14px 0; }
.card { background:var(--card); border:1px solid var(--border);
        border-radius:8px; padding:10px 14px; }
.card .k { font-size:12px; color:var(--muted); }
.card .v { font-size:20px; font-weight:600; }
.chart { background:var(--card); border:1px solid var(--border);
         border-radius:8px; padding:12px; margin-bottom:12px; }
.chart h2 { font-size:13px; font-weight:600; margin:0 0 8px;
            color:var(--muted); }
select, button { background:var(--card); color:var(--fg);
  border:1px solid var(--border); border-radius:6px; padding:4px 10px;
  font-size:12px; }
button:hover { border-color:var(--muted); cursor:pointer; }
#ctl button.danger { color:var(--bad); }
</style>
</head>
<body>
<div class="row">
  <h1 id="run">RDT monitor</h1>
  <span class="pill" id="state">-</span>
  <span class="muted" id="meta"></span>
  <span style="flex:1"></span>
  <label class="muted">points
    <select id="pts"><option>500</option><option selected>1500</option>
    <option>4000</option><option>10000</option></select></label>
  <label class="muted">refresh
    <select id="ref"><option value="2000">2s</option>
    <option value="5000" selected>5s</option><option value="15000">15s</option>
    <option value="0">pause</option></select></label>
  <span id="ctl"></span>
</div>
<div id="bar"><div></div></div>
<div class="row muted" id="sub"></div>
<div class="cards" id="cards"></div>
<div id="charts"></div>
<p class="muted" id="err"></p>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const $ = (id) => document.getElementById(id);
const charts = {};
const ORDER = ["loss", "lr", "grad_norm", "throughput", "tokens"];
let timer = null;

function fmtDur(s) {
  if (!isFinite(s) || s < 0) return "?";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return (h ? h + "h" : "") + String(m).padStart(2, "0") + "m";
}
function fmtNum(v) {
  if (typeof v !== "number") return String(v);
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5))
    return v.toExponential(2);
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(4);
}
function smooth(data, w) {
  if (data.length < w * 2) return null;
  const out = []; let acc = 0; const q = [];
  for (const p of data) {
    q.push(p.y); acc += p.y;
    if (q.length > w) acc -= q.shift();
    out.push({x: p.x, y: acc / q.length});
  }
  return out;
}
function ensureChart(key) {
  if (charts[key]) return charts[key];
  const box = document.createElement("div");
  box.className = "chart";
  box.innerHTML = "<h2>" + key + "</h2><div style='position:relative;height:220px'><canvas></canvas></div>";
  const anchors = Object.keys(charts);
  $("charts").appendChild(box);
  const css = getComputedStyle(document.body);
  const c = new Chart(box.querySelector("canvas"), {
    type: "line",
    data: { datasets: [
      { data: [], borderColor: "#1d9e75", borderWidth: 1.2, pointRadius: 0,
        order: 2 },
      { data: [], borderColor: "#d85a30", borderWidth: 2, pointRadius: 0,
        order: 1 }
    ]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: {
        title: (it) => it.length ? "step " + it[0].parsed.x : "",
        label: (it) => fmtNum(it.parsed.y) } } },
      scales: {
        x: { type: "linear", grid: { color: css.getPropertyValue("--border") },
             ticks: { color: css.getPropertyValue("--muted"), maxTicksLimit: 10 } },
        y: { grid: { color: css.getPropertyValue("--border") },
             ticks: { color: css.getPropertyValue("--muted"),
                      callback: (v) => fmtNum(v) } }
      }
    }
  });
  charts[key] = c;
  return c;
}
async function jget(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}
async function tick() {
  try {
    const n = $("pts").value;
    const [st, hist] = await Promise.all([
      jget("/api/status"), jget("/api/history?n=" + n)]);
    render(st, hist.history || []);
    $("err").textContent = "";
  } catch (e) {
    $("err").textContent = "fetch failed: " + e.message;
  }
}
function render(st, rows) {
  const state = st.state || "unknown";
  $("state").textContent = state;
  $("state").className = "pill " + state;
  const step = st.step || 0, max = st.max_steps;
  const ago = st.updated_at ? Math.round(Date.now() / 1000 - st.updated_at) : "?";
  $("meta").textContent = "step " + step.toLocaleString() +
    (max ? " / " + max.toLocaleString() : "") + " · updated " + ago + "s ago";
  $("bar").firstElementChild.style.width =
    max ? Math.min(100, 100 * step / max) + "%" : "0";
  const rate = st.rate_steps_per_s;
  const eta = (max && rate) ? fmtDur((max - step) / rate)
    : (max && st.elapsed_s && step > 0)
      ? fmtDur(st.elapsed_s / step * (max - step)) : "?";
  $("sub").textContent = "eta " + eta +
    (rate ? " · " + (rate * 60).toFixed(1) + " steps/min" : "") +
    (st.source === "log" ? " · source: log" : "");
  const metrics = st.metrics || {};
  $("cards").innerHTML = Object.keys(metrics).map((k) =>
    "<div class='card'><div class='k'>" + k + "</div><div class='v'>" +
    fmtNum(metrics[k]) + "</div></div>").join("");
  const keys = new Set();
  rows.forEach((r) => Object.keys(r).forEach((k) => {
    if (k !== "step" && typeof r[k] === "number") keys.add(k); }));
  const sorted = [...keys].sort((a, b) =>
    (ORDER.indexOf(a) + 99) % 99 - (ORDER.indexOf(b) + 99) % 99);
  for (const key of sorted.slice(0, 6)) {
    const data = rows.filter((r) => typeof r[key] === "number")
      .map((r) => ({x: r.step, y: r[key]}));
    if (data.length < 2) continue;
    const c = ensureChart(key);
    c.data.datasets[0].data = data;
    const sm = key === "loss" ? smooth(data, Math.max(5, Math.round(data.length / 40))) : null;
    c.data.datasets[1].data = sm || [];
    c.update("none");
  }
}
async function init() {
  try {
    const h = await jget("/api/health");
    document.title = "RDT monitor - " + (h.run || "");
    $("run").textContent = (h.run || "RDT monitor").split("/").pop();
    if (h.control) {
      $("ctl").innerHTML =
        "<button onclick=\\"ctl('save')\\">save</button> " +
        "<button onclick=\\"ctl('eval')\\">eval</button> " +
        "<button class='danger' onclick=\\"ctl('stop')\\">stop</button>";
    }
  } catch (e) { /* header stays generic */ }
  const arm = () => {
    if (timer) clearInterval(timer);
    const ms = Number($("ref").value);
    if (ms) timer = setInterval(tick, ms);
  };
  $("ref").onchange = arm;
  $("pts").onchange = tick;
  arm();
  tick();
}
function ctl(cmd) {
  if (cmd === "stop" && !confirm("stop training?")) return;
  fetch("/api/control", { method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({command: cmd}) }).then(tick);
}
init();
</script>
</body>
</html>
"""


# ----------------------------------------------------------------- one-shot


def _fmt_eta(status: dict) -> str:
    step = status.get("step") or 0
    max_steps = status.get("max_steps")
    if not max_steps or step <= 0:
        return "?"
    rate = status.get("rate_steps_per_s")
    if rate:
        return _fmt_dur((max_steps - step) / rate)
    elapsed = status.get("elapsed_s") or 0
    if not elapsed:
        return "?"
    return _fmt_dur(elapsed / step * (max_steps - step))


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d{h:02d}h"
    return f"{h:02d}:{m:02d}:{s:02d}"


def one_shot(run_dir: str, as_json: bool, source=None) -> int:
    source = source or MonitorSource(run_dir)
    status = source.status()
    if status is None:
        print(f"no status for {run_dir}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0
    metrics = status.get("metrics") or {}
    stale = time.time() - (status.get("updated_at") or 0)
    print(f"state     {status.get('state')} (updated {stale:.0f}s ago)")
    print(f"step      {status.get('step')} / {status.get('max_steps') or '?'}  eta {_fmt_eta(status)}")
    for key in ("loss", "lr", "grad_norm", "tokens", "rec_steps", "throughput"):
        if key in metrics:
            print(f"{key:<9} {metrics[key]}")
    return 0


def control(run_dir: str, command: str) -> int:
    StatusReporter.request(run_dir, command)
    print(f"queued '{command}' for {run_dir}")
    return 0


# ----------------------------------------------------------------- rich TUI


def tui(run_dir: str, refresh: float) -> int:
    try:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print("tui mode needs `pip install rich`", file=sys.stderr)
        return 1

    console = Console()
    keys = _KeyReader()
    queued: list[tuple[float, str]] = []

    def render():
        status = read_status(run_dir) or {}
        history = read_history(run_dir, last_n=120)
        metrics = status.get("metrics") or {}
        state = status.get("state", "unknown")
        step = status.get("step") or 0
        max_steps = status.get("max_steps")
        stale = time.time() - (status.get("updated_at") or time.time())

        prog = Progress(
            TextColumn("[bold]step"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total}  eta " + _fmt_eta(status)),
        )
        prog.add_task("", total=max_steps or max(step, 1), completed=step)

        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column()
        state_style = {"running": "green", "finished": "blue"}.get(state, "yellow")
        table.add_row("state", f"[{state_style}]{state}[/] (updated {stale:.0f}s ago)")
        for key in ("loss", "lr", "grad_norm", "tokens", "rec_steps", "throughput"):
            if key in metrics:
                table.add_row(key, str(metrics[key]))
        table.add_row("run", str(run_dir))

        losses = [h["loss"] for h in history if isinstance(h.get("loss"), (int, float))]
        spark = Text(_sparkline(losses), style="magenta")

        lines = [prog, table, Text("loss "), spark]
        for ts, cmd in queued[-3:]:
            lines.append(Text(f"queued '{cmd}' at {time.strftime('%H:%M:%S', time.localtime(ts))}", style="dim"))
        lines.append(Text("[s]ave  [e]val  [x] stop  [q]uit", style="bold dim"))
        return Panel(Group(*lines), title="RDT pretraining", border_style="bright_black")

    try:
        with keys, Live(render(), console=console, refresh_per_second=4) as live:
            while True:
                key = keys.poll(refresh)
                if key == "q":
                    break
                if key in ("s", "e", "x"):
                    cmd = {"s": "save", "e": "eval", "x": "stop"}[key]
                    StatusReporter.request(run_dir, cmd)
                    queued.append((time.time(), cmd))
                live.update(render())
    except KeyboardInterrupt:
        # Ctrl+C exits the TUI like 'q'; treat it as a clean shutdown.
        pass
    return 0


def _sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return "(no data)"
    values = values[-width:]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return "▄" * len(values) + f"  {hi:.4f}"
    blocks = "▁▂▃▄▅▆▇█"
    out = "".join(blocks[int((v - lo) / (hi - lo) * (len(blocks) - 1))] for v in values)
    return out + f"  [{lo:.4f}, {hi:.4f}]"


class _KeyReader:
    """Raw single-key reader with timeout; degrades to sleep without a TTY."""

    def __enter__(self):
        self._fd = None
        if sys.stdin.isatty():
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        return False

    def poll(self, timeout: float) -> str | None:
        if self._fd is None:
            time.sleep(timeout)
            return None
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None


# ----------------------------------------------------------------- entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_run(p):
        p.add_argument("--run", required=True, help="training output_dir")

    def add_log(p):
        p.add_argument(
            "--log",
            default="",
            help="parse a `step=N k=v` stdout log instead of monitor/ "
            "(for runs without a StatusReporter)",
        )
        p.add_argument(
            "--max-steps",
            type=int,
            default=None,
            help="target step count for progress/ETA in --log mode",
        )

    p_status = sub.add_parser("status", help="print one snapshot and exit")
    add_run(p_status)
    add_log(p_status)
    p_status.add_argument("--json", action="store_true")

    p_serve = sub.add_parser("serve", help="HTTP API + web dashboard")
    add_run(p_serve)
    add_log(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)

    p_tui = sub.add_parser("tui", help="interactive rich dashboard")
    add_run(p_tui)
    p_tui.add_argument("--refresh", type=float, default=1.0)

    p_ctl = sub.add_parser("control", help="queue save/eval/stop")
    p_ctl.add_argument("command", choices=list(VALID_COMMANDS))
    add_run(p_ctl)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "status":
        return one_shot(args.run, args.json, source=_make_source(args))
    if args.mode == "serve":
        return serve(args.run, args.host, args.port, source=_make_source(args))
    if args.mode == "tui":
        return tui(args.run, args.refresh)
    if args.mode == "control":
        return control(args.run, args.command)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
