"""Measure how long RuinedFooocus takes to start and to generate, and what it uses.

Use the app's own interpreter. Nothing here installs, downloads or changes data:

    python tools/benchmark_app.py startup --runs 3 [--data-dir DIR] [-- --cpu --offline]
    python tools/benchmark_app.py generate --model anima --runs 4 --size 1024x1024

`startup` launches the app (`launch.py`) on a free port, times each phase from
its own log lines and HTTP readiness, then stops it. Give `--data-dir` to time a
particular profile; app arguments after `--` are passed through.

`generate` uses a running app started with `--api`: it times image requests
through `/v1/images/generations`. The first request of a model includes loading
it and is reported apart from the later ones. On Windows the GPU memory and
load of the server process are sampled from the system performance counters.

Results print as a Markdown table; `--output` also writes them as JSON.
"""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
STARTUP_MARKERS = {
    "runtime": "Loading the image runtime",
    "interface": "Starting webui",
    "listening": "Running on local URL",
}
DEFAULT_PROMPT = (
    "masterpiece, best quality, an adult woman in a red scarf walking through an "
    "autumn park, golden light, detailed background"
)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def http_status(url, timeout=2):
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status
    except HTTPError as error:
        return error.code
    except (URLError, OSError):
        return None


def stop_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def time_startup(app_args, *, data_dir=None, timeout=600):
    """One launch: seconds from process start to each phase, or None if not seen."""
    port = free_port()
    environment = dict(os.environ, GRADIO_ANALYTICS_ENABLED="False", PYTHONUNBUFFERED="1")
    if data_dir:
        environment["RF_DATA_DIR"] = str(data_dir)
    command = [
        sys.executable,
        "launch.py",
        "--skip-dependency-install",
        "--nobrowser",
        "--port",
        str(port),
        *app_args,
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    seen, tail = {}, []

    def read():
        for line in process.stdout:
            tail.append(line.rstrip())
            del tail[:-40]
            for name, marker in STARTUP_MARKERS.items():
                if name not in seen and marker in line:
                    seen[name] = round(time.perf_counter() - started, 2)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    result = {"port": port}
    try:
        base = f"http://127.0.0.1:{port}"
        while time.perf_counter() - started < timeout:
            if process.poll() is not None:
                result["error"] = "The app exited during startup."
                break
            if "page" not in result and http_status(base + "/") == 200:
                result["page"] = round(time.perf_counter() - started, 2)
            if "page" in result and http_status(base + "/config") == 200:
                result["config"] = round(time.perf_counter() - started, 2)
                break
            time.sleep(0.25)
        else:
            result["error"] = f"Not ready within {timeout} s."
    finally:
        stop_tree(process)
        reader.join(timeout=5)
    result.update(seen)
    if "error" in result:
        result["log_tail"] = tail[-15:]
    return result


def summary(values):
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    return {
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


class GpuSampler:
    """Windows performance counters for one process: dedicated memory and engine load."""

    def __init__(self, pid, interval=1):
        self.pid, self.interval = pid, interval
        self.samples, self._process = [], None

    def __enter__(self):
        if os.name != "nt" or not self.pid:
            return self
        counters = [
            rf"\GPU Process Memory(pid_{self.pid}_*)\Dedicated Usage",
            rf"\GPU Engine(pid_{self.pid}_*)\Utilization Percentage",
        ]
        self._process = subprocess.Popen(
            ["typeperf", *counters, "-si", str(self.interval)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        header = None
        for line in self._process.stdout:
            fields = parse_csv_line(line)
            if not fields:
                continue
            if header is None:
                if fields[0].startswith("(PDH-CSV"):
                    header = fields
                continue
            sample = parse_sample(header, fields)
            if sample is not None:
                self.samples.append(sample)

    def __exit__(self, *exc):
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def window(self, start, end):
        return [sample for sample in self.samples if start <= sample["time"] <= end]


def parse_csv_line(line):
    line = line.strip()
    if not line.startswith('"'):
        return None
    return [field.strip('"') for field in re.findall(r'"[^"]*"', line)]


def parse_sample(header, fields):
    """One typeperf row: memory in bytes and the busiest engine's load in percent."""
    memory, load = 0.0, 0.0
    for name, value in zip(header[1:], fields[1:]):
        try:
            number = float(value)
        except ValueError:
            continue
        if "Dedicated Usage" in name:
            memory += number
        elif "Utilization Percentage" in name and 0 <= number <= 100:
            # A counter whose engine just appeared can report a garbage first value.
            load = max(load, number)
    return {"time": time.perf_counter(), "memory": memory, "load": load}


def pid_on_port(port):
    """The process listening on a local port (Windows netstat), or None."""
    if os.name != "nt":
        return None
    output = subprocess.run(
        ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, check=False
    ).stdout
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
            return int(parts[4])
    return None


def api(base, path, payload=None, timeout=1800):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("RF_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["RF_API_KEY"]
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(base + path, data=data, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def resolve_model(models, wanted):
    """The first image model whose id contains `wanted`, ignoring case."""
    names = [
        item["id"]
        for item in models.get("data", [])
        if item.get("type") == "image" and item.get("generation_available", True)
    ]
    match = [name for name in names if wanted.lower() in name.lower()]
    if not match:
        raise SystemExit(f"No image model matches {wanted!r}. Available: {', '.join(names)}")
    return match[0]


def time_generation(base, model, runs, *, size, steps=None, prompt=DEFAULT_PROMPT, pid=None):
    records = []
    with GpuSampler(pid) as sampler:
        for run in range(runs):
            payload = {"prompt": prompt, "model": model, "n": 1, "size": size, "seed": 1234}
            if steps:
                payload["steps"] = steps
            start = time.perf_counter()
            record = {"model": model, "run": run, "size": size}
            try:
                response = api(base, "/v1/images/generations", payload)
                record["seconds"] = round(time.perf_counter() - start, 2)
                record["images"] = len(response.get("data", []))
            except HTTPError as error:
                record["error"] = (
                    f"HTTP {error.code}: {error.read()[:300].decode('utf-8', 'replace')}"
                )
            except (URLError, OSError) as error:
                record["error"] = str(error)
            samples = sampler.window(start, time.perf_counter() + 0.5)
            if samples:
                record["gpu_memory_gb"] = round(max(s["memory"] for s in samples) / 2**30, 2)
                record["gpu_busy_percent"] = round(statistics.mean(s["load"] for s in samples), 1)
            records.append(record)
            print(json.dumps(record), flush=True)
    return records


def generation_table(records):
    rows = [
        "| Model | Size | First image s | Later images s (median, min-max) | GPU memory GB | GPU busy % |"
    ]
    rows.append("|---|---|---|---|---|---|")
    by_model = {}
    for record in records:
        by_model.setdefault((record["model"], record["size"]), []).append(record)
    for (model, size), items in by_model.items():
        ok = [item for item in items if "seconds" in item]
        first = ok[0]["seconds"] if ok and ok[0]["run"] == 0 else "failed"
        later = summary([item["seconds"] for item in ok if item["run"] > 0])
        memory = summary([item.get("gpu_memory_gb") for item in ok])
        busy = summary([item.get("gpu_busy_percent") for item in ok if item["run"] > 0])
        rows.append(
            f"| {model} | {size} | {first} | "
            + (f"{later['median']} ({later['min']}-{later['max']})" if later else "-")
            + f" | {memory['max'] if memory else '-'} | {busy['median'] if busy else '-'} |"
        )
    return "\n".join(rows)


def startup_table(results):
    rows = ["| Phase | Median s | Min s | Max s |", "|---|---|---|---|"]
    for phase in ("runtime", "interface", "listening", "page", "config"):
        stats = summary([result.get(phase) for result in results])
        if stats:
            rows.append(f"| {phase} | {stats['median']} | {stats['min']} | {stats['max']} |")
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    startup = commands.add_parser("startup", help="Time application startup phases")
    startup.add_argument("--runs", type=int, default=3)
    startup.add_argument("--data-dir", help="Profile folder (RF_DATA_DIR) to start with")
    startup.add_argument("--timeout", type=int, default=600)
    startup.add_argument("app_args", nargs=argparse.REMAINDER, help="After --: app arguments")
    generate = commands.add_parser("generate", help="Time image requests on a running app")
    generate.add_argument("--base-url", default="http://127.0.0.1:7860")
    generate.add_argument(
        "--model", action="append", required=True, help="Image model id or part of it; repeatable"
    )
    generate.add_argument("--runs", type=int, default=3)
    generate.add_argument("--size", default="1024x1024")
    generate.add_argument("--steps", type=int)
    generate.add_argument("--prompt", default=DEFAULT_PROMPT)
    for command in (startup, generate):
        command.add_argument("--output", help="Also write the results here as JSON")
    args = parser.parse_args(argv)

    if args.command == "startup":
        app_args = [value for value in args.app_args if value != "--"]
        results = []
        for run in range(args.runs):
            result = time_startup(app_args, data_dir=args.data_dir, timeout=args.timeout)
            print(
                json.dumps({"run": run, **{k: v for k, v in result.items() if k != "log_tail"}}),
                flush=True,
            )
            if "error" in result:
                print("\n".join(result.get("log_tail", [])), file=sys.stderr)
            results.append(result)
        table = startup_table(results)
    else:
        base = args.base_url.rstrip("/")
        models = api(base, "/v1/models", timeout=30)
        port = int(base.rsplit(":", 1)[1].split("/")[0]) if base.count(":") == 2 else None
        pid = pid_on_port(port) if port else None
        results = []
        for wanted in args.model:
            model = resolve_model(models, wanted)
            results += time_generation(
                base,
                model,
                args.runs,
                size=args.size,
                steps=args.steps,
                prompt=args.prompt,
                pid=pid,
            )
        table = generation_table(results)
    print()
    print(table)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all("error" not in result for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
