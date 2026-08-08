#!/usr/bin/env python3
"""Run one training stage and stop it on a stale heartbeat or hard timeout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heartbeat-file", required=True, type=Path)
    parser.add_argument("--stale-seconds", required=True, type=float)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.stale_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    if args.stale_seconds >= args.timeout_seconds:
        parser.error("stale-seconds must be smaller than timeout-seconds")
    return args


def stop_process(process: subprocess.Popen[object], reason: str) -> int:
    print(f"HEALTH_WATCHDOG {reason}", file=sys.stderr, flush=True)
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - formal runs use Linux
            process.terminate()
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - formal runs use Linux
            process.kill()
        process.wait()
    return 124


def main() -> int:
    args = parse_args()
    heartbeat = args.heartbeat_file.resolve()
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.touch()
    started = time.monotonic()
    environment = os.environ.copy()
    environment["Q_ATTENTION_HEARTBEAT_FILE"] = str(heartbeat)
    process = subprocess.Popen(
        args.command,
        env=environment,
        start_new_session=(os.name == "posix"),
    )
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode
        elapsed = time.monotonic() - started
        if elapsed > args.timeout_seconds:
            return stop_process(
                process,
                f"stage_timeout elapsed_seconds={round(elapsed, 1)}",
            )
        try:
            heartbeat_age = time.time() - heartbeat.stat().st_mtime
        except FileNotFoundError:
            heartbeat_age = elapsed
        if heartbeat_age > args.stale_seconds:
            return stop_process(
                process,
                f"stale_heartbeat age_seconds={round(heartbeat_age, 1)}",
            )
        time.sleep(5.0)


if __name__ == "__main__":
    raise SystemExit(main())
