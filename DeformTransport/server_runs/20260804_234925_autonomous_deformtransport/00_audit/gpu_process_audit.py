#!/usr/bin/env python3
"""Read-only GPU process provenance and liveness audit."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path


PROJECT = Path("/mnt/sdbd/home/liuyu_qyh/DeformTransport")
SECRET_PATTERNS = [
    re.compile(r"(?i)(--?(?:api[-_]?key|token|password|secret))([= ]+)(\S+)"),
    re.compile(r"(?i)\b((?:API_KEY|TOKEN|PASSWORD|SECRET)=)(\S+)"),
]


def run(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda match: match.group(1) + match.group(2) + "<redacted>", text)
    return text


def read_text(path: Path, binary_nul: bool = False) -> tuple[str | None, str | None]:
    try:
        data = path.read_bytes()
        if binary_nul:
            data = data.replace(b"\0", b" ")
        return data.decode("utf-8", errors="replace").strip(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def readlink(path: Path) -> tuple[str | None, str | None]:
    try:
        return os.readlink(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def proc_stat(pid: int) -> dict:
    text, error = read_text(Path(f"/proc/{pid}/stat"))
    if error or not text:
        return {"error": error or "empty"}
    closing = text.rfind(")")
    fields = text[closing + 2 :].split()
    # fields[0] is process state, corresponding to proc stat field 3.
    return {
        "state": fields[0],
        "ppid": int(fields[1]),
        "utime_ticks": int(fields[11]),
        "stime_ticks": int(fields[12]),
        "threads": int(fields[17]),
        "starttime_ticks": int(fields[19]),
    }


def proc_io(pid: int) -> dict:
    text, error = read_text(Path(f"/proc/{pid}/io"))
    if error or text is None:
        return {"error": error or "empty"}
    values = {}
    for line in text.splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip())
    return values


def process_table(pid: int) -> str:
    code, stdout, stderr = run(
        [
            "ps",
            "-p",
            str(pid),
            "-o",
            "pid,user,lstart,etime,state,pcpu,pmem,ppid,args",
            "--cols",
            "4096",
        ]
    )
    return redact(stdout.strip() if code == 0 else f"ERROR: {stderr.strip()}")


def fd_links_in_project(pid: int) -> tuple[list[dict], str | None]:
    links = []
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        entries = list(fd_dir.iterdir())
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    for entry in entries:
        target, error = readlink(entry)
        if error or target is None or str(PROJECT) not in target:
            continue
        target_path = Path(target.replace(" (deleted)", ""))
        stat = None
        try:
            item = target_path.stat()
            stat = {"size": item.st_size, "mtime_ns": item.st_mtime_ns}
        except Exception as exc:
            stat = {"error": f"{type(exc).__name__}: {exc}"}
        links.append({"fd": entry.name, "target": target, "stat": stat})
    return links, None


def main() -> None:
    query_fields = "gpu_uuid,pid,process_name,used_memory"
    code, stdout, stderr = run(
        [
            "nvidia-smi",
            f"--query-compute-apps={query_fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        raise RuntimeError(stderr)

    apps = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4:
            apps.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mib": int(parts[3]),
                }
            )

    code, gpu_stdout, gpu_stderr = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        raise RuntimeError(gpu_stderr)
    gpu_by_uuid = {}
    for line in gpu_stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        gpu_by_uuid[parts[1]] = {
            "index": int(parts[0]),
            "gpu_util_pct": int(parts[2]),
            "memory_used_mib": int(parts[3]),
            "memory_free_mib": int(parts[4]),
            "temperature_c": int(parts[5]),
        }

    code, container_id, _ = run(
        ["docker", "inspect", "--format", "{{.Id}}", "deformtransport-dev"]
    )
    deformtransport_id = container_id.strip() if code == 0 else None

    pids = sorted({item["pid"] for item in apps})
    before = {pid: {"stat": proc_stat(pid), "io": proc_io(pid)} for pid in pids}
    before_fds = {pid: fd_links_in_project(pid) for pid in pids}

    # Five one-second pmon samples distinguish sustained compute from a stale allocation.
    pmon_code, pmon_stdout, pmon_stderr = run(
        ["nvidia-smi", "pmon", "-c", "5", "-d", "1", "-s", "um"], timeout=15
    )

    after = {pid: {"stat": proc_stat(pid), "io": proc_io(pid)} for pid in pids}
    after_fds = {pid: fd_links_in_project(pid) for pid in pids}
    pmon_by_pid: dict[int, list[dict]] = defaultdict(list)
    if pmon_code == 0:
        for line in pmon_stdout.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or not parts[1].isdigit():
                continue
            pid = int(parts[1])
            if pid not in pids:
                continue
            pmon_by_pid[pid].append(
                {
                    "gpu_index": int(parts[0]),
                    "type": parts[2],
                    "sm_pct": parts[3],
                    "mem_pct": parts[4],
                    "enc_pct": parts[5],
                    "dec_pct": parts[6],
                    "command": parts[7],
                }
            )

    records = []
    for app in apps:
        pid = app["pid"]
        status_text, status_error = read_text(Path(f"/proc/{pid}/status"))
        cmdline, cmdline_error = read_text(Path(f"/proc/{pid}/cmdline"), binary_nul=True)
        cwd, cwd_error = readlink(Path(f"/proc/{pid}/cwd"))
        cgroup, cgroup_error = read_text(Path(f"/proc/{pid}/cgroup"))
        stat_before = before[pid]["stat"]
        stat_after = after[pid]["stat"]
        io_before = before[pid]["io"]
        io_after = after[pid]["io"]
        cpu_ticks_delta = None
        if "error" not in stat_before and "error" not in stat_after:
            cpu_ticks_delta = (
                stat_after["utime_ticks"]
                + stat_after["stime_ticks"]
                - stat_before["utime_ticks"]
                - stat_before["stime_ticks"]
            )
        io_delta = {}
        if "error" not in io_before and "error" not in io_after:
            for key in ("rchar", "wchar", "read_bytes", "write_bytes"):
                io_delta[key] = io_after.get(key, 0) - io_before.get(key, 0)

        docker_marker = None
        if cgroup:
            match = re.search(r"(?:docker[/:-]|docker-)([0-9a-f]{12,64})", cgroup)
            if match:
                docker_marker = match.group(1)
        belongs_to_deformtransport = bool(
            docker_marker
            and deformtransport_id
            and deformtransport_id.startswith(docker_marker)
            or docker_marker
            and deformtransport_id
            and docker_marker.startswith(deformtransport_id)
        )

        parent_table = None
        if "error" not in stat_after:
            parent_table = process_table(stat_after["ppid"])

        fd_before, fd_before_error = before_fds[pid]
        fd_after, fd_after_error = after_fds[pid]
        project_link = bool(
            (cwd and str(PROJECT) in cwd)
            or (cmdline and str(PROJECT) in cmdline)
            or fd_before
            or fd_after
        )
        records.append(
            {
                **app,
                "gpu": gpu_by_uuid.get(app["gpu_uuid"]),
                "ps": process_table(pid),
                "cmdline": redact(cmdline) if cmdline else None,
                "cmdline_error": cmdline_error,
                "cwd": cwd,
                "cwd_error": cwd_error,
                "status_excerpt": [
                    line
                    for line in (status_text or "").splitlines()
                    if line.startswith(("Name:", "State:", "Pid:", "PPid:", "Threads:", "VmRSS:"))
                ],
                "status_error": status_error,
                "parent": parent_table,
                "cgroup": cgroup,
                "cgroup_error": cgroup_error,
                "docker_marker": docker_marker,
                "deformtransport_container_id": deformtransport_id,
                "belongs_to_deformtransport_dev": belongs_to_deformtransport,
                "deformtransport_project_link": project_link,
                "project_fd_links_before": fd_before,
                "project_fd_links_after": fd_after,
                "project_fd_error": fd_before_error or fd_after_error,
                "cpu_ticks_delta_over_sample": cpu_ticks_delta,
                "io_delta_over_sample": io_delta,
                "pmon_samples": pmon_by_pid.get(pid, []),
            }
        )

    print(
        json.dumps(
            {
                "audit_epoch": time.time(),
                "sample_seconds": 5,
                "gpu_snapshot": gpu_by_uuid,
                "pmon_error": pmon_stderr.strip() if pmon_code != 0 else None,
                "processes": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
