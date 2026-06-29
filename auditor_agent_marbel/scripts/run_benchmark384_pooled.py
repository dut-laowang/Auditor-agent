from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml


POOL_LIMITS = {
    "research": 3,
    "coding": 3,
    "bargaining": 3,
    "minecraft": 1,
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree_contents(item, target)
        else:
            shutil.copy2(item, target)


def merge_outputs(intermediate: Path, final_dir: Path) -> None:
    merged = final_dir / "merged"
    if merged.exists():
        shutil.rmtree(merged)
    merged.mkdir(parents=True)
    manifest = merged / "run_manifest.jsonl"
    for shard in sorted((intermediate / "shards").glob("*")):
        copy_tree_contents(shard / "configs", merged / "configs")
        copy_tree_contents(shard / "trajectories", merged / "trajectories")
        copy_tree_contents(shard / "marble_results", merged / "marble_results")
        shard_manifest = shard / "run_manifest.jsonl"
        if shard_manifest.exists():
            with shard_manifest.open("r", encoding="utf-8") as src, manifest.open("a", encoding="utf-8") as out:
                for line in src:
                    out.write(line)


def make_scenario_config(base_config: dict[str, Any], scenario: str, path: Path) -> None:
    config = dict(base_config)
    config["scenarios"] = {scenario: base_config["scenarios"][scenario]}
    write_yaml(path, config)


def command_for(
    python_bin: str,
    marble_root: Path,
    run_config: Path,
    attack_spec: Path,
    output_dir: Path,
    offset: int | None,
    limit: int | None,
    clean_only: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "auditor_agent_marbel.runner.run_attack",
        "--marble-root",
        str(marble_root),
        "--run-config",
        str(run_config),
        "--output-dir",
        str(output_dir),
    ]
    if clean_only:
        cmd.append("--clean-only")
    else:
        cmd.extend(["--attack-spec", str(attack_spec)])
        cmd.append("--attacks-only")
    if offset is not None:
        cmd.extend(["--target-offset", str(offset)])
    if limit is not None:
        cmd.extend(["--target-limit", str(limit)])
    return cmd


def count_status(intermediate: Path) -> dict[str, int]:
    counts = {"rows": 0, "completed": 0, "failed": 0, "clean_rows": 0, "attacked_rows": 0}
    for manifest in (intermediate / "shards").glob("*/run_manifest.jsonl"):
        for row in read_jsonl(manifest):
            counts["rows"] += 1
            if row.get("attack_id") in (None, "clean"):
                counts["clean_rows"] += 1
            else:
                counts["attacked_rows"] += 1
            if row.get("status") == "completed":
                counts["completed"] += 1
            elif row.get("status") == "failed":
                counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 384-sample multiscenario ACI-on-MARBLE benchmark with scenario pools.")
    parser.add_argument("--marble-root", required=True, type=Path)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--attack-spec", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--shard-size", type=int, default=3)
    parser.add_argument("--progress-interval", type=int, default=20)
    args = parser.parse_args()

    base_config = load_yaml(args.run_config)
    intermediate = args.work_dir / "intermediate"
    final_dir = args.work_dir / "final"
    configs_dir = intermediate / "scenario_configs"
    shards_dir = intermediate / "shards"
    logs_dir = intermediate / "logs"
    for path in (configs_dir, shards_dir, logs_dir, final_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    jobs_by_scenario: dict[str, deque[dict[str, Any]]] = {}
    for scenario in base_config["scenarios"]:
        scenario_config_path = configs_dir / f"{scenario}.yaml"
        make_scenario_config(base_config, scenario, scenario_config_path)
        jobs: deque[dict[str, Any]] = deque()
        clean_out = shards_dir / f"{scenario}_clean"
        jobs.append(
            {
                "name": f"{scenario}_clean",
                "cmd": command_for(args.python, args.marble_root, scenario_config_path, args.attack_spec, clean_out, None, None, True),
            }
        )
        attacked_total = 2 * 3 * 15
        for offset in range(0, attacked_total, args.shard_size):
            limit = min(args.shard_size, attacked_total - offset)
            shard_name = f"{scenario}_attack_{offset:03d}_{offset + limit:03d}"
            jobs.append(
                {
                    "name": shard_name,
                    "cmd": command_for(
                        args.python,
                        args.marble_root,
                        scenario_config_path,
                        args.attack_spec,
                        shards_dir / shard_name,
                        offset,
                        limit,
                        False,
                    ),
                }
            )
        jobs_by_scenario[scenario] = jobs

    active: dict[subprocess.Popen[Any], dict[str, Any]] = {}
    active_by_scenario: dict[str, int] = defaultdict(int)
    total_jobs = sum(len(jobs) for jobs in jobs_by_scenario.values())
    finished_jobs = 0
    failures: list[dict[str, Any]] = []
    last_progress = 0.0

    def launch_available() -> None:
        for scenario, jobs in jobs_by_scenario.items():
            limit = POOL_LIMITS.get(scenario, 4)
            while jobs and active_by_scenario[scenario] < limit:
                job = jobs.popleft()
                log_path = logs_dir / f"{job['name']}.log"
                handle = log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(job["cmd"], stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd()))
                active[proc] = {"scenario": scenario, "name": job["name"], "log_handle": handle, "log_path": str(log_path)}
                active_by_scenario[scenario] += 1

    launch_available()
    while active:
        time.sleep(1)
        for proc, job in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            job["log_handle"].close()
            del active[proc]
            active_by_scenario[job["scenario"]] -= 1
            finished_jobs += 1
            if code != 0:
                failures.append({"name": job["name"], "scenario": job["scenario"], "exit_code": code, "log_path": job["log_path"]})
            launch_available()

        now = time.time()
        if now - last_progress >= args.progress_interval:
            last_progress = now
            counts = count_status(intermediate)
            progress = {
                "finished_jobs": finished_jobs,
                "total_jobs": total_jobs,
                "active_jobs": len(active),
                "queued_jobs": sum(len(jobs) for jobs in jobs_by_scenario.values()),
                **counts,
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)

    merge_outputs(intermediate, final_dir)
    summary = {
        "total_jobs": total_jobs,
        "finished_jobs": finished_jobs,
        "failures": failures,
        **count_status(intermediate),
        "merged_dir": str((final_dir / "merged").resolve()),
    }
    (final_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
