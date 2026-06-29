from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from auditor_agent_marbel.attacks.config_injection import apply_static_config_attack
from auditor_agent_marbel.attacks.runtime_hooks import installed_runtime_hooks
from auditor_agent_marbel.core.jsonl import append_jsonl, read_jsonl, write_json
from auditor_agent_marbel.core.schemas import AttackSpec, RunTarget
from auditor_agent_marbel.data.sample_loader import load_samples
from auditor_agent_marbel.placement.graph_selector import resolve_placement
from auditor_agent_marbel.topology.topology_builder import apply_topology
from auditor_agent_marbel.logging.trajectory_logger import TrajectoryLogger


def attach_private_user_information(config: Dict[str, Any], attack: Optional[AttackSpec]) -> None:
    if not attack:
        return
    private_info = attack.metadata.get("private_user_information")
    if not private_info:
        return
    task = dict(config.get("task", {}))
    task["private_user_information"] = private_info
    config["task"] = task


def load_run_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_attack_specs(path: Optional[Path]) -> List[AttackSpec]:
    if path is None:
        return []
    return [AttackSpec.from_dict(row) for row in read_jsonl(path)]


def configure_llm_environment(run_config: Dict[str, Any]) -> None:
    api_config = run_config.get("api", {})
    base_url = api_config.get("base_url")
    if base_url:
        os.environ.setdefault("OPENAI_API_BASE", str(base_url))
        os.environ.setdefault("OPENAI_BASE_URL", str(base_url))
    api_key = api_config.get("api_key")
    api_key_env = api_config.get("api_key_env", "OPENAI_API_KEY")
    if api_key and api_key_env:
        os.environ.setdefault(str(api_key_env), str(api_key))


def prepare_config(
    sample: Dict[str, Any],
    scenario: str,
    topology: str,
    run_config: Dict[str, Any],
    attack: Optional[AttackSpec],
) -> Dict[str, Any]:
    config = apply_topology(sample, topology)
    default_llm = run_config.get("default_llm") or run_config.get("api", {}).get("model_name", "gpt-4o-mini")
    config["llm"] = config.get("llm") or default_llm
    overrides = run_config.get("environment_overrides", {}).get(scenario, {})
    environment = dict(config.get("environment", {}))
    environment.update(overrides)
    config["environment"] = environment
    config.setdefault("output", {})
    config.setdefault("memory", {"type": "SharedMemory"})
    metrics = dict(config.get("metrics") or {})
    if not metrics.get("evaluate_llm"):
        metrics["evaluate_llm"] = {"provider": "openai", "model": default_llm}
    config["metrics"] = metrics
    config.setdefault("engine_planner", {"initial_progress": "Starting MARBLE attack run."})
    attach_private_user_information(config, attack)
    return config


def build_targets(
    marble_root: Path,
    run_config: Dict[str, Any],
    attack_specs: Iterable[AttackSpec],
    clean_only: bool,
    attacks_only: bool = False,
) -> List[RunTarget]:
    targets: List[RunTarget] = []
    attacks = list(attack_specs)
    max_attacked_targets = int(run_config.get("max_attacked_targets") or 0)
    attacked_count = 0
    for scenario, scenario_config in run_config.get("scenarios", {}).items():
        samples = load_samples(
            marble_root=marble_root,
            dataset=scenario_config["dataset"],
            sample_ids=scenario_config["sample_ids"],
        )
        topologies = scenario_config.get("topologies", ["graph"])
        for sample in samples:
            sample_id = int(sample.get("task_id"))
            for topology in topologies:
                if not attacks_only:
                    targets.append(RunTarget(scenario, sample_id, topology, sample, None))
                if clean_only:
                    continue
                for attack in attacks:
                    if scenario in attack.scenarios and topology in attack.topologies:
                        if max_attacked_targets and attacked_count >= max_attacked_targets:
                            continue
                        targets.append(RunTarget(scenario, sample_id, topology, sample, attack))
                        attacked_count += 1
    return targets


def write_generated_config(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def is_nonfatal_marble_evaluator_error(error: str) -> bool:
    return (
        'KeyError: \'"rating"\'' in error
        or 'KeyError: \'\\"rating"\'' in error
        or "KeyError: 'target_agent_id'" in error
        or "json.decoder.JSONDecodeError: Unterminated string" in error
    )


def run_one(
    target: RunTarget,
    run_config: Dict[str, Any],
    marble_root: Path,
    output_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    attack = target.attack
    base_config = prepare_config(target.sample, target.scenario, target.topology, run_config, attack)
    placement = resolve_placement(base_config, attack.placement) if attack else {}

    context = {
        "scenario": target.scenario,
        "sample_id": target.sample_id,
        "topology": target.topology,
        "attack_id": attack.attack_id if attack else "clean",
    }
    if attack:
        base_config = apply_static_config_attack(base_config, attack, placement, context)

    config_path = (output_dir / "configs" / f"{target.run_id}.yaml").resolve()
    trajectory_path = (output_dir / "trajectories" / f"{target.run_id}.jsonl").resolve()
    output_config = dict(base_config.get("output", {}))
    output_config["file_path"] = str((output_dir / "marble_results" / f"{target.run_id}.jsonl").resolve())
    Path(output_config["file_path"]).parent.mkdir(parents=True, exist_ok=True)
    base_config["output"] = output_config
    if target.scenario == "coding":
        environment = dict(base_config.get("environment", {}))
        environment["workspace_dir"] = str((output_dir / "workspaces" / target.run_id).resolve())
        base_config["environment"] = environment
    write_generated_config(base_config, config_path)

    run_meta = {
        "run_id": target.run_id,
        "scenario": target.scenario,
        "sample_id": target.sample_id,
        "topology": target.topology,
        "attack_id": attack.attack_id if attack else None,
        "surface": attack.surface if attack else None,
        "objective": attack.objective if attack else None,
        "placement": placement,
        "attack_metadata": attack.metadata if attack else {},
    }
    logger = TrajectoryLogger(trajectory_path, run_meta)
    logger.log("config_prepared", {"config_path": str(config_path)})

    if dry_run:
        logger.close("dry_run")
        return {**run_meta, "status": "dry_run", "config_path": str(config_path)}

    if str(marble_root) not in sys.path:
        sys.path.insert(0, str(marble_root))

    try:
        from marble.configs.config import Config
        from marble.engine.engine import Engine

        old_cwd = Path.cwd()
        marble_runtime_cwd = marble_root / "marble"
        try:
            (marble_runtime_cwd / "logs").mkdir(parents=True, exist_ok=True)
            os.chdir(marble_runtime_cwd)
            with installed_runtime_hooks(attack, placement, context, logger) as hooks:
                config = Config.load(str(config_path))
                engine = Engine(config)
                hooks.install_memory_attack(engine)
                engine.start()
        finally:
            os.chdir(old_cwd)
        logger.close("completed")
        status = "completed"
        error = None
        warning = None
    except Exception as exc:  # pragma: no cover - depends on MARBLE runtime and LLM keys.
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        if is_nonfatal_marble_evaluator_error(error):
            warning = error
            logger.log("evaluator_warning", {"warning": warning, "traceback": traceback.format_exc()})
            logger.close("completed")
            status = "completed"
            error = None
        else:
            warning = None
            logger.log("exception", {"error": error, "traceback": traceback.format_exc()})
            logger.close("failed", error=error)
            status = "failed"

    return {**run_meta, "status": status, "error": error, "warning": warning, "config_path": str(config_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACI-inspired graph-aware attacks on official MARBLE tasks.")
    parser.add_argument("--marble-root", required=True, type=Path, help="Path to official MARBLE repository.")
    parser.add_argument("--run-config", required=True, type=Path, help="Pilot run YAML config.")
    parser.add_argument("--attack-spec", type=Path, help="JSONL attack spec file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs and trajectories without executing MARBLE.")
    parser.add_argument("--clean-only", action="store_true", help="Run only clean baselines.")
    parser.add_argument("--attacks-only", action="store_true", help="Run only attacked targets, without clean baselines.")
    parser.add_argument("--target-offset", type=int, default=0, help="Skip this many planned targets.")
    parser.add_argument("--target-limit", type=int, help="Run at most this many planned targets after offset.")
    args = parser.parse_args()

    marble_root = args.marble_root.resolve()
    if not marble_root.exists():
        raise FileNotFoundError(f"MARBLE root does not exist: {marble_root}")

    run_config = load_run_config(args.run_config)
    configure_llm_environment(run_config)
    attack_specs = [] if args.clean_only else load_attack_specs(args.attack_spec)
    targets = build_targets(marble_root, run_config, attack_specs, clean_only=args.clean_only, attacks_only=args.attacks_only)
    if args.target_offset or args.target_limit is not None:
        start = max(args.target_offset, 0)
        end = None if args.target_limit is None else start + max(args.target_limit, 0)
        targets = targets[start:end]

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run_plan.json",
        {
            "marble_root": str(marble_root),
            "run_config": str(args.run_config),
            "attack_spec": str(args.attack_spec) if args.attack_spec else None,
            "dry_run": args.dry_run,
            "clean_only": args.clean_only,
            "num_targets": len(targets),
        },
    )

    manifest_path = args.output_dir / "run_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    for target in targets:
        result = run_one(target, run_config, marble_root, args.output_dir, args.dry_run)
        append_jsonl(manifest_path, [result])
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
