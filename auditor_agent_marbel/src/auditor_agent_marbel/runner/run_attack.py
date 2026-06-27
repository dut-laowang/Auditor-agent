from __future__ import annotations

import argparse
import json
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


def load_run_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_attack_specs(path: Optional[Path]) -> List[AttackSpec]:
    if path is None:
        return []
    return [AttackSpec.from_dict(row) for row in read_jsonl(path)]


def prepare_config(
    sample: Dict[str, Any],
    scenario: str,
    topology: str,
    run_config: Dict[str, Any],
    attack: Optional[AttackSpec],
) -> Dict[str, Any]:
    config = apply_topology(sample, topology)
    config["llm"] = config.get("llm") or run_config.get("default_llm", "gpt-4o-mini")
    overrides = run_config.get("environment_overrides", {}).get(scenario, {})
    environment = dict(config.get("environment", {}))
    environment.update(overrides)
    config["environment"] = environment
    config.setdefault("output", {})
    config.setdefault("memory", {"type": "SharedMemory"})
    config.setdefault("metrics", {})
    config.setdefault("engine_planner", {"initial_progress": "Starting MARBLE attack run."})
    return config


def build_targets(
    marble_root: Path,
    run_config: Dict[str, Any],
    attack_specs: Iterable[AttackSpec],
    clean_only: bool,
) -> List[RunTarget]:
    targets: List[RunTarget] = []
    attacks = list(attack_specs)
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
                targets.append(RunTarget(scenario, sample_id, topology, sample, None))
                if clean_only:
                    continue
                for attack in attacks:
                    if scenario in attack.scenarios and topology in attack.topologies:
                        targets.append(RunTarget(scenario, sample_id, topology, sample, attack))
    return targets


def write_generated_config(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


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

    config_path = output_dir / "configs" / f"{target.run_id}.yaml"
    trajectory_path = output_dir / "trajectories" / f"{target.run_id}.jsonl"
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

        with installed_runtime_hooks(attack, placement, context, logger) as hooks:
            config = Config.load(str(config_path))
            engine = Engine(config)
            hooks.install_memory_attack(engine)
            engine.start()
        logger.close("completed")
        status = "completed"
        error = None
    except Exception as exc:  # pragma: no cover - depends on MARBLE runtime and LLM keys.
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        logger.log("exception", {"error": error, "traceback": traceback.format_exc()})
        logger.close("failed", error=error)
        status = "failed"

    return {**run_meta, "status": status, "error": error, "config_path": str(config_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACI-inspired graph-aware attacks on official MARBLE tasks.")
    parser.add_argument("--marble-root", required=True, type=Path, help="Path to official MARBLE repository.")
    parser.add_argument("--run-config", required=True, type=Path, help="Pilot run YAML config.")
    parser.add_argument("--attack-spec", type=Path, help="JSONL attack spec file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs and trajectories without executing MARBLE.")
    parser.add_argument("--clean-only", action="store_true", help="Run only clean baselines.")
    args = parser.parse_args()

    marble_root = args.marble_root.resolve()
    if not marble_root.exists():
        raise FileNotFoundError(f"MARBLE root does not exist: {marble_root}")

    run_config = load_run_config(args.run_config)
    attack_specs = [] if args.clean_only else load_attack_specs(args.attack_spec)
    targets = build_targets(marble_root, run_config, attack_specs, clean_only=args.clean_only)

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
