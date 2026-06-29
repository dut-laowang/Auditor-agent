from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VALID_SURFACES = {"input", "profile", "message", "memory", "tool"}
VALID_OBJECTIVES = {"disclosure", "disruption", "hijacking"}
VALID_TOPOLOGIES = {"graph", "star", "tree"}


@dataclass(frozen=True)
class AttackSpec:
    attack_id: str
    surface: str
    objective: str
    scenarios: List[str]
    topologies: List[str]
    placement: Dict[str, Any]
    payload_template: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AttackSpec":
        spec = AttackSpec(
            attack_id=str(data["attack_id"]),
            surface=str(data["surface"]),
            objective=str(data["objective"]),
            scenarios=list(data.get("scenarios", [])),
            topologies=list(data.get("topologies", [])),
            placement=dict(data.get("placement", {})),
            payload_template=str(data["payload_template"]),
            metadata=dict(data.get("metadata", {})),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.surface not in VALID_SURFACES:
            raise ValueError(f"Unsupported attack surface: {self.surface}")
        if self.objective not in VALID_OBJECTIVES:
            raise ValueError(f"Unsupported attack objective: {self.objective}")
        unknown_topologies = set(self.topologies) - VALID_TOPOLOGIES
        if unknown_topologies:
            raise ValueError(f"Unsupported topologies: {sorted(unknown_topologies)}")
        if not self.scenarios:
            raise ValueError(f"Attack {self.attack_id} has no scenarios")
        if not self.topologies:
            raise ValueError(f"Attack {self.attack_id} has no topologies")


@dataclass(frozen=True)
class RunTarget:
    scenario: str
    sample_id: int
    topology: str
    sample: Dict[str, Any]
    attack: Optional[AttackSpec] = None

    @property
    def run_id(self) -> str:
        attack_id = self.attack.attack_id if self.attack else "clean"
        return f"{self.scenario}_task{self.sample_id:04d}_{self.topology}_{attack_id}"

