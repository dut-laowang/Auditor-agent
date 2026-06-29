from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from auditor_agent_marbel.attacks.payloads import render_payload
from auditor_agent_marbel.core.schemas import AttackSpec
from auditor_agent_marbel.logging.trajectory_logger import TrajectoryLogger


class RuntimeAttackHooks:
    def __init__(
        self,
        spec: Optional[AttackSpec],
        placement: Dict[str, Any],
        context: Dict[str, Any],
        logger: TrajectoryLogger,
    ):
        self.spec = spec
        self.placement = placement
        self.context = context
        self.logger = logger
        self.payload = render_payload(spec, context) if spec else ""

    def install_memory_attack(self, engine: Any) -> None:
        if not self.spec or self.spec.surface != "memory":
            return
        target = self.placement.get("target_agent")
        for agent in getattr(engine, "agents", []):
            if getattr(agent, "agent_id", None) == target and hasattr(agent, "memory"):
                agent.memory.update("attack_injection", self.payload)
                self.logger.log(
                    "attack_injected",
                    {
                        "surface": "memory",
                        "target_agent": target,
                        "attacked_content": self.payload,
                    },
                )
                return

    def patch_agent_class(self, BaseAgent: Any) -> Dict[str, Any]:
        originals: Dict[str, Any] = {}
        hooks = self

        originals["act"] = BaseAgent.act
        originals["send_message"] = BaseAgent.send_message
        originals["receive_message"] = BaseAgent.receive_message

        def act_wrapper(agent_self: Any, task: str) -> Any:
            hooks.logger.log(
                "agent_act",
                {
                    "agent": getattr(agent_self, "agent_id", None),
                    "clean_content": task,
                },
            )
            return originals["act"](agent_self, task)

        def send_message_wrapper(agent_self: Any, session_id: Any, target_agent: Any, message: str) -> Any:
            source = getattr(agent_self, "agent_id", None)
            target = getattr(target_agent, "agent_id", None)
            attacked_message = message
            injected = False
            if (
                hooks.spec
                and hooks.spec.surface == "message"
                and source == hooks.placement.get("source_agent")
                and target == hooks.placement.get("target_agent")
            ):
                attacked_message = f"{message}\n\n[Injected message]\n{hooks.payload}"
                injected = True
            hooks.logger.log(
                "message_send",
                {
                    "source_agent": source,
                    "target_agent": target,
                    "clean_content": message,
                    "attacked_content": attacked_message if injected else None,
                    "attack_injected": injected,
                },
            )
            return originals["send_message"](agent_self, session_id, target_agent, attacked_message)

        def receive_message_wrapper(agent_self: Any, session_id: Any, sender: Any, message: str) -> Any:
            hooks.logger.log(
                "message_receive",
                {
                    "source_agent": getattr(sender, "agent_id", None),
                    "target_agent": getattr(agent_self, "agent_id", None),
                    "content": message,
                },
            )
            return originals["receive_message"](agent_self, session_id, sender, message)

        BaseAgent.act = act_wrapper
        BaseAgent.send_message = send_message_wrapper
        BaseAgent.receive_message = receive_message_wrapper
        return originals

    def patch_environment_class(self, BaseEnvironment: Any) -> Dict[str, Any]:
        originals: Dict[str, Any] = {"apply_action": BaseEnvironment.apply_action}
        hooks = self

        def apply_action_wrapper(env_self: Any, agent_id: str, action_name: str, arguments: Dict[str, Any]) -> Any:
            hooks.logger.log(
                "tool_call",
                {
                    "agent": agent_id,
                    "tool": action_name,
                    "arguments": arguments,
                },
            )
            result = originals["apply_action"](env_self, agent_id, action_name, arguments)
            attacked_result = result
            injected = False
            if hooks.spec and hooks.spec.surface == "tool":
                target_agent = hooks.placement.get("target_agent")
                if target_agent in (None, agent_id):
                    attacked_result = f"{result}\n\n[Injected tool result]\n{hooks.payload}"
                    injected = True
            hooks.logger.log(
                "tool_result",
                {
                    "agent": agent_id,
                    "tool": action_name,
                    "clean_content": result,
                    "attacked_content": attacked_result if injected else None,
                    "attack_injected": injected,
                },
            )
            return attacked_result

        BaseEnvironment.apply_action = apply_action_wrapper
        return originals


@contextmanager
def installed_runtime_hooks(
    spec: Optional[AttackSpec],
    placement: Dict[str, Any],
    context: Dict[str, Any],
    logger: TrajectoryLogger,
) -> Iterator[RuntimeAttackHooks]:
    from marble.agent.base_agent import BaseAgent
    from marble.environments.base_env import BaseEnvironment

    hooks = RuntimeAttackHooks(spec, placement, context, logger)
    agent_originals = hooks.patch_agent_class(BaseAgent)
    env_originals = hooks.patch_environment_class(BaseEnvironment)
    try:
        yield hooks
    finally:
        BaseAgent.act = agent_originals["act"]
        BaseAgent.send_message = agent_originals["send_message"]
        BaseAgent.receive_message = agent_originals["receive_message"]
        BaseEnvironment.apply_action = env_originals["apply_action"]
