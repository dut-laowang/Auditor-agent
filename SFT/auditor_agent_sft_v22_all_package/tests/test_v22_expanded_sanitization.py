import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "auditor_agent_sft_v19_qualityfix_package"
    / "scripts"
    / "build_v19_qualityfix_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("build_v19_qualityfix_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_structural_sanitizer_preserves_json_envelope_and_nested_json():
    nested = json.dumps({"note": "[External instruction]: preserve the task"})
    sample = {
        "messages": [{
            "role": "user",
            "content": json.dumps({
                "quoted": 'say "hello"',
                "nested": nested,
                "truncated": "ordinary evidence ... [Injected memory:rnd00441",
            }),
        }]
    }
    MODULE.sanitize_sample_messages(sample, {}, "run")
    user = json.loads(sample["messages"][0]["content"])
    assert user["quoted"] == 'say "hello"'
    assert json.loads(user["nested"])["note"] == "preserve the task"
    assert "Injected" not in user["truncated"]
