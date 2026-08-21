import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("heldout", ROOT / "scripts/build_heldout_splits.py")
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)


def row(i, verdict, topology="tree", surface="message", scenario="research"):
    user = json.dumps({"i": i})
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": "{}"}],
            "metadata": {"run_id": f"r{i}", "sample_uid": f"u{i}", "sample_id": i,
                         "source_type": "x", "verdict": verdict, "topology": topology,
                         "surface": surface, "scenario": scenario}}


class TestHeldout(unittest.TestCase):
    def test_topology_removal_and_no_overlap(self):
        train = [row(1, "attack_success"), row(2, "clean_safe", topology="star")]
        val = [row(3, "attack_failed"), row(4, "clean_safe")]
        a, b = MOD.build_fold(train, val, "topology", "tree")
        self.assertEqual([r["metadata"]["run_id"] for r in a], ["r2"])
        self.assertEqual(MOD.audit(a, b, "topology", "tree")["run_id_overlap"], 0)

    def test_surface_keeps_clean_controls(self):
        train = [row(1, "attack_success"), row(2, "clean_safe", surface="none")]
        val = [row(3, "attack_failed"), row(4, "clean_safe", surface="none")]
        a, b = MOD.build_fold(train, val, "surface", "message")
        self.assertEqual(len(a), 1); self.assertEqual(len(b), 2)


if __name__ == "__main__": unittest.main()
