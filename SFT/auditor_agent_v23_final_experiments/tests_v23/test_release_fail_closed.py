import json
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "auditor_agent_v23_final_experiments"


class ReleaseFailClosed(unittest.TestCase):
    @staticmethod
    def metric(n):
        return {"n": n, "three_class_accuracy": .5, "binary_accuracy": .6,
                "three_class_report": {"macro avg": {"f1-score": .4}, "attack_success": {"recall": .3}},
                "localization": {"component_micro_f1": .2}}

    def test_missing_graph_metrics_remain_tbd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                [sys.executable, str(PKG / "scripts/render_experiment_tables.py"),
                 "--run-dir", str(root / "run"), "--supplement-dir", str(root / "exp"),
                 "--output-dir", str(root / "tables")], check=True, capture_output=True, text=True,
            )
            status = json.loads((root / "tables/TABLE_STATUS.json").read_text(encoding="utf-8"))
            table = (root / "tables/FINAL_FOUR_TABLES.md").read_text(encoding="utf-8")
            self.assertEqual(status["status"], "INCOMPLETE")
            self.assertGreater(status["required_tbd_cells"], 0)
            gsafeguard = next(line for line in table.splitlines() if "G-Safeguard" in line)
            self.assertIn("TBD", gsafeguard)

    def test_xgguard_is_two_phase_and_atomic(self):
        source = (PKG / "server_scripts/run_v22_official_xgguard_once.sh").read_text(encoding="utf-8")
        self.assertIn("train-validation --model-kind xgguard", source)
        self.assertIn("final-test --checkpoint-dir", source)
        self.assertIn("test.incomplete", source)
        self.assertLess(source.index("train-validation --model-kind xgguard"), source.index("final-test --checkpoint-dir"))
        self.assertNotIn(' --train "$TRAIN" --validation "$VAL" --test "$TEST"', source)

    def test_complete_four_table_fixture_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"run";exp=root/"exp"
            paths=[(run/"qwen3_8b_plain_sft_test/metrics.json",6207),(run/"internlm3_8b_sft_test/metrics.json",6207),(run/"modernbert_test/metrics.json",6167),
                   (exp/"baselines/gsafeguard_official_v23_v1/test/metrics.json",6207),(exp/"baselines/tam_official_v23_v1/test/metrics.json",6207),(exp/"baselines/blindguard_official_v23_v1/test/metrics.json",6207),(exp/"baselines/xgguard_official_v23_v2/test/metrics.json",6207),
                   (exp/"components/fixed_cascade/metrics.json",6167),(exp/"components/rule_router/metrics.json",6167),
                   (exp/"heldout/topology__tree/modernbert_ztr_v2/metrics.json",765),(exp/"heldout/surface__message/modernbert_ztr_v2/metrics.json",3041),(exp/"heldout/scenario__research/modernbert_ztr_v2/metrics.json",1064)]
            for path,n in paths:
                value=self.metric(n)
                if "components" in str(path):value.update({"verify_rate":.15,"verify_rows":925,"corrected":1,"corrupted":0})
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding="utf-8")
            agent=self.metric(6167);agent.update({"rows":6167,"verify_rate":.15,"verify_rows":925,"corrected":1,"corrupted":0})
            p=run/"bounded_agent_common6167/AGENT_TEST_COMPARISON.json";p.parent.mkdir(parents=True);p.write_text(json.dumps(agent),encoding="utf-8")
            out=root/"tables";subprocess.run([sys.executable,str(PKG/"scripts/render_experiment_tables.py"),"--run-dir",str(run),"--supplement-dir",str(exp),"--output-dir",str(out)],check=True,capture_output=True,text=True)
            status=json.loads((out/"TABLE_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"],"PASS")
            self.assertEqual(status["required_tbd_cells"],0)
            self.assertEqual(status["tables"],4)

    def test_orchestrator_validates_before_skip(self):
        source = (PKG / "scripts/run_v23_experiment_suite.py").read_text(encoding="utf-8")
        self.assertIn("SKIPPED_VALIDATED", source)
        self.assertIn("valid_metric", source)
        self.assertNotIn("SKIPPED_COMPLETE", source)

    def test_two_gpu_scheduler_spreads_large_jobs_first(self):
        path=PKG/"scripts/run_v23_experiment_suite.py"
        spec=importlib.util.spec_from_file_location("v23_scheduler",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ids=["0","1"];reserved={"0":0.0,"1":0.0};capacity={"0":81.6,"1":81.6}
        exclusive={"0":False,"1":False}
        first=module.choose_gpu(ids,reserved,capacity,34,True,exclusive);reserved[first]+=34;exclusive[first]=True
        second=module.choose_gpu(ids,reserved,capacity,34,True,exclusive)
        self.assertEqual((first,second),("0","1"))
        self.assertIsNone(module.choose_gpu(ids,reserved,capacity,12,False,{"0":True,"1":True}))

    def test_internlm_runtime_dependency_is_pinned(self):
        requirements=(ROOT.parent/"requirements-v23-h100.txt").read_text(encoding="utf-8")
        bootstrap=(ROOT.parent/"run_v23_h100.sh").read_text(encoding="utf-8")
        self.assertIn("sentencepiece==0.2.0",requirements)
        self.assertIn("import accelerate,datasets,numpy,peft,safetensors,scipy,sentence_transformers,sentencepiece",bootstrap)
        self.assertIn("'sentencepiece':'0.2.0'",bootstrap)

    def test_code_update_resume_requires_explicit_opt_in(self):
        source=(PKG/"scripts/run_v23_experiment_suite.py").read_text(encoding="utf-8")
        self.assertIn("V23_ALLOW_CODE_UPDATE_RESUME",source)
        self.assertIn("RUN_CONTRACT_MIGRATIONS.jsonl",source)


if __name__ == "__main__":
    unittest.main()
