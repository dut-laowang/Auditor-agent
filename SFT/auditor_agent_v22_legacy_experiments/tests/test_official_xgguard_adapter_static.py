import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class TestOfficialXGGuardAdapterStatic(unittest.TestCase):
    def test_no_local_xgguard_model_substitute(self):
        source=(ROOT/"scripts/v22_xgguard_official_adapter.py").read_text(encoding="utf8")
        tree=ast.parse(source)
        classes={x.name for x in ast.walk(tree) if isinstance(x,ast.ClassDef)}
        self.assertNotIn("XGGuardModel",classes)
        self.assertIn('wanted={"GCNEncoder","OursMethod","get_score_overall"}',source)
        self.assertIn("exec(compile(module",source)

    def test_runner_smoke_precedes_sealed_test(self):
        source=(ROOT/"server_scripts/run_v22_official_xgguard_once.sh").read_text(encoding="utf8")
        self.assertLess(source.index('--test "$VAL"'),source.index('--test "$TEST"'))
        self.assertIn("OFFICIAL_XGGUARD_SMOKE: PASS",source)

if __name__=="__main__":unittest.main()
