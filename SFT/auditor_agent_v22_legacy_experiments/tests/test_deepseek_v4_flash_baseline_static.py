import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class TestDeepSeekV4FlashBaseline(unittest.TestCase):
    def test_inference_is_label_blind_and_fixed_model(self):
        source=(ROOT/"scripts/v22_deepseek_v4_flash_baseline.py").read_text(encoding="utf-8")
        tree=ast.parse(source)
        infer=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=="infer")
        infer_source=ast.get_source_segment(source,infer)
        self.assertNotIn('messages"][2]',infer_source)
        self.assertIn('MODEL = "deepseek-v4-flash"',source)
        self.assertIn('"thinking": {"type": "disabled"}',source)
        self.assertIn('response_format={"type": "json_object"}',source)

    def test_resume_progress_and_key_safety(self):
        source=(ROOT/"scripts/v22_deepseek_v4_flash_baseline.py").read_text(encoding="utf-8")
        self.assertIn('if item.get("status") == "ok"',source)
        self.assertIn('set_postfix(',source)
        self.assertNotIn('"api_key": key',source)
        runner=(ROOT/"server_scripts/run_v22_deepseek_v4_flash_table1_once.sh").read_text(encoding="utf-8")
        self.assertIn('${EVAL_API_KEY:-}',runner)
        self.assertIn('V22_DEEPSEEK_V4_FLASH_TABLE1_RESULTS.tar.gz',runner)

if __name__=="__main__": unittest.main()
