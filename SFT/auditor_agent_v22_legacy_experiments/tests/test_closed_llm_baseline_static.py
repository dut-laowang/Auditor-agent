import ast,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class TestClosedLLM(unittest.TestCase):
 def test_prepare_separates_labels(self):
  s=(ROOT/'scripts/v22_closed_llm_baseline.py').read_text(encoding='utf8');ast.parse(s)
  self.assertIn("V22_CLOSED_LLM_REQUESTS.jsonl",s);self.assertIn("V22_PRIVATE_GOLD.jsonl",s);self.assertIn("request_package_label_blind",s)
  import sys
  sys.path.insert(0,str(ROOT/'scripts'))
  import v22_closed_llm_baseline as closed
  import v22_deepseek_v4_flash_baseline as deepseek
  self.assertEqual(closed.SYSTEM_SUFFIX,deepseek.SYSTEM_SUFFIX)
 def test_both_providers_and_resume(self):
  s=(ROOT/'scripts/v22_closed_llm_baseline.py').read_text(encoding='utf8')
  self.assertIn("choices=['openai','anthropic']",s);self.assertIn("json_schema",s);self.assertIn("tool_choice",s);self.assertIn("if x.get('status')=='ok'",s)
 def test_runner_does_not_score_or_require_gold(self):
  s=(ROOT/'server_scripts/run_v22_closed_llm_colleague_once.sh').read_text(encoding='utf8')
  self.assertNotIn('PRIVATE_GOLD',s);self.assertNotIn(' score ',s);self.assertIn('RAW_RESULTS.tar.gz',s)
if __name__=='__main__':unittest.main()
