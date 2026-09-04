import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
UNSUP=ROOT/'auditor_agent_gnn_baselines_package/server_scripts/v19_unsupervised_graph_baselines.py'
RUNNER=ROOT/'auditor_agent_v23_final_experiments/server_scripts/run_v23_official_blindguard_once.sh'
ORCH=ROOT/'auditor_agent_v23_final_experiments/scripts/run_v23_experiment_suite.py'
TABLE=ROOT/'auditor_agent_v23_final_experiments/scripts/render_experiment_tables.py'

class BlindGuardContract(unittest.TestCase):
 def test_official_scl_not_tam(self):
  s=UNSUP.read_text(encoding='utf-8')
  self.assertIn('from TAM import GATSCL',s)
  self.assertIn('0.8 * torch.norm',s)
  self.assertIn('self.core.neg_all(self.core.encode',s)
  self.assertIn('normal_only_train=True',s)
 def test_v23_runner_is_normal_only_and_pinned(self):
  s=RUNNER.read_text(encoding='utf-8')
  self.assertIn('--model-kind blindguard',s)
  self.assertIn('--batch-size 1',s)
  self.assertIn('--lr 0.001',s)
  self.assertIn('1889c20a326ba9ba9a6982744d473626e74f9986',s)
  self.assertNotIn('--model-kind tam',s)
 def test_required_by_dag_and_table(self):
  self.assertIn("task('blind','BlindGuard SCL V23'",ORCH.read_text(encoding='utf-8'))
  self.assertIn('blindguard_official_v23_v1/test/metrics.json',TABLE.read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
