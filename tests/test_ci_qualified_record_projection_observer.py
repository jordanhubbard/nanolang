"""I refuse an unqualified Darwin worker and inherited test-limit changes."""
from pathlib import Path
import os,subprocess,sys,tempfile,unittest,yaml
ROOT=Path(__file__).resolve().parents[1]
class ProjectionObserver(unittest.TestCase):
    def test_darwin_uses_the_established_pinned_worker(self):
        workflow=yaml.safe_load((ROOT/'.github/workflows/qualified-record-projection-hosted.yml').read_text())
        self.assertEqual(workflow['jobs']['projection']['strategy']['matrix']['os'],['xcode-27'])
        selects=[s for s in workflow['jobs']['projection']['steps'] if s.get('run')=='python3 scripts/ci_select_capture_clang.py']
        self.assertEqual(len(selects),1)
        self.assertEqual(workflow['env']['PRODUCT_REVISION'],'a52e44117f2623acc401d0fe0c10024a29b130dd')

    def test_inherited_deadline_changes_refuse_before_namespace(self):
        for variable in ['CMD_TIMEOUT','BOOTSTRAP_NANOISA_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS','BOOTSTRAP_NANOISA_SHADOW_TIMEOUT']:
            with self.subTest(variable=variable),tempfile.TemporaryDirectory(prefix='projection-policy-') as directory:
                base=Path(directory);output=base/'result'
                result=subprocess.run([sys.executable,str(ROOT/'scripts/ci_qualified_record_projection.py'),'--old',str(base/'absent-old'),'--product',str(base/'absent-product'),'--output',str(output)],env={**os.environ,variable:'1'},capture_output=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(b'I require original timeout defaults',result.stderr)
                self.assertFalse(output.exists())
if __name__=='__main__':unittest.main()
