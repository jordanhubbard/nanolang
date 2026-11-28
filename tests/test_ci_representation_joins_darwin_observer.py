"""I admit only my actual Darwin producer and preserve every original join method."""
from pathlib import Path
import hashlib,os,subprocess,sys,tempfile,unittest,yaml
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import ci_representation_joins_darwin_restore as restore
from ci_original_qualification_policy import require_original_timeout_defaults
class DarwinJoinObserver(unittest.TestCase):
    def test_selected_worker_toolchain_and_method_limits(self):
        q=yaml.safe_load((ROOT/'.github/workflows/representation-joins-darwin-hosted.yml').read_text())
        self.assertEqual(q['jobs']['original-component']['runs-on'],'xcode-27')
        self.assertEqual(q['env']['PRODUCT_REVISION'],'8e422db88ba093c7d277e93b1e120c236acd85b9')
        def visit(value):
            if isinstance(value,dict):
                if isinstance(value.get('env'),dict):require_original_timeout_defaults(value['env'])
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(q)
        runs=[s.get('run','') for s in q['jobs']['original-component']['steps']]
        self.assertIn('python3 scripts/ci_select_capture_clang.py',runs)
        self.assertTrue(any('--cohort darwin-bootstrap' in s for s in runs))
        self.assertTrue(any('actions/artifacts/15254109040/zip' in s for s in runs))

    def test_linux_elf_and_wrong_macho_headers_refuse(self):
        restore.require_arm64_macho(bytes.fromhex('cffaedfe0c000001'))
        for image in [b'',b'\x7fELF\x02\x01'+b'\0'*20,bytes.fromhex('cffaedfe07000001')]:
            with self.subTest(image=image),self.assertRaisesRegex(ValueError,'ARM64 Mach-O'):restore.require_arm64_macho(image)

    def test_wrong_archive_refuses_before_workspace_creation(self):
        with tempfile.TemporaryDirectory(prefix='darwin-join-archive-') as tmp:
            base=Path(tmp);archive=base/'wrong.zip';archive.write_bytes(b'wrong');workspace=base/'workspace';admission=base/'admission.json'
            with mock.patch.object(sys,'argv',['restore','--archive',str(archive),'--workspace',str(workspace),'--repo',str(base/'absent'),'--admission',str(admission)]),mock.patch.dict(os.environ,{'GITHUB_ACTIONS':'true'}),mock.patch.object(restore.platform,'system',return_value='Darwin'),mock.patch.object(restore.platform,'machine',return_value='arm64'):
                with self.assertRaises(AssertionError):restore.main()
            self.assertFalse(workspace.exists());self.assertFalse(admission.exists())

    def test_timeout_changes_refuse_before_namespace(self):
        for name in ['CMD_TIMEOUT','BOOTSTRAP_NANOISA_TIMEOUT','BOOTSTRAP_NANOISA_SHADOW_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS']:
            with self.subTest(variable=name),tempfile.TemporaryDirectory(prefix='darwin-join-policy-') as tmp:
                base=Path(tmp);out=base/'out';r=subprocess.run([sys.executable,str(ROOT/'scripts/ci_representation_joins_darwin_qualification.py'),'--product',str(base/'absent'),'--observations',str(out),'--producer-admission',str(base/'none'),'--qualified-report',str(base/'none2')],env={**os.environ,name:'1'},capture_output=True,timeout=10)
                self.assertNotEqual(r.returncode,0);self.assertIn(b'I require original timeout defaults',r.stderr);self.assertFalse(out.exists())

    def test_execution_body_is_the_exact_linux33_body(self):
        original=subprocess.check_output(['git','show','586b7df63823bef290caaeea1f7750064ab106e3:scripts/ci_representation_joins_qualification.py'],cwd=ROOT,text=True)
        current=(ROOT/'scripts/ci_representation_joins_darwin_qualification.py').read_text();anchor='from tests import native_process\n'
        self.assertEqual(current[current.index(anchor):],original[original.index(anchor):])
        self.assertIn('suite.countTestCases()==33',current)
        for module in ['representation_facts','representation_shapes','representation_stack','shape_graph','shape_solver','representation_joins']:
            self.assertEqual(current.count("'tests.test_nanolang_translator_"+module+"'"),1)
        self.assertEqual(current.count("'tests.test_nvm2c_representation_join_allocation'"),1)

    def test_qualification_reader_is_the_original_verified_reader(self):
        self.assertEqual(hashlib.sha256((ROOT/'scripts/ci_representation_joins_qualified_linux.py').read_bytes()).hexdigest(),'67b30dfc47c02a18edd074c2668dae9a68c847be6fae15c142066c0768a0bca8')

if __name__=='__main__':unittest.main()
