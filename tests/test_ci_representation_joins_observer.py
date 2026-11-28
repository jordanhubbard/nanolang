"""I check observer admission without restoring or executing compiler products."""
from pathlib import Path
import hashlib, os, subprocess, sys, tempfile, unittest
from unittest import mock
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from ci_original_qualification_policy import require_original_timeout_defaults
from ci_representation_joins_restore import safe_name

class RepresentationJoinObserver(unittest.TestCase):
    def test_retained_paths_are_relative_and_contained(self):
        self.assertEqual(str(safe_name('obj/cache/provider.so')),'obj/cache/provider.so')
        for name in ['', '.', '/tmp/compiler', '../compiler', 'bin/../../compiler']:
            with self.subTest(name=name), self.assertRaises(ValueError): safe_name(name)

    def test_workflow_keeps_all_timeout_defaults(self):
        q=yaml.safe_load((ROOT/'.github/workflows/representation-joins-hosted.yml').read_text())
        def visit(value):
            if isinstance(value,dict):
                if isinstance(value.get('env'),dict): require_original_timeout_defaults(value['env'])
                for child in value.values(): visit(child)
            elif isinstance(value,list):
                for child in value: visit(child)
        visit(q)
        with self.assertRaisesRegex(RuntimeError,'NANO_SHADOW_TIMEOUT_SECONDS'):
            visit({'env':{'NANO_SHADOW_TIMEOUT_SECONDS':'10'}})
        self.assertEqual(q['env']['PRODUCT_REVISION'],'e2ad13e3ee129b926a5cab9b15f22bf61c0ac5d8')
        self.assertEqual(q['permissions'],{'contents':'read','actions':'read'})

    def test_controller_refuses_override_before_namespace(self):
        with tempfile.TemporaryDirectory(prefix='join-observer-policy-') as tmp:
            base=Path(tmp);out=base/'observations'
            command=[sys.executable,str(ROOT/'scripts/ci_representation_joins_qualification.py'),
                '--product',str(base/'absent'), '--observations',str(out),
                '--producer-admission',str(base/'no-admission'), '--qualified-report',str(base/'no-report')]
            result=subprocess.run(command,env={**os.environ,'NANO_SHADOW_TIMEOUT_SECONDS':'10'},capture_output=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'I require original timeout defaults',result.stderr)
            self.assertNotIn(b'FileNotFoundError',result.stderr)
            self.assertFalse(out.exists())

    def test_restore_refuses_wrong_archive_before_workspace_write(self):
        import ci_representation_joins_restore as restore
        with tempfile.TemporaryDirectory(prefix='join-observer-archive-') as tmp:
            base=Path(tmp);archive=base/'wrong.zip';archive.write_bytes(b'wrong')
            output=base/'workspace';admission=base/'admission.json'
            argv=['restore','--archive',str(archive),'--workspace',str(output),
                  '--repo',str(base/'no-repo'),'--admission',str(admission)]
            with mock.patch.object(sys,'argv',argv), mock.patch.dict(os.environ,{'GITHUB_ACTIONS':'true'}), mock.patch.object(restore.platform,'machine',return_value='x86_64'):
                with self.assertRaises(AssertionError): restore.main()
            self.assertFalse(output.exists());self.assertFalse(admission.exists())

    def test_full_prerequisites_and_new_methods_are_selected(self):
        source=(ROOT/'scripts/ci_representation_joins_qualification.py').read_text()
        for module in ['representation_facts','representation_shapes','representation_stack','shape_graph','shape_solver','representation_joins']:
            self.assertEqual(source.count("'tests.test_nanolang_translator_"+module+"'"),1)
        self.assertIn('suite.countTestCases()==32',source)
        self.assertIn("require_original_timeout_defaults(os.environ)",source)
        self.assertIn("q['all_guards_match']",source)
        self.assertIn("q['qualified_guards_match']",source)

    def test_prior_qualification_reader_is_the_verified_original(self):
        reader=ROOT/'scripts/ci_representation_joins_qualified_linux.py'
        self.assertEqual(hashlib.sha256(reader.read_bytes()).hexdigest(),
            '67b30dfc47c02a18edd074c2668dae9a68c847be6fae15c142066c0768a0bca8')

if __name__=='__main__': unittest.main()
