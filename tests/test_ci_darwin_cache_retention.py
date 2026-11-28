"""I retry collection only; I preserve every original Darwin cache recipe."""
from pathlib import Path
import itertools, os, subprocess, sys, tempfile, unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from ci_darwin_cache_retention import GATES, retention_status
from ci_original_qualification_policy import require_original_timeout_defaults

class DarwinCacheRetention(unittest.TestCase):
    def test_upload_success_never_covers_failed_or_missing_product_gates(self):
        for values in itertools.product(['success','failure'],repeat=len(GATES)):
            gates=dict(zip(GATES,values))
            for attempts in itertools.product(['success','failure','skipped'],repeat=3):
                self.assertEqual(retention_status(attempts,gates),
                    all(v=='success' for v in values) and 'success' in attempts)
        with self.assertRaises(ValueError):retention_status(['success']*3,{})
        with self.assertRaises(ValueError):retention_status(['success']*4,dict.fromkeys(GATES,'success'))
        with self.assertRaises(ValueError):retention_status(['unknown']*3,dict.fromkeys(GATES,'success'))

    def test_original_cache_recipes_flags_conditions_and_deadlines_are_exact(self):
        old=yaml.safe_load((ROOT/'.github/workflows/c-repair-original-gates.yml').read_text())['jobs']['original-gates']
        new=yaml.safe_load((ROOT/'.github/workflows/c-repair-darwin-cache-retention.yml').read_text())['jobs']['original-darwin-cache']
        self.assertEqual(new['timeout-minutes'],old['timeout-minutes'])
        self.assertEqual(new['strategy']['matrix'],{'os':['xcode-27'],'gate':['cache']})
        names=['Require storage headroom','Install Darwin dependencies','Select the pinned Darwin compiler and SDK',
            'Install Python dependencies','Record source and tool identities','Build fresh production tools',
            'Check managed STRING publication and exact alias retention','Execute my array repair regression',
            'Execute the complete original cache and worker gates','Verify source and selected tool guards',
            'Print retained bootstrap failure logs']
        original={s.get('name'):s for s in old['steps']};changed={s.get('name'):s for s in new['steps']}
        for name in names:
            with self.subTest(step=name):
                for key in ['run','timeout-minutes','if','env']:
                    self.assertEqual(changed[name].get(key),original[name].get(key),(name,key))
                self.assertNotIn('continue-on-error',changed[name])
        checkout=changed['Check out Darwin product and original test inputs']
        self.assertEqual(checkout['with']['ref'],'${{ env.PRODUCT_REVISION }}')
        self.assertEqual(checkout['with']['path'],'product')

    def test_only_three_upload_steps_retry_and_never_overwrite(self):
        q=yaml.safe_load((ROOT/'.github/workflows/c-repair-darwin-cache-retention.yml').read_text())
        steps=q['jobs']['original-darwin-cache']['steps']
        uploads=[s for s in steps if s.get('uses')=='actions/upload-artifact@v4']
        self.assertEqual(len(uploads),3)
        self.assertEqual([s['id'] for s in uploads],['retain_1','retain_2','retain_3'])
        self.assertEqual(len({s['with']['name'] for s in uploads}),3)
        self.assertEqual(len({s['with']['path'] for s in uploads}),1)
        for i,s in enumerate(uploads,1):
            self.assertTrue(s['continue-on-error']);self.assertFalse(s['with'].get('overwrite',False))
            self.assertIn('always()',s['if'])
            if i>1:self.assertIn("steps.retain_"+str(i-1)+".outcome == 'failure'",s['if'])
        self.assertEqual({s['id'] for s in steps if s.get('continue-on-error')},{'retain_1','retain_2','retain_3'})
        self.assertEqual(steps[-1]['name'],'Require retained bytes and every original gate')
        self.assertNotIn('--observe-only',steps[-1]['run'])

    def test_workflow_keeps_original_timeouts_and_exact_product(self):
        q=yaml.safe_load((ROOT/'.github/workflows/c-repair-darwin-cache-retention.yml').read_text())
        self.assertEqual(q['env']['PRODUCT_REVISION'],'ddd3b5a4ff35678d1906c07c08fc6bbc39b041de')
        def visit(value):
            if isinstance(value,dict):
                if isinstance(value.get('env'),dict):require_original_timeout_defaults(value['env'])
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(q)
        with self.assertRaisesRegex(RuntimeError,'NANO_SHADOW_TIMEOUT_SECONDS'):
            visit({'env':{'NANO_SHADOW_TIMEOUT_SECONDS':'10'}})

    def test_terminal_command_fails_when_all_uploads_fail_and_retains_record(self):
        with tempfile.TemporaryDirectory(prefix='darwin-retention-control-') as tmp:
            output=Path(tmp)/'terminal.json'
            env={**os.environ,'PRODUCT_REVISION':'ddd3b5a4ff35678d1906c07c08fc6bbc39b041de',
                'GITHUB_SHA':'observer-control','GITHUB_RUN_ID':'no-native-run'}
            env.update({'CACHE_GATE_'+name.upper():'success' for name in GATES})
            env.update({'RETENTION_'+str(i):'failure' for i in range(1,4)})
            args=[sys.executable,str(ROOT/'scripts/ci_darwin_cache_retention.py'),'--output',str(output)]
            failed=subprocess.run(args,env=env,capture_output=True,timeout=10)
            self.assertNotEqual(failed.returncode,0);self.assertTrue(output.is_file())
            self.assertIn(b'I retain failure',failed.stderr)
            env.update(RETENTION_2='success',RETENTION_3='skipped')
            passed=subprocess.run(args,env=env,capture_output=True,timeout=10)
            self.assertEqual(passed.returncode,0,passed.stderr)
            env['CACHE_GATE_CACHE']='failure'
            refused=subprocess.run(args,env=env,capture_output=True,timeout=10)
            self.assertNotEqual(refused.returncode,0)

if __name__=='__main__':unittest.main()
