"""I require on-time completion and a normally reaped compiler shadow child."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from tests.native_sdk_runner import run
ROOT=Path(__file__).resolve().parents[1]

class ShadowCompletion(unittest.TestCase):
    def test_exact_decision_boundaries(self):
        report=Path(os.environ['NANO_SHADOW_COMPLETION_REPORT'])
        report.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix='decision-',dir=report))
        binary=work/'probe'
        cc=shlex.split(os.environ['NANO_SHADOW_COMPLETION_CC'])
        flags=shlex.split(os.environ.get('NANO_SHADOW_COMPLETION_CFLAGS',''))
        run(work,'compile',cc+flags+['-std=c11','-D_POSIX_C_SOURCE=200809L','-Wall','-Wextra','-Werror','-I',ROOT/'src',ROOT/'tests/shadow_completion_probe.c','-o',binary],ROOT,timeout=120,track_descendants=True)
        out,err,_=run(work,'probe',[binary],ROOT,track_descendants=True)
        self.assertEqual((out,err),(b'',b''))

    def test_actual_compiler_success_failure_and_live_deadline(self):
        report=Path(os.environ['NANO_SHADOW_COMPLETION_REPORT'])
        report.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix='compiler-',dir=report))
        manifest=json.loads(Path(os.environ['NANO_SHADOW_COMPLETION_COMPILER']).read_text())
        compiler=Path(manifest['path'])
        def identity():
            return dict(sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),bytes=compiler.stat().st_size,mode=compiler.stat().st_mode&0o7777)
        expected={key:manifest[key] for key in ('sha256','bytes','mode')}
        self.assertEqual(identity(),expected)
        (work/'producer-before.json').write_text(json.dumps(manifest,indent=2)+'\n')
        try:
            for name,shadow,success in [('success','assert true',True),('failure','assert false',False),('deadline','while true { }',False)]:
                source=work/(name+'.nano')
                source.write_text('fn main() -> int { return 0 }\nshadow main { '+shadow+' }\n')
                output=work/name;sentinel=b'completion output sentinel\n';output.write_bytes(sentinel)
                shadow_json=work/(name+'-shadows.json')
                out,err,status=run(work,name+'-compile',[compiler,source,'-o',output,'--llm-shadow-json',shadow_json],ROOT,{'NANO_SHADOW_TIMEOUT_SECONDS':'1','NANO_SHADOW_TIMING':None},expected=(0,) if success else (1,),timeout=120,track_descendants=True)
                self.assertFalse(status['timeout'])
                self.assertTrue(status['group_absent'] and status['descendants_absent'])
                if success:
                    info=json.loads(shadow_json.read_text())
                    self.assertTrue(info['completed'] and info['success'])
                    self.assertEqual(info['test_count'],1);self.assertEqual(info['failures'],[])
                    actual,_,_=run(work,name+'-execute',[output],ROOT,track_descendants=True)
                    self.assertEqual(actual,b'')
                else:
                    self.assertEqual(output.read_bytes(),sentinel)
                    self.assertIn(b'I stopped shadow execution after 1 seconds.' if name=='deadline' else b'Shadow tests failed',out+err)
        finally:
            (work/'producer-after.json').write_text(json.dumps(identity(),indent=2)+'\n')
            self.assertEqual(identity(),expected)
