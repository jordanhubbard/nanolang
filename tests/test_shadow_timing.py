"""I qualify bounded timing metadata without executing compiler programs."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest
from tests.native_sdk_runner import run

ROOT = Path(__file__).resolve().parents[1]


class ShadowTiming(unittest.TestCase):
    def test_standalone_controls(self):
        work = Path(tempfile.mkdtemp(prefix='shadow-timing-', dir=os.environ.get('SHADOW_TIMING_REPORT_DIR')))
        print('I retain timing control artifacts at ' + str(work), flush=True)
        cc = shlex.split(os.environ.get('CC', 'cc'))
        compiler = Path(shutil.which(cc[0])).resolve()
        inputs = [ROOT/'src/runtime/shadow_timing.h', ROOT/'tests/shadow_timing_probe.c',
                  ROOT/'tests/test_shadow_timing.py', ROOT/'tests/native_sdk_runner.py', compiler]
        def inventory():
            return {str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                               bytes=p.stat().st_size, mode=p.stat().st_mode & 0o7777) for p in inputs}
        before = inventory()
        (work/'inputs-before.json').write_text(json.dumps(before, indent=2)+'\n')
        executable = work/'probe'
        try:
            run(work,'compiler-version',cc+['--version'],ROOT)
            run(work,'compile',[*cc,'-std=c99','-Wall','-Wextra','-Werror','-pedantic',
                '-I'+str(ROOT/'src'),ROOT/'tests/shadow_timing_probe.c','-o',executable],ROOT)
            for label,value in [('absent',None),('empty',''),('zero','0'),('leading-zero','01'),('word','true')]:
                out,err,_=run(work,label,[executable],ROOT,{'NANO_SHADOW_TIMING':value})
                self.assertEqual(out,b'');self.assertEqual(err,b'')
            for label,args,count in [('enabled',[],2),('cap',['cap'],4096)]:
                out,err,_=run(work,label,[executable,*args],ROOT,{'NANO_SHADOW_TIMING':'1'})
                self.assertEqual(out,b'')
                lines=[line for line in err.splitlines() if line]
                self.assertEqual(len(lines),count)
                records=[]
                for i,line in enumerate(lines):
                    self.assertTrue(line.startswith(b'NANO_SHADOW_TIMING '))
                    row=json.loads(line[len(b'NANO_SHADOW_TIMING '):])
                    self.assertEqual(set(row),{'phase','source','item','ordinal','children','status',
                        'wall_ok','wall_sec','wall_nsec','cpu_ok','user_sec','user_usec','sys_sec','sys_usec'})
                    self.assertEqual(row['source'],7);self.assertEqual(row['item'],11)
                    self.assertEqual(row['ordinal'],i);self.assertEqual(row['children'],i%2)
                    self.assertEqual(row['status'],0);self.assertEqual(row['wall_ok'],1);self.assertEqual(row['cpu_ok'],1)
                    self.assertGreaterEqual(row['wall_sec'],0)
                    self.assertTrue(0 <= row['wall_nsec'] < 1000000000)
                    for field in ('user','sys'):
                        self.assertGreaterEqual(row[field+'_sec'],0)
                        self.assertTrue(0 <= row[field+'_usec'] < 1000000)
                    expected='record_limit' if i==4095 else ('test_end' if i%2 else 'test_start')
                    self.assertEqual(row['phase'],expected)
                    if records:self.assertGreaterEqual((row['wall_sec'],row['wall_nsec']),
                        (records[-1]['wall_sec'],records[-1]['wall_nsec']))
                    records.append(row)
                (work/(label+'-parsed.json')).write_text(json.dumps(records,indent=2)+'\n')
        finally:
            after=inventory();(work/'inputs-after.json').write_text(json.dumps(after,indent=2)+'\n')
            self.assertEqual(before,after)


if __name__ == '__main__':
    unittest.main()
