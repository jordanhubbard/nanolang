"""I observe unchanged C join allocation failures without a pure-language OOM claim."""
from pathlib import Path
import hashlib,json,os,shlex,tempfile,unittest
from tests.nanolang_translator_routes import checked
from tests.native_process import native_link_flags
from tests.test_nanolang_translator_representation_joins import helpers,raw,join,program,packet
ROOT=Path(__file__).resolve().parents[1]
class JoinAllocationFailures(unittest.TestCase):
    def test_branch_plan_and_both_field_clone_failures(self):
        with tempfile.TemporaryDirectory(prefix='nano-join-c-allocation-') as directory:
            work=Path(directory);source=(ROOT/'src/nanoisa/nvm2c.c').read_bytes();actual,types,spans=helpers(source)
            (work/'nvm2c_representation_joins_actual.h').write_bytes(actual)
            (work/'nvm2c_representation_joins_types.h').write_bytes(types)
            manifest=dict(whole_release=False,qualification=False,c_only_allocator_observations=True,nanolang_oom_equivalence=False,source_sha256=hashlib.sha256(source).hexdigest(),actual_helpers_sha256=hashlib.sha256(actual).hexdigest(),spans=spans,runs=[])
            cases={
                'branch':packet([program([raw(0,2),join()],mode=1)]),
                'plan':packet([program([raw(8,3),join(),join()],mode=1)]),
                'record':packet([program([raw(4,fields=(2,2)),join(),join()],mode=1)]),
            }
            for name,data in cases.items():(work/(name+'.bin')).write_bytes(data)
            for mode,flags in [('ordinary',['-O2']),('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
                binary=work/('reference-'+mode)
                checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror','-D_POSIX_C_SOURCE=200809L','-DNANO_JOIN_ALLOC_FAULT_OBSERVER=1',*flags,'-I',work,ROOT/'tests/nvm2c_representation_joins_reference.c',ROOT/'src/nanoisa/isa.c',*native_link_flags(),'-o',binary],build=True)
                for name,fail,calls,blocks,error in [
                    ('branch',0,1,0,''),('plan',0,2,0,''),('record',0,3,2,''),
                    ('branch',1,1,0,'I cannot allocate classifier branch stack'),
                    ('plan',2,2,0,'I cannot allocate scalar join facts'),
                    ('record',2,2,0,'I cannot allocate aggregate field facts'),
                    ('record',3,3,1,'I cannot allocate aggregate field facts'),
                ]:
                    with self.subTest(mode=mode,case=name,fail=fail):
                        text=checked([binary,'--fail-at='+str(fail),work/(name+'.bin')],timeout=60)
                        self.assertEqual(text.splitlines()[-1],f'F|{calls}|{fail}|{blocks}')
                        self.assertEqual(sum(line.startswith('Z|') for line in text.splitlines()),1)
                        self.assertIn('Z|0|'+str(2 if name=='branch' else 3)+'|0|'+str(bool(error)*1)+'|'+error+'|',text)
                        if name=='branch' and fail:
                            self.assertIn('J|0|0|0|0|0|0\n',text);self.assertNotIn('\nT|',text)
                        else:
                            changed=int(name=='plan' and not fail);maps=changed
                            self.assertIn(f'J|1|1|1|{changed}|0|{maps}\n',text)
                        if name=='plan':
                            self.assertIn('T|0|8|-1|0|0|3|0|\n',text)
                            if fail:self.assertNotIn('\nP|',text)
                            else:self.assertIn('P|0|7|30|1|3\n',text)
                        if name=='record':
                            expected='T|0|4|-1|0|0|0|0|\n' if fail==3 else 'T|0|4|-1|1|0|0|0|2,2\n'
                            self.assertIn(expected,text)
                        path=work/(f'{mode}-{name}-{fail}.reports');path.write_text(text)
                        manifest['runs'].append(dict(mode=mode,case=name,fail_at=fail,allocations=calls,owned_field_blocks=blocks,sha256=hashlib.sha256(text.encode()).hexdigest()))
            self.assertEqual(len(manifest['runs']),14)
            (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':unittest.main()
