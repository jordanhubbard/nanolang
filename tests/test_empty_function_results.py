"""I reject empty result bodies before VM execution and preserve C refusals."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import struct
import sys
import tempfile
import unittest
import zlib
from tests.native_process import native_link_flags, run_captured
from tests.nanolang_translator_routes import build_routes
from tests.test_nanolang_translator_module import packet
from tests.test_nanolang_translator_payloads import table, constant, signature

ROOT=Path(__file__).resolve().parents[1]


def module_bytes(version, functions, declared=0):
    # Each definition is (code, parameter tags, result tags). I retain actual
    # legacy/v2 wire formats; no verifier replacement constructs acceptance.
    strings=[b'main',*[f'helper{i}'.encode() for i in range(1,len(functions))]]
    code=b''.join(item[0] for item in functions);offset=0;rows=[]
    for index,(body,params,results) in enumerate(functions):
        if version==2:
            rows.append(struct.pack('<IIQQHHHH',index,index,offset,len(body),len(params),0,declared,0))
        else:
            rows.append(struct.pack('<IHIIHHBB',index,len(params),offset,len(body),len(params),0,results[0] if results else 0,len(results)))
        offset+=len(body)
    if version==2:
        return packet([(2,table(*(constant(5,s) for s in strings))),
                       (3,table(*(signature(params,results) for _,params,results in functions))),
                       (5,table(*rows)),(6,code)],entry=0)
    pool=b''.join(struct.pack('<I',len(s))+s for s in strings)
    sections=[(2,pool),*([(1,code)] if code else []),(3,b''.join(rows))]
    start=32+len(sections)*12;directory=b'';payload=b''
    for kind,data in sections:
        directory+=struct.pack('<III',kind,start+len(payload),len(data));payload+=data
    body=directory+payload
    return struct.pack('<4s7I',b'NVM\x01',2,1,0,len(sections),start,len(pool),zlib.crc32(body))+body


class EmptyFunctionResults(unittest.TestCase):
    def setUp(self):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        self.work=Path(tempfile.mkdtemp(prefix='empty-result-products-',dir=ROOT/'.test_output'))
        self.record=dict(test=self.id(),steps=[],cases=[],limits=['I preserve zero declared-depth absence and existing C optional-body pruning. I require complete VM verification of all functions and prior-output refusal for required invalid C bodies.'])
        print(f'I retain empty-result products at {self.work}',flush=True)
        self.addCleanup(self.save)

    def save(self):
        (self.work/'manifest.json').write_text(json.dumps(self.record,indent=2)+'\n')

    def run_product(self,args,*,build=False):
        name=f'step-{len(self.record["steps"]):04d}'
        r=run_captured(args,cwd=ROOT,timeout=30,build=build)
        (self.work/(name+'.stdout')).write_bytes(r.stdout);(self.work/(name+'.stderr')).write_bytes(r.stderr)
        self.record['steps'].append(dict(name=name,argv=list(map(str,args)),returncode=r.returncode));self.save();return r

    def fixture(self,name,version,functions,declared=0):
        path=self.work/(name+'.nvm');data=module_bytes(version,functions,declared);path.write_bytes(data)
        self.record['cases'].append(dict(name=name,version=version,declared=declared,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));self.save();return path

    def test_result_bodies_refuse_before_execution(self):
        protected=self.work/'prior.c';sentinel=b'I retain my prior C output.\n'
        for version in (1,2):
            for declared in ((0,) if version==1 else (0,1)):
                for tag in range(1,16):
                    for count in (1,2,255):
                        name=f'v{version}-declared-{declared}-tag-{tag}-count-{count}'
                        with self.subTest(case=name):
                            path=self.fixture(name,version,[(b'',[],[tag]*count)],declared)
                            expected=(b'module was built for NanoISA v1' if version==1 else f'I require a body for function[0] with {count} declared result(s).'.encode())
                            for args in ([ROOT/'bin/nano_vm','--verify-only',path],[ROOT/'bin/nano_vm',path]):
                                r=self.run_product(args);self.assertNotEqual(r.returncode,0);self.assertIn(expected,r.stderr)
                            protected.write_bytes(sentinel)
                            r=self.run_product([ROOT/'bin/nvm2c',path,'-o',protected])
                            self.assertNotEqual(r.returncode,0);self.assertEqual(protected.read_bytes(),sentinel)

    def test_nonentry_results_are_not_exempt_from_vm_verification(self):
        for version in (1,2):
            for called in (False,True):
                prefix=bytes([59])+struct.pack('<I',1)+bytes([8]) if called else b''
                main=prefix+bytes([1])+struct.pack('<q',0)+bytes([61])
                path=self.fixture(f'v{version}-helper-called-{called}',version,[(main,[],[1]),(b'',[],[1])])
                for args in ([ROOT/'bin/nano_vm','--verify-only',path],[ROOT/'bin/nano_vm',path]):
                    r=self.run_product(args);self.assertNotEqual(r.returncode,0);self.assertIn(b'module was built for NanoISA v1' if version==1 else b'I require a body for function[1] with 1 declared result(s).',r.stderr)
                if called:
                    protected=self.work/'prior-helper.c';sentinel=b'I retain my prior helper output.\n';protected.write_bytes(sentinel)
                    r=self.run_product([ROOT/'bin/nvm2c',path,'-o',protected]);self.assertNotEqual(r.returncode,0);self.assertEqual(protected.read_bytes(),sentinel)

    def test_empty_void_bodies_and_nonempty_returns_still_run(self):
        compiler=shlex.split(os.environ.get('CC','cc'))
        for version in (1,2):
            path=self.fixture(f'v{version}-void-main',version,[(b'',[],[])])
            for args in ([ROOT/'bin/nano_vm','--verify-only',path],[ROOT/'bin/nano_vm',path]):
                r=self.run_product(args)
                if version==1:self.assertNotEqual(r.returncode,0);self.assertIn(b'module was built for NanoISA v1',r.stderr)
                else:self.assertEqual(r.returncode,0,r.stderr.decode())
            if version==1:continue
            for arguments in (0,1):
                values=(bytes([1])+struct.pack('<q',7))*arguments
                main=values+bytes([59])+struct.pack('<I',1)+bytes([1])+struct.pack('<q',0)+bytes([61])
                path=self.fixture(f'v{version}-void-helper-args-{arguments}',version,[(main,[],[1]),(b'',[1]*arguments,[])],declared=1)
                for args in ([ROOT/'bin/nano_vm','--verify-only',path],[ROOT/'bin/nano_vm',path]):
                    r=self.run_product(args);self.assertEqual(r.returncode,0,r.stderr.decode())
                source=path.with_suffix('.c');native=path.with_suffix('.native')
                r=self.run_product([ROOT/'bin/nvm2c',path,'-o',source]);self.assertEqual(r.returncode,0,r.stderr.decode())
                r=self.run_product([*compiler,'-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',source,
                    ROOT/'bin/nano_aot_runtime.o','-lm',*(['-ldl','-Wl,--export-dynamic'] if sys.platform.startswith('linux') else []),*native_link_flags(),'-o',native],build=True)
                self.assertEqual(r.returncode,0,r.stderr.decode());r=self.run_product([native]);self.assertEqual(r.returncode,0,r.stderr.decode())

    def test_compilers_preserve_empty_void_source_bodies(self):
        source=self.work/'empty_void.nano'
        source.write_text('''fn leave_empty(value: int) -> void {}
shadow leave_empty {
    let value: int = 7
    (leave_empty value)
    assert (== value 7)
}
fn main() -> int {
    (leave_empty 7)
    (println "I preserve an empty void source body.")
    return 0
}
''')
        self.record['routes']=[]
        for name,args in build_routes(source,self.work):
            r=self.run_product(args);self.assertEqual(r.returncode,0,r.stderr.decode())
            self.assertEqual(r.stdout,b'I preserve an empty void source body.\n')
            self.record['routes'].append(dict(name=name,argv=list(map(str,args))));self.save()
