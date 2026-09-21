"""I compare direct LLVM products against every retained VM observation."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from tests import test_file_cyclic
from tests import test_record_array_execution
from tests import test_record_array_vm
from tests import test_record_array_generated

ROOT = Path(__file__).resolve().parents[1]


class RecordArrayLLVM(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command
    providers = test_record_array_generated.RecordArrayGenerated.providers

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-record-array-llvm-'))
        print(f'I retain direct LLVM artifacts at {cls.artifacts}', flush=True)
        cls.cc = shlex.split(os.environ.get('RECORD_GENERATED_CC', 'cc'))
        cls.flags = ['-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra', '-Werror',
                     *shlex.split(os.environ.get('RECORD_GENERATED_CFLAGS', '')),
                     '-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME', '-DNANO_RECORD_ARRAY_GENERATED_PRIVATE',
                     '-Isrc', '-Isrc/nanoisa', '-Itests/nanoisa', '-Iobj/nanoisa']
        cls.vm_objects = shlex.split(os.environ['RECORD_GENERATED_VM_OBJECTS'])
        cls.query_objects = shlex.split(os.environ['RECORD_GENERATED_QUERY_OBJECTS'])
        cls.ldflags = shlex.split(os.environ.get('RECORD_GENERATED_LDFLAGS', '-lm -lcrypto -lffi -pthread'))
        cls.native_ldflags = shlex.split(os.environ.get('RECORD_GENERATED_NATIVE_LDFLAGS', '-lm -pthread'))
        cls.ir_asan = os.environ.get('RECORD_LLVM_IR_ASAN', '0') == '1'
        cls.tools = {name: shlex.split(os.environ.get('RECORD_LLVM_' + name.upper().replace('-', '_'), name))
                     for name in ('clang', 'opt', 'llc', 'nm', 'wasmtime', 'node')}
        os.environ['LSAN_OPTIONS'] = ''
        (cls.artifacts / 'selection.json').write_text(json.dumps({
            'python': sys.executable, 'cc': cls.cc, 'flags': cls.flags, 'tools': cls.tools,
            'vm_inputs': cls.vm_objects, 'query_inputs': cls.query_objects,
            'ldflags': cls.ldflags, 'native_ldflags': cls.native_ldflags,
            'IR_ASan': cls.ir_asan, 'LSAN_OPTIONS': '',
            'product_link_closure': ['actual emitted LLVM', 'record_array_generated_private.c',
                                     'captured replay', 'test allocator only in observed native mode'],
            'wasm_memory': {'initial': 1048576, 'maximum': 67108864, 'refusal_maximum': 4194304},
            'scope': 'Selected rebuilt providers; native IR ASan explicit, no retroactive LLVM UBSan claim.'}, indent=2)+'\n')

    def test_01_emission_bounds(self):
        objects = self.providers('emission', test_record_array_execution.PROVIDERS,
                                 self.query_objects, observed=True)
        output = self.artifacts / 'emission'
        self.command('emission-build', [*self.cc, *self.flags,
            'tests/nanoisa/test_record_array_llvm_emission.c', 'tests/nanoisa/record_array_alloc.c',
            *objects, *self.ldflags, '-o', str(output)])
        result = self.command('emission-run', [str(output)])
        self.assertEqual(result.count(b'all93 emission recipes and all256 decisions'), 2)
        self.assertEqual(result.count(b'actual emission allocation positions in both modes'), 2)

    def test_00_factored_c_bytes(self):
        baseline=Path(os.environ['RECORD_LLVM_C_BASELINE'])
        self.assertTrue(baseline.is_dir())
        objects=self.providers('c-parity',test_record_array_vm.PROVIDERS,
            self.vm_objects,omit=('src/nanovm/vm.c',))
        executable=self.artifacts/'c-parity-capture'
        self.command('c-parity-build',[*self.cc,*self.flags,
            'tests/nanoisa/test_record_array_generated_capture.c','src/nanoisa/nvm2c_record_array_private.c',
            'tests/nanoisa/record_array_alloc.c',*objects,*self.ldflags,'-o',str(executable)])
        corpus=self.artifacts/'c-parity';corpus.mkdir()
        self.command('c-parity-run',[str(executable),str(corpus)])
        proof={}
        for index in range(73):
            for suffix in ('c','replay.c'):
                name=f'product-{index:04d}.'+suffix
                before=(baseline/name).read_bytes();after=(corpus/name).read_bytes()
                self.assertEqual(before,after,name)
                proof[name]=hashlib.sha256(after).hexdigest()
        self.assertEqual(json.loads((corpus/'corpus-counts.json').read_text())['products'],73)
        (self.artifacts/'c-parity.json').write_text(json.dumps({'baseline':str(baseline),'exact':proof},indent=2)+'\n')

    def capture(self, wasm):
        label = 'wasm' if wasm else 'native'
        objects = self.providers(label+'-capture', test_record_array_vm.PROVIDERS,
                                 self.vm_objects, omit=('src/nanovm/vm.c',))
        output = self.artifacts / (label+'-capture')
        self.command(label+'-capture-build', [*self.cc, *self.flags, f'-DRECORD_LLVM_WASM32={int(wasm)}',
            'tests/nanoisa/test_record_array_llvm_capture.c', 'src/nanoisa/nvm2llvm_record_array_private.c',
            'tests/nanoisa/record_array_alloc.c', *objects, *self.ldflags, '-o', str(output)])
        corpus = self.artifacts / (label+'-corpus'); corpus.mkdir()
        result = self.command(label+'-capture-run', [str(output), str(corpus)])
        self.assertIn(b'all93 actual retired operations and all256 decisions', result)
        counts = json.loads((corpus/'corpus-counts.json').read_text())
        self.assertEqual(counts['products'], 73); self.assertEqual(counts['actions'], 404)
        sources = [corpus/f'product-{i:04d}.ll' for i in range(73)]
        replays = [corpus/f'product-{i:04d}.replay.c' for i in range(73)]
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources, *replays]}
        (self.artifacts/(label+'-corpus-before.json')).write_text(json.dumps(manifest, indent=2)+'\n')
        target = json.loads((ROOT/'obj/nanoisa/managed_runtime_ir.json').read_text())['variants']['wasm32' if wasm else 'native']
        for source in sources:
            text = source.read_text()
            self.assertEqual(re.findall(r'^target datalayout = "([^"]+)"$', text, re.M), [target['layout']])
            self.assertEqual(re.findall(r'^target triple = "([^"]+)"$', text, re.M), [target['triple']])
        self.abi_probe(wasm, sources[0])
        return sources, replays, manifest

    @staticmethod
    def abi_signature(line):
        head, args = line.split('(', 1)
        before, name = head.rsplit('@', 1)
        tokens = before.split()
        result = next(t for t in reversed(tokens) if re.fullmatch(r'i\d+|void|ptr|double', t))
        def parameter(value):
            words = value.strip().split()
            return (words[0], 'zeroext' in words, 'signext' in words)
        params = args.split(')', 1)[0]
        return name, (result, 'zeroext' in tokens, 'signext' in tokens,
                      tuple(parameter(v) for v in params.split(',')) if params.strip() else ())

    def abi_probe(self, wasm, emitted):
        label = 'wasm' if wasm else 'native'
        output = self.artifacts/(label+'-actual-c-abi.ll')
        target = ['--target=wasm32-unknown-unknown', '-ffreestanding', '-fno-builtin'] if wasm else []
        self.command(label+'-abi-build', [*self.tools['clang'], *target, '-std=c11', '-O0',
            '-DNANO_RECORD_ARRAY_GENERATED_PRIVATE', '-Isrc/nanoisa', '-S', '-emit-llvm',
            'src/nanoisa/record_array_generated_private.c', '-o', str(output)])
        actual = dict(self.abi_signature(line) for line in output.read_text().splitlines()
                      if line.startswith('define ') and re.search(r'@nrg_\w+\(', line))
        declared = dict(self.abi_signature(line) for line in emitted.read_text().splitlines()
                        if line.startswith('declare ') and '@nrg_' in line)
        self.assertGreater(len(declared), 35)
        for name, signature in declared.items():
            self.assertIn(name, actual); self.assertEqual(signature, actual[name], name)
        (self.artifacts/(label+'-abi-comparison.json')).write_text(json.dumps({
            'declared': declared, 'actual_C': {name: actual[name] for name in declared},
            'equal': True}, indent=2)+'\n')

    def llvm_object(self, source, label, optimization, wasm=False):
        selected = self.artifacts/(label+'.optimized.ll')
        self.command(label+'-opt', [*self.tools['opt'], '-S', '-passes=default<'+optimization+'>',
                                   str(source), '-o', str(selected)])
        if self.ir_asan and not wasm:
            self.assertTrue(any('fsanitize=' in flag and 'address' in flag for flag in self.flags))
            text, count = re.subn(r'^(define [^\n]+) \{', r'\1 sanitize_address {', selected.read_text(), flags=re.M)
            self.assertGreater(count, 0)
            marked = self.artifacts/(label+'.marked.ll'); marked.write_text(text)
            selected = self.artifacts/(label+'.asan.ll')
            self.command(label+'-asan', [*self.tools['opt'], '-S', '-passes=asan', str(marked), '-o', str(selected)])
            self.assertIn('__asan_report_', selected.read_text())
        self.command(label+'-verify', [*self.tools['opt'], '-passes=verify', '-disable-output', str(selected)])
        obj = self.artifacts/(label+'.o')
        self.command(label+'-object', [*self.tools['llc'], '-filetype=obj', '-relocation-model='+('static' if wasm else 'pic'),
                                      str(selected), '-o', str(obj)])
        return obj

    def startup_controls(self, source, wasm, multiple):
        prefix='wasm-startup' if wasm else 'native-startup'
        flags=(['--target=wasm32-unknown-unknown','-ffreestanding','-fno-builtin',
                '-std=c11','-Wall','-Wextra','-Werror','-DNANO_RECORD_ARRAY_GENERATED_PRIVATE',
                '-Isrc/nanoisa','-Itests/nanoisa'] if wasm else self.flags)
        cc=self.tools['clang'] if wasm else self.cc
        runtime=self.artifacts/(prefix+'-runtime.o')
        testing=['-DNMS_TESTING','-DNMS_TEST_ALLOC_HOOKS']
        self.command(prefix+'-runtime-build',[*cc,*flags,*testing,
            '-Dnrg_layout_field=nrg_layout_field_actual','-Dnrg_create=nrg_create_actual',
            '-c','src/nanoisa/record_array_generated_private.c','-o',str(runtime)])
        original=source.read_text()
        corrupted,count=re.subn(r'^(@program = private constant %P \{ i32 )1,',r'\g<1>2,',original,flags=re.M)
        self.assertEqual(count,1)
        bad=self.artifacts/(prefix+'-table-mismatch.ll');bad.write_text(corrupted)
        text=multiple.read_text();line=re.search(r'^@functions = .*$',text,re.M)
        self.assertIsNotNone(line)
        fields=list(re.finditer(r'%Fn \{ i32 (\d+)',line.group(0)))
        self.assertGreaterEqual(len(fields),2)
        field=fields[1];begin=line.start()+field.start(1);end=line.start()+field.end(1)
        different=text[:begin]+str(int(field.group(1))+1)+text[end:]
        callee=self.artifacts/(prefix+'-callee-mismatch.ll');callee.write_text(different)
        for index,selected in enumerate((source,bad,callee)):
            name=prefix+('-table' if index==1 else '-callee' if index==2 else '-abi')
            obj=self.llvm_object(selected,name,'O0',wasm=wasm)
            output=self.artifacts/(name+('.wasm' if wasm else ''))
            linkflags=(['-nostdlib','-Wl,--no-entry','-Wl,--export=nano_main',
                        '-Wl,--initial-memory=1048576','-Wl,--max-memory=67108864'] if wasm else self.native_ldflags)
            self.command(name+'-link',[*cc,*flags,*testing,
                *(['-DNRG_EXPECT_TABLE_REFUSAL'] if index else []),str(obj),
                'tests/nanoisa/test_record_array_llvm_startup.c',str(runtime),*linkflags,'-o',str(output)])
            if wasm:
                for engine in ('wasmtime','node'):
                    self.assertEqual(self.wasm_invoke(name+'-'+engine,engine,output,'nano_main'),0)
            else:self.command(name+'-run',[str(output)])

    def native_faults(self, product, name):
        result = self.command(name+'-baseline', [str(product), '--fault-baseline'])
        rows = re.findall(rb'^NRG_FAULT_BASELINE calls=(\d+) peak=(\d+)$', result, re.M)
        self.assertEqual(len(rows), 1); calls, peak = map(int, rows[0]); self.assertGreater(calls, 0)
        plan = {'calls': calls, 'peak': peak, 'modes': 2, 'complete': False, 'workers': []}
        path = self.artifacts/(name+'-fault-coverage.json'); path.write_text(json.dumps(plan)+'\n')
        cursor = recoveries = 0
        for begin in range(0, calls, 16):
            end = min(begin+16, calls); self.assertEqual(begin, cursor)
            output = self.command(f'{name}-fault-{begin:06d}-{end:06d}',
                [str(product), '--fault-range', str(begin), str(end), str(calls)])
            rows = re.findall(rb'^NRG_FAULT_RANGE begin=(\d+) end=(\d+) calls=(\d+) modes=(\d+) recoveries=(\d+) peak=(\d+)$', output, re.M)
            self.assertEqual(len(rows), 1)
            self.assertEqual(tuple(map(int, rows[0])), (begin, end, calls, 2, 2*(end-begin), peak))
            plan['workers'].append([begin, end, 2, 2*(end-begin)])
            cursor=end; recoveries+=2*(end-begin)
        self.assertEqual(cursor, calls); self.assertEqual(recoveries, 2*calls)
        plan.update(complete=True, recoveries=recoveries); path.write_text(json.dumps(plan, indent=2)+'\n')

    def test_02_native_corpus(self):
        sources, replays, original = self.capture(False)
        self.startup_controls(sources[0],False,next(p for p in sources if int(re.search(r'@functions = private constant \[(\d+)',p.read_text()).group(1))>1))
        for optimization in ('O0', 'O2'):
            for observed in (False, True):
                mode=optimization+('-observed' if observed else '-linked')
                testing=['-DNMS_TESTING', '-DNRG_OBSERVED'] if observed else []
                hooks=['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if observed else []
                runtime=self.artifacts/(mode+'-runtime.o')
                self.command(mode+'-runtime-build', [*self.cc, *self.flags, '-'+optimization, *testing, *hooks,
                    '-c', 'src/nanoisa/record_array_generated_private.c', '-o', str(runtime)])
                if not observed:
                    lifecycle=self.artifacts/(mode+'-lifecycle')
                    self.command(mode+'-lifecycle-build',[*self.cc,*self.flags,'-'+optimization,
                        'tests/nanoisa/test_record_array_generated_lifecycle.c',str(runtime),
                        *self.native_ldflags,'-o',str(lifecycle)])
                    self.assertIn(b'ABI, wrong thread, BUSY, acquired finish, retained result and recovery',
                        self.command(mode+'-lifecycle-run',[str(lifecycle)]))
                for i,(source,replay) in enumerate(zip(sources,replays)):
                    name=f'{mode}-{i:04d}'; obj=self.llvm_object(source,name,optimization)
                    product=self.artifacts/name
                    support=['tests/nanoisa/record_array_alloc.c'] if observed else []
                    self.command(name+'-link', [*self.cc, *self.flags, '-'+optimization, *testing,
                        str(obj), str(replay), str(runtime), *support, *self.native_ldflags, '-o', str(product)])
                    symbols=self.command(name+'-nm', [*self.tools['nm'], '-u', str(product)])
                    for forbidden in (b'vm_core_execute', b'vm_record_array', b'isa_decode', b'nvm_prepare'):
                        self.assertNotIn(forbidden,symbols)
                    if observed:self.native_faults(product,name)
                    else:self.assertIn(b'generated replay observations; status 0', self.command(name+'-run',[str(product)]))
        self.assertEqual(original,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources,*replays]})

    def wasm_invoke(self, name, engine, product, export, arguments=()):
        if engine=='wasmtime':
            result=self.command(name,[*self.tools['wasmtime'],'run','--invoke',export,str(product),*map(str,arguments)])
            rows=result.decode().strip().splitlines(); self.assertEqual(len(rows),1)
            return int(rows[0])
        script=self.artifacts/'wasm-run.cjs'
        if not script.exists():
            script.write_text('const fs=require("fs");\n'
                'const m=new WebAssembly.Module(fs.readFileSync(process.argv[2]));\n'
                'if(WebAssembly.Module.imports(m).length)throw Error("imports");\n'
                'const i=new WebAssembly.Instance(m);\n'
                'const r=i.exports[process.argv[3]](...process.argv.slice(4).map(Number));\n'
                'console.log(String(r));\n')
        result=self.command(name,[*self.tools['node'],str(script),str(product),export,*map(str,arguments)])
        return int(result.decode().strip())

    def wasm_faults(self, product, name, engine):
        packed=self.wasm_invoke(name+'-'+engine+'-baseline',engine,product,'nano_baseline_report')
        calls,peak=packed>>32,packed&0xffffffff; self.assertGreater(calls,0); self.assertGreater(peak,8*1024*1024)
        memory=self.wasm_invoke(name+'-'+engine+'-memory',engine,product,'nano_memory_report')
        before,after=memory>>32,memory&0xffffffff
        self.assertEqual(before,16);self.assertGreater(after,before);self.assertLessEqual(after,1024)
        plan={'calls':calls,'requested_live_peak':peak,'linear_memory_before_pages':before,
              'linear_memory_after_pages':after,'page_bytes':65536,
              'engine':engine,'modes':2,'complete':False,'workers':[]}
        path=self.artifacts/(name+'-'+engine+'-coverage.json'); path.write_text(json.dumps(plan)+'\n')
        cursor=recoveries=0
        for begin in range(0,calls,16):
            end=min(begin+16,calls); self.assertEqual(begin,cursor)
            value=self.wasm_invoke(f'{name}-{engine}-fault-{begin:06d}-{end:06d}',engine,product,
                'nano_range_report',(begin,end,calls))
            self.assertEqual(value,(calls<<32)|2*(end-begin))
            plan['workers'].append([begin,end,2,2*(end-begin)]); cursor=end; recoveries+=2*(end-begin)
        self.assertEqual(cursor,calls); self.assertEqual(recoveries,2*calls)
        plan.update(complete=True,recoveries=recoveries); path.write_text(json.dumps(plan,indent=2)+'\n')

    def test_03_wasm_corpus(self):
        sources,replays,original=self.capture(True)
        self.startup_controls(sources[0],True,next(p for p in sources if int(re.search(r'@functions = private constant \[(\d+)',p.read_text()).group(1))>1))
        flags=['--target=wasm32-unknown-unknown','-std=c11','-ffreestanding','-fno-builtin',
               '-Wall','-Wextra','-Werror','-DNANO_RECORD_ARRAY_GENERATED_PRIVATE','-Isrc/nanoisa','-Itests/nanoisa']
        for optimization in ('O0','O2'):
            for observed in (False,True):
                mode='wasm-'+optimization+('-observed' if observed else '-linked')
                testing=['-DNMS_TESTING','-DNMS_TEST_ALLOC_HOOKS','-DNRG_OBSERVED'] if observed else []
                runtime=self.artifacts/(mode+'-runtime.o')
                self.command(mode+'-runtime-build',[*self.tools['clang'],*flags,'-'+optimization,*testing,
                    '-c','src/nanoisa/record_array_generated_private.c','-o',str(runtime)])
                if not observed:
                    symbols=self.command(mode+'-nm',[*self.tools['nm'],str(runtime)])
                    self.assertNotIn(b'nms_test_',symbols)
                for i,(source,replay) in enumerate(zip(sources,replays)):
                    name=f'{mode}-{i:04d}'; obj=self.llvm_object(source,name,optimization,wasm=True)
                    product=self.artifacts/(name+'.wasm')
                    exports=['nano_main']+(['nano_baseline_report','nano_range_report','nano_memory_report','nano_memory_refusal'] if observed else [])
                    link=[*self.tools['clang'],*flags,'-'+optimization,*testing,'-nostdlib',str(obj),str(replay),str(runtime),
                          '-Wl,--no-entry','-Wl,--initial-memory=1048576','-Wl,--max-memory=67108864',
                          *['-Wl,--export='+export for export in exports],'-o',str(product)]
                    self.command(name+'-link',link)
                    for engine in ('wasmtime','node'):
                        self.assertEqual(self.wasm_invoke(name+'-'+engine+'-run',engine,product,'nano_main'),0)
                        if observed:self.wasm_faults(product,name,engine)
                    if observed and i==0:
                        limited=self.artifacts/(name+'-limited.wasm')
                        limited_link=[arg.replace('--max-memory=67108864','--max-memory=4194304') for arg in link]
                        limited_link[-1]=str(limited); self.command(name+'-limited-link',limited_link)
                        for engine in ('wasmtime','node'):
                            self.assertEqual(self.wasm_invoke(name+'-'+engine+'-memory-limit',engine,limited,'nano_memory_refusal'),0)
        self.assertEqual(original,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources,*replays]})


if __name__=='__main__':
    unittest.main()
