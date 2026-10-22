"""I qualify exact consuming FLOAT locals without widening signatures or fields."""
import copy
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import reconstruct_owned, OwnedAnalyze, OwnedEmit
from nanoisa_reconstruction import Expr, FLOAT, Refusal
from tests import test_file_source_plan as retained_runner
from tests.native_process import native_link_flags
from tests import test_owned_record_patterns as patterns
PREFIX = patterns.PREFIX


class OwnedFloatReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='owned-float-reconstruction-'))
        cls.serial = 0
        cls.tools = Path(os.environ.get('NANOLANG_FLOAT_TOOLS', ROOT/'bin'))
        cls.fact_tool = Path(os.environ.get('NANOLANG_FLOAT_FACTS', cls.tools/'nanoisa_hl_facts'))
        cls.compilers = [cls.tools/'nano_virt'] + [Path(p) for p in shlex.split(
            os.environ.get('NANOLANG_OWNED_RECONSTRUCTION_COMPILERS', ''))]
        print('I retain FLOAT evidence at '+str(cls.work), flush=True)

    def command(self, args, expected=0):
        type(self).serial += 1
        name = 'command-'+str(self.serial)
        try:
            out, err = retained_runner.FileSourcePlan.command.__func__(type(self), name, args, timeout=180)
        except AssertionError:
            status = json.loads((self.work/(name+'-status.json')).read_text())
            self.assertFalse(status['timeout'] or status['cleanup_errors'] or not status['group_absent'])
            self.assertNotEqual(expected, 0, status)
            out = (self.work/(name+'.stdout')).read_bytes()
        status = json.loads((self.work/(name+'-status.json')).read_text())
        if expected == 0:self.assertEqual(status['returncode'], 0)
        else:self.assertNotEqual(status['returncode'], 0)
        return out

    def source(self):
        sources = []
        case = patterns.OwnedRecordPatterns()
        case.check_case = lambda source, *args, **kwargs: sources.append(PREFIX+source)
        case.test_unsafe_pattern_keeps_outer_shadow()
        self.assertEqual(len(sources), 1)
        return sources[0]

    def module(self, text, compiler=None):
        source = self.work/('source-'+str(self.serial)+'.nano');source.write_text(text)
        module = source.with_suffix('.nvm')
        self.command([compiler or self.compilers[0], source, '--emit-nvm', '-o', module])
        self.command([self.tools/'nano_vm', '--verify-only', module])
        self.command([self.tools/'nano_vm', module])
        return module

    def facts(self, module):return json.loads(self.command([self.fact_tool, module]))

    def qualify(self, module):
        facts = self.facts(module)
        for language in ('nano', 'c'):
            output = self.work/('generated-'+str(self.serial)+'.'+language)
            output.write_text(reconstruct_owned(facts, language))
            if language == 'nano':
                for compiler in self.compilers:self.module(output.read_text(), compiler)
            else:
                for opt in ('-O0', '-O2'):
                    native = output.with_suffix('.'+opt[1:]+'.exe')
                    self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')), '-std=c11',opt,
                        '-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                        output,self.tools/'nano_aot_runtime.o','-lm',*native_link_flags(),'-o',native])
                    self.command([native])
        return facts

    def test_unchanged_ordinary_and_independent_original_shadow(self):
        for compiler in self.compilers:self.qualify(self.module(self.source(), compiler))
        driver = os.environ.get('NANOLANG_OWNED_FLOAT_SHADOW_EMITTER')
        source = self.work/'original-shadow-source.nano';source.write_text(self.source())
        assembly = self.work/'original-shadow.nasm'
        assembly.write_bytes(self.command([driver,source,'0','raw']) if driver else
                             (ROOT/'docs/evidence/owned-float-reconstruction-baseline-20261019/original-shadow.nasm').read_bytes())
        module = assembly.with_suffix('.nvm')
        self.command([self.tools/'nanoisa','asm',assembly,'-o',module])
        self.command([self.tools/'nano_vm','--verify-only',module]);self.command([self.tools/'nano_vm',module])
        self.qualify(module)

    def test_finite_literal_bits_across_actual_producers(self):
        facts = self.facts(self.module(self.source()))
        functions = [OwnedAnalyze(facts,i).run() for i in range(len(facts['functions']))]
        emitter = OwnedEmit(functions,'nano',facts['layouts'])
        for magnitude in (0, 1, 0x000fffffffffffff, 0x0010000000000000,
                          0x7fefffffffffffff, 0x3ff0000000000001):
            for_signs = (magnitude, magnitude | (1 << 63))
            for bits in for_signs:
                literal = emitter.expression(Expr(FLOAT,'float_bits',bits))
                source = self.source().replace('2.5', literal)
                for compiler in self.compilers:
                    with self.subTest(bits=hex(bits), compiler=str(compiler)):
                        module = self.module(source,compiler)
                        actual = self.facts(module)
                        # I observe actual decoded literal/negation transport, not
                        # an inspection opcode absent from the original profile.
                        values=[]
                        for function in actual['functions']:
                            for ins in function['code']:
                                if ins['op']=='PUSH_F64':values.append(int(ins['f64_bits'],16))
                                elif ins['op']=='F64_NEG':
                                    self.assertTrue(values);values[-1] ^= 1<<63
                        self.assertIn(bits,values,(hex(bits),compiler,values))

    def test_nonfinite_and_malformed_literal_target_boundaries(self):
        facts = self.facts(self.module(self.source()))
        for bits in (0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000042):
            changed=copy.deepcopy(facts)
            for fn in changed['functions']:
                for ins in fn['code']:
                    if ins['op']=='PUSH_F64':ins['f64_bits']=f'{bits:016x}'
            self.assertIn(f'0x{bits:016x}', reconstruct_owned(changed,'c'))
            with self.assertRaises(Refusal):reconstruct_owned(changed,'nano')
        for text in ('0','x'*16,'0'*17):
            changed=copy.deepcopy(facts)
            next(ins for fn in changed['functions'] for ins in fn['code'] if ins['op']=='PUSH_F64')['f64_bits']=text
            for language in ('nano','c'):
                with self.assertRaises(Refusal):reconstruct_owned(changed,language)

    def test_nonfinite_actual_module_refuses_atomically_only_for_nano(self):
        original=self.module(self.source())
        assembly=self.command([self.tools/'nanoisa','dump',original]).decode()
        for bits in ('7ff0000000000000','fff0000000000000','7ff8000000000042'):
            text=re.sub(r'PUSH_F64 bits:[0-9a-fA-F]{16}', 'PUSH_F64 bits:'+bits, assembly)
            source=self.work/('nonfinite-'+bits+'.nasm');source.write_text(text)
            module=source.with_suffix('.nvm')
            self.command([self.tools/'nanoisa','asm',source,'-o',module])
            self.command([self.tools/'nano_vm','--verify-only',module])
            output=source.with_suffix('.nano');output.write_text('prior-output')
            self.command([sys.executable,ROOT/'scripts/nvm2hl.py','--language','nano',module,'-o',output],expected=1)
            self.assertEqual(output.read_text(),'prior-output')
            c=source.with_suffix('.c')
            self.command([sys.executable,ROOT/'scripts/nvm2hl.py','--language','c',module,'-o',c])
            self.assertIn('0x'+bits,c.read_text())

    def test_float_signature_field_and_uninitialized_refusals(self):
        facts = self.facts(self.module(self.source()))
        cases = []
        changed=copy.deepcopy(facts);changed['layouts'][0]['fields'][0]=FLOAT
        changed['layouts'][0]['field_types'][0]=[FLOAT,4294967295];cases.append(changed)
        changed=copy.deepcopy(facts);changed['functions'][1]['params'][0]=[FLOAT,4294967295]
        changed['functions'][1]['local_types'][0]=[FLOAT,4294967295];cases.append(changed)
        changed=copy.deepcopy(facts);changed['functions'][0]['result']=FLOAT
        changed['functions'][0]['result_type']=[FLOAT,4294967295];cases.append(changed)
        changed=copy.deepcopy(facts)
        # I remove the actual initialization before the source's later read.
        code=changed['functions'][0]['code']
        store=next(i for i, ins in enumerate(code) if ins['op']=='STORE_LOCAL' and
                   changed['functions'][0]['local_types'][ins['arg']][0]==FLOAT)
        code[store]['op']='POP';code[store]['arg']=0;cases.append(changed)
        for case in cases:
            for language in ('nano','c'):
                with self.assertRaises(Refusal):reconstruct_owned(case,language)

    def test_dynamic_nan_copy_and_typed_predicates(self):
        text = PREFIX+'''fn main() -> int {
 let mut huge: float = 10000000000.0
 set huge (* huge huge) set huge (* huge huge) set huge (* huge huge)
 set huge (* huge huge) set huge (* huge huge)
 let nan: float = (- huge huge)
 let copy: float = nan
 assert (!= copy nan) assert (not (== copy nan))
 assert (not (< copy 1.0)) assert (not (> copy 1.0))
 assert (not (<= copy 1.0)) assert (not (>= copy 1.0))
 let owner: Handle = Handle { fd: 7 }
 assert (== (close owner) 7)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        for compiler in self.compilers:self.qualify(self.module(text,compiler))
