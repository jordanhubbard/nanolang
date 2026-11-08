"""I execute the actual C-created source emitter and its binary-string VM output."""
import hashlib
import json
from pathlib import Path
import tempfile

from tests import test_bstring_vm_contracts as binary
from tests.native_process import run_captured

ROOT = binary.ROOT


class BstringSourceVmContracts(binary.BstringVmContracts):
    artifact_directory = 'bstring-source-vm-contracts'
    scope = 'I test actual source NanoISA lowering and VM execution; native BSTRING translation remains separate.'

    @classmethod
    def setUpClass(cls):
        output = ROOT/'.test_output/bstring-source-compiler'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='actual-native-', dir=output))
        compiler = ROOT/'bin/nanoc_c'
        source = ROOT/'src_nano/nanoc_v06.nano'
        cls.compiler = work/'compiler'
        paths = sorted((ROOT/'src_nano').rglob('*.nano'))
        paths += [compiler, ROOT/'modules/compiler_support/compiler_support.nano',
                  ROOT/'modules/nanoisa/nanoisa.nano']
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        product_before = None
        commands = []
        try:
            argv = [compiler, source, '--keep-c', '-o', cls.compiler]
            result = run_captured(argv, cwd=ROOT, timeout=60, build=True)
            (work/'source-build.stdout').write_bytes(result.stdout)
            (work/'source-build.stderr').write_bytes(result.stderr)
            commands.append({'label': 'source-build', 'argv': list(map(str, argv)),
                             'returncode': result.returncode})
            if result.returncode:
                raise AssertionError(result.stdout.decode(errors='replace')+result.stderr.decode(errors='replace'))
            magic = cls.compiler.read_bytes()[:4]
            if magic not in [b'\x7fELF', bytes.fromhex('cffaedfe'), bytes.fromhex('feedfacf')]:
                raise AssertionError('I require an actual native source emitter.')
            product_before = hashlib.sha256(cls.compiler.read_bytes()).hexdigest()
        finally:
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            product_after = hashlib.sha256(cls.compiler.read_bytes()).hexdigest() if cls.compiler.is_file() else None
            (work/'commands.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in commands))
            (work/'inputs.json').write_text(json.dumps({'scope': 'I create the source emitter through the actual C seed with default selected shadows.',
                'native_lifecycle_admitted': False, 'fresh_stages_qualified': False,
                'before': before, 'after': after, 'product_before_run': product_before,
                'product_after_run': product_after}, indent=2)+'\n')
            if before != after or (product_before is not None and product_before != product_after):
                raise AssertionError('I require unchanged source/provider inputs and a stable completed compiler product.')

    def test_lexical_callee_and_constructor_argument_order(self):
        self.exercise('''let mut trace:int=0
fn mark(value:bstring,digit:int)->bstring{
 set trace (+ (* trace 10) digit)
 return value
}
shadow mark {
 let value:bstring=(bstr_new "x")
 set trace 0
 assert (== (mark value 3) value)
 assert (== trace 3)
 (bstr_free value)
}
fn add_two(value:int)->int{return (+ value 2)}
shadow add_two{assert (== (add_two 40) 42)}
fn apply(bstr_length:fn(int)->int,value:int)->int{return (bstr_length value)}
shadow apply{assert (== (apply add_two 40) 42)}
fn forty_two()->int{return 42}
shadow forty_two{assert (== (forty_two) 42)}
fn apply_free(bstr_free:fn()->int)->int{return (bstr_free)}
shadow apply_free{assert (== (apply_free forty_two) 42)}
fn identity(value:bstring)->bstring{return value}
shadow identity {
 let value:bstring=(bstr_new "held")
 assert (== (identity value) value)
 (bstr_free value)
}
fn main()->int{
 assert (== (apply add_two 40) 42)
 assert (== (apply_free forty_two) 42)
 let first:bstring=(bstr_new "left")
 let second:bstring=(bstr_new "right")
 set trace 0
 let joined:bstring=(bstr_concat (mark first 1) (mark second 2))
 assert (== trace 12)
 assert (== (bstr_to_cstr (identity joined)) "leftright")
 (bstr_free first) (bstr_free second) (bstr_free joined)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_binary_owner_operators_refuse_mixed_or_ordered_operands(self):
        for expression in ['(< value value)', '(+ value value)', '(== value "x")']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{let value:bstring=(bstr_new "x") '
                              +expression+' return 0}\nshadow main{assert true}\n', refused=True)
