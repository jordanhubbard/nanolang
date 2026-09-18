"""I qualify conditional source operands through all three canonical producers."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''let mut trace: int = 0
fn mark(digit: int, result: bool) -> bool {
    set trace (+ (* trace 10) digit)
    return result
}
shadow mark { set trace 0 assert (mark 1 true) assert (== trace 1) set trace 0 }
fn main() -> int {
    set trace 0
    assert (not (and (mark 1 false) (mark 2 true)))
    assert (== trace 1)
    set trace 0
    assert (not (and (mark 1 true) (mark 2 false)))
    assert (== trace 12)
    set trace 0
    assert (or (mark 1 true) (mark 2 false))
    assert (== trace 1)
    set trace 0
    assert (or (mark 1 false) (mark 2 true))
    assert (== trace 12)
    set trace 0
    assert (or (and (mark 1 false) (mark 2 true)) (and (mark 3 true) (mark 4 true)))
    assert (== trace 134)
    set trace 0
    let mut index: int = 0
    while (and (< index 2) (mark 5 true)) { set index (+ index 1) }
    assert (== index 2)
    assert (== trace 55)
    (println "selfhost short-circuit")
    return 0
}
shadow main { assert (== (main) 0) }
'''


class SelfhostShortCircuit(unittest.TestCase):
    @classmethod
    def command(cls, *args, ok=True):
        result = subprocess.run(list(map(str, args)), cwd=ROOT,
                                capture_output=True, text=True, timeout=900)
        if (result.returncode == 0) != ok:
            raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}\n{result.stderr}')
        return result

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-selfhost-short-circuit-'))
        print('I retain fresh qualification artifacts at', cls.work, flush=True)
        cls.emitters = []
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            emitter = cls.work / (compiler + '-emit')
            cls.command(ROOT/'bin'/compiler, ROOT/'src_nano/nanoisa_emit.nano', '-o', emitter)
            cls.emitters.append(emitter)

    def test_selected_skipped_nested_and_loop_effects(self):
        source = self.work/'effects.nano'
        source.write_text(SOURCE)
        self.command(ROOT/'bin/nano', source)
        for index, emitter in enumerate(self.emitters):
            with self.subTest(producer=index):
                module = self.work/f'effects-{index}.nvm'
                self.command(emitter, source, '--emit-nvm', '-o', module)
                self.command(ROOT/'bin/nano_vm', '--verify-only', module)
                vm = self.command(ROOT/'bin/nano_vm', module)
                self.assertEqual(vm.stdout, 'selfhost short-circuit\n')
                dump = self.command(ROOT/'bin/nanoisa', 'dump', module).stdout
                self.assertIn('JMP_FALSE', dump)
                self.assertIn('JMP_TRUE', dump)
                self.assertNotIn('BOOL_AND', dump)
                self.assertNotIn('BOOL_OR', dump)
                output = module.with_suffix('.c')
                self.command(ROOT/'bin/nvm2c', module, '-o', output)
                binary = module.with_suffix('.native')
                self.command(os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-Wall',
                             '-Wextra', '-Werror', '-fsanitize=address,undefined',
                             '-fno-sanitize-recover=all', output, '-lm', '-o', binary)
                self.assertEqual(self.command(binary).stdout, vm.stdout)

    def test_wrong_operand_types_preserve_prior_output(self):
        for expression in ('(and 1 true)', '(and false 1)', '(or 1 false)', '(or true 1)'):
            source = self.work/'refused.nano'
            source.write_text('fn main() -> bool { return '+expression+' }\nshadow main { assert true }\n')
            for index, emitter in enumerate(self.emitters):
                with self.subTest(expression=expression, producer=index):
                    output = self.work/'prior.nvm'
                    output.write_bytes(b'prior output')
                    result = self.command(emitter, source, '--emit-nvm', '-o', output, ok=False)
                    self.assertIn('two bool operands', result.stdout)
                    self.assertEqual(output.read_bytes(), b'prior output')


if __name__ == '__main__':
    unittest.main()
