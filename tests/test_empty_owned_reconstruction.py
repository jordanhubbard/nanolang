"""I preserve nominal zero-field owners without inventing source fields."""
import re
import unittest
from tests import test_owned_nested_path_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned

ROOT = support.ROOT
BASELINE = ROOT/'docs/evidence/empty-owned-baseline-20261019'
TWO = '''resource struct EmptyA {}
resource struct EmptyB {}
fn main() -> int {
 let a: EmptyA = EmptyA {}
 let b: EmptyB = EmptyB {}
 let EmptyB {} = b
 let EmptyA {} = a
 return 0
}
shadow main { assert true }
'''


class EmptyOwnedReconstruction(unittest.TestCase):
    setUpClass = support.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command = support.OwnedNestedPathReconstruction.command
    facts = support.OwnedNestedPathReconstruction.facts
    native = support.OwnedNestedPathReconstruction.native
    module = support.OwnedNestedPathReconstruction.module

    def qualify(self, module):
        facts = self.facts(module)
        emitter = OwnedEmit([OwnedAnalyze(facts, i).run() for i in range(len(facts['functions']))],
                            'nano', facts['layouts'])
        counts = [emitter.nano_local_cost(function) for function in emitter.functions]
        for language in ('c', 'nano'):
            output = module.with_suffix('.'+language)
            self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output])
            if language == 'c':
                self.native(output, 0)
            else:
                for compiler in self.compilers:
                    rebuilt = output.with_suffix('.'+compiler.name+'.nvm')
                    self.command([compiler, output, '--emit-nvm', '-o', rebuilt])
                    self.command([ROOT/'bin/nano_vm', '--verify-only', rebuilt]); self.command([ROOT/'bin/nano_vm', rebuilt])
                    regenerated = self.facts(rebuilt)
                    self.assertEqual(regenerated['layouts'], facts['layouts'])
                    self.assertEqual([function['locals'] for function in regenerated['functions']], counts)
                    native = rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c', rebuilt, '-o', native]); self.native(native, 0)
        return facts

    def strict_c(self, source, expected=0):
        for optimization in ('-O0', '-O2'):
            output = source.with_name(source.name+optimization)
            self.command(['cc', '-std=c11', '-pedantic-errors', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                          optimization, source, '-o', output], build=True)
            self.command([output], expected)

    def test_unchanged_source_and_independent_shadow(self):
        modules = [self.module(compiler, (BASELINE/'original.nano').read_text(), 'empty-original')
                   for compiler in self.compilers]
        shadow = self.work/'original-shadow.nvm'
        self.command([ROOT/'bin/nanoisa', 'asm', BASELINE/'original-shadows.nasm', '-o', shadow])
        self.command([ROOT/'bin/nano_vm', '--verify-only', shadow]); self.command([ROOT/'bin/nano_vm', shadow])
        modules.append(shadow)
        for module in modules:
            facts = self.facts(module)
            self.assertEqual([layout['fields'] for layout in facts['layouts']], [[1], []])
            self.qualify(module)
            self.strict_c(module.with_suffix('.c'))

    def test_distinct_empty_nominal_identities(self):
        for compiler in self.compilers:
            module = self.module(compiler, TWO, 'two-empty')
            facts = self.qualify(module)
            self.assertEqual([layout['identity'] for layout in facts['layouts']], [0, 1])
            self.assertEqual([layout['fields'] for layout in facts['layouts']], [[], []])
            self.assertEqual([layout['field_types'] for layout in facts['layouts']], [[], []])
            text = module.with_suffix('.c').read_text()
            self.assertIn('unsigned char nlr_empty;', text)
            self.assertIn('NlrOwned0', text); self.assertIn('NlrOwned1', text)
            self.strict_c(module.with_suffix('.c'))

    def test_zero_pack_preserves_operand_prefix(self):
        text = (BASELINE/'nano_virt.dump.nasm').read_text()
        start = text.index('.function main '); end = text.index('.end', start)
        body = '''.function main 0 2 0 int 1
  PUSH_I64 7
  OWN_PACK 1
  OWN_STORE_LOCAL 0
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  OWN_UNPACK_LOCAL 1
  RET
'''
        assembly = self.work/'prefix.nasm'; assembly.write_text(text[:start]+body+text[end:])
        module = assembly.with_suffix('.nvm')
        self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        self.command([ROOT/'bin/nano_vm', '--verify-only', module]); self.command([ROOT/'bin/nano_vm', module], 7)
        for language in ('c', 'nano'):
            output = module.with_suffix('.'+language)
            self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output])
            if language == 'c':
                self.strict_c(output, 7)
            else:
                for compiler in self.compilers:
                    rebuilt = self.work/(compiler.name+'-prefix.nvm')
                    self.command([compiler, output, '--emit-nvm', '-o', rebuilt])
                    self.command([ROOT/'bin/nano_vm', '--verify-only', rebuilt]); self.command([ROOT/'bin/nano_vm', rebuilt], 7)
                    native = rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c', rebuilt, '-o', native]); self.native(native, 7)

    def test_borrowed_empty_original_is_not_admitted(self):
        text = (ROOT/'tests/nanoisa/fixtures/source_borrow_exclusive.nano').read_text()
        text = 'resource struct Empty {}\n'+text
        text = text.replace('fn main() -> int {', 'fn main() -> int {\n let empty: Empty = Empty {}\n let Empty {} = empty', 1)
        for compiler in self.compilers:
            module = self.module(compiler, text, 'borrowed-empty')
            self.assertNotEqual(self.command([ROOT/'bin/nanoisa_hl_facts', module], None).returncode, 0)
            for language in ('c', 'nano'):
                output = module.with_suffix('.'+language); output.write_text('prior source')
                result = self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output], None)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_text(), 'prior source')

    def test_nominal_transfer_and_consumption_fail_closed(self):
        module = self.module(self.compilers[0], TWO, 'empty-negative')
        text = self.command([ROOT/'bin/nanoisa', 'dump', module]).stdout
        self.assertIn('OWN_PACK 0', text)
        unpack = re.search(r'^  OWN_UNPACK_LOCAL (\d+)\n', text, re.M)
        move = re.search(r'^  OWN_MOVE_LOCAL (\d+)\n', text, re.M)
        self.assertIsNotNone(unpack); self.assertIsNotNone(move)
        variants = {'wrong-nominal': text.replace('OWN_PACK 0', 'OWN_PACK 1', 1),
                    'leak': text[:unpack.start()]+text[unpack.end():],
                    'move-twice': text[:move.end()]+move.group(0)+text[move.end():]}
        for name, assembly_text in variants.items():
            assembly = self.work/(name+'.nasm'); assembly.write_text(assembly_text)
            bad = assembly.with_suffix('.nvm')
            assembled = self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', bad], None)
            if assembled.returncode:
                self.assertRegex(assembled.stderr, r'(?i)verif|owner|layout|type|stack|live|resource|consum')
            else:
                self.assertNotEqual(self.command([ROOT/'bin/nano_vm', '--verify-only', bad], None).returncode, 0)
                for language in ('c', 'nano'):
                    output = bad.with_suffix('.'+language); output.write_text('prior source')
                    result = self.command([ROOT/'bin/nvm2hl', '--language', language, bad, '-o', output], None)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output.read_text(), 'prior source')

    def test_transport_and_advisory_independence(self):
        module = self.module(self.compilers[0], TWO, 'empty-transport')
        text = self.command([ROOT/'bin/nanoisa', 'dump', module]).stdout
        facts = self.facts(module)
        for name, content in (('canonical', text), ('stripped', '\n'.join(line for line in text.splitlines() if not line.startswith('.metadata'))+'\n')):
            assembly = self.work/(name+'.nasm'); assembly.write_text(content)
            rebuilt = assembly.with_suffix('.nvm')
            self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', rebuilt])
            self.command([ROOT/'bin/nano_vm', '--verify-only', rebuilt]); self.command([ROOT/'bin/nano_vm', rebuilt])
            if name == 'canonical': self.assertEqual(module.read_bytes(), rebuilt.read_bytes())
            retained = self.facts(rebuilt)
            self.assertEqual(retained['layouts'], facts['layouts'])
            for language in ('c', 'nano'):
                self.assertEqual(reconstruct_owned(retained, language), reconstruct_owned(facts, language))

    def test_nested_empty_child_and_consuming_result(self):
        text = '''resource struct Empty {}
resource struct Box { child: Empty, marker: int }
fn relay(owned: Empty) -> Empty { return owned }
shadow relay { assert true }
fn main() -> int {
 let empty: Empty = (relay Empty {})
 let box: Box = Box { child: empty, marker: 7 }
 let Box { child, marker } = box
 assert (== marker 7)
 let Empty {} = child
 return 0
}
shadow main { assert (== (main) 0) }
'''
        for compiler in self.compilers:
            module = self.module(compiler, text, 'nested-empty-result')
            facts = self.qualify(module)
            self.assertEqual([layout['fields'] for layout in facts['layouts']], [[], [8, 1]])
            self.assertEqual(facts['functions'][1]['params'], [[8, 0]])
            self.assertEqual(facts['functions'][1]['result_type'], [8, 0])
            self.strict_c(module.with_suffix('.c'))


if __name__ == '__main__':
    unittest.main()
