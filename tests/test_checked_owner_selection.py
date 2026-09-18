"""I select checked scalar closures without discarding source obligations."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.test_affine_contract_boundaries import AffineContractBoundaries, PREFIX
from tests.test_source_borrow_emission import SourceBorrowEmission

ROOT = Path(__file__).resolve().parents[1]
UNUSED = PREFIX + 'fn unused(owner: FileHandle) -> FileHandle { return owner }\n'


class CheckedOwnerSelection(unittest.TestCase):
    command = staticmethod(SourceBorrowEmission.command)
    execute_pair = SourceBorrowEmission.execute_pair

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='nano-checked-selection-')
        cls.work = Path(cls.temporary.name)
        source = cls.work / 'driver.nano'
        source.write_text((ROOT / 'tests/nanoisa/fixtures/checked_owner_selection_driver.nano.txt').read_text())
        cls.drivers = []
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            driver = cls.work / compiler
            cls.command(ROOT / 'bin' / compiler, source, '-o', driver, timeout=900)
            cls.drivers.append(driver)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def source(self, text):
        path = self.work / 'case.nano'
        path.write_text(text)
        return path

    def invoke(self, driver, text, mode='program', first=0, accepted=True, phase=None):
        source = self.source(text)
        output = self.work / 'selected.nasm'
        output.write_bytes(b'prior artifact')
        result = subprocess.run([str(driver), str(source), mode, str(first), str(output)],
                                cwd=ROOT, capture_output=True, text=True, timeout=180)
        if accepted:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotEqual(output.read_bytes(), b'prior artifact')
        else:
            self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_bytes(), b'prior artifact')
            if phase:
                self.assertIn(phase, result.stdout + result.stderr)
                self.assertNotIn('PARSE', result.stdout + result.stderr)
        return output

    def test_original_declaration_probes_keep_whole_source_checks(self):
        cases = []
        original = AffineContractBoundaries()
        original.check_case = lambda name, declaration, accepted: cases.append((name, declaration, accepted))
        for method in sorted(name for name in dir(original) if name.startswith('test_')):
            getattr(original, method)()
        self.assertEqual(sum(accepted for _, _, accepted in cases), 14)
        for name, declaration, accepted in cases:
            text = PREFIX + declaration + '\nfn main() -> int { return 0 }\n'
            canonical = None
            for driver in self.drivers:
                with self.subTest(case=name, producer=driver.name):
                    assembly = self.invoke(driver, text, accepted=accepted, phase=None if accepted else 'TYPECHECK')
                    if accepted:
                        emitted = assembly.read_text()
                        if canonical is None:
                            canonical = emitted
                        else:
                            self.assertEqual(emitted, canonical)
                        self.assertNotIn('consume_handle', emitted)
                        module = self.work / 'probe.nvm'
                        self.command(ROOT / 'bin/nano_asm', assembly, '-o', module)
                        self.execute_pair(module)
            if accepted:
                for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                    with self.subTest(case=name, cli=compiler):
                        module = self.work / 'cli.nvm'
                        self.command(ROOT / 'bin' / compiler, self.source(text), '--emit-nvm', '-o', module)
                        self.execute_pair(module)

    def test_exact_lexical_dependencies_and_selected_suffix(self):
        cases = [
            ('chain', 'fn leaf() -> int { return 3 } fn middle() -> int { return (leaf) } fn main() -> int { assert (== (middle) 3) return 0 }', 'classify', 0, 1),
            ('prior', 'fn main() -> int { let value: int = 3 if true { let value: int = value assert (== value 3) } return 0 }', 'classify', 0, 1),
            ('self', 'fn main() -> int { let value: int = value return 0 }', 'classify', 0, 0),
            ('cycle', 'fn helper() -> int { return (main) } fn main() -> int { return (helper) }', 'classify', 0, 0),
            ('formal', 'fn helper(print: int) -> int { (print 1) return 0 } fn main() -> int { return (helper 1) }', 'classify', 0, 0),
            ('local', 'fn main() -> int { let print: int = 1 (print 1) return 0 }', 'classify', 0, 0),
            ('shadow-owner', 'fn main() -> int { return 0 } shadow main { let owner: FileHandle = FileHandle { fd: 1 } let FileHandle { fd: value } = owner assert (== value 1) }', 'classify-shadows', 0, 2),
            ('shadow-extern', 'extern fn foreign_value() -> int fn main() -> int { return 0 } shadow main { unsafe { assert (== (foreign_value) 0) } }', 'classify-shadows', 0, 0),
            ('suffix-all', 'fn main() -> int { return 0 } shadow main { let owner: FileHandle = FileHandle { fd: 1 } let FileHandle { fd: value } = owner assert (== value 1) } shadow main { assert (== (main) 0) }', 'classify-shadows', 0, 2),
            ('suffix-selected', 'fn main() -> int { return 0 } shadow main { let owner: FileHandle = FileHandle { fd: 1 } let FileHandle { fd: value } = owner assert (== value 1) } shadow main { assert (== (main) 0) }', 'classify-shadows', 1, 1),
        ]
        for name, body, mode, first, expected in cases:
            for driver in self.drivers:
                with self.subTest(case=name, producer=driver.name):
                    result = self.command(driver, self.source(UNUSED + body), mode, first, self.work / 'unused')
                    self.assertEqual(result.stdout.strip(), str(expected))
                    if name in ('shadow-owner', 'shadow-extern', 'suffix-all'):
                        self.invoke(driver, UNUSED + body, 'shadows', first, accepted=False, phase='LOWERING')
                    if expected == 1:
                        assembly = self.invoke(driver, UNUSED + body, 'shadows' if mode.endswith('shadows') else 'program', first)
                        module = self.work / 'control.nvm'
                        self.command(ROOT / 'bin/nano_asm', assembly, '-o', module)
                        self.execute_pair(module)

    def test_full_source_invalid_unused_and_raw_full_module_refusal(self):
        good = UNUSED + 'fn main() -> int { return 0 } shadow main { assert (== (main) 0) }'
        bad = PREFIX + 'fn unused(owner: FileHandle) -> int { return 0 } fn main() -> int { return 0 }'
        bad_type = UNUSED + 'fn bad() -> int { return "wrong" } fn main() -> int { return 0 }'
        for driver in self.drivers:
            with self.subTest(producer=driver.name):
                self.invoke(driver, good, 'whole', accepted=False, phase='LOWERING')
                self.invoke(driver, bad, accepted=False, phase='TYPECHECK')
                self.invoke(driver, bad_type, accepted=False, phase='TYPECHECK')
                self.invoke(driver, good, 'shadows')


if __name__ == '__main__':
    unittest.main()
