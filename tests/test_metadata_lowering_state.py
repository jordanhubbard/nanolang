"""I exercise new ordinary metadata shadows through both repaired compiler stages."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MetadataLoweringState(unittest.TestCase):
    def checked(self, *args):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def exercise(self, empty):
        with tempfile.TemporaryDirectory(prefix='nano-metadata-state-') as tmp:
            work = Path(tmp)
            dependency = work/'metadata_state.nano'
            dependency.write_text('module metadata_state\n' + ('' if empty else
                'pub struct Visible { value: int }\n'
                'pub fn visible() -> int { return 7 }\nshadow visible { assert true }\n'))
            function, record, count = ('', '', 0) if empty else ('visible', 'Visible', 1)
            source = work/'ordinary.nano'
            source.write_text(f'module {json.dumps(str(dependency))} as facts\n'
                'extern fn ___module_function_count_metadata_state() -> int\n'
                'extern fn ___module_function_name_metadata_state(index: int) -> string\n'
                'extern fn ___module_struct_name_metadata_state(index: int) -> string\n'
                'let mut visits: int = 0\n'
                'fn next_index() -> int { set visits (+ visits 1) return 0 }\n'
                'shadow next_index { assert true }\n'
                'fn main() -> int { return 0 }\n'
                'shadow main {\n'
                ' let mut total = 0\n let mut name = ""\n let mut record = ""\n'
                ' unsafe {\n'
                '  set total (___module_function_count_metadata_state)\n'
                '  set name (___module_function_name_metadata_state (next_index))\n'
                '  set record (___module_struct_name_metadata_state 0)\n'
                '  assert (== (___module_function_name_metadata_state -1) "")\n'
                '  assert (== (___module_struct_name_metadata_state 99) "")\n'
                ' }\n'
                f' assert (== total {count})\n assert (== name {json.dumps(function)})\n'
                f' assert (== record {json.dumps(record)})\n assert (== visits 1)\n'
                ' println "metadata shadow checked"\n}\n')
            for stage in ('nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(stage=stage, empty=empty):
                    output = work/(stage+'.nvm')
                    diagnostic = self.checked(ROOT/'bin'/stage, source, '--emit-nvm', '-o', output)
                    self.assertIn('metadata shadow checked', diagnostic)
                    self.checked(ROOT/'bin/nano_vm', '--verify-only', output)
                    self.checked(ROOT/'bin/nano_vm', output)
                    generated, native = work/(stage+'.c'), work/stage
                    self.checked(ROOT/'bin/nvm2c', output, '-o', generated)
                    self.checked('cc', '-std=c11', '-Wall', '-Wextra', '-Werror', generated,
                                 '-lm', '-o', native)
                    self.checked(native)

    def test_nonempty_exports_and_inferred_shadow_assignments(self):
        self.exercise(False)

    def test_empty_exports_and_once_only_index(self):
        self.exercise(True)


if __name__ == '__main__':
    unittest.main()
