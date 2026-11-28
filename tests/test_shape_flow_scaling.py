"""I preserve directional shape flow while measuring duplicate lookup work."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = '''#include <stdlib.h>
#include <stddef.h>
static size_t flow_test_roots, flow_test_allocations, flow_test_fail_at;
static int flow_test_armed;
static void *flow_test_calloc(size_t n, size_t width) {
    if (flow_test_armed && ++flow_test_allocations == flow_test_fail_at) return NULL;
    return calloc(n, width);
}
#define calloc flow_test_calloc
'''


class ShapeFlowScaling(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get('NANO_SHAPE_FLOW_ARTIFACT_DIR')
        if retained:
            self.work = Path(retained).resolve() / self._testMethodName
            self.work.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory(prefix='nano-shape-flow-')
            self.addCleanup(temporary.cleanup)
            self.work = Path(temporary.name)

    def source(self, variant):
        source = (ROOT / 'src/nanoisa/nvm2c_shape.c').read_text()
        if variant == 'previous':
            start = source.index('static int flow_one(')
            end = source.index('\nint nvm_shape_solve_conversions(', start)
            source = source[:start] + (ROOT / 'tests/support/shape_flow_previous.inc').read_text() + source[end:]
        marker = 'NvmShapeId nvm_shape_root(NvmShapeGraph *g, NvmShapeId id) {\n'
        self.assertEqual(source.count(marker), 1)
        source = source.replace(marker, marker + '    if (flow_test_armed) ++flow_test_roots;\n')
        return PREFIX + source + '\n#undef calloc\n' + (ROOT / 'tests/shape_flow_probe.c').read_text()

    def command(self, label, argv, expected=0):
        argv = [str(x) for x in argv]
        result = subprocess.run(argv, capture_output=True, timeout=90)
        (self.work / (label + '.stdout')).write_bytes(result.stdout)
        (self.work / (label + '.stderr')).write_bytes(result.stderr)
        (self.work / (label + '.json')).write_text(json.dumps({
            'argv': argv, 'exit': result.returncode, 'expected_exit': expected,
            'deadline_seconds': 90,
        }, indent=2) + '\n')
        self.assertEqual(result.returncode, expected, result.stderr.decode(errors='replace'))
        self.assertNotIn(b'ERROR:', result.stderr)
        self.assertNotIn(b'runtime error:', result.stderr)
        return result

    def variants(self, variant, expected):
        source = self.work / (variant + '.c')
        source.write_text(self.source(variant))
        results = []
        for mode, flags in [('ordinary', ['-O2']), ('sanitized', ['-O1', '-g',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all'])]:
            with self.subTest(mode=mode):
                binary = self.work / (variant + '-' + mode)
                self.command(mode + '-compile', [*shlex.split(os.environ.get('CC', 'cc')),
                    '-std=c11', '-Wall', '-Wextra', '-Werror', *flags,
                    '-I' + str(ROOT / 'src/nanoisa'), source, '-o', binary])
                functional = self.command(mode + '-functional', [binary])
                strict = self.command(mode + '-strict', [binary, '--strict'], expected)
                self.assertEqual(functional.stdout, strict.stdout)
                rows = [json.loads(line) for line in functional.stdout.splitlines()]
                self.assertEqual(rows[0]['semantic_cases'], 1800)
                self.assertEqual([r['nodes'] for r in rows[1:]], [32, 128, 1024, 4096, 128, 128])
                self.assertEqual([r['cycle'] for r in rows[1:]], [0, 0, 0, 0, 1, 0])
                self.assertEqual([r['wide'] for r in rows[1:]], [0, 0, 0, 0, 0, 1])
                print(variant, mode, functional.stdout.decode(), end='')
                results.append(rows)
        self.assertEqual(results[0], results[1])
        return results[0]

    def test_previous_helper_preserves_graphs_and_exceeds_lookup_budget(self):
        rows = self.variants('previous', 7)
        for row in rows[1:]:
            self.assertGreater(row['root_calls'], row['nodes'] ** 2 // 2)
            self.assertEqual(row['optional_allocations'], 0)


if __name__ == '__main__':
    unittest.main()
