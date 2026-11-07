"""I exercise my actual standalone map runtime before AOT enrollment."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tests/nanoisa/native_map_alloc_probe.h"
#define NVM2C_ABORT() do { fputs("I refuse the map runtime boundary.\n", stderr); _Exit(70); } while (0)
@RUNTIME@
static nmap_value key(uint8_t kind, int64_t number, char *buffer, size_t capacity) {
    snprintf(buffer, capacity, "key-%" PRId64, number);
    return kind == 1 ? (nmap_value){1, number, NULL} : (nmap_value){5, 0, buffer};
}
static nmap_value value(uint8_t kind, int64_t number, char *buffer, size_t capacity) {
    snprintf(buffer, capacity, "value-%" PRId64, number);
    return kind == 1 ? (nmap_value){1, number, NULL} : (nmap_value){5, 0, buffer};
}
static void check_value(nmap_value actual, uint8_t kind, int64_t number) {
    char expected[96]; snprintf(expected, sizeof expected, "value-%" PRId64, number);
    assert(actual.kind == kind);
    if (kind == 1) assert(actual.integer == number);
    else assert(actual.text && strcmp(actual.text, expected) == 0);
    nmap_release_value(actual);
}
static void exercise(uint8_t key_kind, uint8_t value_kind, int extensive) {
    char kb[96], vb[96];
    nmap_t map = nmap_new_pair(key_kind, value_kind), alias = map;
    nmap_value missing = key(key_kind, -91827, kb, sizeof kb);
    assert(nmap_get_key(map, missing).kind == 0 && !nmap_has_key(map, missing));
    assert(nmap_delete_key(map, missing) == alias && nmap_len(map) == 0);
    int count = extensive ? 1024 : 9;
    for (int i = 0; i < count; ++i) {
        nmap_value k = key(key_kind, i, kb, sizeof kb), v = value(value_kind, -i, vb, sizeof vb);
        assert(nmap_set_key(map, k, v) == alias);
        memset(kb, 'X', sizeof kb); memset(vb, 'Y', sizeof vb);
    }
    assert(nmap_len(alias) == (size_t)count && map->capacity > 16);
    for (int i = 0; i < count; ++i)
        check_value(nmap_get_key(alias, key(key_kind, i, kb, sizeof kb)), value_kind, -i);
    int64_t extremes[] = {INT64_MIN, INT64_MAX, -1};
    for (size_t i = 0; i < sizeof extremes / sizeof extremes[0]; ++i) {
        nmap_value k = key(key_kind, extremes[i], kb, sizeof kb);
        nmap_set_key(map, k, value(value_kind, extremes[i], vb, sizeof vb));
        check_value(nmap_get_key(alias, k), value_kind, extremes[i]);
    }
    nmap_value k = key(key_kind, 0, kb, sizeof kb);
    nmap_value retained = nmap_get_key(map, k);
    nmap_set_key(alias, k, value(value_kind, 77, vb, sizeof vb));
    assert(nmap_delete_key(map, k) == alias && !nmap_has_key(alias, k));
    check_value(retained, value_kind, 0);
    assert(nmap_len(map) == (size_t)count + 2);
    nmap_destroy(map);
    assert(ntest_live == 0 && ntest_bytes == 0 && nmap_live_bytes == 0);
}
int main(int argc, char **argv) {
    (void)ntest_realloc;
    if (argc > 1 && strcmp(argv[1], "alloc") == 0) {
        assert(argc == 5);
        ntest_fail = (size_t)strtoull(argv[4], NULL, 10);
        exercise((uint8_t)atoi(argv[2]), (uint8_t)atoi(argv[3]), 0);
        printf("allocations=%zu\n", ntest_calls);
        return 0;
    }
    if (argc > 1) {
        int mode = atoi(argv[1]);
        if (mode == 0) (void)nmap_new_pair(4, 1);
        else if (mode == 1) (void)nmap_new_pair(1, 3);
        else if (mode == 2) (void)nmap_get_key(nmap_new_pair(1, 1), (nmap_value){5, 0, "wrong"});
        else if (mode == 3) (void)nmap_set_key(nmap_new_pair(5, 5), (nmap_value){1, 7, NULL}, (nmap_value){5, 0, "wrong"});
        else if (mode == 4) (void)nmap_set_key(nmap_new_pair(1, 1), (nmap_value){1, 7, NULL}, (nmap_value){5, 0, "wrong"});
        else if (mode == 5) (void)nmap_has_key(nmap_new(1), (nmap_value){5, 0, NULL});
        else if (mode == 6) (void)nmap_has_key(NULL, (nmap_value){1, 7, NULL});
        else if (mode == 7) (void)nmap_get(nmap_new_pair(1, 1), "wrong");
        else if (mode == 8) { nmap_s map = {0}; map.capacity = SIZE_MAX; nmap_grow(&map); }
        return 1;
    }
    for (uint8_t k = 1; k <= 5; k += 4)
        for (uint8_t v = 1; v <= 5; v += 4) exercise(k, v, 1);
    nmap_t map = nmap_new(5);
    char input_key[] = "borrowed", input_value[] = "retained";
    nmap_set(map, input_key, (nmap_value){5, 0, input_value});
    input_key[0] = 'X'; input_value[0] = 'X';
    assert(nmap_has(map, "borrowed") && !nmap_has(map, input_key));
    nmap_value held = nmap_get(map, "borrowed");
    nmap_delete(map, "borrowed"); nmap_destroy(map);
    assert(strcmp(held.text, "retained") == 0); nmap_release_value(held);
    assert(ntest_live == 0 && ntest_bytes == 0 && nmap_live_bytes == 0);
    puts("I retain all four exact scalar pairs, aliases, growth, copies, extremes and zero live allocations.");
    return 0;
}
'''


class MapScalarRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='nano-map-scalar-runtime-')
        work = Path(cls.directory.name)
        source, cls.binary = work / 'runtime.c', work / 'runtime'
        fragment = ROOT / 'src/nanoisa/nvm2c_map_runtime.inc'
        runtime = ''.join(json.loads(line) for line in fragment.read_text().splitlines()
                          if line.startswith('"'))
        source.write_text(HARNESS.replace('@RUNTIME@', runtime))
        command = [*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                   '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                   '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                   '-I', str(ROOT), str(source), '-o', str(cls.binary)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        original = (ROOT / 'tests/nanoisa/test_nvm2c.c').read_text()
        original = original.split('static void test_native_map_runtime(void) {', 1)[1]
        original = original.split('const char *source =', 1)[1].split('    int status = -1;', 1)[0]
        pieces = []
        for line in original.splitlines():
            line = line.strip()
            if line.startswith('"'):
                pieces.append(json.loads(line.removesuffix(';')))
            elif line == '#include "../../src/nanoisa/nvm2c_map_runtime.inc"':
                pieces.append(runtime)
        original_source = work / 'original.c'
        cls.original_binary = work / 'original'
        original_source.write_text(''.join(pieces))
        original_command = command[:-3] + [str(original_source), '-o', str(cls.original_binary)]
        result = subprocess.run(original_command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_harness(self, *arguments, expected=0):
        result = subprocess.run([str(self.binary), *map(str, arguments)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertNotIn('AddressSanitizer', result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        if expected == 70:
            self.assertIn('I refuse the map runtime boundary.', result.stderr)
        return result

    def test_complete_original_string_runtime_fixture(self):
        result = subprocess.run([str(self.original_binary)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('AddressSanitizer', result.stderr)
        self.assertNotIn('runtime error:', result.stderr)

    def test_four_pairs_ownership_growth_and_existing_string_api(self):
        print(self.run_harness().stdout, end='')

    def test_wrong_tags_null_inputs_and_growth_overflow_refuse(self):
        for mode in range(9):
            with self.subTest(mode=mode):
                self.run_harness(mode, expected=70)

    def test_every_allocation_prefix_refuses_for_each_pair(self):
        for key in (1, 5):
            for value in (1, 5):
                baseline = self.run_harness('alloc', key, value, 2**63)
                count = int(baseline.stdout.strip().split('=')[1])
                self.assertGreater(count, 10)
                for prefix in range(count):
                    with self.subTest(key=key, value=value, prefix=prefix):
                        self.run_harness('alloc', key, value, prefix, expected=70)
                print(f'I refuse all {count} allocation prefixes for map pair {key}/{value}.')
