"""I preserve actual selected-driver response quoting and bounded refusal."""
from pathlib import Path
import os
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResponseDriverGrammar(unittest.TestCase):
    def setUp(self):
        self.probe = ROOT / 'obj/test_module_generation_probe'

    def test_response_quoted_backslashes_follow_selected_driver(self):
        compilers = list(dict.fromkeys(filter(None, (shutil.which('cc'), shutil.which('clang')))))
        self.assertTrue(compilers)
        fragments = ["'-DVALUE=a\\b'", '"-DVALUE=a\\qb"',
                     r"'-DVALUE=a\\b'", r'"-DVALUE=a\\b"',
                     r"'-DVALUE=a\"b'", r'"-DVALUE=a\'b"',
                     r'"-DVALUE=a\$b"', r'"-DVALUE=a\`b"']
        with tempfile.TemporaryDirectory(prefix='nano-response-selected-') as tmp:
            directory = Path(tmp)
            source = directory / 'input with spaces.c'
            source.write_text('VALUE\n')
            response = directory / 'flags with spaces.rsp'
            for compiler in compilers:
                # My explicit NanoLang selection wins over an ambient driver.
                ambient = next((c for c in compilers if c != compiler), compiler)
                env = dict(os.environ, CC=ambient, NANO_CC=compiler)
                for fragment in fragments:
                    with self.subTest(compiler=compiler, fragment=fragment):
                        response.write_text(fragment + '\n')
                        captured = subprocess.run([str(self.probe), 'capture-response',
                                                   shlex.quote('@' + str(response))],
                                                  env=env, capture_output=True, timeout=35)
                        self.assertEqual(captured.returncode, 0, captured.stderr)
                        arguments = shlex.split(captured.stdout.decode())
                        self.assertFalse(any(arg.startswith('@') for arg in arguments))
                        prefix = [compiler, '-E', '-P', str(source)]
                        native = subprocess.run([*prefix, '@' + str(response)], capture_output=True, timeout=10)
                        replay = subprocess.run([*prefix, *arguments], capture_output=True, timeout=10)
                        self.assertEqual(native.returncode, 0, native.stderr)
                        self.assertEqual((replay.returncode, replay.stdout),
                                         (native.returncode, native.stdout), replay.stderr)
            observations = []
            for compiler in compilers:
                env = dict(os.environ, NANO_CC=compiler)
                for quote in ("", "'", '"'):
                    for octet in (9, 10, 11, 12, 13, *range(32, 127)):
                        fragment = quote + '-DVALUE=a' + chr(92) + chr(octet) + 'b' + quote
                        response.write_text(fragment + '\n')
                        captured = subprocess.run([str(self.probe), 'capture-response',
                                                   shlex.quote('@' + str(response))],
                                                  env=env, capture_output=True, timeout=35)
                        arguments = shlex.split(captured.stdout.decode()) if captured.returncode == 0 else []
                        prefix = [compiler, '-E', '-P', str(source)]
                        native = subprocess.run([*prefix, '@' + str(response)], capture_output=True, timeout=10)
                        replay = subprocess.run([*prefix, *arguments], capture_output=True, timeout=10)
                        observations.append(dict(compiler=compiler, quote=quote, octet=octet,
                            fragment=fragment, captured_returncode=captured.returncode,
                            captured=captured.stdout.decode(), captured_stderr=captured.stderr.decode(),
                            native_returncode=native.returncode, native_stdout=native.stdout.decode(),
                            native_stderr=native.stderr.decode(), replay_returncode=replay.returncode,
                            replay_stdout=replay.stdout.decode(), replay_stderr=replay.stderr.decode()))
                        with self.subTest(compiler=compiler, quote=quote, octet=octet):
                            self.assertEqual(captured.returncode, 0, captured.stderr)
                            self.assertEqual((replay.returncode, replay.stdout),
                                             (native.returncode, native.stdout), replay.stderr)
            output = ROOT / '.test_output/response-driver-grammar-matrix.json'
            output.parent.mkdir(exist_ok=True)
            output.write_text(json.dumps(observations, indent=2) + '\n')
            marker = directory / 'queried'
            wrapper = directory / 'selected-cc'
            for fault in ('unknown', 'timeout'):
                wrapper.write_text(f'#!{sys.executable}\nimport pathlib,sys,time\n'
                    f'pathlib.Path({str(marker)!r}).write_text("queried")\n' +
                    ('print("unrecognized response grammar")\n' if fault == 'unknown' else 'time.sleep(30)\n'))
                wrapper.chmod(0o700)
                env = dict(os.environ, NANO_CC=str(wrapper), NANO_CAPTURE_TIMEOUT_MS='50')
                response.write_text(r"'-DVALUE=a\b'" + '\n')
                fragment = shlex.quote('@' + str(response))
                with self.subTest(fault=fault):
                    captured = subprocess.run([str(self.probe), 'capture-response', fragment],
                                              env=env, capture_output=True, timeout=5)
                    self.assertEqual(captured.returncode, 0, captured.stderr)
                    self.assertEqual(captured.stdout.decode().strip(), fragment)
                    self.assertTrue(marker.exists())
                marker.unlink()
                response.write_text('-DVALUE=42\n')
                captured = subprocess.run([str(self.probe), 'capture-response', fragment],
                                          env=env, capture_output=True, timeout=5)
                self.assertEqual(captured.returncode, 0, captured.stderr)
                self.assertEqual(shlex.split(captured.stdout.decode()), ['-DVALUE=42'])
                self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
