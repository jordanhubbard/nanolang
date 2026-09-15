"""I test command execution with a fake MAC CLI, never the live ledger."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MacCommandBoundary(unittest.TestCase):
    def test_both_backends_execute_once(self):
        for compiler, bytecode in [("nanoc_c", False), ("nano_virt", True)]:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-mac-boundary-") as temp:
                directory = Path(temp)
                cli = directory / "mac"
                log = directory / "calls.jsonl"
                counter = directory / "counter"
                cli.write_text("#!/usr/bin/env python3\n"
                    "import json, os, sys\n"
                    "with open(os.environ['NANO_MAC_TEST_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                    "print(json.dumps({'id': 'task_fixture'} if 'create' in sys.argv else []))\n")
                cli.chmod(0o755)
                env = dict(os.environ, PATH=str(directory) + os.pathsep + os.environ["PATH"],
                           NANO_MAC_TEST_LOG=str(log))
                source = directory / "main.nano"
                source.write_text('''from "stdlib/mac.nano" import exec_command, CommandResult, mac_create
fn main() -> int {
    let result: CommandResult = (exec_command "printf x >> '@COUNTER@'; printf out; printf err >&2; exit 7")
    assert (== result.stdout "out")
    assert (== result.stderr "err")
    assert (== result.exit_code 7)
    assert (== (mac_create "fixture" "description" 4 "report") "task_fixture")
    return 0
}
shadow main { assert (== (main) 0) }
'''.replace("@COUNTER@", str(counter)))
                output = directory / ("main.nvm" if bytecode else "main")
                command = [str(ROOT / "bin" / compiler), str(source), "-o", str(output)]
                if bytecode:
                    command.append("--emit-nvm")
                built = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=40)
                self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace"))
                self.assertEqual(counter.read_text(), "x")
                run = [str(ROOT / "bin/nano_vm"), str(output)] if bytecode else [str(output)]
                executed = subprocess.run(run, cwd=ROOT, env=env, capture_output=True, timeout=20)
                self.assertEqual(executed.returncode, 0, executed.stderr.decode(errors="replace"))
                self.assertEqual(counter.read_text(), "xx")
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                creates = [call for call in calls if call[:2] == ["task", "create"]]
                self.assertEqual(len(creates), 2)
                self.assertTrue(all(call[2] == "fixture" for call in creates))
                for fixture in ("test_mac_exec_once", "test_mac_exec_once_vm"):
                    target = directory / (fixture + (".nvm" if bytecode else ""))
                    compile_fixture = [str(ROOT / "bin" / compiler),
                        str(ROOT / "tests/unit" / (fixture + ".nano")), "-o", str(target)]
                    if bytecode:
                        compile_fixture.append("--emit-nvm")
                    checked = subprocess.run(compile_fixture, cwd=ROOT, env=env,
                                             capture_output=True, timeout=40)
                    self.assertEqual(checked.returncode, 0, checked.stderr.decode(errors="replace"))
                    invoke = [str(ROOT / "bin/nano_vm"), str(target)] if bytecode else [str(target)]
                    checked = subprocess.run(invoke, cwd=ROOT, env=env, capture_output=True, timeout=20)
                    self.assertEqual(checked.returncode, 0, checked.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
