"""I retain shadow attribution on completion, assertion failure and timeout."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CSeedShadowTrace(unittest.TestCase):
    def compile(self, body, trace=True, seconds="10"):
        with tempfile.TemporaryDirectory(prefix="nano-shadow-trace-") as tmp:
            source = Path(tmp) / 'quoted"source.nano'
            output = Path(tmp) / "program"
            source.write_text(body)
            output.write_bytes(b"prior-output")
            environment = dict(os.environ, NANO_SHADOW_TIMEOUT_SECONDS=seconds)
            environment.pop("NANO_SHADOW_TRACE", None)
            if trace:
                environment["NANO_SHADOW_TRACE"] = "1"
            result = subprocess.run([ROOT / "bin/nanoc_c", source, "-o", output],
                                    cwd=ROOT, env=environment, capture_output=True,
                                    text=True, timeout=30)
            events = [json.loads(line) for line in result.stderr.splitlines()
                      if line.startswith('{"event":"shadow-')]
            for event in events:
                self.assertEqual(event["source"], str(source))
                self.assertGreater(event["line"], 0)
            if result.returncode:
                self.assertEqual(output.read_bytes(), b"prior-output")
            return result, events

    def test_success_and_opt_in(self):
        source = "fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n"
        result, events = self.compile(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([e["event"] for e in events], ["shadow-start", "shadow-complete"])
        self.assertEqual([e["name"] for e in events], ["main", "main"])
        self.assertGreaterEqual(events[1]["elapsed_seconds"], 0)
        self.assertEqual(events[1]["failures"], 0)
        result, events = self.compile(source, trace=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(events, [])

    def test_assertion_failure_is_completed_and_preserves_output(self):
        result, events = self.compile("fn main() -> int { return 0 }\nshadow main { assert false }\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([e["event"] for e in events], ["shadow-start", "shadow-complete"])
        self.assertEqual(events[-1]["failures"], 1)

    def test_timeout_retains_last_start_and_preserves_output(self):
        result, events = self.compile('''fn ready() -> int { return 7 }
shadow ready { assert (== (ready) 7) }
fn stuck() -> void { while true {} }
shadow stuck { (stuck) }
fn main() -> int { return 0 }
shadow main { assert true }
''', seconds="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("I stopped shadow execution after 1 seconds", result.stderr)
        self.assertEqual([(e["event"], e["name"]) for e in events],
                         [("shadow-start", "ready"), ("shadow-complete", "ready"),
                          ("shadow-start", "stuck")])


if __name__ == "__main__":
    unittest.main()
