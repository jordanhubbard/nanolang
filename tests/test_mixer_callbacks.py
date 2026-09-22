"""I exercise retained post-mix callbacks with a real SDL dummy audio device."""
from pathlib import Path
import os
import shlex
import json
import re
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MixerCallbacks(unittest.TestCase):
    def test_every_extern_has_an_explicit_policy(self):
        module = ROOT / "modules/sdl_mixer"
        names = set(re.findall(r"^(?:pub )?extern fn (\w+)\(",
                               (module / "sdl_mixer.nano").read_text(), re.MULTILINE))
        adapters = json.loads((module / "module.json").read_text())["callback_adapters"]
        self.assertEqual(names, set(adapters))
        for name, adapter in adapters.items():
            self.assertEqual(adapter["abi"], "retained_v1")
            self.assertEqual(adapter["execution"],
                             "owner" if name in {"Mix_GetError", "Mix_ClearError"} else "worker")

    def test_native_operations(self):
        env = dict(os.environ, SDL_AUDIODRIVER="dummy")
        sanitizer = env.get("NANO_MIXER_SANITIZER", "")
        self.assertIn(sanitizer, {"", "address,undefined", "thread"})
        extra_flags = ["-g", "-O1", f"-fsanitize={sanitizer}"] if sanitizer else []
        if sanitizer == "address,undefined" and sys.platform == "darwin":
            env["ASAN_OPTIONS"] = "detect_leaks=0"
        compiler = shlex.split(os.environ.get("CC", "cc"))
        flags = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "SDL2_mixer"], text=True)
        with tempfile.TemporaryDirectory(prefix="nano-mixer-native-") as directory:
            if sys.platform == "darwin":
                sdl3 = subprocess.run(["pkg-config", "--variable=libdir", "sdl3"],
                                      capture_output=True, text=True, timeout=10)
                if sdl3.returncode == 0 and sdl3.stdout.strip():
                    library_dir = sdl3.stdout.strip()
                    previous = env.get("DYLD_LIBRARY_PATH", "")
                    env["DYLD_LIBRARY_PATH"] = library_dir + (":" + previous if previous else "")
                    probe = Path(directory) / "loader-probe"
                    built = subprocess.run([
                        *compiler, *extra_flags, "tests/nanovm/sdl_library_probe.c",
                        "-o", str(probe)], cwd=ROOT, capture_output=True, timeout=30)
                    self.assertEqual(built.returncode, 0, built.stderr)
                    missing = subprocess.run([str(probe), str(Path(directory) / "missing.dylib")],
                                             env=env, capture_output=True, timeout=10)
                    self.assertEqual(missing.returncode, 1, missing.stderr)
                    self.assertIn(b"I could not load", missing.stderr)
                    loaded = subprocess.run([str(probe), "libSDL3.dylib"],
                                            env=env, capture_output=True, timeout=10)
                    self.assertEqual(loaded.returncode, 0, loaded.stderr)
                elif sanitizer:
                    self.fail("I require an SDL3 pkg-config library directory for the headless Darwin sanitizer preflight")
            output = Path(directory) / "operations"
            compiled = subprocess.run([
                *compiler, *extra_flags, "-std=c99", "-Wall",
                "-Wextra", "-Werror", "-pthread",
                "tests/nanovm/mixer_native_operations.c",
                "modules/sdl_mixer/sdl_mixer_operations.c",
                "modules/sdl_mixer/sdl_mixer_callbacks.c",
                "-o", str(output), *shlex.split(flags)], cwd=ROOT,
                capture_output=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run([str(output)], cwd=ROOT, env=env,
                                      capture_output=True, timeout=25)
            self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_audio_callbacks_and_isolation(self):
        env = dict(os.environ, SDL_AUDIODRIVER="dummy")
        with tempfile.TemporaryDirectory(prefix="nano-mixer-") as directory:
            output = Path(directory) / "callbacks.nvm"
            compiled = subprocess.run([
                str(ROOT / "bin/nano_virt"), "tests/nanovm/mixer_callbacks.nano",
                "--emit-nvm", "-o", str(output)], cwd=ROOT, env=env,
                capture_output=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run([str(ROOT / "bin/nano_vm"), str(output)],
                                      cwd=ROOT, env=env, capture_output=True, timeout=20)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            isolated = subprocess.run([str(ROOT / "bin/nano_vm"), "--isolate-ffi", str(output)],
                                      cwd=ROOT, env=env, capture_output=True, timeout=20)
            self.assertNotEqual(isolated.returncode, 0)
            self.assertIn(b"isolated FFI", isolated.stderr)


if __name__ == "__main__":
    unittest.main()
