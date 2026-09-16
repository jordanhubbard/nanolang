"""I exercise production array helpers without loading SDL startup code."""
import os
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SdlImageArrays(unittest.TestCase):
    def test_vm_cleanup(self):
        flags = subprocess.run(["pkg-config", "--cflags", "SDL2_image"],
                               capture_output=True, text=True, timeout=15)
        if flags.returncode:
            self.skipTest("SDL_image SDK headers are unavailable")
        with tempfile.TemporaryDirectory(prefix="nano-sdl-vm-") as tmp:
            artifact = Path(tmp) / "fake-sdl.so"
            shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                "-Wall", "-Wextra", "-Werror", *shared, "-DNANO_SDL_VM_FIXTURE",
                "-Isrc", *shlex.split(flags.stdout), "tests/test_sdl_image_arrays.c",
                "src/runtime/dyn_array.c", "src/runtime/gc.c", "src/runtime/gc_struct.c",
                "-o", str(artifact)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            # I load only fake SDL plus the production adapter; no SDL startup.
            env = dict(os.environ, NANO_TEST_SDL_ARRAY_LIBRARY=str(artifact),
                       NANO_TEST_SDL_ARRAY_ONLY="1")
            make = ["make", "test-vm-ffi", "CC=" + os.environ.get("CC", "cc")]
            if os.environ.get("NANO_TEST_OBJ_DIR"):
                make.append("OBJ_DIR=" + os.environ["NANO_TEST_OBJ_DIR"])
            ran = subprocess.run(make, cwd=ROOT, env=env,
                                 capture_output=True, text=True, timeout=180)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn("sdl_image_cleanup_dispatch...", ran.stdout)
            self.assertIn("All 1 SDL VM cleanup tests passed.", ran.stdout)

    def test_array_boundary(self):
        flags = subprocess.run(["pkg-config", "--cflags", "SDL2_image"],
                               capture_output=True, text=True, timeout=15)
        if flags.returncode:
            self.skipTest("SDL_image SDK headers are unavailable")
        cc = shlex.split(os.environ.get("CC", "cc"))
        with tempfile.TemporaryDirectory(prefix="nano-sdl-image-array-") as tmp:
            executable = Path(tmp) / "probe"
            built = subprocess.run([*cc, "-std=c99", "-g", "-Wall", "-Wextra", "-Werror",
                                    "-Isrc", *shlex.split(flags.stdout),
                                    "tests/test_sdl_image_arrays.c", "src/runtime/dyn_array.c",
                                    "src/runtime/gc.c", "src/runtime/gc_struct.c",
                                    "-o", str(executable)], cwd=ROOT, capture_output=True,
                                   text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)
            # I link the manifest sources but never load the resulting library.
            # Loading SDL2 compatibility here could run its fatal-dialog startup.
            metadata = json.loads((ROOT / "modules/sdl_image/module.json").read_text())
            sources = [str(ROOT / "modules/sdl_image" / name) for name in metadata["c_sources"]]
            libraries = subprocess.run(["pkg-config", "--libs", "SDL2_image"],
                                       capture_output=True, text=True, timeout=15)
            self.assertEqual(libraries.returncode, 0, libraries.stderr)
            shared = (["-dynamiclib", "-undefined", "dynamic_lookup"] if sys.platform == "darwin"
                      else ["-shared", "-fPIC"])
            artifact = Path(tmp) / "module.so"
            linked = subprocess.run([*cc, "-std=c99", *shared, *shlex.split(flags.stdout),
                                     *sources, *shlex.split(libraries.stdout), "-o", str(artifact)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            symbols = subprocess.run(["nm", "-g", str(artifact)], capture_output=True,
                                     text=True, timeout=15)
            self.assertEqual(symbols.returncode, 0, symbols.stderr)
            for name in ("nl_img_load_icon_batch", "nl_img_destroy_texture_batch",
                         "nl_img_get_supported_formats"):
                self.assertIn(name + "__nano_array_abi", symbols.stdout)
        checked = subprocess.run([*cc, "-std=c99", "-fsyntax-only", *shlex.split(flags.stdout),
                                  "modules/sdl_image/sdl_image_helpers.c",
                                  "modules/sdl_image/sdl_image_arrays.c"], cwd=ROOT,
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
