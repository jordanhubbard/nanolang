"""I qualify private non-emitting union native target preparation."""
import os
from pathlib import Path
import shlex
import unittest
from tests.test_record_array_origins import RecordArrayOrigins, PROVIDERS

class UnionFieldNativeTarget(RecordArrayOrigins):
    def test_complete_origin_and_allocation_controls(self):
        modules = shlex.split(os.environ.get("UNION_NATIVE_TARGET_MODULES", ""))
        self.assertIn(len(modules), (0, 6))
        for observed in (False, True):
            mode = "instrumented" if observed else "linked"
            products = []
            for source in PROVIDERS:
                if observed and Path(source).stem == "managed_array_shapes":
                    continue
                obj = self.artifacts / f"union-native-target-{mode}-{Path(source).stem}.o"
                flags = ["-DRA_ALLOC_WRAP", "-include", "tests/nanoisa/record_array_alloc.h"] if observed else []
                self.command(f"union-native-target-{mode}-{Path(source).stem}-build",
                    [*self.cc, *self.flags, *flags, "-c", source, "-o", str(obj)])
                products.append(str(obj))
            exe = self.artifacts / ("union-native-target-" + mode)
            self.command("union-native-target-" + mode + "-build", [*self.cc, *self.flags,
                *(["-DUFP_ALLOC"] if observed else []), "tests/nanoisa/test_union_field_native_target.c",
                "tests/nanoisa/record_array_alloc.c", *products, *self.objects,
                *self.ldflags, "-o", str(exe)])
            output = self.command("union-native-target-" + mode + "-run", [str(exe), *(modules if not observed else [])])
            self.assertIn(b"observed native target checks" if observed else b"native target checks", output)
            print(output.decode().strip(), flush=True)

# Avoid collecting the imported base test as a second, unrelated gate.
del RecordArrayOrigins
if __name__ == "__main__":
    unittest.main()
