"""I carry required instrumentation into native product links."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProductInstrumentLinkFlags(unittest.TestCase):
    def exported_flags(self, *, cflags: str, ldflags: str,
                       nano_ldflags: str | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="nano-product-flags-") as raw:
            probe = Path(raw) / "probe.mk"
            probe.write_text(
                f"include {ROOT / 'Makefile.gnu'}\n"
                "print-product-link-flags:\n"
                "\t@printf '%s\\n' \"$$NANO_LDFLAGS\"\n"
            )
            env = os.environ.copy()
            # I test these declared inputs independently of the parent make's
            # command-line assignments, which travel through MAKEFLAGS too.
            for name in ("MAKEFLAGS", "MAKEOVERRIDES", "MFLAGS"):
                env.pop(name, None)
            if nano_ldflags is None:
                env.pop("NANO_LDFLAGS", None)
            else:
                env["NANO_LDFLAGS"] = nano_ldflags
            completed = subprocess.run(
                ["make", "--no-print-directory", "-f", str(probe),
                 "print-product-link-flags", f"CFLAGS={cflags}",
                 f"LDFLAGS={ldflags}"],
                cwd=ROOT, env=env, text=True, capture_output=True, check=True,
            )
            return completed.stdout.strip()

    def test_coverage_and_sanitizer_links_receive_only_required_flags(self):
        observed = self.exported_flags(
            cflags="-Wall -fprofile-arcs -ftest-coverage -I/private/input",
            ldflags="-lm -fsanitize=address,undefined -L/private/lib",
        )
        self.assertEqual(
            observed,
            "-fprofile-arcs -fsanitize=address,undefined -ftest-coverage",
        )

    def test_unrelated_build_flags_are_not_exported(self):
        self.assertEqual(
            self.exported_flags(cflags="-Wall -I/private/input",
                                ldflags="-lm -L/private/lib"),
            "",
        )

    def test_explicit_product_link_flags_win(self):
        self.assertEqual(
            self.exported_flags(
                cflags="-fprofile-arcs -ftest-coverage",
                ldflags="-fsanitize=address",
                nano_ldflags="-L/declared/product -lfixture",
            ),
            "-L/declared/product -lfixture",
        )


if __name__ == "__main__":
    unittest.main()
