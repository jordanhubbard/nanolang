#!/usr/bin/env python3
"""I run each declared make target as one case and write the retained test report.

usage: run-cases POLICY REPORT_PATH ACTION_ID. A case passes only when its make target exits zero.
I reproduce what `make test` does around its cases: point bin/nanoc at the C reference compiler, hand that
compiler to every case as NANOLANG_COMPILER, bound each case by TEST_TIMEOUT, and restore the bootstrap link.
make's own CMD_TIMEOUT default (600s) kills the self-hosted compile when the host is loaded, so I raise it to the
case bound; the case bound still limits every case."""
import json
import os
import subprocess
import sys

TEST_TIMEOUT = 1800
policy, report_path, action = sys.argv[1:4]
cases = [c["id"] for c in json.load(open(policy))["cases"] if c["action"] == action]
reference = os.path.abspath("bin/nanoc_c")


def point_at(target):
    if os.path.lexists("bin/nanoc"):
        os.remove("bin/nanoc")
    os.symlink(target, "bin/nanoc")


def restore():
    """Mirror `make test`: the staged link only when the third bootstrap is built, otherwise the reference."""
    staged = os.path.exists(".bootstrap3.built") and os.path.exists("bin/nanoc_stage2")
    point_at("nanoc_stage2" if staged else "nanoc_c")


results = []
point_at("nanoc_c")
try:
    for case in cases:
        try:
            code = subprocess.run(["make", case, f"NANOLANG_COMPILER={reference}", f"CMD_TIMEOUT={TEST_TIMEOUT}"], timeout=TEST_TIMEOUT).returncode
            reason = None if code == 0 else f"make {case} exited {code}"
        except subprocess.TimeoutExpired:
            code, reason = 124, f"make {case} exceeded {TEST_TIMEOUT}s"
        results.append({"id": case, "outcome": "passed" if code == 0 else "failed", "reason": reason})
finally:
    restore()
with open(report_path, "w") as out:
    json.dump({"schema": "literate-ai/retained-project-test-report@1", "action": action, "cases": results}, out, sort_keys=True)
sys.exit(0 if all(r["outcome"] == "passed" for r in results) else 1)
