"""I read exact immutable capture fixtures from their retained evidence manifest."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/capture-complete-hosted-20261122'


def retained_bytes(logical):
    from scripts.ci_capture_assembler_profile import retained_capture_bytes
    return retained_capture_bytes(logical)


def fixed_profile(directory):
    from scripts.ci_capture_assembler_profile import materialize_retained_profile
    materialize_retained_profile(directory, machine='x86_64')


def pre_admission_inputs():
    """I keep historical qualifier premises separate from current product tests."""
    return {name: retained_bytes('corrected/' + name) for name in
            ('src/module_builder.c', 'tests/test_source_snapshots.py')}
