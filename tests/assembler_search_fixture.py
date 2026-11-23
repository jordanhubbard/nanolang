"""I preserve my package-fixture protocol without starting an interpreter per query."""
import shlex

IMPORT = 'from tests.characterize_source_snapshot import measure, require_consistent\n'
ADDED_IMPORT = 'from tests.assembler_search_fixture import pkg_config_script\n'


def pkg_config_script(mapping):
    """I emit the same known-package cflags and empty exists/libs responses."""
    return ('#!/bin/sh\nnano_fixture_cflags=0\nnano_fixture_package=\n'
            'for nano_fixture_arg do\n'
            '  if [ "$nano_fixture_arg" = --cflags ]; then nano_fixture_cflags=1; fi\n'
            '  nano_fixture_package=$nano_fixture_arg\ndone\n'
            '[ "$nano_fixture_cflags" = 1 ] || exit 0\n'
            'case "$nano_fixture_package" in\n' +
            ''.join('  ' + shlex.quote(name) + ") printf '%s\\n' " + shlex.quote(value) + ';;\n'
                    for name, value in mapping.items()) +
            '  *) exit 1;;\nesac\n')


# I restrict continuity to this one retained fixture statement and one import.
LEGACY = '                    pkg.write_text(f\'#!{sys.executable}\\nimport sys\\nif "--cflags" in sys.argv: print({mapping!r}[sys.argv[-1]])\\n\')'
REPLACEMENT = '                    pkg.write_text(pkg_config_script(mapping))'


def upgrade_source(source):
    if source.count(LEGACY) != 1 or source.count(IMPORT) != 1 or ADDED_IMPORT in source:
        raise RuntimeError('I refuse a source change beyond my declared package-fixture replacement.')
    return source.replace(IMPORT, IMPORT + ADDED_IMPORT).replace(LEGACY, REPLACEMENT)
