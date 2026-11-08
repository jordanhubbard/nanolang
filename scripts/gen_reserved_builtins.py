#!/usr/bin/env python3
"""I share the closed reserved-definition policy between my two source checkers."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    options = argparse.ArgumentParser()
    options.add_argument('--check', action='store_true')
    args = options.parse_args()
    spec = json.loads((ROOT/'spec/reserved_builtin_names.json').read_text())
    names = spec.get('names') if isinstance(spec, dict) else None
    if (not isinstance(spec, dict) or spec.get('schema') != 'nanolang.reserved_builtin_names.v1'
            or not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name)
                   for name in names)
            or len(names) != len(set(names))):
        raise SystemExit('I require a unique closed reserved-definition name set.')
    outputs = {
        ROOT/'src/generated/reserved_builtin_names.h':
            '/* I generate this definition policy from spec/reserved_builtin_names.json. */\n'
            'static const char *const builtin_function_names[] = {\n'
            + ''.join(f'    "{name}",\n' for name in names) + '};\n',
        ROOT/'src_nano/generated/reserved_builtin_names.nano':
            '# I generate this definition policy from spec/reserved_builtin_names.json.\n'
            'pub fn reserved_builtin_name(name: string) -> bool {\n'
            + ''.join(f'    if (== name "{name}") {{ return true }}\n' for name in names)
            + '    return false\n}\nshadow reserved_builtin_name {\n'
            + '    for name in ["print", "println", "format", "str_length", "min", "cast_float"] {\n'
            + '        assert (reserved_builtin_name name)\n    }\n'
            + '    for name in ["str_trim", "exp", "array_pop", "array_reverse", "map", "filter", "reduce", "array_map", "array_filter", "array_fold", "__nano_module_1_print", "user_function"] {\n'
            + '        assert (not (reserved_builtin_name name))\n    }\n}\n',
    }
    for path, output in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != output:
                raise SystemExit(f'I need to regenerate {path.relative_to(ROOT)}')
        else:
            path.write_text(output)


if __name__ == '__main__':
    main()
