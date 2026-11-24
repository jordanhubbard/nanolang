"""I share real metadata providers and exact private C definition lookup."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = ROOT / 'tests/support/public_c_fixture_providers.c'


def command_with_public_c_providers(args):
    args = list(args)
    fixture = any(Path(str(arg)).parent == ROOT / 'tests' and
                  Path(str(arg)).name.startswith('test_public_c_') and
                  Path(str(arg)).name.endswith('_api.c') for arg in args)
    if fixture and not any(str(arg) == str(PROVIDER) for arg in args):
        args.append(PROVIDER)
    return args


def private_c_helper_prefix(source, name):
    pattern = (r'(?m)^static[ \t]+(?:inline[ \t]+)?'
               r'(?:(nano_cb_[0-9]+_)unused[ \t]+)?'
               r'(?:const[ \t]+char[ \t]*\*|int)[ \t]*'
               r'(nano_cb_[0-9]+_)' + re.escape(name) + r'[ \t]*\([^)]*\)[ \t]*\{')
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1:
        raise AssertionError('I require one exact private C helper definition: ' + name)
    annotation, prefix = matches[0].groups()
    if annotation is not None and annotation != prefix:
        raise AssertionError('I require one namespace for a private helper and its annotation')
    return prefix
