"""I distinguish native host artifact work from NanoLang-generated C in bootstrap checks."""
import json
import os
import sys
from pathlib import Path


def retained_input_names(name, metadata):
    """I admit only declared source snapshots and the builder's host objects."""
    names = [f'{name}.o']
    for group, key in enumerate(('c_sources', 'shared_c_sources')):
        count = len(metadata.get(key, []))
        for index in range(count):
            names.extend(f'__snapshot_{group}_{index}.{suffix}' for suffix in ('i', 's'))
            names.append(f'__shared_{index}.o' if group else
                         f'{name}.o' if count == 1 else f'{name}_{index}.o')
    return sorted(set(names))


def main(config):
    args = sys.argv[1:]
    probe = any(arg in ('-E', '-###', '-print-prog-name=as', '--version',
                                  '-Wl,--version', '-Wl,-version_details') for arg in args)
    active = os.environ.get('NANOLANG_BOOTSTRAP_NO_CC') == '1'

    def host_input(value):
        path = Path(value).resolve()
        if str(path) in config['native_sources']:
            return True
        retained_file = config.get('retained_host_inputs')
        if retained_file and Path(retained_file).exists():
            retained = json.loads(Path(retained_file).read_text())
            stage = path.parent
            if (stage.name.startswith('.nano-build-') and
                    path.name in retained.get(str(stage.parent), [])):
                return True
            # I allow cached relinks only from the seed's exact published generation.
            if (stage.name.startswith('.nano-gen-') and
                    path.name in retained.get(str(stage), [])):
                return True
        if any(path.is_relative_to(Path(root)) for root in config['module_build_roots']):
            return True
        # The module builder compiles retained native-host inputs here while
        # validating its cache. Product/shadow C uses a different staging area.
        return any(part.startswith('nano-gcc-check-') for part in path.parts)

    if active:
        inputs = []
        skip = False
        for arg in args:
            if skip:
                skip = False
                continue
            if arg in ('-o', '-MF', '-MT', '-MQ'):
                skip = True
                continue
            if not arg.startswith('-') and Path(arg).suffix in ('.c', '.i', '.s', '.S', '.o', '.a'):
                inputs.append(arg)
        primary = os.environ.get('NANO_AS_CAPTURE_PRIMARY')
        if primary:
            inputs.append(primary)
        if not probe and (not inputs or not all(host_input(value) for value in inputs)):
            with open(config['native_marker'], 'a') as log:
                log.write(json.dumps({'message': 'I rejected unclassified compiler work.',
                                      'inputs': inputs, 'options': [arg for arg in args if arg.startswith('-')]}) + '\n')
            sys.exit(91)
        with open(config['probe_log'], 'a') as log:
            log.write('host-cache-probe\n' if probe else 'native-host-artifact\n')
    os.execvp(config['compiler'][0], config['compiler'] + args)
