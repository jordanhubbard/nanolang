#!/usr/bin/env python3
"""I bootstrap raw NanoISA generations, then translate their native products."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.bootstrap_native_guard import retained_input_names


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inputs(root):
    names = ['Makefile.gnu', 'scripts/bootstrap_nanoisa.py', 'tests/bootstrap_native_guard.py']
    for directory in ('src_nano', 'src', 'modules', 'std', 'stdlib'):
        base = root / directory
        if base.exists():
            names.extend(str(p.relative_to(root)) for p in base.rglob('*')
                         if p.is_file() and p.suffix in ('.nano', '.c', '.h', '.json')
                         and not any(part.startswith('.') for part in p.relative_to(base).parts))
    return {name: digest(root / name) for name in sorted(names)}


class Bootstrap:
    def __init__(self, root, timeout):
        self.root = root.resolve()
        self.timeout = timeout
        self.receipt = self.root / 'bin/nanoc_bootstrap.json'
        self.env = {**os.environ, 'NANOLANG_ROOT': str(self.root),
                    'NANO_VM': str(self.root / 'bin/nano_vm'),
                    'NANO_MODULE_PATH': str(self.root / 'modules')}
        self.env.pop('NANOLANG_BOOTSTRAP_NO_CC', None)
        self.manifest = {}

    def save(self):
        (self.work / 'manifest.json').write_text(json.dumps(self.manifest, indent=2) + '\n')

    def publish_receipt(self):
        self.save()
        temporary = self.receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.manifest, indent=2) + '\n')
        temporary.replace(self.receipt)

    def run(self, label, argv, env=None):
        command = [str(arg) for arg in argv]
        index = len(self.manifest['steps'])
        log_path = self.work / f'{index:02d}-{label}.log'
        entry = {'label': label, 'argv': command, 'log': str(log_path)}
        self.manifest['steps'].append(entry)
        self.save()
        print(f'I run {label}; I retain its log at {log_path}.', flush=True)
        started = time.monotonic()
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, cwd=self.root, env=env or self.env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=self.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                entry['interrupted_or_timed_out'] = True
                entry['elapsed_seconds'] = time.monotonic() - started
                self.save()
                raise
        entry.update(exit_code=status, elapsed_seconds=time.monotonic() - started,
                     log_sha256=digest(log_path))
        self.save()
        if status:
            raise RuntimeError(f'I failed {label} with status {status}; see {log_path}.')

    def imports(self, module):
        dump = subprocess.check_output([self.root / 'bin/nanoisa', 'dump', module],
                                       cwd=self.root, text=True, timeout=self.timeout)
        libraries = sorted({shlex.split(line)[1] for line in dump.splitlines()
                            if line.startswith('.import ') and shlex.split(line)[1]})
        if not all(Path(name).is_absolute() for name in libraries):
            raise RuntimeError('I require absolute, immutable host-library paths.')
        return {name: digest(name) for name in libraries}

    def tool_inputs(self):
        names = ['nano_virt', 'nano_vm', 'nanoisa', 'nvm2c', 'nano_aot_runtime.o']
        if sys.platform.startswith('linux'):
            names.append('nano_as_capture.so')
        return {name: digest(self.root / 'bin' / name) for name in names}

    def check_inputs(self):
        if self.manifest['sources'] != source_inputs(self.root):
            raise RuntimeError('I require unchanged bootstrap source inputs; rebuild Stage 1.')
        if self.manifest['tools'] != self.tool_inputs():
            raise RuntimeError('I require unchanged bootstrap tools; rebuild Stage 1.')
        hosts = self.manifest['hosts']
        if {name: digest(name) for name in hosts} != hosts:
            raise RuntimeError('I require unchanged host-library bytes; rebuild Stage 1.')
        if (self.work / 'unexpected-native-compiler').exists():
            raise RuntimeError('I rejected native code generation during a VM generation.')

    def check_artifacts(self, names):
        for name in names:
            if digest(self.root / 'bin' / name) != self.manifest['artifacts'][name]:
                raise RuntimeError(f'I require the recorded bootstrap artifact: {name}.')

    def guarded_environment(self):
        guard = shlex.quote(str(self.work / 'guard-native-compiler'))
        env = {**self.env, 'CC': guard, 'NANO_CC': guard}
        if sys.platform.startswith('linux'):
            env['NANO_AS_CAPTURE_HELPER'] = str(self.root / 'bin/nano_as_capture.so')
        return env

    def record_artifact(self, path, name):
        destination = self.root / 'bin' / name
        path.replace(destination)
        self.manifest['artifacts'][name] = digest(destination)
        self.save()
        return destination

    def generation(self, label, compiler):
        self.check_inputs()
        output = self.work / f'{label}.nvm'
        env = {**self.guarded_environment(), 'NANOLANG_BOOTSTRAP_NO_CC': '1'}
        self.run(label, [self.root / 'bin/nano_vm', compiler, '--',
                        'src_nano/nanoc_v06.nano', '--emit-nvm', '-o', output], env)
        self.run(label + '-verify', [self.root / 'bin/nano_vm', '--verify-only', output])
        if self.imports(output) != self.manifest['hosts']:
            raise RuntimeError('I require the exact seed host closure in every generation.')
        self.check_inputs()
        return self.record_artifact(output, f'nanoc_{label}.nvm')

    def native(self, label, module):
        source = self.work / f'{label}.c'
        binary = self.work / label
        self.run(label + '-translate', [self.root / 'bin/nvm2c', module, '-o', source])
        host_flags = ['-rdynamic', '-ldl'] if sys.platform.startswith('linux') else (
            ['-Wl,-E'] if sys.platform.startswith('freebsd') else [])
        command = self.manifest['compiler'] + ['-std=c11'] + self.manifest['cflags']
        command += [str(source), str(self.root / 'bin/nano_aot_runtime.o'), '-lm']
        command += host_flags + self.manifest['ldflags'] + ['-o', str(binary)]
        self.run(label + '-native', command)
        # I keep the seed's compiler identity for dependency shadow cache checks.
        # Native product compilation is allowed here, after the VM generation.
        hello = self.work / (label + '-hello')
        self.run(label + '-hello-compile', [binary, 'examples/language/nl_hello.nano',
                                         '-o', hello], self.guarded_environment())
        self.run(label + '-hello-execute', [hello])
        return self.record_artifact(binary, f'nanoc_{label}')

    def stage1(self):
        base = self.root / 'obj/bootstrap-nanoisa'
        base.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix='run-', dir=base))
        compiler = shlex.split(self.env.get('NANO_CC') or self.env.get('CC') or 'cc')
        if not compiler:
            raise RuntimeError('I require a host C compiler command.')
        self.manifest = {'version': 1, 'phase': 'building-stage1', 'work': str(self.work),
                         'root': str(self.root), 'sources': source_inputs(self.root),
                         'tools': self.tool_inputs(), 'compiler': compiler,
                         'cflags': shlex.split(self.env.get('NANO_CFLAGS') or '-O1'),
                         'ldflags': shlex.split(self.env.get('NANO_LDFLAGS') or self.env.get('LDFLAGS') or ''),
                         'artifacts': {}, 'steps': [], 'stage_timeout_seconds': self.timeout}
        metadata, native_sources, module_roots = {}, [], []
        for name in ('compiler_support', 'nanoisa', 'std', 'file_source_inputs', 'file_source_catalog'):
            root = self.root / 'modules' / name
            metadata[name] = json.loads((root / 'module.json').read_text())
            native_sources.extend(str((root / source).resolve())
                                  for key in ('c_sources', 'shared_c_sources')
                                  for source in metadata[name].get(key, []))
            module_roots.append(str(root / '.build'))
        retained_file = self.work / 'retained-host-inputs.json'
        config = {'compiler': compiler, 'native_sources': native_sources,
                  'retained_host_inputs': str(retained_file), 'module_build_roots': module_roots,
                  'native_marker': str(self.work / 'unexpected-native-compiler'),
                  'probe_log': str(self.work / 'host-cache-probes.log')}
        guard = self.work / 'guard-native-compiler'
        guard.write_text('#!' + sys.executable + '\nconfig = ' + repr(config) + '\n' +
                         (self.root / 'tests/bootstrap_native_guard.py').read_text() + '\nmain(config)\n')
        guard.chmod(0o755)
        seed = self.work / 'seed.nvm'
        self.run('seed', [self.root / 'bin/nano_virt', 'src_nano/nanoc_v06.nano',
                          '--emit-nvm', '--strip-debug', '-o', seed], self.guarded_environment())
        self.run('seed-verify', [self.root / 'bin/nano_vm', '--verify-only', seed])
        self.manifest['hosts'] = self.imports(seed)
        retained = {}
        for library in self.manifest['hosts']:
            path = Path(library)
            name = path.name.removeprefix('lib').split('.')[0]
            if not path.parent.name.startswith('.nano-gen-') or name not in metadata:
                raise RuntimeError(f'I cannot pin this host artifact: {library}.')
            inputs = retained_input_names(name, metadata[name])
            retained[str(path.parent.parent)] = inputs
            retained[str(path.parent)] = [value for value in inputs if value.endswith(".o")]
        retained_file.write_text(json.dumps(retained, indent=2) + '\n')
        seed = self.record_artifact(seed, 'nanoc_seed.nvm')
        first = self.generation('stage1', seed)
        self.native('stage1', first)
        self.check_inputs()
        self.manifest['phase'] = 'stage1'
        self.publish_receipt()

    def load(self):
        self.manifest = json.loads(self.receipt.read_text())
        if self.manifest.get('version') != 1 or self.manifest.get('root') != str(self.root):
            raise RuntimeError('I require a bootstrap receipt for this checkout.')
        self.work = Path(self.manifest['work'])
        self.check_inputs()

    def stage2(self):
        self.load()
        self.check_artifacts(['nanoc_seed.nvm', 'nanoc_stage1.nvm', 'nanoc_stage1'])
        first = self.root / 'bin/nanoc_stage1.nvm'
        second = self.generation('stage2', first)
        self.compare(first, second)
        self.native('stage2', second)
        self.check_inputs()
        self.manifest.update(phase='stage2', raw_stage1_stage2_equal=True)
        self.publish_receipt()

    @staticmethod
    def compare(first, second):
        if first.read_bytes() != second.read_bytes():
            raise RuntimeError('I require byte-identical raw Stage 1 and Stage 2 modules; I do not normalize them.')

    def verify(self):
        self.load()
        if self.manifest['phase'] != 'stage2':
            raise RuntimeError('I require both completed bootstrap generations before installation.')
        self.check_artifacts(['nanoc_seed.nvm', 'nanoc_stage1.nvm', 'nanoc_stage2.nvm',
                              'nanoc_stage1', 'nanoc_stage2'])
        for label in ('seed', 'stage1', 'stage2'):
            module = self.root / 'bin' / f'nanoc_{label}.nvm'
            self.run(label + '-final-verify', [self.root / 'bin/nano_vm', '--verify-only', module])
            if self.imports(module) != self.manifest['hosts']:
                raise RuntimeError('I require the unchanged host closure before installation.')
        self.compare(self.root / 'bin/nanoc_stage1.nvm', self.root / 'bin/nanoc_stage2.nvm')
        print('I compared raw Stage 1 and Stage 2 modules: they are byte-identical in this build.')
        print('I have not established reproducibility across clean environments or proved compiler semantic correctness.')
        self.publish_receipt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('stage1', 'stage2', 'verify'))
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('I require a positive stage timeout.')
    try:
        getattr(Bootstrap(ROOT, args.timeout), args.stage)()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
