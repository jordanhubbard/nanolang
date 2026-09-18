#!/usr/bin/env python3
"""I publish freestanding Wasm from my verified scalar NanoISA lowering."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path,
                        help='I write a complete module here; otherwise I write binary stdout.')
    args = parser.parse_args()
    translator = os.environ.get('NANO_NVM2LLVM', str(Path(__file__).resolve().parent / 'nvm2llvm'))
    llc = os.environ.get('NANO_LLC', 'llc')
    linker = os.environ.get('NANO_WASM_LD', 'wasm-ld')
    try:
        source = args.input.resolve(strict=True)
        output = args.output.absolute() if args.output else None
        if output and (source == output.resolve() or
                       (output.exists() and os.path.samefile(source, output))):
            raise ValueError('I require distinct source and output files')
        with tempfile.TemporaryDirectory(prefix='.nano-wasm-',
                                         dir=output.parent if output else None) as tmp:
            work = Path(tmp)
            ir, obj, module = (work / name for name in ('input.ll', 'input.o', 'output.wasm'))
            commands = ([translator, str(source), '--entry-name', 'nano_entry', '--runtime-target', 'wasm32', '-o', str(ir)],
                        [llc, '-mtriple=wasm32-unknown-unknown', '-filetype=obj',
                         str(ir), '-o', str(obj)],
                        [linker, '--no-entry', '--export=nano_entry', '--export-if-defined=nano_try_entry',
                         '--export-if-defined=nano_dispose', '--fatal-warnings',
                         str(obj), '-o', str(module)])
            for command in commands:
                result = subprocess.run(command, capture_output=True)
                if result.returncode:
                    sys.stderr.buffer.write(result.stdout + result.stderr)
                    raise ValueError('I could not complete my Wasm translation')
            with module.open('rb') as completed:
                if completed.read(8) != b'\0asm\x01\0\0\0':
                    raise ValueError('I require a complete core Wasm module')
                if output:
                    os.fsync(completed.fileno())
                else:
                    completed.seek(0)
                    shutil.copyfileobj(completed, sys.stdout.buffer)
                    sys.stdout.buffer.flush()
            if output:
                os.replace(module, output)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        print('I did not publish Wasm output', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
