#!/usr/bin/env python3
"""I reconstruct bounded executable C or NanoLang regions from one NanoISA module."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from nanoisa_reconstruction import analyze, Emit, Refusal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=('c', 'nano'), required=True)
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()
    temporary = None
    try:
        source = args.input.resolve(strict=True)
        if args.output and (source == args.output.resolve() or
                            args.output.exists() and os.path.samefile(source, args.output)):
            raise Refusal('I require separate input and output files')
        result = subprocess.run([ROOT / 'bin/nanoisa_hl_facts', source],
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise Refusal(result.stderr.strip() or 'I could not read verified scalar facts')
        facts = json.loads(result.stdout)
        text = Emit(analyze(facts), args.language).run(facts['entry'])
        if args.output:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.absolute().parent,
                                             prefix='.nvm2hl-', delete=False) as file:
                temporary = file.name
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, args.output)
            temporary = None
        else:
            sys.stdout.write(text)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        print('I did not publish reconstructed source', file=sys.stderr)
        return 1
    finally:
        if temporary:
            os.unlink(temporary)


if __name__ == '__main__':
    raise SystemExit(main())
