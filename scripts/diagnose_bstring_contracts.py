#!/usr/bin/env python3
"""I retain actual C/VM bstring baselines before changing their representation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.native_process import run_captured


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probes():
    def program(body):
        return 'fn main()->int{\n'+body+'\n return 0\n}\nshadow main{assert true}\n'
    return {
        'binary-nul': program(''' let a:u8=65 let zero:u8=0 let b:u8=66
 let bytes:array<u8> = [a,zero,b]
 let value:bstring=(bstr_new_binary bytes)
 (println (bstr_length value))
 (println (bstr_byte_at value 0))
 (println (bstr_byte_at value 1))
 (println (bstr_byte_at value 2))
 let text:string=(bstr_to_cstr value)
 (println (str_length text))
 (bstr_free value)'''),
        'unicode-and-ownership': program(''' let value:bstring=(bstr_new "Aé茶")
 (println (bstr_length value))
 (println (bstr_utf8_length value))
 (println (bstr_utf8_char_at value 1))
 (println (bstr_utf8_char_at value 2))
 (println (bstr_byte_at value 1))
 let tail:bstring=(bstr_new "!")
 let joined:bstring=(bstr_concat value tail)
 (bstr_free value) (bstr_free tail)
 (println (bstr_to_cstr joined))
 let owned:string=(str_concat (bstr_to_cstr joined) "kept")
 (bstr_free joined) (println owned)'''),
        'invalid-utf8': program(''' let bad:u8=255 let bytes:array<u8> = [bad]
 let raw:string=(string_from_bytes bytes)
 let value:bstring=(bstr_new raw)
 (println (bstr_validate_utf8 value))
 (println (bstr_utf8_length value))
 (println (bstr_utf8_char_at value 0))
 (bstr_free value)'''),
        'substring-bounds': program(''' let value:bstring=(bstr_new "hello")
 let negative:bstring=(bstr_substring value -1 2)
 let beyond:bstring=(bstr_substring value 99 2)
 let clipped:bstring=(bstr_substring value 3 99)
 (println (bstr_length negative))
 (println (bstr_length beyond))
 (println (bstr_to_cstr clipped))
 (bstr_free negative) (bstr_free beyond) (bstr_free clipped) (bstr_free value)'''),
        'byte-negative': program(''' let value:bstring=(bstr_new "hello")
 (println (bstr_byte_at value -1)) (bstr_free value)'''),
        'byte-beyond': program(''' let value:bstring=(bstr_new "hello")
 (println (bstr_byte_at value 99)) (bstr_free value)'''),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    out = options.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT/'bin/nanoc_c', ROOT/'bin/nano_virt', ROOT/'bin/nano_vm',
              ROOT/'tests/test_bstring.nano', ROOT/'src/stdlib_runtime.c',
              ROOT/'src/nanovirt/codegen.c', ROOT/'src/runtime/nl_string.c',
              ROOT/'src/nanovm/vm_builtins.c', Path(__file__).resolve()]
    before = {str(p): digest(p) for p in inputs}
    cases = [('whole-fixture', ROOT/'tests/test_bstring.nano')]
    for label, text in probes().items():
        source = out/(label+'.nano')
        source.write_text(text)
        cases.append((label, source))
    records = []
    try:
        for label, source in cases:
            for provider in ['nanoc_c', 'nano_virt']:
                work = out/(label+'-'+provider)
                work.mkdir(exist_ok=True)
                product = work/('program' if provider == 'nanoc_c' else 'program.nvm')
                source_before = digest(source)
                commands = []
                def command(name, argv, build=False):
                    argv = list(map(str, argv))
                    result = run_captured(argv, cwd=ROOT, timeout=60, build=build)
                    (work/(name+'.stdout')).write_bytes(result.stdout)
                    (work/(name+'.stderr')).write_bytes(result.stderr)
                    commands.append({'label': name, 'argv': argv, 'returncode': result.returncode})
                    return result.returncode
                build_argv = [ROOT/'bin'/provider, source, '-o', product]
                if provider == 'nano_virt':
                    build_argv.append('--emit-nvm')
                else:
                    build_argv.append('--keep-c')
                built = command('build', build_argv, True)
                product_before = digest(product) if product.is_file() else None
                magic = product.read_bytes()[:4].hex() if product.is_file() else None
                if built == 0:
                    if provider == 'nanoc_c':
                        if magic not in ['7f454c46', 'cffaedfe', 'feedfacf']:
                            raise AssertionError('I require an actual native baseline product.')
                        command('run', [product])
                    elif command('verify', [ROOT/'bin/nano_vm', '--verify-only', product]) == 0:
                        command('run', [ROOT/'bin/nano_vm', product])
                product_after = digest(product) if product.is_file() else None
                assert source_before == digest(source) and product_before == product_after
                record = {'case': label, 'provider': provider, 'commands': commands,
                          'source': str(source), 'source_before': source_before,
                          'source_after': digest(source), 'product_magic': magic,
                          'product_before_run': product_before, 'product_after_run': product_after,
                          'scope': 'I retain observations; boundary failures are not semantic acceptance.'}
                (work/'record.json').write_text(json.dumps(record, indent=2)+'\n')
                records.append(record)
                print(label, provider, [(c['label'], c['returncode']) for c in commands], flush=True)
        whole = [r for r in records if r['case'] == 'whole-fixture']
        assert len(whole) == 2 and all(all(c['returncode'] == 0 for c in r['commands']) for r in whole)
    finally:
        after = {str(p): digest(p) for p in inputs}
        (out/'observations.json').write_text(json.dumps({'scope': 'I retain C/VM baselines, not a binary-string semantic gate.',
            'before': before, 'after': after, 'cases': records}, indent=2)+'\n')
        assert before == after


if __name__ == '__main__':
    main()
