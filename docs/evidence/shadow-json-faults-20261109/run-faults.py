"""I compare actual stdio fault outcomes without changing production code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'shadow-json-span-comparison-20261109'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(output):
    output.mkdir(parents=True,exist_ok=False)
    inputs=[HERE/'stream-faults.c',Path(__file__).resolve(),SOURCE/'original.inc',SOURCE/'candidate.inc',ROOT/'src/eval.c']
    before={str(p):digest(p) for p in inputs}
    compiler=shlex.split(os.environ.get('CC','cc'))
    tool=Path(shutil.which(compiler[0]) or compiler[0]).resolve(strict=True)
    tool_before=digest(tool)
    results={};operations=[];products={}
    for variant in ('original','candidate'):
        for optimization in ('-O0','-O2'):
            label=variant+optimization
            binary=output/label
            argv=[*compiler,'-std=c99',optimization,'-g','-Wall','-Wextra','-Werror','-UNDEBUG',
                  '-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer',
                  '-DTRACE_HELPERS='+json.dumps(str(SOURCE/(variant+'.inc'))),HERE/'stream-faults.c','-o',binary]
            for action,command in [('build',argv),('run',[binary])]:
                if action=='run': product_before=digest(binary)
                result=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
                (output/(label+'-'+action+'.stdout')).write_bytes(result.stdout)
                (output/(label+'-'+action+'.stderr')).write_bytes(result.stderr)
                operations.append({'label':label+'-'+action,'argv':list(map(str,command)),'returncode':result.returncode})
                (output/'operations.json').write_text(json.dumps(operations,indent=2)+'\n')
                assert result.returncode==0,result.stderr.decode(errors='replace')
                if action=='run':
                    assert not result.stderr and product_before==digest(binary)
                    rows=[json.loads(line) for line in result.stdout.splitlines()]
                    assert len(rows)==144 and len({(r['input'],r['mode'],r['threshold'],r['buffered']) for r in rows})==144
                    results[label]=rows
            data=binary.read_bytes()
            assert b'__asan_init' in data and b'__ubsan_handle' in data
            if data[:4]==b'\xcf\xfa\xed\xfe': assert struct.unpack_from('<I',data,12)[0]==2
            else: assert data[:4]==b'\x7fELF'
            products[label]={'sha256_before_after':product_before,'bytes':len(data),'magic':data[:4].hex()}
        assert results[variant+'-O0']==results[variant+'-O2']
    compared=('helper_errno','helper_error','flush','flush_errno','flush_error','close','close_errno','delivered_hex')
    differences=[]
    for original,candidate in zip(results['original-O0'],results['candidate-O0']):
        key={k:original[k] for k in ('input','mode','threshold','buffered')}
        assert all(candidate[k]==v for k,v in key.items())
        fields=[k for k in compared if original[k]!=candidate[k]]
        if fields: differences.append({'case':key,'different_fields':fields,'original':original,'candidate':candidate})
    after={str(p):digest(p) for p in inputs}
    assert before==after and tool_before==digest(tool)
    report={'inputs_before':before,'inputs_after':after,'tool':{'argv':compiler,'path':str(tool),'sha256_before_after':tool_before},'operations':operations,'products':products,'cases_per_image':144,'actual_native_images':4,'o0_o2_outcomes_identical':True,'candidate_stream_equivalent_on_matrix':not differences,'differences':differences,'production_changed':False,'scope':'I compare delivered bytes, error/errno and flush/close outcomes, not callback counts. This is actual native stdio with deterministic short writes and injected errors; I do not infer whole-factory gain or complete SDK closure.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'differences':len(differences),'candidate_qualified':not differences,'images':4}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    run(parser.parse_args().output.resolve())
