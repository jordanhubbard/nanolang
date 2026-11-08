"""I preserve original generic-record fixtures and exact empty-array storage."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest

from tests.native_process import native_link_flags, run_captured

ROOT=Path(__file__).resolve().parents[1]
PROVIDERS=os.environ.get('NANO_EMPTY_RECORD_PROVIDERS',
    'nanoc_c,nano_virt,nanoc_stage1,nanoc_stage2').split(',')


class EmptyRecordArrayContexts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        allowed={'nanoc_c','nano_virt','nanoc_stage1','nanoc_stage2'}
        if not PROVIDERS or len(set(PROVIDERS))!=len(PROVIDERS) or set(PROVIDERS)-allowed:
            raise AssertionError('I require explicit unique source providers.')
        missing=[name for name in PROVIDERS if not (ROOT/'bin'/name).is_file()]
        if missing:raise AssertionError('I require actual source providers: '+', '.join(missing))

    def work(self, label):
        output=ROOT/'.test_output/empty-record-array-contexts'
        output.mkdir(parents=True,exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=label+'-',dir=output))

    def command(self, work, label, argv, *, build=False, accepted=True):
        result=run_captured(argv,cwd=ROOT,text=True,timeout=60,build=build)
        (work/(label+'.stdout')).write_text(result.stdout)
        (work/(label+'.stderr')).write_text(result.stderr)
        with (work/'commands.jsonl').open('a') as log:
            log.write(json.dumps({'label':label,'argv':list(map(str,argv)),
                'returncode':result.returncode})+'\n')
        if accepted:self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result

    def original_fixture(self, name):
        source=ROOT/'tests/unit'/name
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        for provider in PROVIDERS:
            with self.subTest(provider=provider,source=name):
                work=self.work(source.stem+'-'+provider)
                compiler=ROOT/'bin'/provider
                before=hashlib.sha256(compiler.read_bytes()).hexdigest()
                native=work/'native'
                if provider=='nanoc_c':
                    self.command(work,'source-build',[compiler,source,'-o',native],build=True)
                else:
                    module,generated=work/'module.nvm',work/'native.c'
                    self.command(work,'source-build',[compiler,source,'--emit-nvm','-o',module],build=True)
                    self.command(work,'verify',[ROOT/'bin/nano_vm','--verify-only',module])
                    self.command(work,'vm',[ROOT/'bin/nano_vm',module])
                    dump=self.command(work,'dump',[ROOT/'bin/nanoisa','dump',module]).stdout
                    if provider=='nano_virt':
                        # Both unchanged empty Box<Plain> payloads need record storage.
                        self.assertGreaterEqual(len(re.findall(r'\bARR_LITERAL 8 0\b',dump)),2,dump)
                    self.command(work,'aot',[ROOT/'bin/nvm2c',module,'-o',generated],build=True)
                    self.command(work,'native-build',[
                        *shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')),
                        '-std=c11','-O1','-Wall','-Wextra','-Werror',
                        '-fsanitize=address,undefined','-fno-sanitize-recover=all',generated,
                        ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),
                        *(['-Wl,--export-dynamic','-ldl'] if sys.platform.startswith('linux') else []),
                        '-o',native],build=True)
                self.command(work,'native',[native])
                after=hashlib.sha256(compiler.read_bytes()).hexdigest()
                (work/'inputs.json').write_text(json.dumps({'source':str(source),
                    'source_before':source_hash,'source_after':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'provider':str(compiler),'provider_before':before,'provider_after':after},indent=2)+'\n')
                self.assertEqual(before,after)
                self.assertEqual(source_hash,hashlib.sha256(source.read_bytes()).hexdigest())

    def test_original_native_generic_contexts(self):
        self.original_fixture('test_native_generic_contexts.nano')

    def test_original_selfhost_generic_contexts(self):
        self.original_fixture('test_selfhost_generic_contexts.nano')

    def refused(self, item):
        source=('struct Plain {value:int} struct Other {value:int} '
            'union Box<T>{Some{values:array<T>}} '
            'fn main()->int{let bad:Box<Plain> = Box.Some{values:['+item+']} return 0} '
            'shadow main{assert true}\n')
        for provider in PROVIDERS:
            with self.subTest(provider=provider,item=item):
                work=self.work('refusal-'+provider)
                path,output=work/'main.nano',work/'prior'
                path.write_text(source)
                original=b'I preserve my prior compiler output.\n'
                output.write_bytes(original)
                argv=[ROOT/'bin'/provider,path,'-o',output]
                if provider!='nanoc_c':argv.append('--emit-nvm')
                result=self.command(work,'refusal',argv,build=True,accepted=False)
                self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                self.assertRegex(result.stdout+result.stderr,r'(?i)(record|array element|declared.*type)')
                self.assertEqual(output.read_bytes(),original)

    def test_wrong_nominal_record_preserves_output(self):
        self.refused('Other {value:7}')

    def test_scalar_element_preserves_output(self):
        self.refused('7')
