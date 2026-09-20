"""I qualify immutable binding bytes; generated shadows remain unexecuted text."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_source_plan as retained_runner
from tests.file_binding_corpus import corpus

ROOT=Path(__file__).resolve().parents[1]
PROVIDERS=('src/nsi_file_binding.c','src/nsi_file_plan.c','src/nsi.c','src/cJSON.c','src/utf8.c')

class FileBindingPlan(unittest.TestCase):
    # I reuse the qualified file-backed runner: first terminal, bounded group
    # cleanup, durable stdout/stderr and explicit empty LSAN_OPTIONS.
    command=classmethod(retained_runner.FileSourcePlan.command.__func__)

    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-file-binding-',dir=os.environ.get('NANO_FILE_BINDING_REPORT_DIR')))
        print(f'I retain strict binding artifacts at {cls.work}',flush=True)
        cls.cc=shlex.split(os.environ.get('NANO_FILE_BINDING_CC','cc'))
        cls.flags=shlex.split(os.environ.get('NANO_FILE_BINDING_CFLAGS',''))
        cls.flags+=['-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1',
                    '-I',str(ROOT/'src'),'-I',str(ROOT/'tests'),'-I',str(cls.work)]
        if os.environ.get('NANO_FILE_BINDING_SANITIZERS','0')=='1':
            cls.flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
        cls.links=shlex.split(os.environ.get('NANO_FILE_BINDING_LDFLAGS',''))+['-lm']
        cls.cases=corpus(ROOT,cls.work)
        cls.objects={}
        for mode in ('linked','instrumented'):
            rows=[]
            for source in PROVIDERS:
                obj=cls.work/(mode+'-'+Path(source).stem+'.o')
                hooks=['-include',str(ROOT/'tests/file_binding_hooks.h')] if mode=='instrumented' else []
                cls.command(mode+'-provider-'+Path(source).stem,[*cls.cc,*cls.flags,*hooks,'-c',source,'-o',obj])
                rows.append(str(obj))
            cls.objects[mode]=rows
        identities={path:{'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'bytes':Path(path).stat().st_size}
                    for paths in cls.objects.values() for path in paths}
        (cls.work/'providers.json').write_text(json.dumps(identities,indent=2)+'\n')
        (cls.work/'scope.json').write_text(json.dumps({'providers':PROVIDERS,'source_cases':len(cls.cases),
            'instrumentation':'all malloc/calloc/realloc/free/strdup references in the five fresh providers; fixture/libc internal allocations excluded',
            'sanitizer_scope':'fresh selected providers, binding fixture and unchanged NSI/generator/descriptor neighbors',
            'generated_source':'exact retained forward text only; no parser, service, publication or shadow execution'},indent=2)+'\n')

    def test_complete_document_lifetime_and_allocation(self):
        for mode in ('linked','instrumented'):
            exe=self.work/('binding-'+mode)
            options=['-DBINDING_INSTRUMENT'] if mode=='instrumented' else []
            self.command(mode+'-build',[*self.cc,*self.flags,*options,'tests/test_nsi_file_binding.c',*self.objects[mode],*self.links,'-o',exe])
            out,_=self.command(mode+'-run',[exe,self.work,self.work/'expected.json',ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt'],timeout=900)
            self.assertIn(b'PASS strict File binding ',out)
            self.assertEqual((self.work/('actual-'+mode+'-interface.json')).read_bytes(),(self.work/'expected.json').read_bytes())
            self.assertEqual((self.work/('actual-'+mode+'-binding.nano.txt')).read_bytes(),(ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_bytes())
            for case in self.cases:
                self.assertEqual(out.count(('CASE '+case['name']+' ').encode()),1)
            if mode=='instrumented':
                self.assertIn(b'NAMED_FAILURE index=1 ',out)
                self.assertIn(b'NAMED_FAILURE index=2 ',out)
                for name in ('catalog','alloc-array.json','alloc-callback.json','alloc-async.json'):
                    self.assertIn(('FAULT_SUMMARY '+name+' ').encode(),out)
            print(out.splitlines()[-1].decode(),flush=True)

    def test_unchanged_legacy_neighbors(self):
        ordinary={Path(p).name.removeprefix('linked-'):p for p in self.objects['linked']}
        shared=[ordinary[name] for name in ('nsi.o','cJSON.o','utf8.o')]
        for name,source,extra,args in (
            ('nsi','tests/test_nsi.c',[],[]),
            ('nsi-generator','tests/test_nsi_gen.c',['src/nsi_gen.c'],[]),
            ('nsi-file-plan','tests/test_nsi_file_plan.c',[ordinary['nsi_file_plan.o']],['tests/fixtures/nsi_file_plan.json'])):
            exe=self.work/name
            self.command(name+'-build',[*self.cc,*self.flags,source,*shared,*extra,*self.links,'-o',exe])
            out,_=self.command(name+'-run',[exe,*args],timeout=180)
            self.assertIn(b'PASS' if name=='nsi-file-plan' else b'0 failed',out)
            print(name,out.splitlines()[-1].decode(),flush=True)

if __name__=='__main__':
    unittest.main()
