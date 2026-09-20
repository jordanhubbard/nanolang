"""I qualify explicit publication, not generated source execution."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import unittest
from tests import test_file_source_plan as retained_runner
ROOT=Path(__file__).resolve().parents[1]

class FileBindingPublisher(unittest.TestCase):
    command=classmethod(retained_runner.FileSourcePlan.command.__func__)
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-file-publisher-',dir=os.environ.get('NANO_FILE_PUBLISH_REPORT_DIR')))
        print('I retain publication artifacts at',cls.work,flush=True)
        cls.cc=shlex.split(os.environ.get('NANO_FILE_PUBLISH_CC','cc'))
        cls.flags=shlex.split(os.environ.get('NANO_FILE_PUBLISH_CFLAGS',''))+['-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I',str(ROOT/'src')]
        if os.environ.get('NANO_FILE_PUBLISH_SANITIZERS','0')=='1':cls.flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
        cls.links=shlex.split(os.environ.get('NANO_FILE_PUBLISH_LDFLAGS',''))
        marker=cls.work/'required-cppflags.h';marker.write_text('#ifndef FILE_PUBLISH_CPPFLAGS_PRESENT\n#error I require CPPFLAGS during provider compilation\n#endif\n')
        make_flags=cls.flags+['-include',str(marker)]
        cls.obj=cls.work/'make-obj';cls.bin=cls.work/'make-bin';cls.exe=cls.bin/'nsi-file-binding'
        cls.command('actual-make',[shutil.which('make'),'-f','Makefile.gnu','-j2','CC='+shlex.join(cls.cc),'CPPFLAGS=-DFILE_PUBLISH_CPPFLAGS_PRESENT','CFLAGS='+shlex.join(make_flags),'LDFLAGS='+shlex.join(cls.links),'OBJ_DIR='+str(cls.obj),'BIN_DIR='+str(cls.bin),'nsi-file-binding'],timeout=300)
        cls.providers=[cls.obj/'file-binding-publisher'/(n+'.o') for n in ('nsi_file_binding','nsi_file_plan','nsi','cJSON','utf8')]
        paths=cls.providers+[cls.obj/'file-binding-publisher/nsi_file_publish.o',cls.obj/'file-binding-publisher/nsi_file_binding_main.o',cls.exe]
        (cls.work/'providers.json').write_text(json.dumps({str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths},indent=2)+'\n')
        (cls.work/'scope.json').write_text(json.dumps({'provider_mode':'all7 explicit Make providers freshly built with selected compiler/flags; CPPFLAGS required by CFLAGS forced header','instrumented':'publisher+CLI included with real-operation hooks;5 strict providers linked without custom malloc hooks','close_error':'real close followed by injected reporting failure; not arbitrary OS close disposition','source_execution':False},indent=2)+'\n')
    def test_api_and_faults(self):
        for mode in ('linked','instrumented'):
            exe=self.work/('fixture-'+mode);where=self.work/('api-'+mode);where.mkdir()
            options=['-DPUBLISH_INSTRUMENT'] if mode=='instrumented' else []
            providers=self.providers+([] if options else [self.obj/'file-binding-publisher/nsi_file_publish.o'])
            self.command(mode+'-build',[*self.cc,*self.flags,*options,'tests/test_nsi_file_publish.c',*providers,*self.links,'-o',exe])
            out,_=self.command(mode+'-run',[exe,'tests/fixtures/nsi_file_plan.json',where],timeout=180)
            self.assertIn(b'PASS File publisher',out)
            if options:
                self.assertIn(b'FAULT index=',out);self.assertIn(b'CLI_FAULT index=',out);self.assertIn(b'CLI_EINTR interrupts=65',out)
    def test_actual_cli(self):
        self.command('actual-cli',[sys.executable,'tests/file_binding_publish_cli.py',self.exe,self.work/'cli','tests/fixtures/nsi_file_plan.json'],timeout=300)
    def test_strict_binding_and_legacy_neighbors(self):
        env={'NANO_FILE_BINDING_CC':shlex.join(self.cc),'NANO_FILE_BINDING_CFLAGS':os.environ.get('NANO_FILE_PUBLISH_CFLAGS',''),'NANO_FILE_BINDING_SANITIZERS':os.environ.get('NANO_FILE_PUBLISH_SANITIZERS','0'),'NANO_FILE_BINDING_REPORT_DIR':str(self.work)}
        self.command('strict-neighbors',[sys.executable,'-m','unittest','-f','-v','tests.test_nsi_file_binding'],timeout=900,extra=env)
if __name__=='__main__':unittest.main()
