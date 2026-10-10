"""I compare nominal service body checks through both actual source routes."""
import collections
import os
import shlex
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DECL = 'service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"\n'
POSITIVE = '''fn pass(file: File) -> File { return file }
fn helper(file: &mut File) -> WriteResult { return (write_byte &mut file 255) }
fn main() -> int {
    let opened: OpenResult = (temp)
    match opened {
        Ok(file) => {
            let mut owned: File = (pass file)
            let written: WriteResult = (helper &mut owned)
            match written { Ok(count) => { assert (== count 1) } Error(error) => { assert (>= error.status 0) } }
            let positioned: PositionResult = (rewind &mut owned)
            match positioned { Ok() => {} Error(error) => { assert (not error.eof) } }
            let read: ReadResult = (read_byte &mut owned)
            match read { Ok(octet) => { assert (>= octet.value 0) } Error(error) => { assert false } }
            let closed: CloseResult = (close owned)
            match closed { Ok() => {} Error(error) => { assert false } }
        }
        Error(error) => { assert (>= error.host_errno 0) }
    }
    return 0
}
shadow main { assert true }
'''

class ServiceBodies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='nano-service-bodies-')
        cls.work = Path(cls.directory.name).resolve()
        cls.addClassCleanup(cls.directory.cleanup)
        (cls.work/'interface.nsi.json').write_bytes((ROOT/'tests/fixtures/nsi_file_plan.json').read_bytes())
        cls.probes = []
        producers = [[ROOT/'bin/nano_virt']]
        driver = os.environ.get('NANO_SERVICE_BODY_DRIVER_MODULE')
        if driver:
            producers.append([ROOT/'bin/nano_vm', driver, '--'])
        for i, command in enumerate(producers):
            module = cls.work/f'probe-{i}.nvm'
            cls.run_command([*command, ROOT/'tests/service_bodies.nano', '--emit-nvm', '-o', module], timeout=300)
            cls.probes.append([ROOT/'bin/nano_vm', module, '--'])
            source=module.with_suffix('.c');native=module.with_suffix('.native')
            cls.run_command([ROOT/'bin/nvm2c',module,'-o',source],timeout=120)
            cls.run_command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC','cc')),'-std=c11','-O1','-g',
                '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-o',native,'-lm',
                *(['-ldl'] if sys.platform.startswith('linux') else [])],timeout=300)
            cls.probes.append([native])
        cls.drivers = [[ROOT/'bin/nanoc_c','--emit-nvm'], [ROOT/'bin/nano_virt', '--emit-nvm']]
        if driver:
            cls.drivers.append([ROOT/'bin/nano_vm', driver, '--', '--emit-nvm'])

    @staticmethod
    def run_command(command, timeout=60, expected=0):
        run = subprocess.run(list(map(str,command)), cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        if run.returncode != expected:
            raise AssertionError(f'{command}\nexit={run.returncode}\n{run.stdout}\n{run.stderr}')
        return run.stdout

    def check(self, source, status):
        path=self.work/'body.nano';path.write_text(source)
        reports=[self.run_command([os.environ.get('NANO_SERVICE_BODY_C_RUNNER',ROOT/'obj/test_service_bodies'),path,str(status)])]
        for command in self.probes:
            reports.append(self.run_command([*command,path,str(status)]))
        calls=[collections.Counter(line for line in report.splitlines() if line.startswith('CALL ')) for report in reports]
        for actual in calls[1:]:self.assertEqual(calls[0],actual,reports)
        self.check_drivers(path,status)
        return reports

    def check_drivers(self,path,status):
        for command in self.drivers:
            output=self.work/'prior.nvm';output.write_bytes(b'prior')
            run=subprocess.run(list(map(str,[*command,path,'-o',output])),cwd=ROOT,text=True,capture_output=True,timeout=60)
            if run.returncode==0:
                self.assertEqual(status,0)
                self.assertTrue(output.read_bytes().startswith(b'NVM'))
                continue
            message=run.stdout+run.stderr
            if status in (1,3):self.assertIn('I cannot type-check File service bodies:',message)
            else:
                self.assertRegex(message,r'I have not resolved File service declarations|I require --allow-temporary-files|I require --allow-tcp-connections|I cannot lower File source')
                self.assertNotIn('I cannot type-check File service bodies:',message)
            self.assertEqual(output.read_bytes(),b'prior')

    def test_complete_calls_fields_helpers_and_generated_shadows(self):
        self.check(DECL+POSITIVE,0)
        destination=self.work/'generated'
        self.run_command([ROOT/'bin/nsi-file-binding',ROOT/'tests/fixtures/nsi_file_plan.json','--file-binding-dir',destination])
        generated=(destination/'binding.nano').read_text()
        self.assertEqual(generated,(ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text())
        reports=self.check(generated+'\nfn main()->int { return 0 }\n',0)
        self.assertTrue(all('SHADOWS 5' in report for report in reports))

    def test_string_length_builtin_facts(self):
        reports=self.check(DECL+'fn main()->int { return (str_length "abc") }\n',0)
        self.assertTrue(all('CALL str_length 0 0' in report for report in reports),reports)

    def test_nominal_refusals(self):
        cases={
            'fabricated-file':'fn main()->int { let file: File = 0 return 0 }',
            'wrong-result':'fn main()->int { let value: ReadResult = (temp) return 0 }',
            'method-arity':'fn main()->int { let value: OpenResult = (temp 1) return 0 }',
            'borrow-mode':'fn bad(file: File)->WriteResult { return (write_byte file 1) } fn main()->int{return 0}',
            'immutable-root':'fn bad(file: File)->WriteResult { return (write_byte &mut file 1) } fn main()->int{return 0}',
            'borrow-escape':'fn bad(file: &mut File)->File { return file } fn main()->int{return 0}',
            'unknown-field':'fn bad(error: FileError)->int { return error.missing } fn main()->int{return 0}',
            'file-field':'fn bad(file: File)->int { return file.value } fn main()->int{return 0}',
            'pure-service':'pure fn bad()->OpenResult { return (temp) } fn main()->int{return 0}',
            'wrong-return':'fn bad()->File { return (temp) } fn main()->int{return 0}',
            'missing-return':'fn bad(file: File)->File { if true { return file } } fn main()->int{return 0}',
            'missing-arm':'fn bad(value: OpenResult)->void { match value { Ok(file)=>{} } } fn main()->int{return 0}',
            'duplicate-arm':'fn bad(value: OpenResult)->void { match value { Ok(file)=>{} Ok(other)=>{} } } fn main()->int{return 0}',
            'wrong-payload':'fn bad(value: CloseResult)->void { match value { Ok(number)=>{} Error(error)=>{} } } fn main()->int{return 0}',
            'missing-payload':'fn bad(value: OpenResult)->void { match value { Ok()=>{} Error(error)=>{} } } fn main()->int{return 0}',
            'wrong-helper':'fn helper(file: File)->File{return file} fn main()->int{let f: File=(helper 0) return 0}',
            'shadow-body':'fn main()->int{return 0} shadow main { let file: File=0 }',
        }
        for name,body in cases.items():
            with self.subTest(case=name):self.check(DECL+body+'\n',1)

    def test_unsupported_is_not_checked_and_later_bodies_still_checked(self):
        self.check(DECL+'struct Ordinary { field:int }\nfn main()->int { return 0 }\n',2)
        self.check(DECL+'struct Ordinary { field:int }\nfn main()->int { let f:File=0 return 0 }\n',1)
        self.check(DECL+'fn outer(n:int)->fn()->int { return fn()->int { return n } } fn main()->int { return 0 }\n',2)

    def test_local_bound(self):
        parameters=','.join(f'p{i}:int' for i in range(1025))
        self.check(DECL+f'fn many({parameters})->void {{ return }} fn main()->int{{return 0}}\n',3)

    def test_imported_helper_aliases_and_distinct_nominal_bodies(self):
        library=self.work/'binding.nano'
        other=self.work/'other.nano'
        bridge=self.work/'bridge.nano'
        other.write_text(DECL)
        bridge.write_text('pub use "binding.nano" as files\n')
        helper='pub fn keep(file:File)->File { return file }\n'
        cases=[
            ('module "binding.nano" as files\nfn pass(file:files.File)->files.File { return (files.keep file) }\n','',0),
            ('module "binding.nano" as files\nfn pass()->files.File { return (files.keep 0) }\n','',1),
            ('module "binding.nano" as files\n','shadow keep { let fabricated:File=0 }\n',1),
            ('module "binding.nano" as files\nmodule "other.nano" as other\nfn pass(file:other.File)->files.File { return (files.keep file) }\n','',1),
            ('module "bridge.nano" as api\nfn open()->api.files.OpenResult { return (api.files.temp) }\nfn write(file:&mut api.files.File)->api.files.WriteResult { return (api.files.write_byte &mut file 1) }\n','',0),
        ]
        for i,(source,extra,status) in enumerate(cases):
            with self.subTest(case=i):
                library.write_text(DECL+helper+extra)
                path=self.work/'root.nano';path.write_text(source+'fn main()->int { return 0 }\n')
                self.run_command([os.environ.get('NANO_SERVICE_BODY_C_RUNNER',ROOT/'obj/test_service_bodies'),path,str(status)])
                self.check_drivers(path,status)
