"""I preserve compiler line boundaries and measure actual native length work."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import tempfile
import time
from unittest.mock import patch
import unittest
from tests import nanolang_translator_routes as routes

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'tests/fixtures/compiler_line_splitting_prior.nano'
PRIOR_SHA='79c1d70587c7d78d8787bebd41b6cc7c765e9fad8c630af74d8bf63c98945176'
MARKER='I preserved compiler line boundaries.\n'


def unit(text):
    start=text.index('fn split_lines(')
    end=text.find('\nfn parse_import_path_from_line(',start)
    return text[start:end if end>=0 else len(text)].strip()+'\n'


def inputs():
    return ['', 'hello', '\n', '\n\n', '\na', 'a\n', 'a\n\nb',
            'a\r\nb\r\n', 'café\n茶\n', 'left|right\n# import "path.nano"',
            *['x'*n for n in (256,1024,4096)], 'ab\r\n'*128+'tail']


def program(body):
    literal=lambda s:json.dumps(s,ensure_ascii=False)
    rows=[body,'fn main() -> int {']
    for index,text in enumerate(inputs()):
        expected=text.split('\n')
        rows.extend([f' let lines_{index}: array<string> = (split_lines {literal(text)})',
                     f' assert (== (array_length lines_{index}) {len(expected)})'])
        for at,value in enumerate(expected):
            rows.append(f' assert (== (at lines_{index} {at}) {literal(value)})')
    rows.extend([' (println "I preserved compiler line boundaries.")',' return 0','}',
                 'shadow main { assert (== (main) 0) }'])
    return '\n'.join(rows)+'\n'


class CompilerLineSplitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior=unit(PRIOR.read_text())
        if hashlib.sha256(cls.prior.encode()).hexdigest()!=PRIOR_SHA:
            raise AssertionError('I require my unchanged prior compiler function and shadow.')
        cls.current=unit((ROOT/'src_nano/nanoc_v06.nano').read_text())
        for assertion in ['assert (== (array_length lines) 1)',
                          'assert (== (at lines 0) "hello")']:
            if assertion not in cls.current:
                raise AssertionError('I retain my original line-splitting shadow assertions.')
        output=ROOT/'.test_output/compiler-line-splitting'
        output.mkdir(parents=True,exist_ok=True)
        cls.work=Path(tempfile.mkdtemp(prefix='run-',dir=output))

    def setUp(self):
        original=routes.run_captured
        def observed(command, **kwargs):
            index=len(list(self.work.glob('command-*.json')))
            prefix=self.work/f'command-{index:03d}'
            started=time.monotonic()
            result=original(command,**kwargs)
            prefix.with_suffix('.stdout').write_bytes(result.stdout)
            prefix.with_suffix('.stderr').write_bytes(result.stderr)
            prefix.with_suffix('.json').write_text(json.dumps({
                'argv':list(map(str,command)), 'exit':result.returncode,
                'seconds':time.monotonic()-started},indent=2)+'\n')
            return result
        observer=patch.object(routes,'run_captured',observed)
        observer.start()
        self.addCleanup(observer.stop)

    def test_boundaries_through_all_five_routes(self):
        self.assertTrue(routes.source_compiler_command(),
                        'I require an explicit real NanoLang compiler product for all five routes.')
        for label,body in [('prior',self.prior),('current',self.current)]:
            with self.subTest(algorithm=label):
                directory=self.work/label;directory.mkdir()
                source=directory/'lines.nano';source.write_text(program(body))
                products=routes.build_routes(source,directory)
                self.assertEqual(len(products),5)
                for route,command in products:
                    with self.subTest(route=route):
                        self.assertEqual(routes.checked(command),MARKER)

    def test_native_length_work_against_prior_algorithm(self):
        observed={}
        for label,body in [('prior',self.prior),('current',self.current)]:
            directory=self.work/('work-'+label);directory.mkdir()
            source,module,generated,measured,native=[directory/n for n in
                ['lines.nano','lines.nvm','lines.c','measured.c','measured']]
            source.write_text(program(body))
            routes.checked([ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module],build=True)
            routes.checked([ROOT/'bin/nvm2c',module,'-o',generated],build=True)
            text=generated.read_text();marker='#include <string.h>\n'
            self.assertGreaterEqual(text.count(marker),1)
            instrumentation=r'''
#include <stdint.h>
#include <stdio.h>
static volatile uint64_t line_length_calls, line_length_bytes;
static __attribute__((noinline)) size_t line_counted_strlen(const char *text) {
    size_t length = strlen(text);
    line_length_calls++; line_length_bytes += length;
    return length;
}
static __attribute__((destructor)) void line_report_length_work(void) {
    fprintf(stderr, "NANO_LINE_WORK {\"calls\":%llu,\"requested_bytes\":%llu}\n",
            (unsigned long long)line_length_calls, (unsigned long long)line_length_bytes);
}
#define strlen line_counted_strlen
'''
            measured.write_text(text.replace(marker,marker+instrumentation,1))
            routes.checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-O1','-g',
                '-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                '-fno-sanitize-recover=all',measured,ROOT/'bin/nano_aot_runtime.o',
                '-lm',*routes.native_link_flags(),'-o',native],build=True)
            completed=routes.run_captured([native],timeout=60,cwd=ROOT)
            self.assertEqual(completed.returncode,0,completed.stderr.decode(errors='replace'))
            self.assertEqual(completed.stdout.decode(),MARKER)
            lines=completed.stderr.decode().splitlines()
            self.assertEqual(len(lines),1,completed.stderr.decode())
            self.assertTrue(lines[0].startswith('NANO_LINE_WORK '))
            observed[label]=json.loads(lines[0].removeprefix('NANO_LINE_WORK '))
        total=sum(len(s.encode()) for s in inputs())
        (self.work/'length-work.json').write_text(json.dumps({'input_bytes':total,'observed':observed},indent=2)+'\n')
        self.assertLess(observed['current']['requested_bytes'],64*total+65536)
        self.assertGreater(observed['prior']['requested_bytes'],10*observed['current']['requested_bytes'])
        self.assertGreater(observed['prior']['calls'],5*observed['current']['calls'])


if __name__=='__main__':unittest.main()
