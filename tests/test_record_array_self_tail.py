"""I qualify singleton self-tail authority through actual consumers and source."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_record_array_vm as vm_support
from tests.native_process import run_captured, native_compile_timeout

ROOT=Path(__file__).resolve().parents[1]

class RecordArraySelfTail(unittest.TestCase):
    def test_instrumented_runtime(self):
        work=Path(tempfile.mkdtemp(prefix='nano-record-self-tail-'))
        print('I retain self-tail runtime evidence at',work,flush=True)
        def run(name,args):
            build=name.endswith('-build')
            r=run_captured(list(map(str,args)),cwd=ROOT,timeout=native_compile_timeout(default=900) if build else 180,build=build)
            (work/(name+'.stdout')).write_bytes(r.stdout);(work/(name+'.stderr')).write_bytes(r.stderr)
            self.assertEqual(r.returncode,0,r.stdout.decode(errors='replace')+r.stderr.decode(errors='replace'))
            return r
        cc=shlex.split(os.environ.get('RECORD_ARRAY_TAIL_CC','cc'))
        flags=[*shlex.split(os.environ.get('RECORD_ARRAY_TAIL_CFLAGS','')),'-std=c11','-D_DEFAULT_SOURCE','-O1','-g','-Wall','-Wextra','-Werror',
               '-fsanitize=address,undefined','-fno-sanitize-recover=all',
               '-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME','-Isrc','-Isrc/nanoisa']
        flags += shlex.split(os.environ.get('RECORD_ARRAY_TAIL_CPPFLAGS',''))
        providers=list(dict.fromkeys([*vm_support.PROVIDERS,'src/nanoisa/nvm_v2_convert.c',
            'src/nanoisa/nvm_v2_module.c','src/nanoisa/nvm_format.c','src/nanoisa/nvm2c.c']))
        names={Path(p).stem for p in providers}
        common=[p for p in shlex.split(os.environ['RECORD_ARRAY_TAIL_OBJECTS']) if Path(p).stem not in names]
        common=sorted(common,key=lambda p:p.endswith('.a'))
        ld=shlex.split(os.environ.get('RECORD_ARRAY_TAIL_LDFLAGS','-lm -lcrypto -lffi -pthread'))
        objects=[]
        for source in providers:
            obj=work/(Path(source).stem+'.o')
            hooks=['-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h'] if source in vm_support.PROVIDERS else []
            run(obj.stem+'-build',[*cc,*flags,*hooks,'-c',source,'-o',obj]);objects.append(obj)
        exe=work/'self-tail'
        run('harness-build',[*cc,*flags,'tests/nanoisa/test_record_array_self_tail.c',
            'tests/nanoisa/record_array_alloc.c',*objects,*common,*ld,'-o',exe])
        run('runtime',[exe,work])
        run('allocation-prefixes',[exe,'allocations'])
        headroom=work/'headroom'
        run('headroom-build',[*cc,*flags,'tests/nanoisa/test_record_array_self_tail_headroom.c',
            'tests/nanoisa/record_array_alloc.c',*[p for p in objects if p.stem!='vm'],*common,*ld,'-o',headroom])
        run('headroom',[headroom])
        proof=work/'proof'
        run('proof-build',[*cc,*flags,'-DRA_WHITEBOX','tests/nanoisa/test_record_array_self_tail_proof.c',
            'tests/nanoisa/record_array_alloc.c',*[p for p in objects if p.stem!='managed_array_shapes'],*common,*ld,'-o',proof])
        run('proof',[proof])
        sources=sorted(work.glob('tail-*.c'));self.assertEqual(len(sources),14)
        for source in sources:
            for opt in ('-O0','-O2'):
                binary=work/(source.stem+opt[1:])
                run(binary.name+'-build',[*cc,'-std=c11',opt,'-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-lm','-o',binary])
                run(binary.name,[binary])

    def test_actual_source_all_leaves_depths(self):
        work=Path(tempfile.mkdtemp(prefix='nano-self-tail-source-'))
        print('I retain self-tail source evidence at',work,flush=True)
        seed=Path(os.environ.get('NANO_SELF_TAIL_SEED',str(ROOT/'bin/nano_virt')));vm=Path(os.environ.get('NANO_SELF_TAIL_VM',str(ROOT/'bin/nano_vm')))
        isa=Path(os.environ.get('NANO_SELF_TAIL_ISA',str(ROOT/'bin/nanoisa')));native=Path(os.environ.get('NANO_SELF_TAIL_NATIVE',str(ROOT/'bin/nvm2c')))
        cc=shlex.split(os.environ.get('RECORD_ARRAY_TAIL_CC','cc'))
        def run(name,args):
            build=name.endswith('-build')
            r=run_captured(list(map(str,args)),cwd=ROOT,timeout=native_compile_timeout(default=900) if build else 180,build=build)
            (work/(name+'.stdout')).write_bytes(r.stdout);(work/(name+'.stderr')).write_bytes(r.stderr)
            self.assertEqual(r.returncode,0,r.stdout.decode(errors='replace')+r.stderr.decode(errors='replace'));return r.stdout
        for leaf,left,right in [('int','7','11'),('u8','7','11'),('float','7.25','11.5'),('bool','false','true'),('string','"seven"','"eleven"')]:
            for depth in (1,2,3):
                name=f'{leaf}-{depth}';typ=leaf;a=left;b=right;read='result.values'
                declarations=''
                if leaf=='u8':
                    declarations='let left_octet:u8=7 let right_octet:u8=11'
                    a='left_octet';b='right_octet'
                for _ in range(depth):typ=f'array<{typ}>';a=f'[{a}]';b=f'[{b}]';read=f'(at {read} 0)'
                source=f'''struct Box {{values:{typ}}}
fn swap(a:Box,b:Box,n:int)->Box {{
 if (<= n 0) {{return a}}
 return (swap b a (- n 1))
}}
shadow swap {{{declarations} let result:Box=(swap Box {{values:{a}}} Box {{values:{b}}} 3) assert (== {read} {right})}}
fn main()->int {{{declarations} let result:Box=(swap Box {{values:{a}}} Box {{values:{b}}} 4095)
 assert (== {read} {right}) return 0}}
shadow main {{assert (== (main) 0)}}
'''
                src=work/(name+'.nano');src.write_text(source);module=work/(name+'.nvm')
                run(name+'-compile',[seed,src,'--emit-nvm','-o',module]);run(name+'-verify',[vm,'--verify-only',module]);run(name+'-vm',[vm,module])
                dump=run(name+'-dump',[isa,'dump',module]);self.assertIn(b'.ownership ',dump);self.assertIn(b'TAIL_CALL',dump)
                c=work/(name+'.c');run(name+'-translate',[native,module,'-o',c])
                text=c.read_text();self.assertIn('goto L_tco;',text)
                c.write_text('#define main fixture_main\n'+text+'\n#undef main\nint main(void){nmap_collection_budget=1;int r=fixture_main();nmap_collect();if(nrec_owned_head || nagg_live_bytes)abort();return r;}\n')
                for opt in ('-O0','-O2'):
                    binary=work/(name+opt[1:]);run(binary.name+'-build',[*cc,'-std=c11',opt,'-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',binary]);run(binary.name,[binary])

    def test_new_allocations_and_detached_global_history(self):
        work=Path(tempfile.mkdtemp(prefix='nano-self-tail-history-'))
        print('I retain self-tail historical alias evidence at',work,flush=True)
        seed=Path(os.environ.get('NANO_SELF_TAIL_SEED',str(ROOT/'bin/nano_virt')));vm=Path(os.environ.get('NANO_SELF_TAIL_VM',str(ROOT/'bin/nano_vm')))
        isa=Path(os.environ.get('NANO_SELF_TAIL_ISA',str(ROOT/'bin/nanoisa')));native=Path(os.environ.get('NANO_SELF_TAIL_NATIVE',str(ROOT/'bin/nvm2c')))
        cc=shlex.split(os.environ.get('RECORD_ARRAY_TAIL_CC','cc'))
        def run(name,args):
            build=name.endswith('-build')
            r=run_captured(list(map(str,args)),cwd=ROOT,timeout=native_compile_timeout(default=900) if build else 180,build=build)
            (work/(name+'.stdout')).write_bytes(r.stdout);(work/(name+'.stderr')).write_bytes(r.stderr)
            self.assertEqual(r.returncode,0,r.stdout.decode(errors='replace')+r.stderr.decode(errors='replace'))
        for depth in (1,3):
         typ='int';initial='99';next_value='remaining';first='7';read='VALUE.values'
         for _ in range(depth):
          typ=f'array<{typ}>';initial=f'[{initial}]';next_value=f'[{next_value}]';first=f'[{first}]';read=f'(at {read} 0)'
         src=work/f'history-{depth}.nano';src.write_text(f'''struct Box {{values:{typ}}}
        let mut saved:Box=Box {{values:{initial}}}
        fn repeat(value:Box,remaining:int)->Box {{
         if (<= remaining 0) {{return value}}
         set saved value
         let next:Box=Box {{values:{next_value}}}
         return (repeat next (- remaining 1))
        }}
        shadow repeat {{let result:Box=(repeat Box {{values:{first}}} 3) assert (== {read.replace('VALUE','result')} 1)}}
        fn main()->int {{
         let first:Box=Box {{values:{first}}}
         let detached:{typ} = first.values
         let result:Box=(repeat first 4096)
         assert (== {read.replace('VALUE','first')} 7)
         assert (== {read.replace('VALUE','saved')} 2)
         assert (== {read.replace('VALUE','result')} 1)
         assert (== {read.replace('VALUE.values','detached')} 7)
         return 0
        }}
        shadow main {{assert (== (main) 0)}}
        ''')
         module=src.with_suffix('.nvm');name=f'history-{depth}'
         run(name+'-compile',[seed,src,'--emit-nvm','-o',module])
         run(name+'-verify',[vm,'--verify-only',module]);run(name+'-vm',[vm,module])
         run(name+'-dump',[isa,'dump',module]);self.assertIn('.ownership ',(work/(name+'-dump.stdout')).read_text())
         c=src.with_suffix('.c');run(name+'-translate',[native,module,'-o',c]);text=c.read_text();self.assertIn('goto L_tco;',text)
         c.write_text('#define main fixture_main\n'+text+'\n#undef main\nint main(void){nmap_collection_budget=1;int r=fixture_main();for(size_t i=0;i<sizeof nglobal/sizeof nglobal[0];i++)nglobal[i]=(nmap_value){0};nmap_collect();if(nrec_owned_head||nagg_live_bytes)abort();return r;}\n')
         for opt in ('-O0','-O2'):
          binary=work/(name+opt[1:]);run(binary.name+'-build',[*cc,'-std=c11',opt,'-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',binary]);run(binary.name,[binary])

if __name__=='__main__':unittest.main()
