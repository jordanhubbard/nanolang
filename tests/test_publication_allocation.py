"""I qualify selected actual source-publication allocation boundaries."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest
from tests.native_process import run_captured

ROOT=Path(__file__).resolve().parents[1]
PROVIDERS=[
    'src/nanoisa/assembler.c','src/nanoisa/nvm_format.c','modules/nanoisa/nanoisa.c',
    'src/nanoisa/managed_array_shapes.c','src/nanoisa/verifier.c',
    'src/nanoisa/verifier_types.c','src/nanovm/vm_decode.c',
    'src/nanoisa/ownership_contracts.c','src/nanoisa/nvm_v2_layouts.c',
    'src/nanoisa/retained_layouts.c',
]

CLEANUP_PROVIDERS=['src/nanoisa/nvm_v2_'+name+'.c' for name in ('module','convert','constants','signatures','functions','imports')]

class PublicationAllocation(unittest.TestCase):
    def test_actual_publisher_and_query_prefixes(self):
        self.qualify(False)

    def test_nested_publisher_and_query_prefixes(self):
        self.qualify(True)

    def test_nested_string_publisher_and_query_prefixes(self):
        self.qualify("string")

    def test_nested_primitive_publisher_and_query_prefixes(self):
        for leaf in ('u8','float','bool'):
            with self.subTest(leaf=leaf):
                self.qualify(leaf)

    def test_depth_three_publisher_and_query_prefixes(self):
        for leaf in ('depth3_bool','depth3_string'):
            with self.subTest(leaf=leaf):
                self.qualify(leaf)

    def test_map_publisher_and_query_prefixes(self):
        self.qualify('map')

    def test_map_delete_publisher_and_query_prefixes(self):
        self.qualify('map_delete')

    def qualify(self, nested):
        work=Path(tempfile.mkdtemp(prefix='nano-publication-allocation-'))
        print('I retain publication allocation evidence at',work,flush=True)
        clean_env=os.environ.copy()
        for key in ('PUBLICATION_FAIL','PUBLICATION_PERSISTENT','PUBLICATION_GROWTH_FAIL'):
            clean_env.pop(key,None)
        def run(name,args,env=None):
            build=name.startswith('build-') or name=='link'
            try:
                result=run_captured(args,cwd=ROOT,env=env or clean_env,timeout=30,build=build)
            except subprocess.TimeoutExpired as error:
                (work/(name+'.stdout')).write_bytes(error.output or b'')
                (work/(name+'.stderr')).write_bytes(error.stderr or b'')
                self.fail(f'{name}: process group exceeded its {"build" if build else "execution"} budget')
            stdout,stderr=result.stdout,result.stderr
            (work/(name+'.stdout')).write_bytes(stdout)
            (work/(name+'.stderr')).write_bytes(stderr)
            self.assertEqual(result.returncode,0,f'{name}: '+stdout.decode(errors='replace')+stderr.decode(errors='replace'))
            return stdout.decode()
        cc=shlex.split(os.environ.get('PUBLICATION_CC','cc'))
        flags=[*shlex.split(os.environ['PUBLICATION_CFLAGS']),'-O1','-g',
               '-fsanitize=address,undefined','-fno-sanitize-recover=all',
               '-Isrc/nanovirt','-Isrc/nanoisa','-Itests/nanoisa']
        ldflags=shlex.split(os.environ['PUBLICATION_LDFLAGS'])
        objects=[]
        for source in [*PROVIDERS,*CLEANUP_PROVIDERS]:
            output=work/(Path(source).stem+'.o')
            hooks=['-Dfree=pa_free','-Drealloc=pa_cleanup_realloc'] if source in CLEANUP_PROVIDERS else ['-DPUBLICATION_ALLOC_WRAP']
            run('build-'+output.stem,[*cc,*flags,*hooks,
                '-include','tests/nanoisa/publication_alloc.h','-c',source,'-o',output])
            objects.append(str(output))
        # I instrument the actual include and proof callees, beginning only when
        # publication is reached. Frontend/code generation allocations stay out.
        source=(ROOT/'src/nanovirt/codegen.c').read_text()
        marker='#include "ordinary_authority.inc"'
        self.assertEqual(source.count(marker),1)
        replacement='''#include "publication_alloc.h"
#define malloc pa_malloc
#define calloc pa_calloc
#define realloc pa_realloc
#define free pa_free
#define publish_ordinary_authority publication_under_test
#include "ordinary_authority.inc"
#undef publish_ordinary_authority
#undef malloc
#undef calloc
#undef realloc
#undef free
static void publish_ordinary_authority(CG *cg,ASTNode *program,ModuleList *modules) {
    pa_active=true;
    publication_under_test(cg,program,modules);
    pa_active=false;
}
'''
        generated=work/'codegen.c';generated.write_text(source.replace(marker,replacement))
        output=work/'codegen.o'
        run('build-codegen',[*cc,*flags,'-c',generated,'-o',output]);objects.append(str(output))
        output=work/'publisher-main.o'
        run('build-cli',[*cc,*flags,'-Dmain=publisher_cli_main','-c','src/nanovirt/main.c','-o',output]);objects.append(str(output))
        excluded={str(Path('obj')/Path(p).relative_to('src').with_suffix('.o')) for p in [*PROVIDERS,*CLEANUP_PROVIDERS] if p.startswith('src/')}
        excluded.add('obj/nanoisa/nanoisa_facade.o')
        excluded.update(['obj/nanovirt/codegen.o','obj/nanovirt/main.o'])
        common=[p for p in shlex.split(os.environ['PUBLICATION_OBJECTS'])
                if not any(p==item or p.endswith('/'+item) for item in excluded)]
        common=list(dict.fromkeys(common))
        common=[p for p in common if not p.endswith('.a')]+[p for p in common if p.endswith('.a')]
        executable=work/'publication'
        run('link',[*cc,*flags,'tests/nanoisa/test_publication_allocation.c',
                   'tests/nanoisa/publication_alloc.c',*objects,*common,*ldflags,'-o',executable])
        fixture=work/'fixture.nano';module=work/'fixture.nvm';assembly=work/'fixture.nasm'
        fixture.write_text('struct Box { values: array<int> }\nfn main() -> int {\n'
                           'let values: array<int> = [7, 11]\nlet box: Box = Box { values: values }\n'
                           'assert (== (array_length box.values) 2)\nreturn 0\n}\n')
        if nested:
            fixture.write_text(fixture.read_text().replace('array<int>', 'array<array<int>>').replace('[7, 11]', '[[7], [11]]'))
        if nested == 'string':
            fixture.write_text(fixture.read_text().replace('array<array<int>>', 'array<array<string>>').replace('[[7], [11]]', '[["first"], ["second"]]'))
        if nested in ('u8','float','bool'):
            values={'u8':'[[300], [11]]','float':'[[7.25], [11.5]]','bool':'[[true], [false]]'}
            fixture.write_text(fixture.read_text().replace('array<array<int>>',f'array<array<{nested}>>').replace('[[7], [11]]',values[nested]))
        if nested in ('depth3_bool','depth3_string'):
            leaf=nested.removeprefix('depth3_')
            values='[[[true]], [[false]]]' if leaf=='bool' else '[[["first"]], [["second"]]]'
            fixture.write_text(fixture.read_text().replace('array<array<int>>',f'array<array<array<{leaf}>>>').replace('[[7], [11]]',values))
        if nested == 'map':
            fixture.write_bytes((ROOT/'tests/nanoisa/fixtures/map_record_fields.nano').read_bytes())
        if nested == 'map_delete':
            source=(ROOT/'tests/unit/test_map_field_operations.nano').read_text()
            self.assertEqual(source.count('HashMap<string,string>'),1)
            self.assertEqual(source.count('"kept"'),2)
            fixture.write_text(source.replace('HashMap<string,string>','HashMap<string,int>').replace('"kept"','42'))
        seed=Path(os.environ.get('PUBLICATION_SEED', str(ROOT/'bin/nano_virt')))
        dumper=Path(os.environ.get('PUBLICATION_NANOISA', str(ROOT/'bin/nanoisa')))
        run('source-compile',[seed,fixture,'--emit-nvm','-o',module])
        assembly.write_text(run('source-dump',[dumper,'dump',module]))
        self.assertIn('.ownership "03000000',assembly.read_text())
        for mode in ('publisher','query'):
            args=[executable,mode,fixture,assembly,work/'output.nvm','reserved']
            baseline=run(mode+'-baseline',args)
            count=int(re.search(r'calls=(\d+)',baseline)[1]);self.assertGreater(count,0)
            print(mode,'allocation positions:',count,flush=True)
            for persistent in (False,True):
                for position in range(count):
                    env=clean_env.copy();env['PUBLICATION_FAIL']=str(position)
                    if persistent:env['PUBLICATION_PERSISTENT']='1'
                    run(f'{mode}-{int(persistent)}-{position}',args,env)
            print(f'I qualified {mode}: {count} transient/persistent positions',flush=True)
        # I separately reach the assembler's first function-buffer growth.
        grown=work/'grown.nasm'
        grown.write_text(assembly.read_text().replace('  PUSH_I64 0\n  RET','  PUSH_I64 0\n  POP\n'*450+'  PUSH_I64 0\n  RET'))
        args=[executable,'query',fixture,grown,work/'growth-output.nvm','reserved']
        run('query-growth-baseline',args)
        env=clean_env.copy();env['PUBLICATION_GROWTH_FAIL']='1'
        run('query-growth-failure',args,env)
        persistent=env.copy();persistent['PUBLICATION_PERSISTENT']='1'
        run('query-growth-persistent',args,persistent)
        args[1]='save'
        run('save-growth-baseline',args)
        run('save-growth-failure',args,env)
        run('save-growth-persistent',args,persistent)
