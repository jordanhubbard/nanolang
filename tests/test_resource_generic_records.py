"""I check retained record/generic classification without claiming native layouts."""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = '''resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
union Marker<T> { Mark { number: int } }
'''
CASES = [
    ('owned', 'struct Outer { boxed: Box<Handle> }', 'Outer', True, False),
    ('ordinary', 'struct Outer { boxed: Box<int> }', 'Outer', False, False),
    ('phantom', 'struct Outer { boxed: Marker<Handle> }', 'Outer', False, False),
    ('nested_generic', 'struct Outer { boxed: Box<Box<Handle>> }', 'Outer', True, False),
    ('nested_records', 'struct Inner { boxed: Box<Handle> } struct Outer { inner: Inner }', 'Outer', True, False),
    ('collection', 'struct Outer { boxed: Box<array<Handle>> }', 'Outer', True, True),
    ('record_cycle', 'struct Outer { children: array<Outer> }', 'Outer', False, False),
    ('generic_cycle', 'union Cycle<T> { Next { next: Cycle<T> }, None {} } struct Outer { cycle: Cycle<int> }', 'Outer', False, False),
    ('recursive_collection', 'union Chain<T> { More { children: array<Chain<T>> }, Value { value: T }, None {} } struct Outer { chain: Chain<Handle> }', 'Outer', True, True),
    ('formal_shadows_record', 'resource struct T { fd: int } union Wrap<T> { Some { boxed: Box<T> } } struct Outer { value: Wrap<int> }', 'Outer', False, False),
    ('concrete_formal_name', 'resource struct T { fd: int } struct Outer { value: Box<T> }', 'Outer', True, False),
    ('fixed_union_carrier', 'struct Outer { value: Box<Handle> } union Carrier { Some { value: Outer }, None {} }', 'Carrier', True, False),
]

class ResourceGenericRecords(unittest.TestCase):
    def test_retained_classification_matrix(self):
        source = '''#include "nanolang.h"
#include "resource_tracking.h"
#include <stdio.h>
int g_argc=0; char **g_argv=NULL; char g_project_root[4096]=".";
const char *get_project_root(void) { return g_project_root; }
static int check(const char *label, const char *source, const char *name, int owned, int collection) {
 int count=0; Token *tokens=tokenize(source,&count); if (!tokens) return 0;
 ASTNode *program=parse_program(tokens,count); if (!program) { free_tokens(tokens,count); return 0; }
 clear_module_cache(); Environment *env=create_environment(); typecheck_set_current_file("<record-generic-classification>");
 int accepted=type_check(program,env);
 int actual_owned=is_resource_type(env,name), actual_collection=has_resource_collection_payload(env,name);
 TypeInfo queried={0};queried.base_type=TYPE_STRUCT;queried.generic_name=(char*)name;
 int metadata_owned=is_resource_type_info(env,&queried), metadata_collection=has_resource_collection_type_info(env,&queried);
 StructDef *record=env_get_struct(env,name);
 if (record && record->field_count==1 && record->field_type_info) {
  if (record->field_type_info[0]) {
   metadata_owned=is_resource_type_info(env,record->field_type_info[0]);
   metadata_collection=has_resource_collection_type_info(env,record->field_type_info[0]);
  }
 }
 int ok=accepted && actual_owned==owned && actual_collection==collection && metadata_owned==owned && metadata_collection==collection;
 if (!ok) fprintf(stderr,"%s: accepted=%d owned=%d collection=%d\\n",label,accepted,actual_owned,actual_collection);
 free_environment(env);free_ast(program);free_tokens(tokens,count);return ok;
}
int main(void) {
'''
        for label, declaration, queried, owned, collection in CASES:
            program = PREFIX + declaration + '\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'
            source += ' if (!check(' + ','.join([json.dumps(label), json.dumps(program), json.dumps(queried), str(int(owned)), str(int(collection))]) + ')) return 1;\n'
        source += 'return 0; }\n'
        with tempfile.TemporaryDirectory(prefix='nano-record-classification-') as directory:
            work=Path(directory); harness=work/'check.c'; harness.write_text(source)
            lines=subprocess.check_output(['make','-n','test-typechecker'],cwd=ROOT,text=True).splitlines()
            command=next(shlex.split(line) for line in lines if 'tests/test_typechecker.c' in line and '-o' in line)
            command[command.index('tests/test_typechecker.c')]=str(harness)
            command[command.index('-o')+1]=str(work/'check')
            compiled=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=120)
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            result=subprocess.run([str(work/'check')],cwd=ROOT,text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__': unittest.main()
