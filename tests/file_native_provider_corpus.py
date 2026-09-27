"""I exercise the actual provider preparer with private real compiler objects."""
import json
from pathlib import Path

RUNTIME_SOURCES=(
 'list_int','list_string','list_LexerToken','list_token','list_CompilerDiagnostic',
 'list_CompilerSourceLocation','token_helpers','gc','dyn_array','hashmap_bootstrap',
 'gc_struct','nl_string','cli','regex','list_ASTLet','list_ASTFunction','list_ASTNumber',
 'list_ASTFloat','list_ASTString','list_ASTBool','list_ASTIdentifier','list_ASTBinaryOp',
 'list_ASTCall','list_ASTModuleQualifiedCall','list_ASTArrayLiteral','list_ASTStmtRef',
 'list_ASTBlock','list_ASTUnsafeBlock','list_ASTPrint','list_ASTAssert','list_ASTFieldAccess',
 'list_ASTSet','list_ASTIf','list_ASTWhile','list_ASTFor','list_ASTReturn','list_ASTShadow',
 'list_ASTStruct','list_ASTStructLiteral','list_ASTEnum','list_ASTUnion','list_ASTUnionConstruct',
 'list_ASTMatch','list_ASTImport','list_ASTOpaqueType','list_ASTServiceDecl',
 'list_ASTTupleLiteral','list_ASTTupleIndex')

def setup(root,work):
    repo=work/'provider-repo';repo.mkdir();(repo/'src').symlink_to(root/'src',target_is_directory=True)
    (repo/'modules').mkdir();(repo/'modules/std').symlink_to(root/'modules/std',target_is_directory=True)
    for name in ('a','b','c','broken','conflict','dep'):
        directory=repo/'modules'/name;directory.mkdir();(directory/'main.nano').write_text('')
    def metadata(name,value):(repo/'modules'/name/'module.json').write_text(json.dumps(dict(name=name,**value)))
    (repo/'modules/a/same.c').write_text('#if PROFILE != 23\n#error I require the final ordered profile\n#endif\nint provider_a(void){return PROFILE;}\n')
    (repo/'modules/b/same.c').write_text('int provider_b(void){return 19;}\n')
    (repo/'modules/dep/dep.c').write_text('int provider_dependency(void){return 3;}\n')
    (repo/'modules/b/alias.c').symlink_to(repo/'modules/a/same.c')
    (repo/'modules/broken/fail.c').write_text('#error I am the required failing provider\n')
    metadata('a',dict(c_sources=['same.c'],cflags=['-DPROFILE=11']))
    metadata('b',dict(c_sources=['same.c','alias.c'],cflags=['-UPROFILE','-DPROFILE=23'],dependencies=['dep']))
    metadata('c',dict(shared_c_sources=['../a/same.c'],cflags=[]))
    metadata('dep',dict(c_sources=['dep.c']))
    metadata('broken',dict(c_sources=['fail.c']))
    metadata('conflict',dict(c_sources=['../a/same.c'],c_compiler='c++'))
    cases=[]
    for name,modules,ok,count in [('unique',['a','b','c'],True,3),('required-failure',['a','b','broken'],False,4),('conflicting-language',['a','b','conflict'],False,0)]:
        manifest=work/(name+'.paths');manifest.write_text('\n'.join(str(repo/'modules'/m/'main.nano') for m in modules))
        cases.append(dict(name=name,manifest=manifest,ok=ok,objects=count))
    # Every runtime input is an actual symlink to a selected regular source.
    # Two inventory entries share one canonical source in the alias variant.
    runtime=work/'runtime-alias';runtime.mkdir()
    paths=[*(f'src/runtime/{name}.c' for name in RUNTIME_SOURCES),'src/utf8.c',
           'modules/std/fs.c','modules/std/process.c','modules/std/collections/collections.c',
           'modules/std/json/json.c','src/cJSON.c']
    assert len(paths)==54
    for path in paths:
        dest=runtime/path;dest.parent.mkdir(parents=True,exist_ok=True)
        target=root/('src/runtime/list_ASTFloat.c' if path=='src/runtime/list_ASTBool.c' else path)
        dest.symlink_to(target)
    return repo,cases,runtime
