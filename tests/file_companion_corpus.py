"""I retain graph inputs and independently specified declaration expectations."""
import json
from pathlib import Path

DECL='service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"\n'

def corpus(root, work):
    document=(root/'tests/fixtures/nsi_file_plan.json').read_bytes()
    publisher=(root/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text()
    cases=[]
    def add(name, files, status=1, required=(), absent=(), env=None):
        directory=work/name;directory.mkdir()
        for path,text in files.items():
            target=directory/path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(text if isinstance(text,bytes) else text.encode())
        for parent in {p.parent for p in directory.rglob('*.nano')}:
            (parent/'interface.nsi.json').write_bytes(document)
        cases.append(dict(name=name,entry=str(directory/'main.nano'),status=status,
                          required=list(required),absent=list(absent),env=env or {}))
    add('publisher',{'main.nano':publisher},required=('File','temp','close'))
    add('ordinary',{'main.nano':'fn main() -> int { return 37 }\nshadow main { assert (== (main) 37) }\n'},0)
    add('global-vs-local',{'main.nano':DECL+'let value: int = 7\nfn answer() -> int { let local: int = 3 return local }\nshadow answer { let hidden: int = 3 assert (== (answer) hidden) }\n'},required=('value','answer'),absent=('local','hidden'))
    add('qualified',{'main.nano':'import "./dep.nano" as F\nfn main() -> int { return 0 }\n','dep.nano':DECL+'pub fn answer() -> int { return 3 }\nfn private_answer() -> int { return 4 }\n'},required=('F','answer','private_answer','temp'))
    add('nested-reexport',{'main.nano':'import "./mid.nano" as Outer\nfn main() -> int { return 0 }\n','mid.nano':'pub use "./dep.nano" as Inner\n','dep.nano':DECL+'pub fn answer() -> int { return 3 }\n'},required=('Outer','Inner','temp'))
    add('selective',{'main.nano':'from "./dep.nano" import temp as make_temp, File\nfn main() -> int { return 0 }\n','dep.nano':DECL},required=('make_temp','File'))
    add('same-module-twice',{'main.nano':'import "./dep.nano" as A\nimport "./dep.nano" as B\n','dep.nano':DECL},required=('A','B','temp'))
    add('different-module-same-catalog',{'main.nano':'import "./a.nano" as A\nimport "./b.nano" as B\n','a.nano':DECL,'b.nano':DECL},required=('A','B'))
    add('collision',{'main.nano':DECL+'fn temp() -> int { return 0 }\n'},2)
    add('cycle',{'main.nano':DECL+'import "./dep.nano"\n','dep.nano':'import "./main.nano"\n'},5)
    add('metadata-unresolved',{'main.nano':DECL,'module.json':'{"name":"unknown","headers":["unknown.h"]}\n'},5)
    add('nul',{'main.nano':DECL.encode()+b'\0hidden'},2)
    add('utf8',{'main.nano':DECL.encode()+b'\xff'},2)
    add('configured-files',{'main.nano':'import "./dep.nano" as F\n','dep.nano':DECL},3,env={'NANO_IMPORT_MAX_FILES':'1'})
    add('configured-lines',{'main.nano':DECL+'# extra\n'},3,env={'NANO_IMPORT_MAX_LINES_PER_FILE':'1'})
    add('wildcard',{'main.nano':'from "./dep.nano" import *\n','dep.nano':DECL+'pub fn answer() -> int { return 3 }\nfn hidden() -> int { return 4 }\n'},required=('answer','temp'))
    add('namespace-selective-reexport',{'main.nano':'from "./mid.nano" import Inner as Renamed\n','mid.nano':'pub use "./dep.nano" as Inner\n','dep.nano':DECL},required=('Inner','Renamed','temp'))
    add('private-selective',{'main.nano':'from "./dep.nano" import hidden\n','dep.nano':DECL+'fn hidden() -> int { return 4 }\n'},5)
    add('ordinary-kinds',{'main.nano':DECL+'pub struct Record { value: int }\npub enum Color { Red, Blue }\npub union Choice { None {}, Some { value: int } }\nopaque type External\nfn answer() -> int { return 3 }\nlet top: int = 3\n'},required=('Record','Color','Choice','External','answer','top'))
    for count in (256,257):
        add('ordinary-limit-'+str(count),{'main.nano':DECL+''.join(f'fn ordinary_{i}() -> int {{ return {i} }}\n' for i in range(count))},1 if count==256 else 3)
    for count in (64,65):
        add('alias-limit-'+str(count),{'main.nano':''.join(f'from "./dep.nano" import temp as alias_{i}\n' for i in range(count)),'dep.nano':DECL},1 if count==64 else 3)
    for count in (16,17):
        files={'main.nano':''.join(f'import "./dep{i}.nano"\n' for i in range(count))}
        # Separate roots avoid unqualified catalog collisions while retaining
        # every service request in the actual graph.
        files['main.nano']=''.join(f'import "./dep{i}.nano" as F{i}\n' for i in range(count))
        files.update({f'dep{i}.nano':DECL for i in range(count)})
        # Qualified visibility also consumes aliases; I test the request limit
        # directly in the snapshot controls, not claim this graph isolates it.
        add('many-service-graph-'+str(count),files,3)
    (work/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    return cases

NANO_REPORT=r'''
fn companion_emit_span(value: string) -> void {
    let size: int = (str_length value)
    (print (int_to_string size)) (print ":")
    let digits: string = "0123456789abcdef"
    let mut i: int = 0
    while (< i size) {
        let c: int = (char_at value i)
        (print (str_substring digits (/ c 16) 1))
        (print (str_substring digits (% c 16) 1))
        set i (+ i 1)
    }
}
shadow companion_emit_span { assert (== (str_length "abc") 3) }
fn companion_emit_number(value: int) -> void { (print " ") (print (int_to_string value)) }
shadow companion_emit_number { assert (== 1 1) }
fn companion_report(path: string) -> int {
    let merged: MergeResult = (merge_with_imports_mode path true)
    if (== (array_length merged.files) 0) { (println "COLLECTION_REFUSED") return 0 }
    let result: FileResolution = (file_resolution_prepare merged.files merged.original_sources merged.original_parsers merged.repo_root resolve_import_path)
    (print "STATUS ") (println (int_to_string result.status))
    if (!= result.status 1) { return 0 }
    (print "COUNTS") (companion_emit_number (array_length result.origins))
    (companion_emit_number (array_length result.rows)) (companion_emit_number (array_length result.plan.rows)) (println "")
    let mut i: int = 0
    while (< i (array_length result.origins)) {
        let origin: FileParsedOrigin = (at result.origins i)
        (print "ORIGIN ") (companion_emit_span origin.path) (print " ") (companion_emit_span origin.source) (println "")
        set i (+ i 1)
    }
    set i 0
    while (< i (array_length result.rows)) {
        let row: FileVisibility = (at result.rows i)
        (print "ROW ") (companion_emit_span row.origin) (print " ") (companion_emit_span row.qualifier)
        (print " ") (companion_emit_span row.name) (print " ") (companion_emit_span row.target_origin) (print " ") (companion_emit_span row.target_name)
        (companion_emit_number row.id) (companion_emit_number row.target) (companion_emit_number row.kind) (companion_emit_number row.ordinal)
        if row.exported { (companion_emit_number 1) } else { (companion_emit_number 0) }
        if row.service { (companion_emit_number 1) } else { (companion_emit_number 0) }
        (println "") set i (+ i 1)
    }
    set i 0
    while (< i (array_length result.plan.rows)) {
        let row: FileSourceRow = (at result.plan.rows i)
        (print "PLAN ") (companion_emit_span row.module_id) (print " ") (companion_emit_span row.name)
        (companion_emit_number row.id) (companion_emit_number row.target) (companion_emit_number row.request)
        (companion_emit_number row.kind) (companion_emit_number row.ordinal) (companion_emit_number row.category)
        (companion_emit_number row.input_mode) (companion_emit_number row.result_ordinal)
        (companion_emit_number row.global_layout) (companion_emit_number row.import_index)
        (companion_emit_number row.line) (companion_emit_number row.column) (println "") set i (+ i 1)
    }
    set i 0
    while (< i (array_length result.snapshots)) {
        let snapshot: FileSourceSnapshot = (at result.snapshots i)
        (print "SNAPSHOT ") (companion_emit_span snapshot.module_path) (print " ") (companion_emit_span snapshot.original)
        (print " ") (companion_emit_span snapshot.canonical) (print " ") (companion_emit_span snapshot.generated)
        (print " ") (companion_emit_span snapshot.catalog) (println "") set i (+ i 1)
    }
    return 0
}
shadow companion_report { assert (== 1 1) }
fn main() -> int {
    let options: CompileOptions = (parse_args)
    return (companion_report options.input_file)
}
shadow main { assert (== 1 1) }
'''

def report_source(root):
    source=(root/'src_nano/nanoc_v06.nano').read_text()
    assert source.count('fn main() -> int {')==1
    # I change only the fixture entry name, retaining the real collector and
    # every original driver/module shadow. No alternate visibility grammar.
    source=source.replace('fn main() -> int {','fn companion_driver_main() -> int {')
    source=source.replace('shadow main {','shadow companion_driver_main {')
    return source+'\n'+NANO_REPORT

PROVIDER_REPORT=r'''
fn companion_provider_probe(manifest: string, runtime: string) -> int {
    let files: array<string> = (split_lines (file_read manifest))
    let root: string = (getenv "COMPANION_PROVIDER_ROOT")
    let directory: string = (mktemp_dir "nano_provider_fixture_")
    if (== directory "") { return 2 }
    let result: ModuleBuildFlags = (collect_module_build_flags files root runtime directory true)
    (print "PROVIDER_STATUS ")
    if result.ok { (println "0") } else { (println (int_to_string result.first_status)) }
    let mut i: int = 0
    while (< i (array_length result.runtime_sources)) {
        (print "RUNTIME ") (println (at result.runtime_sources i)) set i (+ i 1)
    }
    set i 0
    while (< i (array_length result.objects)) {
        (print "OBJECT ") (println (at result.objects i)) set i (+ i 1)
    }
    let clean: bool = (native_provider_cleanup result)
    set i 0
    while (< i (array_length result.objects)) {
        assert (not (file_exists (at result.objects i))) set i (+ i 1)
    }
    (native_cleanup directory false)
    assert clean
    assert (not (file_exists directory))
    return 0
}
shadow companion_provider_probe { assert (== 1 1) }
fn main() -> int {
    let options: CompileOptions = (parse_args)
    return (companion_provider_probe options.input_file options.output_file)
}
shadow main { assert (== 1 1) }
'''

def provider_report_source(root):
    source=(root/'src_nano/nanoc_v06.nano').read_text()
    assert source.count('fn main() -> int {')==1
    return source.replace('fn main() -> int {','fn companion_driver_main() -> int {').replace('shadow main {','shadow companion_driver_main {')+'\n'+PROVIDER_REPORT
