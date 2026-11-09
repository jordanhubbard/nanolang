#!/usr/bin/env python3
"""I measure a private fresh-record binding candidate without changing production."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
PIN = "93474a7bbd6405259a83218997ba764f0b9f808b"
UNIT = r'''
#include "nanolang.h"
#include "runtime/gc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static size_t allocations;
static int counting;
#ifdef COUNT_ALLOCATIONS
static void *count_malloc(size_t n) {
    if (counting) ++allocations;
    return malloc(n);
}
#define malloc count_malloc
#endif
#include ENV_SOURCE
#ifdef COUNT_ALLOCATIONS
#undef malloc
#endif
#include "eval/record_carrier_cleanup.inc"

static Environment frame;
static void clear_binding(void) {
    assert(frame.symbol_count == 1);
    free(frame.symbols[0].name);
    free(frame.symbols[0].struct_type_name);
    frame.symbol_count = 0;
}
static void check_binding(ASTNode *expression, Type type, Value fresh, int moved) {
    StructValue *original = fresh.as.struct_val;
    env_define_evaluated_var(&frame, "local", type, TYPE_UNKNOWN, NULL, true,
                             fresh, expression);
    Value held = frame.symbols[0].value;
    assert(held.type == VAL_STRUCT);
    assert((held.as.struct_val == original) == moved);
    if (!moved) discard_literal_record(original);
    discard_literal_record(held.as.struct_val);
    clear_binding();
}
static void controls(void) {
    ASTNode literal = {.type=AST_STRUCT_LITERAL};
    ASTNode identifier = {.type=AST_IDENTIFIER};
    ASTNode call = {.type=AST_CALL};
    char name[] = "é";
    char field[] = "茶";
    char *names[] = {field};
    Value text = create_string("held");
    Value fresh = create_struct(name, names, &text, 1);
    gc_release(text.as.string_val);
    name[0] = 'x'; field[0] = 'x';
    assert(!strcmp(fresh.as.struct_val->struct_name, "é"));
    assert(!strcmp(fresh.as.struct_val->field_names[0], "茶"));
    env_define_evaluated_var(&frame, "local", TYPE_STRUCT, TYPE_UNKNOWN, NULL,
                             true, fresh, &literal);
    StructValue *held = frame.symbols[0].value.as.struct_val;
    assert((held == fresh.as.struct_val) == CANDIDATE);
    assert(!strcmp(held->field_values[0].as.string_val, "held"));
    if (!CANDIDATE) discard_literal_record(fresh.as.struct_val);
    assert(!strcmp(held->field_values[0].as.string_val, "held"));
    discard_literal_record(held); clear_binding();
    check_binding(&literal, TYPE_STRUCT, create_struct("", NULL, NULL, 0), CANDIDATE);
    check_binding(&identifier, TYPE_STRUCT, create_struct("", NULL, NULL, 0), 0);
    check_binding(&call, TYPE_STRUCT, create_struct("", NULL, NULL, 0), 0);
    check_binding(NULL, TYPE_STRUCT, create_struct("", NULL, NULL, 0), 0);
    for (int flag=0; flag<3; ++flag) {
        fresh=create_struct("Flag", NULL, NULL, 0);
        fresh.is_return=flag==0; fresh.is_break=flag==1; fresh.is_continue=flag==2;
        check_binding(&literal, TYPE_STRUCT, fresh, 0);
    }
    for (int mut=0; mut<2; ++mut) {
        fresh=create_struct("Borrowed", NULL, NULL, 0);
        env_define_evaluated_var(&frame,"local",mut?TYPE_BORROW_MUT:TYPE_BORROW_SHARED,
                                 TYPE_UNKNOWN,NULL,true,fresh,&literal);
        assert(frame.symbols[0].value.as.struct_val==fresh.as.struct_val);
        clear_binding(); discard_literal_record(fresh.as.struct_val);
    }
    Value array=create_array(VAL_INT,1,1);
    ((long long*)array.as.array_val->data)[0]=42;
    Value contents[]={create_string("nested"),array};
    char *inner_names[]={"text","array"};
    Value inner=create_struct("Inner",inner_names,contents,2);
    gc_release(contents[0].as.string_val);
    char *outer_names[]={"inner"};
    fresh=create_struct("Outer",outer_names,&inner,1);
    discard_literal_record(inner.as.struct_val);
    env_define_evaluated_var(&frame,"local",TYPE_STRUCT,TYPE_UNKNOWN,NULL,true,fresh,&literal);
    held=frame.symbols[0].value.as.struct_val;
    StructValue *nested=held->field_values[0].as.struct_val;
    assert(!strcmp(nested->field_values[0].as.string_val,"nested"));
    assert(nested->field_values[1].as.array_val==array.as.array_val);
    if(!CANDIDATE) discard_literal_record(fresh.as.struct_val);
    assert(!strcmp(nested->field_values[0].as.string_val,"nested"));
    discard_literal_record(held); clear_binding();
    assert(((long long*)array.as.array_val->data)[0]==42);
    free(array.as.array_val->data);free(array.as.array_val);
}
int main(void) {
    gc_init();
    frame.symbol_capacity=8;
    frame.symbols=calloc(8,sizeof(Symbol));
    controls();
    GCStats before=gc_get_stats();
    ASTNode literal={.type=AST_STRUCT_LITERAL};
    char *names[]={"a","b","c","d"};
    Value values[]={create_int(1),create_int(2),create_int(3),create_int(4)};
    long long checksum=0;
    allocations=0;
    clock_t start=clock();
    for(int i=0;i<100000;++i) {
        Value fresh=create_struct("Point",names,values,4);
        counting=1;
        env_define_evaluated_var(&frame,"local",TYPE_STRUCT,TYPE_UNKNOWN,NULL,true,fresh,&literal);
        counting=0;
        StructValue *held=frame.symbols[0].value.as.struct_val;
        checksum+=held->field_values[3].as.int_val;
        if(!CANDIDATE) discard_literal_record(fresh.as.struct_val);
        discard_literal_record(held);clear_binding();
    }
    double seconds=(double)(clock()-start)/CLOCKS_PER_SEC;
    assert(checksum==400000);
#ifdef COUNT_ALLOCATIONS
    assert(allocations==(CANDIDATE?0:300000));
#endif
    GCStats after=gc_get_stats();
    assert(before.current_usage==after.current_usage && before.num_objects==after.num_objects);
    env_symbol_index_invalidate(&frame);free(frame.symbols);
    printf("{\"candidate\":%d,\"checksum\":%lld,\"binding_mallocs\":%zu,\"cpu_seconds\":%.9f}\n",
           CANDIDATE,checksum,allocations,seconds);
    return 0;
}
'''


def main():
    os.chdir(ROOT)
    output = Path(tempfile.mkdtemp(prefix="fresh-record-bindings-", dir=ROOT / ".test_output"))
    report = {"scope": "I measure an isolated local-binding candidate, not whole compiler timing, safe frame reclamation or SDK closure.",
              "source_pin": PIN, "output": str(output), "commands": [], "runs": []}
    report["collector_source"] = Path(__file__).read_text()
    report["storage_cap_bytes"] = 32 * 1024**2
    report["storage_samples"] = []
    digest = lambda b: hashlib.sha256(b).hexdigest()
    paths = ["src/env.c", "src/nanolang.h", "src/runtime/gc.c", "src/runtime/gc_struct.c",
             "src/runtime/dyn_array.c", "src/eval/record_carrier_cleanup.inc"]
    report["before"] = {p: digest(Path(p).read_bytes()) for p in paths}
    try:
        for p in paths:
            assert subprocess.check_output(["git", "show", f"{PIN}:{p}"]) == Path(p).read_bytes(), p
        original = Path("src/env.c").read_text()
        start = original.index("static void env_define_var_impl(")
        end = original.index("/* Get variable */", start)
        candidate = original[start:end]
        replacements = [
            ("static void env_define_var_impl(Environment *env, const char *name, Type type, Type element_type, TypeInfo *type_info, bool is_mut, Value value, bool retain_string_aliases) {",
             "static void env_define_var_impl(Environment *env, const char *name, Type type, Type element_type, TypeInfo *type_info, bool is_mut, Value value, bool retain_string_aliases, bool copy_records) {"),
            ("if (value.type == VAL_STRUCT && value.as.struct_val &&\n        type != TYPE_BORROW_SHARED", "if (copy_records && value.type == VAL_STRUCT && value.as.struct_val &&\n        type != TYPE_BORROW_SHARED"),
            ("is_mut, value, true);", "is_mut, value, true, true);"),
            ("!evaluated_string_is_fresh(expression, value));", "!evaluated_string_is_fresh(expression, value),\n                        !(expression && expression->type == AST_STRUCT_LITERAL &&\n                          value.type == VAL_STRUCT && value.as.struct_val &&\n                          !value.is_return && !value.is_break && !value.is_continue));"),
            ("is_mut, create_string(text), false);", "is_mut, create_string(text), false, true);")]
        for old, new in replacements:
            assert candidate.count(old) == 1, old
            candidate = candidate.replace(old, new)
        candidate = original[:start] + candidate + original[end:]
        report["original_source"] = original
        report["candidate_source"] = candidate
        report["unit_source"] = UNIT
        (output / "unit.c").write_text(UNIT)
        cc = shutil.which("cc")
        report["compiler"] = {"path": cc, "sha256": digest(Path(cc).read_bytes()),
                              "version": subprocess.check_output([cc, "--version"], text=True)}
        def storage(reserve=0):
            size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            report["storage_samples"].append(size)
            assert size + reserve < report["storage_cap_bytes"], (size, reserve)
        def run(argv, timeout=60, env=None):
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
            entry = {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            report["commands"].append(entry)
            assert result.returncode == 0, entry
            storage()
            return result
        environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
        report["run_env"] = {k: environment[k] for k in ["ASAN_OPTIONS", "UBSAN_OPTIONS"]}
        report["images_before"] = {}
        report["images_after"] = {}
        for label, source in [("original", original), ("candidate", candidate)]:
            folder = output / label; folder.mkdir()
            (folder / "env.c").write_text(source)
        for mode, count in [("counted", 1), ("sanitized", 1), ("plain", 20)]:
            for label in ["original", "candidate"]:
                folder = output / label
                storage(reserve=10 * 1024**2)
                argv = [cc, "-std=c11", "-D_DARWIN_C_SOURCE", "-D_POSIX_C_SOURCE=200809L", "-O2",
                        "-Wall", "-Wextra", "-Werror", "-UNDEBUG",
                        "-ffunction-sections", "-fdata-sections", f'-DENV_SOURCE="{folder / "env.c"}"',
                        f'-DCANDIDATE={int(label == "candidate")}', "-Isrc", str(output / "unit.c"),
                        "src/runtime/gc.c", "src/runtime/gc_struct.c", "src/runtime/dyn_array.c",
                        "-Wl,-dead_strip", "-o", str(folder / mode)]
                if mode == "counted": argv += ["-DCOUNT_ALLOCATIONS"]
                if mode == "sanitized": argv += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                run(argv)
                image = folder / mode
                report["images_before"][str(image)] = digest(image.read_bytes())
            for pair in range(count):
                order = ["original", "candidate"] if pair % 2 == 0 else ["candidate", "original"]
                for label in order:
                    result = run([str(output / label / mode)], env=environment)
                    report["runs"].append({"mode": mode, "pair": pair, "variant": label, "result": json.loads(result.stdout)})
            for label in ["original", "candidate"]:
                image = output / label / mode
                data = image.read_bytes()
                report["images_after"][str(image)] = digest(data)
                assert report["images_before"][str(image)] == digest(data)
                compressed = image.with_suffix(".gz")
                compressed.write_bytes(gzip.compress(data, mtime=0))
                storage()
                assert digest(gzip.decompress(compressed.read_bytes())) == digest(data)
                image.unlink()  # I retain the exact image bytes losslessly within the original cap.
                storage()
        assert report["images_before"] == report["images_after"]
        report["after"] = {p: digest(Path(p).read_bytes()) for p in paths}
        assert report["before"] == report["after"]
        storage()
        report["ok"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (EVIDENCE / "measurement.json.gz").write_bytes(gzip.compress(json.dumps(report, indent=2).encode(), mtime=0))
        print(json.dumps({"output": str(output), "ok": report.get("ok", False), "error": report.get("error")}))


if __name__ == "__main__":
    main()
