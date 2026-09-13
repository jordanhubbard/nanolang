"""I compare native forwarded responses with my existing driver decoder.

Run python3 -m tests.characterize_linker_response_grammar [compiler].
--require-equivalent rejects any admitted decoding that changes native results.
--require-retained-equivalent checks the byte-preserving fixture prototype.
--require-captured-equivalent checks my C graph-capture mechanism.
This is an experiment, not production forwarded-response capture.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from tests import test_bytecode_shadows as shadows


def measure(compiler):
    probe = shadows.ROOT / "obj/test_module_generation_probe"
    cases = []
    with tempfile.TemporaryDirectory(prefix="nano-linker-grammar-") as tmp:
        directory = Path(tmp)

        def run(args):
            return subprocess.run(list(map(str, args)), cwd=directory, capture_output=True, timeout=15)

        member, obj, archive = directory / "member.c", directory / "member.o", directory / "selected.a"
        member.write_text("long long selected(void) { return 42; }\n")
        for command in ([compiler, "-fPIC", "-c", member, "-o", obj], ["ar", "rcs", archive, obj]):
            result = run(command)
            if result.returncode: raise RuntimeError(result.stderr.decode())
        source = directory / "main.c"
        source.write_text("extern long long selected(void); long long answer(void) { return selected(); }\n")
        for name in ("with space.a", "back\\slash.a", "comma,name.a"):
            shutil.copyfile(archive, directory / name)
        (directory / "inner.rsp").write_text("selected.a\n")
        (directory / "other.rsp").write_text("selected.a\n")
        (directory / "symlink.rsp").symlink_to("inner.rsp")
        (directory / "hardlink.rsp").hardlink_to(directory / "inner.rsp")
        retained_dir = directory / "retained"
        retained_dir.mkdir()
        # I know this fixture's nested graph explicitly. This is not a parser.
        # Resolved paths keep separate retained paths even when their payloads
        # agree; aliases of one resolved path keep the same retained path.
        retained_nested = {}
        retained_paths = {}
        for spelling in ("inner.rsp", "other.rsp", "./inner.rsp", "symlink.rsp", "hardlink.rsp"):
            source_path = (directory / spelling).resolve()
            if source_path not in retained_paths:
                retained = retained_dir / f"nested-{len(retained_paths)}.rsp"
                retained.write_bytes(source_path.read_bytes())
                retained_paths[source_path] = retained
            retained = retained_paths[source_path]
            retained_nested[spelling] = retained
        response_dir = directory / "responses"
        response_dir.mkdir()
        response = response_dir / "outer.rsp"
        inputs = {
            "plain": "selected.a\n",
            "single_quotes": "'with space.a'\n",
            "double_quotes": '"with space.a"\n',
            "embedded_quotes": 'sel"ected".a\n',
            "escaped_space": "with\\ space.a\n",
            "quoted_backslash": "'back\\\\slash.a'\n",
            "comma": "comma,name.a\n",
            "vertical_tab": "selected.a\v-lm\n",
            "form_feed": "selected.a\f-lm\n",
            "nested_cwd_relative": "@inner.rsp\n",
            "repeated_response": "@inner.rsp @inner.rsp\n",
            "distinct_equal_responses": "@inner.rsp @other.rsp\n",
            "aliased_response_spelling": "@inner.rsp @./inner.rsp\n",
            "symlink_response_alias": "@inner.rsp @symlink.rsp\n",
            "hardlink_response_alias": "@inner.rsp @hardlink.rsp\n",
            "unterminated_quote": "'selected.a\n",
        }

        def link(label, arguments):
            output = directory / (label + ".so")
            result = run([compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC",
                          source, "-o", output, *arguments])
            answer = None
            if result.returncode == 0:
                loaded = run([sys.executable, "-c", "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
                              "lib.answer.restype=ctypes.c_int64; print(lib.answer())", output])
                if loaded.returncode == 0: answer = int(loaded.stdout)
            return {"status": result.returncode, "answer": answer, "error": result.stderr.decode()[-400:]}

        for name, contents in inputs.items():
            for spelling in ("inner.rsp", "other.rsp"):
                (directory / spelling).write_text("selected.a\n")
            (directory / "hardlink.rsp").unlink(missing_ok=True)
            (directory / "hardlink.rsp").hardlink_to(directory / "inner.rsp")
            assert (directory / "hardlink.rsp").stat().st_ino == (directory / "inner.rsp").stat().st_ino
            response.write_text(contents)
            captured = run([probe, "capture-response", shlex.quote("@" + str(response))])
            if captured.returncode: raise RuntimeError(captured.stderr.decode())
            words = shlex.split(captured.stdout.decode())
            admitted = not any(word.startswith("@") for word in words)
            native = link(name + "-native", ["-Wl,@" + str(response)])
            candidate = link(name + "-candidate", [word for value in words for word in ("-Xlinker", value)]) if admitted else None
            graph = run([probe, "capture-link-response", "apple" if sys.platform == "darwin" else "gnu",
                         directory, response])
            if graph.returncode: raise RuntimeError("I could not capture the fixture response graph")
            graph_path = Path(graph.stdout.decode().strip())
            retained_contents = contents
            collapsed_contents = contents
            for spelling, path in retained_nested.items():
                retained_contents = retained_contents.replace("@" + spelling, "@" + str(path))
                collapsed_contents = collapsed_contents.replace("@" + spelling, "@" + str(retained_nested["inner.rsp"]))
            # Neither retained link may depend on the original response graph.
            response.unlink()
            for spelling in ("inner.rsp", "other.rsp", "hardlink.rsp"):
                (directory / spelling).unlink()
            retained_response = retained_dir / "outer.rsp"
            retained_response.write_text(retained_contents)
            retained = link(name + "-retained", ["-Wl,@" + str(retained_response)])
            captured_graph = link(name + "-captured", ["-Wl,@" + str(graph_path)])
            retained_response.write_text(collapsed_contents)
            collapsed = link(name + "-collapsed", ["-Wl,@" + str(retained_response)])
            case = {"case": name, "driver_decoder_admitted": admitted, "decoded_words": words,
                    "native": native, "candidate": candidate, "retained": retained,
                    "collapsed": collapsed, "captured": captured_graph}
            case["equivalent"] = equivalent(case)
            case["retained_equivalent"] = outcomes_agree(native, retained)
            case["collapsed_equivalent"] = outcomes_agree(native, collapsed)
            case["captured_equivalent"] = outcomes_agree(native, captured_graph)
            cases.append(case)
    return {"platform": sys.platform, "compiler": compiler, "cases": cases}


def equivalent(case):
    if not case["driver_decoder_admitted"]: return True
    return outcomes_agree(case["native"], case["candidate"])


def outcomes_agree(native, candidate):
    return (native["status"] == candidate["status"] and native["answer"] == candidate["answer"]
            and (native["status"] != 0 or native["answer"] == 42))


def require_equivalent(result):
    if not result["cases"] or any(not equivalent(case) for case in result["cases"]):
        raise SystemExit("I cannot substitute my driver decoder for this linker response grammar.")


def require_retained_equivalent(result, field="retained"):
    if not result["cases"] or any(not outcomes_agree(case["native"], case[field])
                                  for case in result["cases"]):
        raise SystemExit("I cannot substitute these retained linker responses.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", nargs="?", default="cc")
    parser.add_argument("--require-equivalent", action="store_true")
    parser.add_argument("--require-retained-equivalent", action="store_true")
    parser.add_argument("--require-captured-equivalent", action="store_true")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if not compiler: raise SystemExit("I need a C compiler executable")
    result = measure(compiler)
    print(json.dumps(result, indent=2))
    if args.require_equivalent: require_equivalent(result)
    if args.require_retained_equivalent: require_retained_equivalent(result)
    if args.require_captured_equivalent: require_retained_equivalent(result, "captured")
