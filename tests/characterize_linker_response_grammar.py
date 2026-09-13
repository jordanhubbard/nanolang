"""I compare native forwarded responses with my existing driver decoder.

Run python3 -m tests.characterize_linker_response_grammar [compiler].
--require-equivalent rejects any admitted decoding that changes native results.
This is a parser experiment, not production forwarded-response capture.
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
            response.write_text(contents)
            captured = run([probe, "capture-response", shlex.quote("@" + str(response))])
            if captured.returncode: raise RuntimeError(captured.stderr.decode())
            words = shlex.split(captured.stdout.decode())
            admitted = not any(word.startswith("@") for word in words)
            native = link(name + "-native", ["-Wl,@" + str(response)])
            candidate = link(name + "-candidate", [word for value in words for word in ("-Xlinker", value)]) if admitted else None
            case = {"case": name, "driver_decoder_admitted": admitted, "decoded_words": words,
                    "native": native, "candidate": candidate}
            case["equivalent"] = equivalent(case)
            cases.append(case)
    return {"platform": sys.platform, "compiler": compiler, "cases": cases}


def equivalent(case):
    if not case["driver_decoder_admitted"]: return True
    native, candidate = case["native"], case["candidate"]
    return (native["status"] == candidate["status"] and native["answer"] == candidate["answer"]
            and (native["status"] != 0 or native["answer"] == 42))


def require_equivalent(result):
    if not result["cases"] or any(not equivalent(case) for case in result["cases"]):
        raise SystemExit("I cannot substitute my driver decoder for this linker response grammar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", nargs="?", default="cc")
    parser.add_argument("--require-equivalent", action="store_true")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if not compiler: raise SystemExit("I need a C compiler executable")
    result = measure(compiler)
    print(json.dumps(result, indent=2))
    if args.require_equivalent: require_equivalent(result)
