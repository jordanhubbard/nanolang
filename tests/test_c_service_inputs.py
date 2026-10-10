"""I exercise strict companions through both actual C frontend drivers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CServiceInputs(unittest.TestCase):
    def test_transitive_companions_and_prior_output(self):
        declaration = 'service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"\n'
        catalog = (ROOT / "tests/fixtures/nsi_file_plan.json").read_bytes()
        with tempfile.TemporaryDirectory(prefix="nano-c-inputs-") as directory:
            work = Path(directory)
            library = work / "library"
            library.mkdir()
            (library / "binding.nano").write_text(declaration)
            (work / "bridge.nano").write_text('module "library/binding.nano" as files\n')
            root = work / "root.nano"
            root.write_text('module "bridge.nano" as bridge\nfn main() -> int { return 0 }\nshadow main { assert true }\n')
            companion = library / "interface.nsi.json"
            output = work / "result"
            for compiler in ("nanoc_c", "nano_virt"):
                for mode in ("valid", "invalid", "missing", "symlink", "output-alias"):
                    with self.subTest(compiler=compiler, mode=mode):
                        companion.unlink(missing_ok=True)
                        companion.write_bytes(catalog)
                        output.write_bytes(b"prior-output")
                        if mode == "invalid": companion.write_text("{}")
                        if mode == "missing": companion.unlink()
                        if mode == "symlink":
                            target = work / "catalog.json"
                            target.write_bytes(catalog)
                            companion.unlink()
                            companion.symlink_to(target)
                        before = companion.read_bytes() if companion.exists() else None
                        command = [ROOT / "bin" / compiler, root, "-o", companion if mode == "output-alias" else output]
                        if compiler == "nano_virt": command.append("--emit-nvm")
                        result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                                capture_output=True, text=True, timeout=60)
                        self.assertNotEqual(result.returncode, 0)
                        text = result.stdout + result.stderr
                        if mode in ("invalid", "missing", "symlink"):
                            self.assertIn("I cannot acquire the immutable companion", text)
                        elif mode == "output-alias":
                            self.assertIn("I will not replace a File companion input", text)
                        else:
                            self.assertIn("I require --allow-temporary-files", text)
                        self.assertEqual(output.read_bytes(), b"prior-output")
                        if before is not None: self.assertEqual(companion.read_bytes(), before)


    def test_tcp_inputs_do_not_grant_source_execution(self):
        with tempfile.TemporaryDirectory(prefix="nano-tcp-admission-") as directory:
            work=Path(directory)
            source=work/'binding.nano'
            source.write_text('service "nsi:nanolang/net" catalog 1 from "interface.nsi.json"\nfn main() -> int { return 0 }\nshadow main { assert true }\n')
            companion=work/'interface.nsi.json'
            catalog=(ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes()
            companion.write_bytes(catalog)
            for compiler in ("nanoc_c","nano_virt","nanoc_stage1","nanoc_stage2"):
                with self.subTest(compiler=compiler):
                    output=work/(compiler+'.out');output.write_bytes(b'prior-output')
                    args=[ROOT/'bin'/compiler,source,'-o',output,'--allow-temporary-files']
                    if compiler!='nanoc_c':args.append('--emit-nvm')
                    result=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=60)
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(),b'prior-output')
                    self.assertEqual(companion.read_bytes(),catalog)


if __name__ == "__main__":
    unittest.main()
