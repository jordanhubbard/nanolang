"""I check complete namespace facts and actual driver publication guards."""
import os
import shlex
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"\n'


class ServiceNamespace(unittest.TestCase):
    def test_graph_namespace_and_driver_refusals(self):
        driver = os.environ.get("NANO_SERVICE_NAMESPACE_DRIVER_MODULE")
        drivers = [[ROOT / "bin/nanoc_c"], [ROOT / "bin/nano_virt", "--emit-nvm"]]
        if driver:
            drivers.append([ROOT / "bin/nano_vm", driver, "--", "--emit-nvm"])
        cases = {
            "bare-import": ('import bare as files\n', '', '', True),
            "bare-selection": ('from bare import File as Handle\n', '', '', True),
            "annotations": ('module "one/binding.nano" as files\nfn preserve(value: files.File) -> files.File { let local: files.File = value return local }\nstruct Wrapper { values: array<files.File>, callback: fn(files.File)->files.File }\nfn factory() -> fn(files.File)->files.File { return preserve }\n', '', '', True),
            "reexported-annotations": ('module "bridge.nano" as bridge\nfn preserve(value: bridge.files.File) -> bridge.files.File { let local: bridge.files.File = value return local }\n', 'pub use "one/binding.nano" as files\n', '', True),
            "union-annotations": ('from "one/binding.nano" import File as Handle\nmodule "one/binding.nano" as files\nunion Envelope { HandleArm { value: Handle }, OpenArm { value: files.OpenResult } }\nunion Generic<Handle> { Value { payload: Handle } }\n', '', '', True),
            "borrowed-annotation": ('module "one/binding.nano" as files\nfn observe(value: &mut files.File) -> void { return }\n', '', '', True),
            "qualified": ('module "bridge.nano" as bridge\n', 'pub use "one/binding.nano" as files\n', '', True),
            "selective": ('from "one/binding.nano" import File as Handle\nmodule "one/binding.nano" as files\n', '', '', True),
            "late-collision": ('from "one/binding.nano" import File\nstruct File { value: int }\n', '', '', False),
            "own-collision": ('module "one/binding.nano" as files\n', '', 'struct File { value: int }\n', False),
            "missing-selection": ('from "one/binding.nano" import Missing\n', '', '', False),
            "repeated-alias": ('module "one/binding.nano" as files\nmodule "alias.nano" as files\n', '', '', True),
            "distinct-alias": ('module "one/binding.nano" as files\nmodule "two/binding.nano" as files\n', '', '', False),
            "distinct-types": ('module "one/binding.nano" as files\nmodule "two/binding.nano" as other\n', '', '', True),
            "formatted": ('  module\n "one/binding.nano" as files\n', '', '', True),
            "header-adjacent": ('module root from "one/binding.nano" import File\nstruct File { value: int }\n', '', '', False),
            "lambda-local": ('module "one/binding.nano" as files\nfn closure() -> fn() -> int { return fn() -> int { return 1 } }\n', '', '', True),
            "comment": ('/* module "absent.nano" as absent */\nmodule "one/binding.nano" as files\n', '', '', True),
        }
        with tempfile.TemporaryDirectory(prefix="nano-service-namespace-") as directory:
            work = Path(directory).resolve()
            for name in ("one", "two"):
                (work / name).mkdir()
                (work / name / "interface.nsi.json").write_bytes((ROOT / "tests/fixtures/nsi_file_plan.json").read_bytes())
                (work / name / "binding.nano").write_text(SERVICE)
            bare = work / "modules/bare"
            bare.mkdir(parents=True)
            (bare / "bare.nano").write_text(SERVICE)
            (bare / "interface.nsi.json").write_bytes((ROOT / "tests/fixtures/nsi_file_plan.json").read_bytes())
            (work / "alias.nano").symlink_to(work / "one/binding.nano")
            for name, (root_text, bridge, extra, accepted) in cases.items():
                with self.subTest(case=name):
                    root = work / "root.nano"
                    root.write_text(root_text + 'fn main() -> int { return 0 }\nshadow main { assert true }\n')
                    (work / "bridge.nano").write_text(bridge)
                    (work / "one/binding.nano").write_text(SERVICE + extra)
                    report = subprocess.run([os.environ.get("NANO_C_SERVICE_NAMESPACE_RUNNER", str(ROOT / "obj/test_service_namespace")), str(root),
                        "accept" if accepted else "reject", "bridge.files.File", "Handle", "files.File", "other.File", "files.Missing"],
                        cwd=ROOT, text=True, capture_output=True, timeout=30)
                    self.assertEqual(report.returncode, 0, report.stdout + report.stderr)
                    if accepted:
                        self.assertIn("PLAN ", report.stdout)
                        if name == "qualified":
                            self.assertIn(f"NAME bridge.files.File 8 {work}/one/binding.nano File", report.stdout)
                        if name == "selective":
                            self.assertIn(f"NAME Handle 8 {work}/one/binding.nano File", report.stdout)
                        if name == "distinct-types":
                            self.assertIn(f"NAME other.File 8 {work}/two/binding.nano File", report.stdout)
                        self.assertIn("MISSING files.Missing", report.stdout)
                    for command in drivers:
                        with self.subTest(driver=str(command[0])):
                            output = work / "prior.nvm"
                            output.write_bytes(b"prior-output")
                            run = subprocess.run([*map(str, command), str(root), "-o", str(output)],
                                cwd=ROOT, text=True, capture_output=True, timeout=90)
                            self.assertNotEqual(run.returncode, 0)
                            expected = "I have not resolved File service declarations" if accepted else "I cannot resolve the complete File service namespace"
                            self.assertRegex(run.stdout + run.stderr, expected + (r"|I require --allow-temporary-files|I cannot lower File source" if accepted else ""))
                            self.assertEqual(output.read_bytes(), b"prior-output")

    def test_mixed_catalog_namespace_and_types(self):
        with tempfile.TemporaryDirectory(prefix="nano-mixed-namespace-") as directory:
            work=Path(directory).resolve()
            args=[work/"one/binding.nano",work/"two/binding.nano",work/"bridge.nano",work/"root.nano"]
            for path in args:
                path.parent.mkdir(exist_ok=True)
            args[0].write_text(SERVICE)
            args[1].write_text(SERVICE.replace("filesystem","net"))
            for path,fixture in zip(args[:2],("nsi_file_plan.json","nsi_socket_plan.json")):
                (path.parent/"interface.nsi.json").write_bytes((ROOT/"tests/fixtures"/fixture).read_bytes())
            args[2].write_text('pub use "two/binding.nano" as net\n')
            base='module "one/binding.nano" as files\nmodule "bridge.nano" as bridge\nfrom "two/binding.nano" import Conn as Handle, Endpoint as Address\n'
            def checked(command):
                run=subprocess.run(list(map(str,command)),cwd=ROOT,text=True,capture_output=True,timeout=240)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                return run.stdout
            for extra,accepted in (("",True),('struct Address { value: int }\n',False),('from "two/binding.nano" import Socket\n',False)):
                args[3].write_text(base+extra+'fn main() -> int { return 0 }\nshadow main { assert true }\n')
                report=checked([os.environ.get("NANO_C_SERVICE_NAMESPACE_RUNNER",str(ROOT/"obj/test_service_namespace")),args[3],
                    "accept" if accepted else "reject","Handle","Address","bridge.net.Conn","files.File","bridge.net.Socket"])
                if accepted:
                    self.assertIn("PLAN 29",report)
                    self.assertIn("MISSING bridge.net.Socket",report)
                    for compiler in ("nanoc_c","nano_virt"):
                        output=work/(compiler+".out");output.write_bytes(b"prior")
                        command=[ROOT/"bin"/compiler,args[3],"-o",output,"--allow-temporary-files"]
                        if compiler=="nano_virt":command.append("--emit-nvm")
                        run=subprocess.run(list(map(str,command)),cwd=ROOT,capture_output=True,text=True,timeout=90)
                        self.assertNotEqual(run.returncode,0)
                        self.assertIn("I have not connected TCP wire and runtime lowering",run.stdout+run.stderr)
                        self.assertEqual(output.read_bytes(),b"prior")
            for compiler in ("nano_virt","nanoc_stage2"):
                module=work/(compiler+".nvm")
                checked([ROOT/"bin"/compiler,ROOT/"tests/service_namespace_tcp.nano","--emit-nvm","-o",module])
                self.assertIn("PASS mixed File TCP namespace identities",checked([ROOT/"bin/nano_vm",module,"--",*args]))
                source=module.with_suffix(".c");native=module.with_suffix(".native")
                checked([ROOT/"bin/nvm2c",module,"-o",source])
                cc=shlex.split(os.environ.get("CC","cc"))
                checked([*cc,"-std=c11","-O1","-g","-fsanitize=address,undefined","-fno-sanitize-recover=all",source,"-o",native,"-lm",*(["-ldl"] if sys.platform.startswith("linux") else [])])
                self.assertIn("PASS mixed File TCP namespace identities",checked([native,*args]))

    def test_independent_nano_identity_and_native_catalog_bridge(self):
        with tempfile.TemporaryDirectory(prefix="nano-namespace-identity-") as directory:
            work = Path(directory).resolve()
            args = [work / "one/binding.nano", work / "two/binding.nano", work / "bridge.nano", work / "root.nano"]
            for path in args:
                path.parent.mkdir(exist_ok=True)
                path.write_text("# source identity\n")
            for path in args[:2]:
                (path.parent / "interface.nsi.json").write_bytes((ROOT / "tests/fixtures/nsi_file_plan.json").read_bytes())
            def checked(command):
                run = subprocess.run(list(map(str, command)), cwd=ROOT, text=True, capture_output=True, timeout=240)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                return run.stdout
            compilers = [[ROOT / "bin/nano_virt"]]
            driver = os.environ.get("NANO_SERVICE_NAMESPACE_DRIVER_MODULE")
            if driver:
                compilers.append([ROOT / "bin/nano_vm", driver, "--"])
            for index, compiler in enumerate(compilers):
                module = work / f"namespace-{index}.nvm"
                checked([*compiler, ROOT / "tests/service_namespace.nano", "--emit-nvm", "-o", module])
                self.assertIn("PASS independent Nano namespace identities", checked([ROOT / "bin/nano_vm", module, "--", *args]))
                source, native = module.with_suffix(".c"), module.with_suffix(".native")
                checked([ROOT / "bin/nvm2c", module, "-o", source])
                cc = shlex.split(os.environ.get("CC", "cc"))
                checked([*cc, "-std=c11", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all", source, "-o", native, "-lm", *(["-ldl"] if sys.platform.startswith("linux") else [])])
                self.assertIn("PASS independent Nano namespace identities", checked([native, *args]))
