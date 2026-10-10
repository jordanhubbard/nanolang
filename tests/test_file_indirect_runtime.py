"""I qualify private indirect carriers and preserve both older carrier corpora."""
import unittest
from tests import test_file_cyclic_runtime as carrier
from tests import test_file_cyclic as bounded


class FileIndirectRuntime(unittest.TestCase):
    fixture = "tests/nanoisa/test_file_indirect_runtime.c"
    artifact_prefix = "nano-file-indirect-runtime-"
    setUpClass = classmethod(carrier.FileCyclicRuntime.setUpClass.__func__)
    qualify = carrier.FileCyclicRuntime.qualify

    def command(self, name, args, run=False):
        stdout = bounded.FileCyclic.command(self, name, args)
        if run:
            self.assertIn(b"manual cyclic carrier/fuel checks", stdout)
            self.assertIn(b"private indirect carrier checks", stdout)
            print(stdout.decode(errors="replace").strip(), flush=True)

    def test_instrumented_indirect_and_existing_carriers(self):
        self.qualify("instrumented", True)

    def test_linked_indirect_and_existing_carriers(self):
        self.qualify("linked", False)


if __name__ == "__main__":
    unittest.main()
