"""I execute actual source-created binary-string modules through native translation."""
from tests import test_bstring_native_contracts as native
from tests import test_bstring_source_vm as source


class BstringSourceNativeContracts(native.BstringNativeContracts):
    artifact_directory = 'bstring-source-native-contracts'
    scope = 'I execute modules emitted by the actual C-created source compiler in the VM and translated sanitized native programs; fresh/integrated stages remain separate.'

    @classmethod
    def setUpClass(cls):
        source.BstringSourceVmContracts.setUpClass()
        cls.compiler = source.BstringSourceVmContracts.compiler
