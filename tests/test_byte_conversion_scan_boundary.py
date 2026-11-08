"""I retain complete byte conversion above my former emitted-C scan cap."""
import unittest

from tests import test_bstring_c_contracts as interpreted
from tests import test_bstring_native_contracts as native
from tests import test_bstring_source_vm as source


PROGRAM = '''fn grow(rounds:int)->string{
 let mut text:string="A"
 let mut i:int=0
 while (< i rounds){set text (str_concat text text) set i (+ i 1)}
 return text
}
shadow grow{assert (== (grow 0) "A") assert (== (grow 3) "AAAAAAAA")}
fn main()->int{
 let prefix:string=(grow 26)
 let text:string=(str_concat prefix "AAAAAAAAAAAAAAAAZ")
 let bytes:array<u8> =(bytes_from_string text)
 assert (== (array_length bytes) 67108881)
 assert (== (at bytes 0) 65)
 assert (== (at bytes 67108863) 65)
 assert (== (at bytes 67108864) 65)
 assert (== (at bytes 67108880) 90)
 return 0
}
# I execute the large production body separately; my useful helper shadow stays small.
shadow main{assert true}
'''


class ByteConversionScanBoundary(unittest.TestCase):
    command = native.BstringNativeContracts.command
    exercise = native.BstringNativeContracts.exercise
    compiler = native.BstringNativeContracts.compiler
    artifact_directory = 'byte-conversion-scan-boundary'
    scope = 'I execute above-cap conversion through actual C-created VM and sanitized AOT products.'

    def test_c_seed_native_keeps_complete_input(self):
        interpreted.BstringCContracts.exercise(self, PROGRAM)

    def test_managed_vm_and_native_keep_complete_input(self):
        self.exercise(PROGRAM)


class SourceByteConversionScanBoundary(unittest.TestCase):
    command = native.BstringNativeContracts.command
    exercise = native.BstringNativeContracts.exercise
    test_managed_vm_and_native_keep_complete_input = ByteConversionScanBoundary.test_managed_vm_and_native_keep_complete_input
    artifact_directory = 'byte-conversion-source-scan-boundary'
    scope = 'I execute above-cap conversion through my actual C-created source compiler, managed VM and sanitized AOT.'

    @classmethod
    def setUpClass(cls):
        source.BstringSourceVmContracts.setUpClass()
        cls.compiler = source.BstringSourceVmContracts.compiler

