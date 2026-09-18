"""I refuse unavailable array values while preserving declared scalar names."""
import os
from pathlib import Path
import re
import unittest
from tests import test_public_c_nonfinite_format as base
ROOT=Path(__file__).resolve().parents[1]
class ArrayProfile(unittest.TestCase):
    setUp=base.PublicCFormat.setUp
    run_cmd=base.PublicCFormat.run_cmd
    compile=base.PublicCFormat.compile
    emit=base.PublicCFormat.emit
    def test_registry_array_families_have_exact_profile_boundaries(self):
        registry=(ROOT/'src/builtins_registry.c').read_text()
        names={re.match(r'\s*\{"([^"]+)"',line).group(1) for line in registry.splitlines()
               if re.match(r'\s*\{"',line) and re.search(r'\bA\b',line)}
        code=(ROOT/'src/c_backend.c').read_text().split('static bool cb_array_builtin',1)[1].split('};',1)[0]
        self.assertEqual(names,set(re.findall(r'"([a-z_]+)"',code)))
    def test_api_array_contexts_preserve_outputs_and_recover(self):
        api=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_array_profile_api.c','-o',api])
        output=self.work/'recovered.c';self.run_cmd([api,output])
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_checked_source_array_refusals(self):
        sources=[
            'fn main()->int{let x:array<int>=[] return 0}',
            'fn main()->int{let x:array<float>=[1.0,2.0] return 0}',
            'fn main()->int{return (array_length [1,2])}',
            'fn values()->array<int>{return [1,2]} shadow values{assert (== (array_length (values)) 2)} fn main()->int{return 0}',
            'fn consume(x:array<int>)->int{return (array_length x)} shadow consume{assert (== (consume [1]) 1)} fn main()->int{return 0}',
            'let values:array<int>=[1,2] fn main()->int{return 0}',
            'struct Holder{values:array<int>} fn main()->int{return 0}',
            'union Choice{Some{values:array<int>},None{}} fn main()->int{return 0}',
            'fn main()->int{let x:array<array<int>>=[[1],[2]] return 0}',
            'fn main()->int{let x:array<string>=(str_split "a,b" ",") return 0}',
        ]
        for index,source in enumerate(sources):
            path=self.work/f'ordinary{index}.nano';path.write_text(source+'\nshadow main{assert true}\n')
            output=self.work/'previous.c';output.write_text('previous')
            result=self.run_cmd([ROOT/'bin/nanoc_c','--target','c',path,'-o',output],expected=1)
            self.assertIn('array value or call ABI',result.stderr)
            self.assertEqual(output.read_text(),'previous')
    def test_declared_scalar_builtin_names_remain_ordinary(self):
        source='fn array_length(x:int)->int{return (+ x 1)} shadow array_length{assert (== (array_length 6) 7)} fn main()->int{assert (== (array_length 6) 7) return 0}\nshadow main{assert true}\n'
        output,_=self.emit(source)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
if __name__=='__main__':unittest.main()
