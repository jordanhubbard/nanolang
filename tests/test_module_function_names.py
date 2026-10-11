"""I preserve module filename bytes in native function symbols and calls."""
import json
import unittest
from tests import test_cseed_generic_functions as base


class ModuleFunctionNames(unittest.TestCase):
    run_command = base.CseedGenericFunctions.run_command
    execute = base.CseedGenericFunctions.execute

    def check_names(self, names):
        files, imports, checks = {}, [], []
        for i, name in enumerate(names):
            value = 7 + i
            filename = name + '.nano'
            files[filename] = f'''fn private_value()->int{{return {value}}}
shadow private_value {{assert (== (private_value) {value})}}
pub fn read()->int{{return (private_value)}}
shadow read {{assert (== (read) {value})}}
'''
            imports.append(f'module {json.dumps(filename, ensure_ascii=False)} as m{i}')
            checks.append(f'''assert (== (m{i}.read) {value})
    let callback{i}:fn()->int = m{i}.read
    assert (== (callback{i}) {value})''')
        imports.append(f'from {json.dumps(names[0]+".nano", ensure_ascii=False)} import read as selected')
        source = '\n'.join(imports)+'\nfn main()->int{\n'+ '\n'.join(checks)+'''
    assert (== (selected) 7)
    return 0
}
shadow main {assert (== (main) 0)}
'''
        self.execute(source, files=files)

    def test_punctuation_and_encoded_looking_names_remain_distinct(self):
        self.check_names(['a-b', 'a_b', 'a b', '__nano_hex_612d62'])

    def test_unicode_digits_and_dots_remain_distinct(self):
        self.check_names(['café', 'cafe', '9probe', 'probe.aux'])

    def test_long_common_prefixes_remain_distinct(self):
        self.check_names(['a'*120+'-x', 'a'*120+'-y', 'a'*120+'_x'])
