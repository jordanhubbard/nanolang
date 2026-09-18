from pathlib import Path
from tests.test_source_borrow_emission import SourceBorrowEmission, ROOT
case=SourceBorrowEmission()
case.work=Path('/tmp/nanolang-temporary-owner-828edaac/smoke-artifacts')
case.work.mkdir()
source=case.work/'ordered.nano'
source.write_text(case.temporary_owner_fixture())
baseline=None
for compiler in ('nano_virt','nanoc_stage1','nanoc_stage2'):
 module=case.work/(compiler+'.nvm')
 case.command(ROOT/'bin'/compiler,source,'--emit-nvm','-o',module)
 actual=case.command(ROOT/'bin/nanoisa','dump',module).stdout
 if baseline is None: baseline=actual
 case.assertEqual(actual,baseline)
 case.execute_pair(module,expected_output=b'ABCABCABC')
 print(compiler, 'exact canonical module, mandatory shadows and VM/native order PASS',flush=True)
