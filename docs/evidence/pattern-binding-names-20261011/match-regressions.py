from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path.cwd()))
from tests.test_match_block_semantics import MatchBlockSemantics
MatchBlockSemantics.compilers=lambda self:[Path.cwd()/'bin/nanoc_c',Path('/private/tmp/nl51-pattern-names/seed'),Path.cwd()/'bin/nano_virt']
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MatchBlockSemantics))
raise SystemExit(not result.wasSuccessful())
