"""I preserve explicit foreign identity across current compiler AST boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=r'''import "src_nano/compiler/nominal_bindings.nano"
fn main() -> int {
    let tokens: List<LexerToken> = (tokenize_string "extern /* retained */\nstruct Foreign { fd: int }\nstruct Local {}\nresource struct Owner { fd: int }" "foreign.nano" (list_CompilerDiagnostic_new))
    let parsed: Parser = (parse_program tokens (list_LexerToken_length tokens) "foreign.nano")
    assert (not (parser_has_error parsed))
    assert (parser_get_struct_def parsed 0).is_extern
    assert (not (parser_get_struct_def parsed 1).is_extern)
    assert (not (parser_get_struct_def parsed 2).is_extern)
    assert (parser_get_struct_def parsed 2).is_resource
    assert (== (parser_get_struct_def parsed 0).line 2)
    (mb_reset [])
    assert (nb_register parsed "__test_")
    (nb_rewrite parsed)
    assert (parser_get_struct_def parsed 0).is_extern
    let other_tokens: List<LexerToken> = (tokenize_string "struct Foreign { fd: int }\n\n\nstruct Foreign { fd: int }" "other.nano" (list_CompilerDiagnostic_new))
    let copied: Parser = (parse_program other_tokens (list_LexerToken_length other_tokens) "other.nano")
    (list_ASTStruct_set copied.structs 0 (parser_get_struct_def parsed 0))
    (mb_reset [1, 3])
    assert (not (nb_register copied "__test_"))
    (mb_reset [])
    return 0
}
shadow main { assert (== (main) 0) }
'''

class ExternRecords(unittest.TestCase):
    def test_fresh_three_stage_parser_and_nominal_boundaries(self):
        with tempfile.TemporaryDirectory(prefix='nano-extern-records-') as d:
            directory=Path(d);source=directory/'main.nano';source.write_text(SOURCE)
            for compiler in ['nanoc_c','nanoc_stage1','nanoc_stage2']:
                with self.subTest(compiler=compiler):
                    output=directory/compiler
                    build=subprocess.run([str(ROOT/'bin'/compiler),str(source),'-o',str(output)],cwd=ROOT,capture_output=True,text=True,timeout=300)
                    self.assertEqual(build.returncode,0,build.stdout+build.stderr)
                    run=subprocess.run([str(output)],cwd=ROOT,capture_output=True,text=True,timeout=30)
                    self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                    self.assertIn('colliding foreign record declarations',run.stdout)

if __name__=='__main__':unittest.main()
