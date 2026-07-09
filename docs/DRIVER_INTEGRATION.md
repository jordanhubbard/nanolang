# Driver Integration Guide

## Overview

This document describes how to integrate `src_nano/driver.nano` with the existing compiler phases to create a fully self-hosted compiler driver.

## Current Status

✅ **Phase 1 Complete**: Interface documentation
- Driver skeleton created
- Command-line argument parsing
- Compilation pipeline structure
- Phase interfaces documented

🔄 **Phase 2 In Progress**: Stub integration
- Enable phase imports
- Wire up actual function calls
- Replace string-based CompilationResult with proper types

## Architecture

### Existing Compiler Phases

The modular compiler (`src_nano/compiler_modular.nano`) already has working phases:

1. **Lexer** (`lexer_main.nano`):
   ```nano
   fn lex_phase_run(source: string, filename: string) -> LexPhaseOutput
   ```
   - **Input**: Source code string, filename
   - **Output**: `{ tokens, token_count, had_error, diagnostics }`

2. **Parser** (`parser.nano`):
   ```nano
   fn parse_phase_run(lex_output: LexPhaseOutput) -> ParsePhaseOutput
   ```
   - **Input**: Lex output
   - **Output**: `{ parser, had_error, diagnostics }`

3. **TypeChecker** (`typecheck.nano`):
   ```nano
   fn typecheck_phase(parser_state: Parser) -> TypecheckPhaseOutput
   ```
   - **Input**: Parser state
   - **Output**: `{ had_error, diagnostics }`

4. **Transpiler** (`transpiler.nano`):
   ```nano
   fn transpile_phase(parser_state: Parser, c_file: string) -> TranspilePhaseOutput
   ```
   - **Input**: Parser state, output C file path
   - **Output**: `{ c_source, had_error, diagnostics }`

### Phase Output Types

All phases return structured outputs with:
- `had_error: bool` - Whether the phase failed
- `diagnostics: List<CompilerDiagnostic>` - Detailed error/warning messages

This allows uniform error handling across all phases.

## Integration Steps

### Step 1: Enable Imports (SAFE)

Uncomment the imports in `driver.nano`:

```nano
import "src_nano/compiler/ir.nano"
import "lexer_main.nano" as Lexer
import "parser.nano" as Parser  
import "typecheck.nano" as TypeCheck
import "transpiler.nano" as Transpile
```

**Safety**: These modules already exist and compile successfully in `compiler_modular.nano`.

### Step 2: Update CompilationResult Type

Replace string-based output with structured types:

```nano
union CompilationPhaseResult {
    LexResult { output: LexPhaseOutput },
    ParseResult { output: ParsePhaseOutput },
    TypeCheckResult { output: TypecheckPhaseOutput },
    TranspileResult { output: TranspilePhaseOutput },
    CCompileResult { binary_path: string },
    Error { message: string, phase: int }
}
```

### Step 3: Wire Up Lexer

```nano
fn run_lexer(input_file: string, verbose: bool) -> CompilationPhaseResult {
    if verbose {
        (println "Phase 1: Lexing...")
    }
    
    /* Read source file */
    let source: string = (file_read input_file)
    
    /* Run lexer */
    let lex_output: LexPhaseOutput = (Lexer.lex_phase_run source input_file)
    
    if lex_output.had_error {
        /* Format diagnostics using error_messages.nano */
        let error_msg: string = (format_diagnostics lex_output.diagnostics)
        return CompilationPhaseResult.Error { 
            message: error_msg, 
            phase: CompilationPhase.PHASE_LEX 
        }
    }
    
    if verbose {
        (print "  Tokens: ")
        (println (int_to_string lex_output.token_count))
    }
    
    return CompilationPhaseResult.LexResult { output: lex_output }
}
```

### Step 4: Wire Up Parser

```nano
fn run_parser(lex_output: LexPhaseOutput, verbose: bool) -> CompilationPhaseResult {
    if verbose {
        (println "Phase 2: Parsing...")
    }
    
    let parse_output: ParsePhaseOutput = (Parser.parse_phase_run lex_output)
    
    if parse_output.had_error {
        let error_msg: string = (format_diagnostics parse_output.diagnostics)
        return CompilationPhaseResult.Error {
            message: error_msg,
            phase: CompilationPhase.PHASE_PARSE
        }
    }
    
    if verbose {
        (print "  Functions: ")
        (println (int_to_string parse_output.parser.functions_count))
    }
    
    return CompilationPhaseResult.ParseResult { output: parse_output }
}
```

### Step 5: Wire Up TypeChecker

```nano
fn run_typechecker(parser_state: Parser, verbose: bool) -> CompilationPhaseResult {
    if verbose {
        (println "Phase 3: Type checking...")
    }
    
    let type_output: TypecheckPhaseOutput = (TypeCheck.typecheck_phase parser_state)
    
    if type_output.had_error {
        let error_msg: string = (format_diagnostics type_output.diagnostics)
        return CompilationPhaseResult.Error {
            message: error_msg,
            phase: CompilationPhase.PHASE_TYPECHECK
        }
    }
    
    if verbose {
        (println "  Type check complete")
    }
    
    return CompilationPhaseResult.TypeCheckResult { output: type_output }
}
```

### Step 6: Wire Up Transpiler

```nano
fn run_transpiler(parser_state: Parser, c_file: string, verbose: bool) -> CompilationPhaseResult {
    if verbose {
        (println "Phase 4: Transpiling to C...")
    }
    
    let transpile_output: TranspilePhaseOutput = (Transpile.transpile_phase parser_state c_file)
    
    if transpile_output.had_error {
        let error_msg: string = (format_diagnostics transpile_output.diagnostics)
        return CompilationPhaseResult.Error {
            message: error_msg,
            phase: CompilationPhase.PHASE_TRANSPILE
        }
    }
    
    if verbose {
        (print "  Generated C file: ")
        (println c_file)
    }
    
    return CompilationPhaseResult.TranspileResult { output: transpile_output }
}
```

### Step 7: Wire Up C Compiler

```nano
fn run_cc(c_file: string, output_file: string, verbose: bool) -> CompilationPhaseResult {
    if verbose {
        (println "Phase 5: Compiling C code...")
    }
    
    /* Build cc command */
    let runtime_files: string = "src/runtime/list_int.c src/runtime/list_string.c src/runtime/list_token.c src/runtime/token_helpers.c src/runtime/gc.c src/runtime/dyn_array.c src/runtime/gc_struct.c src/runtime/nl_string.c src/runtime/cli.c src/runtime/schema_lists.c"
    
    let cc_cmd: string = (str_concat "cc -o " output_file)
    set cc_cmd (str_concat cc_cmd " ")
    set cc_cmd (str_concat cc_cmd c_file)
    set cc_cmd (str_concat cc_cmd " -Isrc ")
    set cc_cmd (str_concat cc_cmd runtime_files)
    
    if verbose {
        (println "  Command:")
        (println cc_cmd)
    }
    
    /* Execute cc command via std::process */
    let cc_output: Output = (Process.run cc_cmd)
    
    if (!= cc_output.code 0) {
        let error_msg: string = (str_concat "C compilation failed:\n" cc_output.stderr)
        return CompilationPhaseResult.Error {
            message: error_msg,
            phase: CompilationPhase.PHASE_CC
        }
    }
    
    if verbose {
        (println "  Binary created:")
        (println output_file)
    }
    
    return CompilationPhaseResult.CCompileResult { binary_path: output_file }
}
```

### Step 8: Update compile() Pipeline

```nano
fn compile(args: CompilerArgs) -> int {
    if args.verbose {
        (println "=== NanoLang Self-Hosted Compiler ===")
        (println args.input_file)
    }
    
    /* Phase 1: Lex */
    let lex_result: CompilationPhaseResult = (run_lexer args.input_file args.verbose)
    match lex_result {
        Error(e) => {
            (println e.message)
            return 1
        }
        LexResult(lex) => {
            /* Phase 2: Parse */
            let parse_result: CompilationPhaseResult = (run_parser lex.output args.verbose)
            match parse_result {
                Error(e) => {
                    (println e.message)
                    return 1
                }
                ParseResult(parse) => {
                    /* Phase 3: Type check */
                    let type_result: CompilationPhaseResult = (run_typechecker parse.output.parser args.verbose)
                    match type_result {
                        Error(e) => {
                            (println e.message)
                            return 1
                        }
                        TypeCheckResult(type_out) => {
                            /* Phase 4: Transpile */
                            let c_file: string = (str_concat args.output_file ".c")
                            let transpile_result: CompilationPhaseResult = (run_transpiler parse.output.parser c_file args.verbose)
                            match transpile_result {
                                Error(e) => {
                                    (println e.message)
                                    return 1
                                }
                                TranspileResult(transpile) => {
                                    /* Phase 5: Compile C */
                                    let cc_result: CompilationPhaseResult = (run_cc c_file args.output_file args.verbose)
                                    match cc_result {
                                        Error(e) => {
                                            (println e.message)
                                            return 1
                                        }
                                        CCompileResult(cc) => {
                                            if args.verbose {
                                                (println "")
                                                (println "✅ Compilation successful!")
                                            }
                                            return 0
                                        }
                                        _ => { return 1 }
                                    }
                                }
                                _ => { return 1 }
                            }
                        }
                        _ => { return 1 }
                    }
                }
                _ => { return 1 }
            }
        }
        _ => { return 1 }
    }
}
```

## Safety Considerations

### Why This Is Safe

1. **No changes to existing compiler phases** - We're only calling existing, tested functions
2. **Isolated development** - `driver.nano` is a new file, not used by CI/CD yet
3. **Incremental integration** - Can test each phase independently
4. **Fallback available** - `compiler_modular.nano` remains as reference
5. **Optional adoption** - Won't affect Stage 1/2/3 until we explicitly switch

### Testing Strategy

1. **Phase-by-phase testing**:
   ```bash
   # Test lexer only
   ./bin/nanoc src_nano/driver.nano -o bin/driver_test
   
   # Test with simple programs
   ./bin/driver_test tests/test_hello.nano -o bin/hello_test
   ```

2. **Parallel validation**:
   ```bash
   # Compare with existing compiler
   ./bin/nanoc test.nano -o bin/test_old
   ./bin/driver_test test.nano -o bin/test_new
   diff bin/test_old bin/test_new
   ```

3. **CI/CD validation**:
   - All existing tests must pass
   - No regressions in Stage 1/2/3 builds
   - Driver tests run separately

## Migration Path

### Phase 3: Full Integration (Future)

Once Phase 2 is complete and tested:

1. Create `bin/nanoc_self_hosted` using the new driver
2. Add make target: `make self-hosted`
3. Run full test suite with self-hosted compiler
4. Compare performance/correctness with C compiler

### Phase 4: Replace C Compiler (nanolang-alp.11)

When ready for full self-hosting:

1. Update `Makefile` to use driver for Stage 2
2. Add `verify-no-nanoc_c` target
3. Update CI/CD to enforce self-hosting
4. Archive C compiler as `bin/nanoc_bootstrap`

## Next Steps

1. ✅ Create integration document (this file)
2. 🔄 Implement Step 1-2 (imports + types)
3. ⏳ Implement Step 3-6 (phase integration)
4. ⏳ Implement Step 7 (C compiler invocation)
5. ⏳ Implement Step 8 (pipeline orchestration)
6. ⏳ Testing and validation
7. ⏳ Migration to self-hosted build

## References

- `src_nano/compiler_modular.nano` - Reference implementation
- `src_nano/driver.nano` - Current driver skeleton
- `src_nano/compiler/error_messages.nano` - Error formatting
- `docs/AFFINE_TYPES_DESIGN.md` - Resource tracking integration
- `.factory/PROJECT_RULES.md` - Dual implementation requirements

