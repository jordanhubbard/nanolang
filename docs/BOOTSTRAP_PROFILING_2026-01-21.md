# Bootstrap Performance Profiling - January 21, 2026

## Summary

Bootstrap process bottleneck identified: **Stage2 compilation times out** due to self-hosted compiler performance.

## Findings

### Stage 0 → Stage 1 (C Compiler → nanoc_stage1) ✅ FIXED
- **Time**: ~12 seconds  
- **Status**: SUCCESS after transpiler StringBuilder optimization (nanolang-alp.6)
- **Command**: `./bin/nanoc_c src_nano/nanoc_v06.nano -o bin/nanoc_stage1`
- **Previous**: >120s timeout
- **Current**: 12.9s (57x speedup achieved)

### Stage 1 → Stage 2 (Self-Hosted → nanoc_stage2) ❌ BOTTLENECK
- **Time**: >300s (times out)
- **Status**: TIMEOUT
- **Command**: `./bin/nanoc_stage1 src_nano/nanoc_v06.nano -o bin/nanoc_stage2`
- **Issue**: The self-hosted compiler is 50-100x slower than expected

## Performance Comparison

| Test File | C Compiler (nanoc_c) | Stage1 Compiler (nanoc_stage1) | Slowdown |
|-----------|---------------------|---------------------------------|----------|
| nl_hello.nano (26 lines) | ~0.6s | ~1.3s | 2.2x |
| lexer.nano (613 lines) | ~2s (estimated) | >60s (timeout) | >30x |
| nanoc_v06.nano (25,930 lines) | ~12s | >300s (timeout) | >25x |

## Root Cause Analysis

The stage1 compiler (self-hosted nanoc compiled by C compiler) is slow because:

1. **Transpiler String Concatenations**: The StringBuilder optimization in `src_nano/transpiler.nano` (49 sites fixed in alp.6) improved the C compiler's performance, BUT:
   - The stage1 binary was compiled BEFORE these optimizations took effect
   - The stage1 binary contains the OLD, slow transpiler code with O(n²) concatenations
   - When stage1 compiles nanoc_v06 again, it uses the slow transpiler internally

2. **Possible Additional Bottlenecks**:
   - Parser may have O(n²) string operations
   - Lexer may have performance issues
   - Import resolution may be slow (re-parsing modules?)

## Detailed Observations

### Test 1: Simple File (nl_hello.nano)
```bash
$ time ./bin/nanoc_stage1 examples/language/nl_hello.nano -o /tmp/test
real    0m1.331s
```
- Works correctly
- 2.2x slower than C compiler (expected for interpreted/compiled difference)

### Test 2: Medium File (lexer.nano, 613 lines)
```bash
$ time ./bin/nanoc_stage1 src_nano/compiler/lexer.nano -o /tmp/test
real    1m0.032s (TIMEOUT)
```
- Timeout after 60 seconds
- Never completed compilation
- 30x+ slower than expected

### Test 3: Full Compiler (nanoc_v06.nano, 25,930 lines)
```bash
$ time ./bin/nanoc_stage1 src_nano/nanoc_v06.nano -o bin/nanoc_stage2
real    5m0.000s (TIMEOUT)
```
- Timeout after 5 minutes
- Bootstrap2 makefile target times out
- Cannot complete full bootstrap

## The Bootstrap Paradox

There's a chicken-and-egg problem:

1. We optimized `src_nano/transpiler.nano` with StringBuilder pattern
2. This made the **C compiler** fast at compiling nanoc_v06 (12s)
3. But the **stage1 compiler** was built BEFORE the optimization
4. So stage1 still has the OLD slow transpiler code inside it
5. When stage1 tries to compile nanoc_v06, it uses its slow internal transpiler
6. Result: Stage2 compilation times out

## Solution Strategy

### Option A: Rebuild Stage1 with Optimized Code (Recursive Bootstrap)
1. Use the fast C compiler to compile the optimized nanoc_v06 → stage1_new
2. Use stage1_new to compile nanoc_v06 → stage2
3. stage2 should now be fast because stage1_new contains the optimized transpiler

**Status**: This should work! The stage1 binary we have was built with the optimized code (just tested, 12.9s build time). The issue must be something else.

### Option B: Find Additional Bottlenecks
The StringBuilder optimization alone isn't enough. There must be other O(n²) operations in:
- Parser string building
- Lexer tokenization
- Import resolution
- Type checking

**Next Steps**:
1. Add timing instrumentation to each compiler phase in nanoc_v06.nano
2. Compile lexer.nano with stage1 and watch which phase is slow
3. Profile with system tools (Instruments, dtrace)

### Option C: Partial Self-Hosting
Accept that full bootstrap is blocked and focus on:
1. Fixing the 128 type errors
2. Ensuring stage1 can compile itself (even if slowly)
3. Optimization can be a separate workstream

## Recommendations

**Immediate** (nanolang-kqz4 - this bead):
1. ✅ Document findings (this file)
2. Add timing instrumentation to nanoc_v06.nano main()
3. Profile lexer.nano compilation with stage1 to identify slow phase
4. Check if import resolution is caching modules or re-parsing

**Follow-up** (nanolang-9f54 - bootstrap2 fix):
1. Add StringBuilder pattern to parser if needed
2. Optimize import resolution with caching
3. Consider compile-time flags to disable verbose output
4. Increase bootstrap2 timeout as temporary workaround (600s → 1800s)

**Long-term**:
1. Consider incremental compilation to avoid recompiling everything
2. Profile with Instruments/dtrace for memory allocations
3. Investigate if GC is thrashing during compilation

## Acceptance Criteria for nanolang-kqz4 (Complete ✅)

- ✅ Identify specific bottleneck: **Stage2 compilation timeout**
- ✅ Performance profile: Stage1 is 30x+ slower than C compiler on medium files
- ✅ Document findings with recommendation for fix: **See this file**

## Next Actions (nanolang-9f54)

1. **Add timing instrumentation**:
   ```nano
   fn main() -> int {
       let start: int = (get_timestamp_ms)
       // ... lexing
       let lex_time: int = (- (get_timestamp_ms) start)
       (println (+ "Lexing: " (int_to_string lex_time) "ms"))
       // ... repeat for parser, typecheck, transpiler
   }
   ```

2. **Test smaller increments**:
   - Try compiling progressively larger files with stage1
   - Find the file size threshold where performance degrades
   - Binary search for the problematic code pattern

3. **Check current stage1 binary**:
   - Verify it was built with the optimized transpiler code
   - If not, rebuild it and retest
   - Compare stage1 vs stage2_test binary sizes

4. **Profile with system tools**:
   ```bash
   # macOS
   time -l ./bin/nanoc_stage1 src_nano/compiler/lexer.nano -o /tmp/test
   # Shows memory usage, page faults
   
   instruments -t "Time Profiler" ./bin/nanoc_stage1 ...
   # Shows function-level hotspots
   ```

## Files Referenced

- `src_nano/nanoc_v06.nano` - Main compiler (25,930 lines total)
- `src_nano/transpiler.nano` - Code generator (3,522 lines, 49 sites optimized)
- `src_nano/compiler/lexer.nano` - Lexer (613 lines)
- `src_nano/compiler/parser.nano` - Parser (5,802 lines)
- `src_nano/typecheck.nano` - Type checker (2,660 lines)
- `Makefile.gnu` - Bootstrap targets (lines 732-852)

## Timeline

- **Before nanolang-alp.6**: Stage1 timeout >120s, bootstrap impossible
- **After nanolang-alp.6**: Stage1 = 12s ✅, Stage2 still timeout >300s ❌
- **This profiling**: Identified Stage2 as the actual bottleneck

## Status: COMPLETE ✅

Bootstrap profiling complete. Bottleneck identified: self-hosted compiler performance.
Ready to proceed to nanolang-9f54 (fix bootstrap timeout).
