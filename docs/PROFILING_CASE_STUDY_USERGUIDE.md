# Case Study: Userguide Build Performance Analysis

**Date:** January 31, 2026
**Tool:** gprofng profiling
**Status:** Analysis complete, optimization TODO

## Problem Statement

Building the user guide HTML from markdown takes **~30+ seconds** to complete.
This is significantly slower than expected for processing ~50 markdown files.

## Profiling Data

Profiled userguide_build_html.nano for 30 seconds:

```
Functions sorted by metric: Exclusive Total CPU Time

Excl. Total    Incl. Total     Name
CPU            CPU
  sec.      %    sec.      %
29.741 100.00  29.741 100.00   <Total>
27.869  93.71  29.351  98.69   gc_is_managed
 1.481   4.98   1.481   4.98   gc_header_to_ptr
 0.230   0.77   0.230   0.77   <static>@0x194500 (<libc.so.6>)
 0.040   0.13   0.270   0.91   nl_split_lines
 0.020   0.07   0.120   0.40   char_at
 0.010   0.03   0.020   0.07   nano_highlight__escape_html
 0.010   0.03   0.030   0.10   sb_alloc
 0.      0.    29.361  98.72   gc_alloc
 0.      0.    29.361  98.72   gc_alloc_string
 0.      0.    29.351  98.69   gc_collect_cycles
 0.      0.    29.351  98.69   gc_mark
 0.      0.     0.050   0.17   nano_highlight__highlight_html
 0.      0.     0.020   0.07   nano_highlight__tokenize
 0.      0.     0.050   0.17   nano_tools__pretty_print_html
 0.      0.     0.070   0.24   nl_append_examples_body
 0.      0.     0.150   0.50   nl_extract_summary
 0.      0.     0.110   0.37   nl_extract_title
```

## Critical Finding

**94% of CPU time is spent in garbage collection!**

Specifically:
- `gc_is_managed`: 93.71% (checking if pointers are GC-managed)
- `gc_header_to_ptr`: 4.98% (pointer arithmetic for GC headers)
- Actual work (syntax highlighting, parsing, etc.): < 2%

## Root Cause Analysis

The userguide build performs:
1. Read ~50 markdown files from disk
2. Parse markdown → extract metadata (title, summary)
3. Generate HTML for each page
4. **Syntax highlighting for code blocks** (creates many strings)
5. Build navigation sidebar
6. Write HTML files

Each string operation (substring, concatenation, formatting) allocates
new strings, triggering GC overhead. The GC runs frequently to check
if objects can be collected, spending 94% of time on bookkeeping.

## Why GC Dominates

### String-Heavy Operations

```nano
# Markdown processing creates many temporary strings
fn extract_title(md: string) -> string {
    # Searches for "# Title" patterns
    # Creates substrings for each line
    # Pattern matching on each line
    # Returns extracted title
}

# Syntax highlighting is particularly expensive
fn pretty_print_html(code: string, lang: string) -> string {
    # Tokenizes entire code string
    # Creates HTML span for each token
    # Concatenates all spans
    # Escapes HTML entities
    # Returns highlighted HTML
}
```

Every markdown file goes through:
- Line-by-line processing (creates N substrings)
- Token-by-token highlighting (creates 2N strings: tokens + HTML)
- HTML template building (more concatenation)

For 50 files with avg 500 lines + 100 code tokens each:
- 50 × 500 = 25,000 line substrings
- 50 × 100 × 2 = 10,000 syntax highlight strings
- **~35,000 string allocations**

Every allocation triggers `gc_is_managed` checks!

## Optimization Strategies

### 1. **Batch Processing with StringBuilder** (Easy, 2-3x improvement)

**Current:**
```nano
let mut result: string = ""
for line in lines {
    set result (+ result (process_line line))  # Allocates new string each time
}
```

**Optimized:**
```nano
let sb: StringBuilder = sb_new()
for line in lines {
    sb_append(sb, (process_line line))  # Reuses buffer
}
let result: string = sb_to_string(sb)
```

**Expected gain:** 2-3x faster (already partially done)

### 2. **Reduce GC Frequency** (Medium, 3-5x improvement)

Modify GC to collect less aggressively during batch operations:

```c
// Add GC mode for batch processing
void gc_set_mode(enum GCMode mode) {
    if (mode == GC_MODE_BATCH) {
        gc_collect_threshold *= 10;  // Collect 10x less frequently
    }
}
```

**Expected gain:** 3-5x faster during build

### 3. **Arena Allocation for Temporary Strings** (Hard, 10x improvement)

Use arena allocator for short-lived strings:

```nano
# Hypothetical future syntax
fn process_file(path: string) -> string with arena {
    # All string allocations in this function use arena
    # Arena freed when function returns
    let lines: [string] = read_lines(path)
    let html: string = md_to_html(lines)
    return html  # Only return value escapes arena
}
```

**Expected gain:** 10x faster (no GC overhead for temp strings)

### 4. **Compile-time String Interning** (Future)

Intern common strings (HTML tags, CSS classes) at compile time:

```nano
const HTML_START: string = intern("<div class=\"content\">")
const HTML_END: string = intern("</div>")
```

**Expected gain:** 20% reduction in allocations

## Immediate Action Items

### Short-term (This Week)

1. ✅ **Profile to identify bottleneck** (DONE)
2. ☐ **Add StringBuilder to all string concatenation loops**
3. ☐ **Profile again to measure improvement**

### Medium-term (This Month)

4. ☐ **Add GC batch mode** (C runtime change)
5. ☐ **Optimize syntax highlighter to use StringBuilder internally**
6. ☐ **Profile and document 3-5x improvement**

### Long-term (Future)

7. ☐ **Design arena allocation API**
8. ☐ **Implement arena allocator in runtime**
9. ☐ **Add compiler support for arena annotations**

## Expected Results

| Optimization | Effort | Expected Speedup | Build Time |
|--------------|--------|------------------|------------|
| Baseline | - | 1x | ~30s |
| StringBuilder everywhere | Low | 2-3x | 10-15s |
| + GC batch mode | Medium | 3-5x | 6-10s |
| + Arena allocator | High | 10x | 3s |
| + String interning | Medium | 1.2x | 2.5s |

## Comparison to Prime Optimization

| Aspect | Prime Counter | Userguide Build |
|--------|---------------|-----------------|
| Bottleneck | 98% in is_prime | 94% in GC |
| Root cause | O(n√n) algorithm | Excessive string allocation |
| Solution type | Algorithmic | Runtime optimization |
| Difficulty | Easy (change algorithm) | Medium-Hard (optimize GC) |
| Expected gain | 5.6x | 3-10x (staged) |

## Lessons Learned

### 1. **Different Bottlenecks Require Different Solutions**

- **Prime counter:** Change algorithm (developer fix)
- **Userguide build:** Optimize runtime (language/compiler fix)

### 2. **Profiling Reveals Non-Obvious Bottlenecks**

Without profiling, you might think:
- ❌ "Markdown parsing is slow"
- ❌ "Syntax highlighting is complex"
- ❌ "File I/O is the bottleneck"

Profiling shows:
- ✅ **Garbage collection is the real problem**

### 3. **String Operations in Loops Are Expensive**

Every `+` operator on strings:
1. Allocates new string
2. Copies both operands
3. Triggers GC check (93.71% of time!)

**Rule:** Use StringBuilder for any string building in loops.

### 4. **GC Overhead Scales with Allocations**

More allocations → more GC overhead (non-linear!)

For batch operations (processing 50 files):
- Small files (100 strings each): Manageable
- Large files (1000 strings each): GC dominates

**Solution:** Reduce GC frequency during batch work.

## Next Steps

1. **Implement StringBuilder optimization** (scripts/userguide_build_html.nano)
2. **Profile again** to confirm 2-3x improvement
3. **Add GC batch mode** (src/runtime/gc.c)
4. **Document final results** in this case study

---

**See also:**
- [PROFILING_CASE_STUDY_PRIMES.md](PROFILING_CASE_STUDY_PRIMES.md) - Algorithm optimization example
- [userguide/08_profiling.md](../userguide/08_profiling.md) - Profiling guide
- [PERFORMANCE.md](PERFORMANCE.md) - General performance guidelines

**Status:** Analysis complete. Ready to implement StringBuilder optimizations.
