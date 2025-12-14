# How Stray Files `-o` and `a.out` Were Created

## The Problem

During testing, two unexpected files appeared in the project root:
- `-o` (74KB executable)
- `a.out` (93KB executable)

## Root Cause Analysis

### File 1: `-o` (Before Fix)

**What happened:**

The test script called:
```bash
./bin/nanoc tests/selfhost/test_arithmetic_ops.nano -o /tmp/test_arith
```

**Before the fix (commit 0be1904), nanoc_v05.nano parsed arguments as:**
```nano
let input: string = (get_argv 1)   /* "tests/selfhost/test_arithmetic_ops.nano" */
let output: string = (get_argv 2)  /* "-o" <- TREATED AS OUTPUT FILENAME! */
                                    /* /tmp/test_arith was ignored */
```

**Result:** The compiler tried to create an executable literally named `-o` in the current directory.

**The fix (commit 0c2ad84):**
```nano
let arg2: string = (get_argv 2)

if (str_equals arg2 "-o") {
    /* Now correctly recognizes -o as a flag */
    let output: string = (get_argv 3)  /* /tmp/test_arith */
    (compile_file input output)
} else {
    /* arg2 is the output filename */
    let output: string = arg2
    (compile_file input output)
}
```

### File 2: `a.out`

**What happened:**

When `bin/nanoc_c` (the C reference compiler) is called without an output file:

**From src/main.c line 451:**
```c
const char *output_file = "a.out";  /* Default output filename */
```

**Scenarios that create `a.out`:**

1. **Direct test run without output:**
   ```bash
   ./bin/nanoc_c tests/test_something.nano
   # Creates: a.out (shadow tests only, but still generates binary)
   ```

2. **Missing -o argument:**
   ```bash
   ./bin/nanoc tests/test_something.nano
   # Internally calls: bin/nanoc_c tests/test_something.nano
   # Creates: a.out
   ```

3. **Standard C compiler behavior:**
   - If gcc/clang don't get `-o`, they create `a.out`
   - nanoc_c follows this convention

## Verification

```bash
$ file ./a.out
./a.out: Mach-O 64-bit executable arm64

$ ./a.out
All standard library tests passed!
```

The `a.out` file is a test binary (test_stdlib_comprehensive.nano) created during test runs.

## Solution

Added `a.out` to `.gitignore` to prevent accidental commits:

```gitignore
# Default compiler output files
a.out
```

The `-o` file issue is resolved by the fix in commit 0c2ad84 which properly handles the `-o` flag.

## Timeline

1. **Before 0c2ad84**: Self-hosted compiler didn't understand `-o` flag
   - Test scripts: `nanoc input.nano -o output`
   - Compiler saw: `output = "-o"` (wrong!)
   - Created: file named `-o`

2. **After 0c2ad84**: Self-hosted compiler handles `-o` correctly
   - Test scripts: `nanoc input.nano -o output` ✅
   - Compiler sees: `flag = "-o", output = "output"` (correct!)
   - Creates: file with correct name

3. **`a.out` always existed**: Standard default output name
   - Solution: Added to `.gitignore`
   - Not a bug, just needs to be ignored

## Prevention

- ✅ Self-hosted compiler now handles `-o` flag correctly
- ✅ `a.out` added to `.gitignore`
- ✅ No more stray files in project root
