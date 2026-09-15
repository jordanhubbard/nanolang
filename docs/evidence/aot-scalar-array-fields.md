# Integer-array and string-array record fields

I preserve integer-array and string-array kinds in aggregate field facts and
generated storage. Packing stores the matching array pointer; extraction
checks the field's runtime kind and produces the corresponding typed array
temporary. Function argument/result facts and local record copies retain
these distinctions. This uses the existing array representation; it does not
introduce deep copying, new ownership guarantees, or unbounded array storage.

I still reject nested record fields and record-array fields. A record-array
field needs its element record's shape carried through the containing record;
remembering only the outer array kind would lose that information. This
remaining work is necessary for the compiler's parser state.

## Verification

`make -j1 test-nvm2c` passes 1,043 checks on Darwin. New executables exercise
empty and populated arrays of both supported element kinds through branches,
record construction, a second function call, local copies and extraction.
They check array lengths and element values. Negative cases reject conflicting
array representations for one function parameter and unsupported nested
aggregate fields. I updated the pre-existing nested-record test to expect the
more precise diagnostic; it still requires rejection. `git diff --check`
passes.

`make -j1 test-one-ir-compiler` remains failing at function 20, now explicitly
reporting `AGG_PACK field requires unsupported nested aggregate shape facts`.
This step supports scalar-element array fields, not arbitrary recursive
aggregate shapes or full compiler execution. MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71` remains open.
