# My optional record projection checkpoint

I retain the original nested generic record-array acceptance source unchanged
in `original.nano`. At base commit `529c42073`, my C bytecode producer executes
its mandatory shadows and my VM executes the resulting module. The original
native translator refuses its final field read with an optional/record shape
conflict. `nvm2c.log.gz` retains that diagnostic.

I separate the optional carrier from its present record shape. A present read
copies a plain record into a tagged snapshot; absent reads retain VOID. Checked
field consumers reject missing values, wrong tags, null pointers and non-record
aggregates. My root tracer retains snapshot children through replacement,
array growth and collection. Allocation failure leaves the source array intact.
My existing native representation does not preserve record identity; tagged
record comparisons remain refused before publication and retain prior output.

I build the changed translator object and executable under
`/tmp/pr522-optional-record`, using the normal Make compile/link recipes with
only their output paths and translator object path replaced. Other linked
objects remain the existing ordinary build. This keeps the concurrently
running scalar-global regression's `bin/nvm2c` unchanged.

My focused command is:

```sh
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
NANO_OPTIONAL_RECORD_TRANSLATOR=/tmp/pr522-optional-record/nvm2c \
python3 -m unittest -v tests.test_native_optional_records
```

Eight methods pass. Generated C uses C11, warnings as errors, ASan/UBSan,
nonrecovering sanitizer errors and `detect_leaks=1`. They cover the original
source, five present/absent indices, missing-field traps, generated roots across
mutation/growth, forced collection with owned children, snapshot allocation
failure, wrong runtime tags and six comparison refusals. Six unchanged scalar
optional-array methods also pass with the same isolated translator and native
sanitizer settings.

I separately rebuild only the translator translation unit with Homebrew LLVM,
ASan/UBSan and `-O1`, and link it to ordinary dependencies. The same eight
methods pass with leak detection. This is scoped translator instrumentation,
not a claim that its dependencies were instrumented.

I retain the initial full-regression failure: the new comparison check queried
missing shape IDs, producing 54 failures. I add the missing-ID guards before
rerunning that gate. The final terminal and source hashes identify the corrected
checkpoint. The corrected full translator regression passes 2,428 assertions.
The shape suite passes 1,500 checks; opcode-coverage/sanitizer-driver tests pass
four methods.

I also retain the separate scalar-global broad run's failed terminal. All 43
assertion failures report that Apple's sanitizer runtime does not support
`detect_leaks=1`; Make stopped before the scalar-union target. The corrected run
selects Homebrew LLVM through `NANO_NATIVE_TEST_CC`, keeps leak detection, and
runs both original unittest modules. It remains separate from this native
repair's qualification.

This checkpoint does not restore the self-hosted frontend's record-array union
admission, the other retained compatibility cases, or full release readiness.
