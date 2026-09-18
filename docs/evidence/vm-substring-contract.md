# My checked VM substring evidence

I test the bounded prerequisite `task_ce840367841a4bdb94ab69fd2446b635`
on Linux ARM64, based on merged `b784e4d8`. My contract preceded implementation
in `3c98524d`. I clip by checked remaining length, release all three popped
operands on success and error, and report allocation failure before publishing
a string result. My LLVM/Wasm substring refusal is unchanged.

I passed the fresh `test-vm-substring-contract` target in three configurations:

- Default GCC build: `/tmp/nanolang-substring-default-final.log`.
- GCC switch dispatch with ASan/UBSan and leak detection:
  `/tmp/nanolang-substring-switch-final.log`.
- Clang computed-goto dispatch with ASan/UBSan and leak detection:
  `/tmp/nanolang-substring-clang-final.log`.

The sanitizer runs compile the actual VM and heap implementation with
`-fsanitize=address,undefined -fno-sanitize-recover=all`; the switch run also
uses `-DNANO_NO_COMPUTED_GOTO`. Clang uses the installed GCC 13 support directory.
I check six small ordinary byte slices, including embedded NUL, empty, end and
clipped ranges. My real opcode test checks deterministic allocator refusal,
unchanged caller ownership, empty transient frames/stack, same-instance recovery,
eight repeated successful calls, non-integer index fallback and type-error
operand cleanup. All checks passed after the final test addition.

I preserve historical artifacts without executing them. These are corrected
ordinary acceptance and controlled allocation tests, not a malformed-input
reproduction. I do not claim Darwin acceptance, translator substring admission,
or completion of the managed-runtime parent.
