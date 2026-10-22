# I retain nonzero subnormal assembler operands

The unchanged FLOAT source parser and C-hosted shadow emitter accept the retained decimal token. Actual libc strtod consumes all of it and returns bits0000000000000001 with ERANGE; my assembler previously rejected every errno. I now accept ERANGE only when fpclassify is FP_SUBNORMAL. Zero-underflow, overflow, malformed operands and other errors retain refusal; raw bits syntax is unchanged.

I compiled the changed assembler in isolation. Initial linkage exposed old retained nvm_format objects missing the three newer advisory codec functions; I retained the linker failure and compiled current nvm_format with matching flags. No installed object or library was overwritten. Existing NanoISA tests pass3018/3018. Focused tests cover exact signed minimum/maximum subnormal, normal extremes, signedzero, rawNaN payload and error cases. Clang and GCC16 strict syntax passed for assembler, facts and the focused test.

An unintended draft text replacement briefly touched parse_int64; I restored it byte-for-byte and reran the assembler and3018-neighbor tests before handoff. Integer overflow refusal controls remain explicit in the final focused test.

Installed Stage1/Stage2 link by an absolute dlopen path to their original host library and still reproduce the subnormal refusal. Separate generated-C copies use only a replaced private library literal, checked by reverse substitution; native rebuild evidence belongs to the FLOAT component. This is not a fresh bootstrap or Linux qualification.
