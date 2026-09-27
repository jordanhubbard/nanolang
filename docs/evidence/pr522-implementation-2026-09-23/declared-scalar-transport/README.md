# My declared scalar artifact transport checkpoint

I retain import kind 4 through the v2 codec and module bridge. I validate exact
counted absolute paths, nonempty symbols, embedded NUL refusal and the bounded
scalar signature, including unused imports. I test heterogeneous parameter
order and exhaust all 256 tag values against the scalar parameter/result set.

On Darwin, `make test-nvm-v2-convert test-nvm-v2-imports test-nvm-v2-module`
passes. After adding malformed wire signatures, `make test-nvm-v2-convert
test-verifier` passes 913 bridge checks and 98 verifier tests. The import and
whole-module suites pass 38 and 43 checks respectively; callback and allocation
failure prerequisites also pass. These logs record ordinary builds, not a new
sanitizer qualification.

I still refuse executable kind-4 modules. Assembler/disassembler support,
matching VM/native libffi execution, string cleanup, source emission and
package linkage remain required. This checkpoint does not repair the three
native module-linking tests or complete task_b8838417bbc54fb98a4c49eea1b0885a.
