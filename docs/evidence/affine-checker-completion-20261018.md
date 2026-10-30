# My bounded affine checker-completion evidence

I implement the bounded completion child of #998 on `fix/5.1-affine-facts`,
starting from `cbfa8c872`. I retain the actual resource-flow checker result,
not a syntactic guess. My UNKNOWN/COMPLETED advisory record does not supply
individual move/borrow/path facts or runtime authority. Full #998 remains open.

My C codegen gate passes 92 cases, including parsed UNKNOWN, actual successful
checking, external UNKNOWN, prior-error reset and repeated successful checking
(`/private/tmp/nano-998-affine-c-retention.log`). My final advisory codec gate
passes 1002 checks, including both assembler directives, malformed records,
reserved bytes, unknown versions, duplicate targets and canonical transport
(`/private/tmp/nano-998-affine-trailing-build.log`). My allocation gate passes
48 boundaries, preserving existing unrelated metadata across failure and
recovery (`/private/tmp/nano-998-affine-alloc-final.log`).

My changed C-hosted Nano compiler builds with full dependency/root shadows and
a bounded 300-second shadow deadline. I retain its final build terminal at
`/private/tmp/nano-998-affine-selfhost-final-build.log` and product at
`/private/tmp/nano-998-affine-selfhost`. This is not a fresh NanoISA stage.

My five final source methods pass in 75.229 seconds through the C seed and
that changed producer (`/private/tmp/nano-998-affine-final-paired2.log`). I
exercise called shared/exclusive borrows, consumption and branch joins through
four actual resource fixtures. Both producers retain UNKNOWN for two checked
generic specializations while their ordinary checked caller is COMPLETED.
No-resource checking remains UNKNOWN. Four invalid resource uses preserve the
previous output. Stripping, forging, duplicating and changing the version of
advisory facts preserve VM results and generated native C; actual native
execution uses ASan/UBSan.

My separate C-hosted parser/checker fixture executes raw UNKNOWN to checked
COMPLETED transitions and resets a deliberately stale external-declaration
fact. Its native compiler/checker execution is not sanitizer-qualified here.
I preserve two failed fixture terminals: the first combined an unsupported
external declaration with the bounded resource emitter
(`/private/tmp/nano-998-affine-checker.log`); the second passed shadows but lacked
explicit owning-module imports for native lexer/parser declarations
(`/private/tmp/nano-998-affine-final-paired.log`). I separate the first control
and add those imports without changing product admission or assertions.

My raw CLI build uses root shadows after the same changed dependencies passed
the full compiler build. Four numeric reference fixtures retain exact canonical
equality after normalizing only their independently checked raw UNKNOWN versus
checked COMPLETED payloads. I forbid executable string/host instructions and
shared non-affine payload references in this deliberately narrow comparator;
all remaining CODE, strings, layouts, ownership and metadata lines must match.
Each raw artifact also executes in my VM and sanitized native C
(`/private/tmp/nano-998-affine-raw-compare.log`).

All 15 neighboring declared purity, inferred effects and match-coverage methods
pass through both producers in 8.447 seconds
(`/private/tmp/nano-998-affine-neighbors.log`). My serial strict gate passes 222
Clang and 222 GCC compilations; the relink fixture also passes
(`/private/tmp/nano-998-affine-fast.log`, `/private/tmp/nano-998-affine-relink.log`).
Schema regeneration and diff whitespace checks pass.

I do not launch another full bootstrap in this isolated tree. Fresh Stage1/Stage2
and Linux/Darwin aggregate qualification remain pending. My new Make target is
`test-function-affine-metadata`; `NANOLANG_AFFINE_COMPILERS` selects checked
artifact producers, and `NANOLANG_AFFINE_NATIVE_COMPILER` selects the native
compiler for the parser/checker fixture (default: my C seed).
