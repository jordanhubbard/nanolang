# Named lexical functions and recursive environments

I parse a named nested declaration as an immutable lexical closure binding. My
ASTFunction retains a separate `self_name` while its generated function identity
stays distinct. I check that body with its self signature and lexical captures;
I diagnose known lexical calls before declaration or outside their scope. My
selfhost emitter initializes an unnamed self-closure slot from the complete
capture environment, so ordinary, tail-position and escaping recursive calls use
my existing indirect-call path. Parameters and later locals can shadow the self
spelling. Schema clones preserve the new field, and both generators agree.

My C checker now resolves named lexical functions by source position, module
owner and enclosing scope. Parameter visibility ends with the declared body. My
C producer retains source names for closure bindings and disambiguates duplicate
wire labels; an unambiguous function name remains unchanged. My VM retains the
active closure on direct calls to the same function, including frame-reusing tail
calls. Native translation passes the current environment on such direct recursive
calls and still requires an explicit environment for other captured targets.

My shared fixtures cover ordinary and escaping closures, recursive captures,
grandparent captures, same-spelling scopes, parameter shadowing, imported module
shadowing, invalid signatures and refusal before artifact publication. Metadata
checks retain exact canonical cycles, advisory stripping and VM/native execution.
The legacy C-native emitter remains a separate open gap: my direct-native audit
emits undeclared `nl_helper`/`nl_sum` calls for these named declarations. I retain
those failures as a legacy compatibility follow-up. My 5.1 contract in
`docs/RELEASE_5.1_SCOPE.md` and `docs/NANOISA_ONLY.md` requires native products
through NanoISA and `nvm2c`; extending AST-to-C is not a release gate. I do not
close the complete producer or release contract from this bounded acceptance.

I retain the initial scope-shadow rejection failure, duplicate-label roundtrip
failure, leaked C parameter, and imported callback-signature mismatch. The first
C index gate used the old private lookup signature; the corrected unit retains
its original reference comparisons. An initial VM fixture declared zero indirect
results despite returning int; I correct that declaration without dropping the
assertions. The first ASan test executable overflowed its stack in main; the
-fno-inline build passes the same complete VM suite. I retain both logs.

My VM gate passes 274,654 checks, including explicit direct and tail recursive
closure ownership and allocation-failure recovery. The instrumented changed VM
object and test executable pass the same count with ASan/UBSan/leak checks. My
instrumented native translator also translates the retained recursive-capture
module, and its sanitized C product returns the expected result. All 2,443 native
translator checks and 90 bytecode checks pass. Environment/index/typechecker
gates pass. A default C-only functional suite skips one selfhost-specific control;
that control passes separately through the new native compiler product.

My final compiler audit after the module-owner correction passes all seven
steps with all fourteen production inputs unchanged. The resulting compiler
passes 21 source/metadata methods with sanitized native execution and 22
nominal/layout methods. The corrected C frontend independently passes the same
21 source/metadata methods; the typechecker, 60 environment-scoping checks and
ten lexical-boundary methods also pass. I retain the prior audit under
`before-module-owner/` and record final source and product hashes.
Full producer/reconstruction acceptance and exact Linux/Darwin release
qualification remain open. Legacy C-native lowering remains separately open.
