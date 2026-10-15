# Legacy binding source visibility and discarded native bodies

I previously retained legacy group symbols without definition locations. Later
passes treated them as visible before their initializers, so a later
`array_length` local hid an earlier builtin call. My prior [probe](../legacy-par-let-20261011/sibling-builtin.json)
printed C type errors but returned success. I now record the group location and
start lexical visibility at its body, after all independent initializers.
Existing lexical scope bounds prevent a retained local from affecting other
functions. I retain rejection when an integer binding is actually visible at a
call, with unchanged prior output.

I also explicitly discard the legacy body result in native C with `(void)(...)`.
Bare arithmetic, literal and compound values previously failed strict unused-value
warnings. The cast preserves body effects and accepts a void-valued call.

My shared tests cover same-line and multiline groups, an earlier function with
a later retained local, arithmetic, float, string, array, tuple and record bodies,
and a void call with an observed side effect. I execute all cases through C-native,
C-NanoISA and the retained selfhosted native product from the previous legacy
compiler audit. I preserve the independent/malformed/immutable/scope rejection
controls. The selfhosted compiler source is unchanged in this change.

Typechecker, environment scoping/index, transpiler and bytecode component gates
pass, including 60 scoping checks, ten lexical-boundary methods, two assertion
methods and 90 bytecode checks. My first command used a nonexistent `test-env`
target after passing the typechecker gate; I retain that log and the corrected
component command's result. All sixteen combined metadata/legacy methods pass with Clang ASan/UBSan and leak checks enabled for native products.

Named nested functions, the broader producer/reconstruction contract and final
exact Linux/Darwin release qualification remain open.
