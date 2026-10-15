# My isolated enum candidate

I prepare candidate.patch outside my frozen bootstrap source tree. I compile
separate instrumented parser, environment, checker and codegen objects and link
a separate NanoVirt executable; I do not rebuild native NanoLang modules or
replace bootstrap tools. Bootstrap source/tool/pinned-host hashes remain unchanged.

My first local enum candidate passes literals, returns, locals, record fields
and distinct-declaration refusal but fails an enum-array identity check. Declared
elements retain legacy struct annotations while literals carry enum annotations.
I normalize these to the declared enum identity before exact binding, and retain
owned binding names while temporary normalized trees are freed. All 24 methods
then pass under compiler-component ASan/UBSan, including the 21 existing generic
controls and three enum cases; generated programs run NanoVM and sanitized C.

Cross-module cases expose unfinished work. The module checker treats enum
function results as structs. My candidate corrects that return context, then
fails because separate modules cannot both declare Shade. Its complete expanded
run remains 24 passed, two failed. The same imported cases also fail through the
fresh self-hosted compiler: lowering lacks the concrete enum identity, and the
conflicting-owner case reaches unsupported local Shade instead of an identity
refusal. These are not successful module-identity qualifications.

I retain the candidate, tests and failed logs for integration after fixing module
ownership; I have not applied this patch to production sources. The full enum,
tuple and generic requirements remain open under #976.
