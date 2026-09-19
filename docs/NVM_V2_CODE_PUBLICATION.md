# I publish only complete converted CODE

I track task_d34bb03dee3b404508faf8006b1fceef, delegated from private File hosted
qualification. I preserve that lane's exact CODE containment and qualified trees.
Static review found that nvm_v2_to_nvm_module ignores nvm_append_code failure:
the helper returns zero for both a successful first append and failed growth.
A later successful allocation does not establish that CODE was copied. I do not
execute an old faulty module or reproduce an exploitation path.

## I retain the existing conversion boundary

The inverse bridge validates nonnull module/output arguments before clearing
*out to NULL, checks service metadata first, allocates a private module, and
publishes its pointer only at final success. I keep that contract: this API does
not preserve an arbitrary preexisting output sentinel on ordinary failure. I
preserve existing first-error order, unrelated validation and all cleanup. Legacy
allocation failures remain NVM_V2_ERR_TRUNCATED; I add no status enum or append ABI.

At the existing CODE stage I require a nonnull source for nonempty CODE and a
size representable in uint32 before allocation or copying. I reserve the exact
requested size only when it exceeds the new module's current capacity, assigning
realloc to a temporary pointer. Failure frees the private module through its
existing destructor with the original allocation intact and returns TRUNCATED
immediately. Success updates pointer/capacity, copies all bytes, then commits
code_size. Empty CODE remains valid at this layer and reaches unchanged later
validation. Successful CODE content/length are exact, including embedded zeros.

This converter-local reserve avoids relying on the append helper's ambiguous
zero result or doubling arithmetic. I leave nvm_append_code and its other callers
unchanged; I make no claim to have audited or repaired every append caller.
I do not change ownership/service/profile admission, max-stack computation,
serialization or function metadata. Public outputs remain unpublished on error.

## I test corrected conversion without execution

A dedicated C fixture constructs ordinary v2 modules with patterned CODE below,
at and above the initial4096-byte capacity. Where structural validity matters,
it uses existing ordinary construction/serialization helpers and valid bounded
instructions; no converted CODE is executed. Success checks exact bytes/size,
separate owned storage and lifetime after freeing the input. Empty CODE and
nonempty-null/over-u32 refusal controls do not dereference missing storage.

Fixture-only allocator wrappers instrument converter and format translation
units. They identify the private module's initial CODE allocation and fail only
its actual converter realloc request above capacity. The failure is transient:
subsequent allocations would succeed, so a checked immediate refusal cannot be
mistaken for a persistent allocator shutdown. I require exactly one site hit,
TRUNCATED, NULL output, unchanged input bytes/pointers and zero tracked temporary
allocations. A fresh conversion in the same process then succeeds and preserves
exact CODE. Allocation-prefix controls, where added, retain their actual status
attribution rather than permitting arbitrary errors. Invalid arguments retain
the existing pre-clear behavior. I do not execute refused or partially converted
modules, mutate a caller-owned source or use historical failed artifacts.

I run the focused conversion fixture under strict GCC/Clang and scoped sanitizers
with exact instrumentation attribution, then existing v2 conversion and relevant
owned/service transport neighbors. Providers are prepared once and inventories
remain stable around phases. Any fixture/setup failure is retained before a
reviewed correction. Full fixture/source diff is independently reviewed before
new builds or gates. This closes only shared CODE publication taskd34; private
hosted File/runtime/source acceptance remains with its existing owner.

## My corrective checkpoint and exact nonexecuting probe

The production delta replaces only inverse conversion's CODE append call with
checked exact reserve/copy. A new C fixture uses CODE-only ordinary v2 containers,
no functions/entry or service metadata; these are representation tests, not a
claim that arbitrary patterned bytes are executable. Sizes0/4095/4096/4097/8193
cover empty, existing storage and growth, exact bytes including zeros and input
lifetime independence. Existing converter neighbors retain actual function/profile
coverage separately.

The instrumented format allocator records the initial calloc(4096,1) CODE pointer.
The converter realloc observer requires that exact pointer and request8193,
returns NULL once, disarms immediately and checks no later allocator attempt before
TRUNCATED/NULL return. Every instrumented allocation has a tracked pointer;
destructor frees must match, leaving zero live blocks. A fresh same-process
conversion succeeds with exact bytes. Missing CODE, over-u32 size and invalid
arguments preserve their existing output rules; a bad earlier constant retains
SECTION_TYPE before a simultaneous CODE-stage error. Input bytes and module
fields remain unchanged. No returned module is executed and no old implementation
is run. Only converter/format translation units receive allocator instrumentation;
linked ordinary validators are not claimed as universally fault-instrumented.
