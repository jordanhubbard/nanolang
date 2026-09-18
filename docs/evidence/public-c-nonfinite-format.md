# My public C signed formatting evidence

I tested reviewed production4cd7a635 plus unreachable-branch cleanup6be7fd4c
with frozen final byte-observer harnessf9c36dda. My [manifest](public-c-nonfinite-format.json)
records eleven verified source/harness/tool hashes and ten retained logs. My
[contract](../PUBLIC_C_NONFINITE_FORMAT_CONTRACT.md) precedes implementation under
child132ebac6 of signed-formatting parente92.

I consume unchanged shared NL_BINARY64_FORMAT_SOURCE under the backend's selected
private namespace. Exact unbound float_to_string creates stable snapshots; direct
print/println selects nonfinite text without allocating. I preserve existing finite
%g output, input bits, once-only operands and declaration/lexical/callee precedence.

My snapshot list retains owned text until generated-process exit. I register cleanup
once, only record registration after success, free an unlinked node on registration
failure, and release every linked node at exit. This preserves aliases across globals,
returns and repeated calls. It is not early reclamation or a memory/RSS bound.

| Frozen check | Result |
| --- | --- |
| Formatting GCC | 4 methods, 1.330 seconds |
| Formatting Clang | 4 methods, 1.788 seconds |
| Adjacent scalar GCC | 8 methods, 4.019 seconds |
| Adjacent scalar Clang | 8 methods, 4.540 seconds |
| Existing C backend programs | 7 pass, no skips |

My actual generated C runs under C99/C11, O0/O2, ASan/UBSan and default Linux leak
checking. Thirteen values distinguish both quiet/signaling NaN signs, infinities,
signed zeros and finite endpoints. Exact stdout observes conversion and printing;
integer observers retain original bits and count effects. Independent C strcmp
checks snapshot identity/content. Three hundred repeated conversions retain earlier
aliases, and allocation/atexit failure controls check exact cleanup counts and
controlled nonzero diagnostics. API controls preserve unknown refusal, prior output,
later recovery and scoped names. No shared header changed and no bootstrap is claimed
for this C-only emitter slice.

I retain the first parser refusals from invalid fixture set= syntax and the later
source assertion failures where STRING EQ emitted C pointer equality. The latter is
separate task4c6d924e0aa242df950f662f61981664: generated source establishes that
boundary, not a formatting-byte failure. I did not replay the failed binaries.
The final scoped fixture uses independent stdout/strcmp rather than claiming source
string equality is fixed. Both historical source fixtures and logs remain retained.

Full public C string equality, other GNU string/block paths, concat ownership and
portability6ade remain required independent work. Parent e92 retains Darwin and
all-route acceptance; these Linux observations do not close it. Scalar policy5009,
canonical callback d099 and full release also remain open.
