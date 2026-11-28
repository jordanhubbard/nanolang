# I preserve each record-list declaration owner

At source `97bb958fd062f6a76db6e938a89f73b38fb0304b`, I extend my existing selected enum-list builtin binding to the actual selected record declaration. Lexical callable resolution still returns before this fallback. I retain constructor result identity for inferred lists and make repeated getter metadata checking idempotent. I change no C layout, helper ABI, capture code, runtime helper, assertion, deadline or compiler pin.

My original source5bf C seed fails the new private-record module case: its annotation emits an owner-qualified List, but its short helper name registers a second flat List with an undeclared C element type. I retain this red compiler, complete diagnostic and generated C. The first repair passes normal native cases, then Linux LSan reports getter-name leaks in two cases. I preserve that candidate, images, inputs and failure streams. The final repair retains or replaces the checked getter name without losing its allocation.

My six controls cover private/public module records, same-spelling records with different field shapes, declared owners independent of filenames, a lexical callable named like a list helper, and refusal of cross-owner lists with prior output intact. One module uses an inferred constructor binding. Every added function has a useful shadow.

| Final observation | Result |
| --- | --- |
| Normal C seed | six controls pass, 12.103s |
| ASan/UBSan C seed, original LSan enabled | six controls pass, 12.895s |
| Production C seed | six controls pass, 11.245s |
| Actual installed Clang18 native consumers | six controls pass, 14.736s |
| Original enum-list neighbors | 21 methods pass, 26.153s |

I compile only the changed checker object and link each profile against its guarded, complete original5bf provider closure and actual traced linker command. Final profiles retain `detect_leaks=1:halt_on_error=1` and UBSan halt behavior. Each positive control executes its native product; each refusal checks the diagnostic and exact prior bytes. Clang products retain its actual version in their ELF comment section. This is focused Linux C-seed evidence, not final-source Darwin, private full-source, default ownership, bootstrap or whole-release acceptance.

Run `python3 read_archive.py`. I verify 5,926 file records, every compressed/raw chunk, parent archive bindings, original source/image/provider identities, traced link arguments, the first sanitizer failure, final Git source/test bytes, six actual case outcomes per profile, Clang native products and all21 original enum controls. My manifest SHA256 is `404856b441a461ed6e711fe01a85a865538786868951253d24be17652e396905`. Large unchanged provider bytes reuse the [fresh Linux archive](../list-source-linux-20261124/README.md).

I also preserve the command-parser and missing-document setup failures. An initial packaging manifest pointed reused chunks at their immediate parent's object directory rather than following the parent's references. I retain that manifest and the deterministic reference repair; no native evidence bytes change. The independent reader verifies every final reference.

I keep #1248's remaining paired qualification and #1229/#1247/#1116/#976 open. The separate original full source qualification still fails its ten-second supervisor deadline after1,070 completions. I preserve every old root and the NAS/TCC/migration pause.
