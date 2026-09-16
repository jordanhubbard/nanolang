# My boolean array representation

I give boolean arrays a distinct compiler representation while sharing the
integer-sized native array storage. My shape graph retains a boolean element
kind. I do not treat integer and boolean arrays as interchangeable.

I preserve this distinction through empty construction, literals, append,
indexed reads and writes, locals, duplication, aggregate fields, ordinary
calls and tail returns. Reading a boolean element produces a boolean value,
not an integer with the same bits.

My tagged globals carry the array representation with the handle. Tagged
reads restore boolean tags; writes and appends require boolean values. Aliases
observe mutation through raw and tagged handles. Tagged out-of-range reads
return void, as in my existing tagged array adapter. Raw array out-of-range
behavior remains the separately tracked native-array boundary work.

My tests cover three construction forms, both caller orders, ordinary and tail
calls, aggregate fields, alias-visible mutation, global storage, exact element
tags and rejection of integer/boolean substitutions. I separately test raw
arrays without globals and tagged printing as `[true, false]`.

I run `make -j1 test-nvm2c`, `make test-nvm2c-sanitizers` and
`make -j1 test-one-ir-compiler`. Full compiler acceptance clears boolean-array
append in `typecheck_local_lets` but remains incomplete: a tail call in
`env_get_type` (311), offset 273, conflicts in the recursive shape graph.
I track that as `task_d1cdf01edf9040b2885cb31134688288`.

Normal and fresh ASan/UBSan runs pass 1,572 AOT and 994 shape checks each.
Sanitizer leak detection is disabled; these runs do not establish leak freedom.
