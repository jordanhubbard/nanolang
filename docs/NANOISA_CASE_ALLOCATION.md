# My case-conversion allocation prerequisite

I record `task_543fe0e46aa34404b5cae96935267d10` before changing my VM.
Static inspection found STR_TO_LOWER/UPPER checks its scratch allocation but
publishes the final string result without checking allocation failure.

I preserve ASCII-only stored-byte conversion: A..Z becomes a..z for lower,
a..z becomes A..Z for upper, and every other byte (including NUL/high bytes)
remains unchanged. I keep immutable output, exact stored length, and
existing source-type refusal. I check scratch length-plus-terminator before
allocation, and check final allocation before publishing a string value.
Both failure paths release the input owner and any scratch storage they own.

I validate fresh corrected-source short and long ordinary byte strings,
unchanged caller aliases, deterministic final-result allocation failure,
interned empty output and subsequent successful invocation. My VM may reuse
an existing equal string before allocation; managed fresh-result behavior
does not establish allocation-event equivalence. I run focused
sanitizers and full VM regression acceptance. I do not replay historical
failed artifacts or require a pre-fix failure demonstration.

Managed LLVM/Wasm conversion remains a separate contract after this repair.
Full runtime51da, Darwin managed7ba and historical evaluator791a stay open.
