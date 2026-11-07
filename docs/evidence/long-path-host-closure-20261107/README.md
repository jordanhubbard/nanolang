# My refused long-path bootstrap

At source `89a8f2eac6cafb7b824a08f298b80b69994f30c8` I run the new complete
Stage 1 check in a fresh detached worktree at a 316-character absolute path.
I omit only historical evidence; I keep native source paths and explicitly
isolate object, binary and module-cache outputs. I retain the raw worktree,
commands, logs, modules and both host generations at the path in my summary.

Seed emission and verification pass. Stage 1 self-compilation passes in
905.683 seconds and its VM verification passes in 0.272 seconds. The next
closure check refuses before native translation: `I require the exact seed
host closure in every generation.` I do not retry past or weaken this check.

Both modules import six host libraries. Five paths remain identical;
`file_product` changes generation path. The two libraries have different
bytes, while all 81 assembly snapshots, 82 objects, 81 dependency files and
81 include records remain identical. Only the dylib changes. Neither
file_product generation has saved cache-control records.

I track this newly exposed failure under #1077. Source inspection identifies
a necessary limit to correct: the shared-link recipe accepts a 65,537-byte
buffer, but Darwin's dependency-recording wrapper uses only 8,192 bytes.
The 35 shared object paths at this retained location alone exceed that
smaller limit. An ordinary link can succeed while dependency capture is
omitted, preventing cache reuse. I still require a direct control and a
complete fresh bootstrap after the correction before I call it the cause
and fix of this observed generation change.

My import-name boundary repair is exercised by successful self-compilation,
but complete #1055 and release qualification remain open. This record is a
failed native check, not a retained-execution receipt or admission.
