I repair ordinary List<int> record-field lowering under #976 on release/5.1-completion-20261007. Scalar writes now propagate through shared array aliases, and still-unresolved concrete ARR_NEW integer constructors retain their storage evidence after shape inference. Unknown parameters remain refused rather than guessed.

I retain the original mutation failure and a separately discovered empty-list failure. Six bytecode methods exercise VM/C AOT parity, scalar aliases, empty fields, conflicting writes and unresolved parameters. Both retained self-hosted compilers pass the 16-method nominal-order suite with the corrected translator.

I also retranslate the preceding pinned compiler module, link a native compiler and pass all eight recorded stages, including 16 nominal-order methods and 16 payload/resource/layout methods, with unchanged module/source hashes. Evidence is docs/evidence/list-record-shapes-20261011. This does not replace the exact-candidate frontend/platform release gates, and #976 remains open.

My final make test-nvm2c gate passes all prerequisites, the six new alias methods and 2,443 structured C tests. The pinned-compiler check takes 29.185 seconds.
