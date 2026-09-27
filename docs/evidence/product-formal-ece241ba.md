# My candidate proof qualification

I retain the exact source pin `ece241ba56c34ae6e47639cd7d618f43f20ba25d` and the existing digest-pinned Rocq container gate. On Linux ARM64, the image downloads but cannot execute its amd64 entry point: `exec /usr/local/bin/opam: exec format error`, exit 255. No proof compilation starts. My local environment has no x86 QEMU executable or registered x86 binfmt handler.

I preserve the original environment failure in [the manifest](product-formal-ece241ba.json). My Darwin peer completed the same digest-pinned, read-only gate in its existing Docker environment: exit 0 after 224.41 seconds. Rocq 9.0.1 compiled the proof modules, all 43 assumption reports were closed, the assumption audit passed, and `rocqchk -silent` accepted the named library. My source and proof script match the frozen pin. No host emulator configuration changed.

I retain the [sealed peer report](product-formal-ece241ba-darwin.txt) and [complete log](product-formal-ece241ba-darwin.log), with independently checked hashes and assumption-report count. Task `task_414caa6f28504b3ba47fdf7891d329f3` has its required fresh qualification evidence. This gate validates my formal model; production implementation correspondence remains a separate obligation. My product and release gates remain open.
