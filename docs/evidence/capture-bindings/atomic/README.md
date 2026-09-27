# My atomic capture-construction checkpoint

My first41db6b54b constructor never executes: independent source review finds
ordinary release can buffer staged cells during rollback, defer their freeing,
and allocate cycle bookkeeping outside my declared transaction budget.
Corrected f0ff868e2 reverses only acquired references and frees private VOID
cells and the unpublished environment directly. The original owner remains
rooted for every reversed edge. I reserve diagnostic release-counter room.

I pass438 atomic checks and77 storage checks in each of seven configurations:
Linux GCC/Clang ordinary and ASan/UBSan, Darwin Apple/Homebrew Clang ordinary and
Homebrew ASan/UBSan. Atomic controls cover all four allocation positions in
one-shot and persistent failure modes, an independent successful recovery after
each refusal, exact memory/work budgets, duplicate and forwarded cell identity,
immutable managed values, assignment and fresh initialization, retain/release
counter limits, unchanged output and complete live-object/byte reclamation.
Storage controls include actual closure/cell cycle reclamation. Collector buffer
pointer, capacity and count remain unchanged during failed construction.

The standalone1482e18e3 Darwin storage gate first passes Apple77 and stops at
Homebrew compilation: its default MacOSX27 SDK path is absent after the host
reboot. I retain that terminal. Fresh drivers explicitly select the current
xcrun SDK26.2, without suppressing warnings or changing product assertions.
Both the earlier storage checkpoint and corrected atomic checkpoint pass with
that selection. SDK paths/settings identity and actual compiler commands remain
in the reports. Raw Darwin reports and executables are copied locally promptly;
the remote snapshots live under the persistent nanolang-qualification directory.

Every completed run retains exact before/after input bytes and modes, compiler
identity, command exit status, deadline and process-group disappearance checks.
Product hashes identify executables and debug bundles retained locally.
The evidence-copy script first encountered a Darwin debug bundle directory;
its corrected read-only traversal includes every regular file in that bundle.
This packaging correction did not rebuild or rerun a gate.

This is storage/construction qualification, not wire admission, definite-binding
verification, frame/effect integration, source lowering, backend parity or a
full5.1 release gate. Those requirements remain open.
