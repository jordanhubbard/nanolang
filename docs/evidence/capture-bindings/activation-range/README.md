# My activation range preparation checkpoint

My reviewed8240c0800 helper prepares local binding state with an explicit
initialized range. Ordinary calls keep the first-arity rule through the existing
API. Effect activations can prepare their nonzero parameter range without
borrowing ownership of the lexical owner's binding state or moving operands.
I validate ranges with subtraction before allocation and copy all exact modes.

I pass122 storage checks and485 existing atomic/environment checks in each of
seven configurations: Linux GCC/Clang ordinary and ASan/UBSan; Darwin Apple and
Homebrew Clang ordinary and Homebrew ASan/UBSan. Additive controls check nonzero
starts, empty end/zero states, excessive ranges, invalid modes, allocation
refusal with unchanged output/accounting, managed parameter reads and complete
cleanup. All14 source/driver input identities remain exact.

Reports retain commands, compiler/SDK identities, exit statuses, deadlines and
process-group cleanup. Product hashes identify locally retained binaries/debug
bundles. Raw Darwin products and reports are copied locally; the remote snapshot
lives under /Users/jkh/nanolang-qualification/nanolang-binding-range-8240c0800.
These manifests do not claim a vendored binary archive.

This is range preparation, not effect execution qualification. Complete lexical
owner lookup, frame/return/unwind integration, definite initialization, wire
admission, source/backend parity and my full5.1 acceptance remain open.
