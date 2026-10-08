# My NanoISA product driver checkpoint

I remove the legacy transpiler import and source-to-C path from `nanoc_v06.nano`.
My default product is a sibling `.nvm`; explicit `-o` requests native output
through module assembly, `nvm2c`, and `cc`. Source-C output uses the same
translator. I execute selected bytecode shadows before module/native publication.

I build this driver through C-seed bytecode, `nvm2c`, and strict C11 compilation.
My corrected CLI passes all 20 methods; four additional product methods check
module defaults, failed-shadow preservation, source-only behavior, and tool
lookup outside the checkout through absolute paths and PATH. My publication
helper passes five methods through both VM and native callers with
ASan/UBSan/LSan, including quoted paths, host imports, failed tools, malformed
modules, directory destinations, and retained C. Eight Make dependency methods
pass. I retain source, binary, and log hashes in `manifest.json`.

My first prepared CLI run fails four methods: missing string-search host
adapters, tool discovery outside the checkout, and an obsolete error oracle.
The corrected driver retains the search fixture's byte-offset and empty-needle
assertions. My translator passes all 2,431 structured-C checks. A combined helper
run accidentally selects Apple cc because Homebrew LLVM has no cc executable;
its native cases refuse unsupported leak detection. I retain that terminal and
repeat the helper with `/opt/homebrew/opt/llvm/bin/clang`, without disabling
leak detection. The broad compiler-product rerun uses an explicit cc symlink to
that compiler and is still pending at this checkpoint.

I do not claim a completed product cutover or release. My Make bootstrap still
needs two raw self-hosted module generations, immutable host closure, mandatory
raw comparison, and separate native translation. Full compiler qualification,
remaining product dependency removal, and the exact release gates remain open.
