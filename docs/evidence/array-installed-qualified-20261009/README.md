# My installed array and slice qualification

I qualify 2250aed60 on Darwin with a fresh bootstrap and unchanged before/after
HEAD, tracked source hashes and user-file hash. The full Make command passes in
942.507 seconds: four source/native slice methods, five checked contextual-array
methods and two native identity methods. The bootstrap completes all stages;
installed Stage1 and Stage2 module bytes have the same SHA256,
e73dd0c4fed8c19fd4b4cd7b14efe818f4d3938b44fdd9df69c9c970d7d2f288.

The slice gate exercises C-seed, Stage1, Stage2 and NanoVirt source routes,
normal shadows, once-only arguments, extreme signed bounds, exact binary64
bits, scalar storage, nested child identity, record values and generated-native
null refusal. All-kinds modules run verified VM and both strict GNU/LLVM C11
ASan/UBSan executables. LLVM uses detect_leaks=1; GNU uses detect_leaks=0 because
of the separately retained empty-main leak-runtime hang on this host. I do not
claim GNU leak qualification or Linux qualification.

Before removing the unused legacy helper-extraction harness, I add explicit
outer-array inequality assertions for integer and record slices. The same full
four-method installed slice suite passes again in 24.214 seconds, with all
original cases intact. This later fixture run is separate from the pinned
bootstrap gate and has its own source and logs. The retired C harness remains
in Git history; its source-level storage assertions now run through actual
NanoISA products. The generated null boundary checks the current invariant
trap rather than the former DynArray helper's null-as-empty behavior.

I retain bootstrap logs/manifest, complete command output, text fixture and
native artifacts, module hashes and an inventory of binary artifacts. Full
release candidate, Linux and remaining legacy compiler retirement gates stay
open under #979 and the full 5.1 scope.
