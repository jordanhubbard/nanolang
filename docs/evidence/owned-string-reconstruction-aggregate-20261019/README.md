# Aggregate owned STRING reconstruction

At integrated source f1f9683f1, all fourteen methods pass in 79.632 seconds:
ten STRING methods plus four prior byte-output controls. Producers are current
NanoVirt and both installed stages from the successful 282d7a9f1 bootstrap.
Ten pinned tools/providers remain unchanged during the run. The facts tool alone
was rebuilt from current source with Homebrew Clang; no frontend/bootstrap was
repeated or claimed.

The unchanged ordinary demo and independently emitted original-shadow executable
retain exact two-line/five-line output. Reconstructed Nano local counts remain
3/5/11 and 11/5/11/3. Original and regenerated VM/native paths, O0/O2 ASan/UBSan
with NDEBUG, assertion prefixes, canonical roundtrips, name/advisory independence,
shared literal/name indexes, exact bytes, invalid-UTF8 target distinction and
1MiB publication/refusal boundaries pass. Prior output bytes stay exact.

The original-shadow input is produced by the pinned C-hosted existing driver from
the component baseline; it is not claimed as a fresh-stage-built shadow driver.
Its regenerated outputs are exercised through all three aggregate producers.
The source compiler/runtime providers remain the existing pinned generations.

The retained command uses NANOLANG_OWNED_RECONSTRUCTION_COMPILERS set to absolute
bin/nanoc_stage1 and bin/nanoc_stage2 paths,
NANOLANG_OWNED_STRING_SHADOW_EMITTER=/private/tmp/owned-string-baseline-20261019/shadow_driver,
and NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang. tests.log identifies all
fourteen methods and their retained artifact directories; providers.json records
full hashes before/after. Hosted Linux/Darwin qualification and full #999 remain open.
