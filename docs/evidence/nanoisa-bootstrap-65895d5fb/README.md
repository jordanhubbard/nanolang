# My clean NanoISA Make bootstrap at 65895d5fb

I run the real Make Stage 1, Stage 2 and Stage 3 targets in a clean detached
checkout at `65895d5fb669a7a5f1f96ded1c2f9c5759feb088` on Darwin. My runner
then executes `verify-bootstrap`, all 24 installed CLI/product methods, and
five publication methods whose producer is the newly built Stage 2 compiler.
Every command passes; the checkout remains clean and the commit unchanged.

My C seed produces verified bytecode. NanoVM executes its first self-hosted
generation in 252.278 seconds and the second in
249.538 seconds. Both raw modules contain
482,084 bytes with SHA-256 `014e173bfaf5efaa297d15465bcc49bc47f934af88e08f98426d79838039c93d`. I compare these raw
bytes without normalization. Both generations retain the seed's exact three
host-library paths and hashes, with no NanoLang native-code-generation call
during VM generation. Declared native host artifact/cache work is permitted
by the retained guard; it is recorded separately from language code generation.

I separately translate each self-hosted module through `nvm2c` and `cc`, then
execute each native compiler to build and run hello. Stage 3 checks the recorded
source, tool, module, native binary and host hashes before installation, and
executes the installed compiler with `nanoc_c` temporarily absent. Separate
Make invocations preserve the tool identities. The installed CLI and publication
checks pass with ASan/UBSan/LSan where selected by their harnesses.

I retain the complete receipt, runner, logs and compressed raw modules. Their
absolute host imports refer to the retained temporary checkout/cache; these are
qualification evidence, not portable release artifacts. This source pin predates
the computed-call repair and does not establish final-release qualification,
complete language parity, Linux acceptance, or completion of the full 5.1 scope.
