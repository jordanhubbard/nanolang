# I expose checked indirect File execution

I accept serialized File modules through `nvm_file_execute_indirect_bytes` and
`nvm2c_emit_file_indirect_bytes` in `nanoisa/file_indirect_public.h`. I retain a
distinct owning indirect plan. My old acyclic and cyclic entrypoints keep their
original admission rules.

I require a temporary-files host grant for execution and explicit revision 1
`NvmFileIndirectOptions`. The instruction limit is 0 through 1,000,000; zero
means zero. Initializer, entry and selected callees share that limit. I validate
all callable candidates and ownership facts before acquiring host resources.
I publish only an integer or boolean scalar after successful destruction and
cleanup; failed calls leave output storage untouched.

My public VM, generated native functions and emission query share the host gate.
A busy call refuses before inspecting its input, options or output addresses.
Outside that busy case, callers provide valid, immutable, disjoint input storage
for the invocation. A grant is host authority, not a sandbox for arbitrary C.

I emit `nvm_file_indirect_program_<identifier>(grant, options, scalar)`, with the
same identifier rules as my other public File emitters. Generated C includes
installed headers and links `libnano_file_runtime.a`. It contains real native
functions and checked target dispatch; it does not link the VM or emitter.
I check public/native ABI revisions and copied plan facts before execution.

```sh
nano_vm --allow-temporary-files --file-indirect \
  --file-instruction-limit 1000000 program.nvm
nvm2c --file-temporary --file-indirect --entry-name program program.nvm -o program.c
```

I reject duplicate/conflicting profiles, missing grants or limits, malformed
limits, and incompatible VM execution modes. Emission preserves the existing
staged-output contract. `make -f Makefile.gnu install-file-public-runtime
PREFIX=/chosen/prefix` installs the archive and its header closure; full
installation uses the same target.

I route File product publication and selected shadows through this public
indirect plan. Existing C/Nano lowerers still need paired callable source
integration; public bytecode admission does not establish that source feature.
My target analysis honors the existing `PUSH_VOID; STORE_LOCAL` end of a copy
local's lifetime, including forgetting callable identity. It cannot erase an
affine owner or a borrowed formal, and a later read requires a fresh binding.

I qualify public/private VM and native O0/O2 against the same serialized corpus,
with grant/reentry, fuel, cleanup and allocation controls. Installed consumers
cover the 256-shared-formal call, public ABI and copied-reference-map mutation,
C/C++ headers, CLI selection and native linkage. These are scoped tests, not
full 5.1 platform or release acceptance.
