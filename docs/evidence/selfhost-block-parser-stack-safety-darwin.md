# I pass the corrected parser matrix before an explicit tool prerequisite stops me

I qualify the iterative statement-list parser under
`task_a613986ffa6f476293e3befa8d9accfd`, its route parent
`task_bd8ebe2ba75457af1686f095275229dd` and installed-product parent
`task_e8d860a16da0464891dd32e91c42bef1`. This is bounded Darwin evidence. It
does not qualify a product candidate or authorize publication.

## Frozen input and retained history

I used a fresh detached checkout at
`603785c9c295cf2c780fea9633aa6d86f37744f4`. The checkout and evidence roots
were:

- `/private/tmp/nanolang-parser-stack-gate.qETcTo`
- `/private/tmp/nanolang-parser-stack-evidence.iPhoJA`

The before and after maps each contain all6,431 tracked files and are
byte-identical. Their map SHA-256 is
`51013841632ebcc2be76c602335a68a3586b86684294795364d5799cac37f3e9`.
The checkout was clean before and after the run.

I did not copy, execute or minimize the historical crashing compiler. I retain
its existing signal-11 log, observation files and macOS crash report separately.

## Fresh bootstrap and parser controls

`make -j8 bootstrap` passed in311.68 seconds. Stage1 and Stage2 each compiled
and ran the hello smoke, the installed compiler ran without `bin/nanoc_c`, and
the parser shadows ran during the mandatory shadow checks. Those shadows cover:

- 521 source-ordered ordinary block statements;
- 520 source-ordered unsafe-block statements;
- owned-pattern and tuple-destructure expansion order; and
- missing closing braces for both ordinary and unsafe blocks.

The bootstrap log SHA-256 is
`bf0f1757d6ac79410372609c280afb9038460b4ebfe1be72bd157bb3559d597e`.
The focused C-frontend runner build passed in0.47 seconds; its log SHA-256 is
`72afa408fdf54a56cf433d51745496bf01762d70e7bfa0f310c9fdd30511af38`.

The selected binaries were unchanged across the subsequent gates:

| Artifact | SHA-256 |
|---|---|
| `bin/nanoc_c` | `7b1367443edf8066cdd414bcca7ccea57818b7cd1bbb6051602dbc6b53d5f2e9` |
| `bin/nanoc_stage1` | `86daed4b5bd8af59c535a6abee9a197f8481f8056422c599b67260555f017813` |
| `bin/nanoc_stage2` | `b00462b3b332bc6ab0d719b61ee3315e393b2b5eb52f177cb419698a747dca2b` |
| `obj/test_affine_c_frontend` | `ae0ce285294bec9a65d8ce1007339c65b73dcb00f868371b2c6bda57516ee112` |

The build selected Apple Clang21.0.0 at
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`,
whose SHA-256 is
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.
I separately record `/usr/bin/cc`, `/usr/bin/make` and `/usr/bin/git` as Apple
dispatcher entrypoints; their shared file hash is not an implementation hash.

## Complete corrected route matrix

I ran the complete frozen matrix with a new private cache and an external
observation directory:

```text
AFFINE_PROFILE_EVIDENCE_DIR=/private/tmp/nanolang-parser-stack-evidence.iPhoJA/route-observations \
NANO_BUILD_CACHE=/private/tmp/nanolang-parser-stack-cache.ugEaM4 \
python3 -m unittest -f -v tests.test_affine_contract_boundaries
```

All20 unittest methods passed in2.57 seconds. The run exercised the unchanged
36-source manifest with14 semantic positives and22 semantic negatives. It
produced144 route observations, including the corrected `owner_257` Stage1 and
Stage2 routes and the prior-output publication checks. The test log SHA-256 is
`081064006a1401a62eb71dfa6097bef37f95d02cc737a72d3f4b45c11224f5d0`.
The complete observation-file map SHA-256 is
`38164fdd8662a028a1b1c4efba0a8d17f4e60e9609e146efc661bf01f0808c6`.

This passing matrix establishes the bounded stack repair across the formerly
crashing route. It does not by itself complete the ordered independent gates.

## First later terminal: missing interpreter prerequisite

I next ran the separately required callable and union public-C tests. All three
callable methods and
`test_api_nominal_refusal_output_and_recovery` passed. The next method,
`test_empty_union_and_distinct_nominal_results`, ended before exercising its
source because `/private/tmp/nanolang-parser-stack-gate.qETcTo/bin/nano` did
not exist. The command exited1 after4.24 seconds. Its log SHA-256 is
`6a164f7020e36f5b36edb15a1189b3b07455fac489f0444b2cc41e3620677783`.

I stopped there. I did not build `bin/nano`, rerun the method, proceed to the
checked-selection or aggregate affine gates, or change production. I record the
missing prerequisite as `task_bd8e2d91943ae17059d289e13ebc34fe`. A corrected
qualification must prepare and freeze `bin/nano` before beginning this ordered
follow-up. I do not treat this setup terminal as a parser or union semantic
failure.

## I inventory the corrected setup before building it

I statically inspected every unreached gate before preparing another checkout.
The two remaining union-result methods use:

- qualified `bin/nanoc_c` for public-C emission;
- `bin/nano` for the ordinary interpreter route;
- `bin/nano_virt` and `bin/nano_vm` for bytecode emission, verification and
  execution; and
- the selected host C compiler for four strict generated-C variants.

The checked-owner-selection class uses:

- qualified `bin/nanoc_c`, `bin/nanoc_stage1` and `bin/nanoc_stage2` to build
  three test-only selection drivers and exercise both explicit compiler routes;
- `bin/nanoisa` for assembly;
- `bin/nano_vm` for verification and execution; and
- `bin/nvm2c` plus the selected host C compiler for strict native execution.

The remaining aggregate affine modules use the same three qualified compilers,
plus `bin/nano_virt` and `bin/nano_vm`. `tests.test_affine_frontend_parity`
checks the three native compiler routes. `tests.test_owned_record_patterns`
adds NanoVirt/VM execution. The already passing
`tests.test_affine_contract_boundaries` module is the third component of
`tests/test_affine_selfhost.sh`; I do not repeat it in the corrected run.

The corrected build set is exactly:

```text
bin/nano
bin/nano_virt
bin/nano_vm
bin/nanoisa
bin/nvm2c
```

I will create a fresh detached checkout at exact `603785c9`, copy only the
three qualified compiler binaries from the retained passing checkout, verify
their hashes before and after the build and gate sequence, and create the
`bin/nanoc` link to the copied Stage2 compiler. All five remaining executables
are built from the fresh checkout. I freeze the tracked-source map, actual
Apple Clang, Python and built executable hashes before running the two
remaining union methods, checked-owner selection, affine frontend parity and
owned-record patterns in that order. I stop and seal the first new terminal.
