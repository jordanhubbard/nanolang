# My source consuming-call evidence

I qualify `task_c5208ffb6e494d7691ba9089c6a8e208` against my
[source contract](../NANOISA_SOURCE_CONSUMING_CALLS.md), after merged runtime
PR709. Contract commit `380d8d75` precedes production `76ec04a9`. I do not
change runtime authority, source syntax or the separate borrowed signature ABI.

I recognize plain resource formals in both NanoISA producers and route them
through my checked ownership profile. One owned formal starts live in helper
slot0 with mode0. A named whole-owner argument must be live, unborrowed,
exact-layout and not a pending disposal holder before `OWN_MOVE_LOCAL` and
`CALL 1`. Existing borrowed-only signatures continue through `CALL_REF`.

## My full source gate

At test checkpoint `27bfb8cd`, fresh bootstrap and required compiler, VM,
translator, text and local-binding tools pass. I also build the declared
`borrow_shadow_names` probe before running all 31 methods in
`tests.test_source_borrow_emission`: **31 pass in 335.237 seconds**.

My three new methods cover:

- Leaf and nested resource arguments; int/bool results; repeated calls; formal
  and helper-local observations; declaration-order metadata with reordered
  source fields; explicit consumption; reaching branches and zero/entered loops.
- C-seed bytecode, C-seed/Stage1/Stage2-built source emitters and canonical
  Stage1/Stage2 output, with identical canonical metadata, lexical names,
  stripped-name execution, verification and native ASan/UBSan/LSan results.
- Selected shadows from all three producer tools, false helper assertions,
  terminal cleanup and mandatory shadow failure preventing publication.
- Eight semantic refusal families: repeat use after transfer, same-shaped
  distinct nominal records, unconsumed parameters, constructed/projected
  actuals, deeper helper graphs, mixed borrowed/owned signatures and two owned
  parameters. Both checked and raw producers preserve previous output; parse
  failures do not count as ownership refusals.

All 28 previous source-borrow methods remain intact and pass. Separate gates
pass 90 compiler-core checks, affine 441/751, helper-local lifecycle 1,309 plus
106 heap-fault checks, and consuming runtime 3,091 plus 104 heap-fault and 90
frame/contract preflight checks. Logs:

- `/tmp/nanolang-source-consuming-bootstrap.log`
- `/tmp/nanolang-source-consuming-paired.log`
- `/tmp/nanolang-source-consuming-authority.log`

## My additive integration

I merge canonical `94cca515` at `f06fa84e`. PR710 changes float text assembly/
disassembly; PR711 is a documentation inventory. No compiler/frontend or
`src_nano` source changed in this restack. I resolve only the roadmap conflict,
preserving the component inventory and completed runtime4ef evidence.

All three production files remain byte-identical to reviewed `76ec04a9`.
The C seed, both stage compilers and C-seed-built source emitter retain the
SHA-256 identities recorded in `source-consuming-tool-hashes.txt`.
After rebuilding the affected NanoISA tools, I run the three new methods with
the retained C-seed source emitter, a fresh C-seed shadow tool and canonical
Stage1/Stage2 compilers: **3 pass in 25.239 seconds**. This focused restack check
does not repeat the earlier Stage1/Stage2-built raw emitter checks; their source
and producing compiler binaries are unchanged. Logs are
`/tmp/nanolang-source-consuming-integrated-tools.log` and
`/tmp/nanolang-source-consuming-integrated-focused.log`.

I leave constructed/computed/projected actuals, mixed signatures, owned
results, deeper graphs and the full normative ownership matrix outside this
bounded admission. No historical failing compiler artifact or malformed
module was executed. Task completion follows actual canonical merge.
