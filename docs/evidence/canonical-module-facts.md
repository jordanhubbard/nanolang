# My canonical module facts

I separate source facts from legacy C helper emission, retain exact imported module identities, and pass explicit facts into both program and shadow NanoISA lowering. All eight introspection operations lower directly to constants and indexed scalar selection. An index expression executes once; an absent export yields an empty string. I validate every intrinsic declaration, including unused declarations, before publication. Ordinary functions with similar names retain their bodies.

I also replace the module loader's imported transpiler suffix helper with my string builtin. My static literal import closure contains 22 files, no unresolved imports, no legacy transpiler and no C introspection helper emitter. This closes a transitive dependency gap; it does not explain the separately retained native startup failure.

My C-seed-built canonical driver passed seven module-facts/source-closure tests, five canonical VM-shadow tests, nine canonical output/artifact tests, and two product-route tests. The facts cases cover all eight operations, public/private and empty exports, single index evaluation, unsafe/FFI flags, existing export shadows, identity collisions, malformed signatures, and VM/native/C products. Generated native fixtures passed ASan/UBSan. A further static closure gate checks the import boundary.

The first artifact test run used a driver in `/tmp` without its sibling capture helper and refused two host artifact builds. I retained that log and supplied the checkout's `NANO_AS_CAPTURE_HELPER` explicitly; all nine cases then passed. I did not bypass capture or alter its assertions.

At source pin `3f01a2dbe79c5ffef76f24a8f8da428959349100`, with the explicit capture helper and default shadow timeout, my driver emitted the full compiler module. Two consecutive NanoVM compiler generations then passed in 289.331 and 288.755 seconds. Both verified modules are **244,236 bytes**, raw SHA-256 **55c880e27269ea5c3f8e8ed5d1aced04bc306b0c2a4d7acf6e498d80c4b90a0a**. The second generation compiled hello, whose module verified and executed with the expected greeting. Source commit, clean tree and capture-helper hash were unchanged during the run.

I compared generation 1 with generation 2 without canonicalizing bytecode or paths. This run used the ordinary host toolchain for declared host artifacts, not the earlier PR497 compiler-call instrumentation. It proves the recorded raw VM fixed point and small second-generation acceptance, not native self-compilation or every release gate. The source pin precedes the separate default-output policy change.

Retained evidence is `/tmp/nanolang-facts-fixedpoint/manifest.json`, its stage logs and both modules; the runner is `/tmp/nanolang-facts-fixedpoint.py`. Earlier focused logs are `/tmp/nanolang-canonical-facts-{driver-build,tests,selfmodule}.log` and `/tmp/nanolang-facts-adjacent-{shadow,product,output,output-helper}.log`.

This draft is stacked on the NanoISA-only product cutover PR522. Task `task_faa47ec22a2545348aa9c9d705580321` stays open until integration gates pass. The native startup investigation, complete ownership IR and broader release acceptance remain open.
