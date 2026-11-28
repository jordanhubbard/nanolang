# My public ARRAY allocation-prefix diagnosis

I retain the original post-bootstrap failure under #1166/#1161/#976, the measured partial source improvement, and a separate complete passing Clang18 public gate. I keep my GCC13 sanitizer configuration and paired/final-head acceptance open.

## My original failure and measurement

At frozen bootstrap pin `c908523c3`, my original public ARRAY target passes its instrumented bridge, then its actual `public-api-allocation-prefixes flow 1` guest exceeds120 seconds. The original method records one timeout error in281.103s; Make exits2 after283.818s. MAP and the later controller steps never start. My separately launched original scalar CLI/product gate passes all24 methods in3.316s. My bootstrap source/tool/host/artifact guards remain unchanged.

My scheduled and preceding-collector first-success controls each measure472 tracked allocations and complete in milliseconds. Both full flow1 campaign controls exceed the original120-second guest limit. The preceding image replaces the actual heap/collector/FFI providers and preserves all other providers and flags; no GC regression follows from the timeout.

My first private baseline link refuses the missing `vm_gc_maybe_collect` call from `vm_ffi.o`. The changed `bstring_intrinsics.inc` belongs to `vm_ffi.c`, not `vm.c`. I retain that setup failure and rebuild the actual preceding FFI includer; I add no stub. My original private count wrapper also collides with the fixture's main wrapper; I preserve the refusal and correct only private entry naming/include binding. A bounded GDB line-breakpoint attempt returns no count and establishes no cause. The later direct count controls supply that measurement.

A30-second private progress observer advances through successive prefixes; I do not infer a single stuck fault. My direct sample of the unchanged actual guest contains3000 main-thread CPU PCs:2950 (98.33%) lie in ASan fake-stack allocation/modulo. A separately interrupted original guest stack reaches `merge`/`global_merge` through record-array verification in `vm_init`. I retain exact image identities, loaded maps, PCs, source/disassembly and symbolizer commands. My sampler has a separate successful ASan-preload count control; I stop only private process groups after the bounded windows. These observations are not full guest acceptance.

## My source improvement and remaining GCC13 failure

I replace private aggregate-by-value merge inputs with const pointers to the same stable lvalues, including the array-origin and union include callers. I preserve every merged field, update and worklist decision, with no sanitizer exception. My private flow1 candidate passes all472 transient and persistent prefixes and recovery checks in6.036s.

My promoted source passes all five original GCC13 flow-tag campaigns. Its complete original target still times out at mutation1 after120 seconds: one method error in196.782s, Make2 after252.770s, with MAP unstarted. A separate3006-PC mutation sample places2574 (85.63%) in ASan fake-stack allocation/modulo. I do not call this full repair.

I reproduce missing fake-frame allocation-flag cleanup independently. My small merge control retains address/undefined sanitization and default use-after-return checking. A private metadata observer reads the documented fake-stack API and its saved flag; only that metadata observer avoids address instrumentation. System GCC13.3 prints `during=1 after=1` and returns the diagnostic108; Clang18 prints `during=1 after=0` and returns0. The actual original merge disassembly likewise retires and poisons its fake frame without clearing the saved allocation flag.

This matches [GCC's missing FakeStack flag-cleanup fix](https://github.com/gcc-mirror/gcc/commit/aac7bb773f7d1c98ee980492fd2a8785f3153212). I retain the exact primary patch and web-tool extraction of the runtime sources. A separate raw runtime-source fetch returns HTTP404; I do not claim raw source bytes for that retrieval. I preserve the selected host runtime ELF images used for symbolization. I change no system compiler, CI default, ASan option, guest deadline, prefix loop or assertion.

## My separate passing Linux controls

With the already available Clang18 compiler route, the complete original public targets pass: ARRAY one method in421.564s and MAP three methods in65.370s; Make exits0 after489.427s. The ARRAY gate includes all five tags for both flow472 and mutation1845 transient/persistent prefix campaigns:23170 fault executions, each followed by its original recovery check, with163475 printed allocation-check assertions. Original generated native, transport, ownership/refusal and identity controls remain selected. MAP retains its original allocation-prefix, computed-goto and switch/native methods. I do not attribute uniform sanitizer coverage to providers the original tests leave ordinary.

My neighboring original union-field proof gate also passes its complete method in13.431s under GCC13, exercising the shared merge helper's union caller. This passes a separate concern; it does not qualify GCC13's failed public sanitizer campaign.

The [manifest](manifest.json) retains1604 logical files,1198920038 raw bytes and173226516 unique compressed bytes. I verify every raw and compressed SHA256 after writing. I retain original and private actual guest images/providers, sources, sampler/observer/control builders, exact terminals, loaded runtime ELF images, successful Clang18 public products and neighboring proof logs. The original successful bootstrap remains a separate snapshot. My pure translator, installed host-library closure, ownership, paired/final-head and full5.1 release obligations remain open.
