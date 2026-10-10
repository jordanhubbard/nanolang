# My managed list release correction

I lower `list_*_free` through the C-seed NanoVirt and self-hosted emitters by evaluating the receiver once, dropping that evaluated reference, and clearing a named compiler root. Other owners retain their references. I keep actual declarations ahead of name-based lowering and refuse mismatched receivers.

`make -j2 test-nanoisa-list-free test-nanovirt` exits zero: all ten list methods and all 90 NanoVirt checks pass. The paired suite executes both producers in NanoVM and strict ASan/UBSan native translations. It covers repeated int/string allocation, temporary evaluation, declaration precedence, record aliases and retained string children, bounds and operand refusals, plus existing insertion coverage.

My observer stops at real print boundaries while the guest frame remains alive and drains deferred cycle suspects. Heap object count increases on allocation, stays unchanged while another owner remains, and returns to baseline after the final release. A control retaining that final owner stays above baseline after collection. Both producers pass. The original before-collection expectation failed because the runtime deliberately defers buffered zero-reference objects; I retain that evidence and do not claim immediate physical reclamation without collection.

The initial declaration-precedence fixture incorrectly redefined a reserved builtin. I use a nonreserved `list_Custom_free` declaration for the corrected control. The initial observer also used the v1-only loader and statement-form print; the corrected observer uses the production module loader and both frontends' call-form print. The initial generic record receiver refused because its checked nominal element is stored as `struct_type_name`; I preserve that identity alongside complete TypeInfo metadata.

This component gate is not fresh installed-stage or full parser qualification. Actual Json host imports, remaining generic receiver forms, and the unchanged complete parser corpus remain required under #978.

My canonical compiler component rebuilt successfully from `src_nano/nanoc_v06.nano`. Compiling the actual `scripts/gen_compiler_schema.nano --emit-nvm` now advances beyond list cleanup and refuses `unsupported extern result or symbol nl_json_free` (exit 1). This establishes the next actual blocker; it does not qualify Json host ownership or ABI.
