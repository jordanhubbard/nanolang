# My function index during emission

I use this design under #1255. My current evidence is the exact `771a143304` compiler, its published phase observation and its shadow-lowering stacks. The original emitter helper still exceeds 300 seconds. This design is not a performance or release result.

## Lifetime and authority

I build a private name index only after the existing preflight checks and owned-borrow delegation, immediately before classic module or shadow emission. I clear it at entry and on every return through the existing entrypoint signatures, and from `nisa_reset`. My other emission state is already sequential and non-reentrant; this index has that same boundary.

My audit finds function insertion and lambda naming in parsing, then merged-name binding in `bind_merged_functions` before `nanoisa_check_and_emit_native`. Nominal rewriting preserves function names and order. `parser_set_function_affine` preserves name, line and body while stamping checking metadata. Classic emission contains no function-table writes: shadow and generic functions are local records, not insertions into `Parser.functions`. My production lookup callers pass the current parser through the same emission path.

I therefore hold names and declaration order fixed within one classic emission. I do not retain an index between parsers or emissions. Standalone lookups use the existing scans. Name replacement between emissions remains live; an unexpected length change during an indexed call selects the existing scan. This is an audited implementation invariant, not a formal proof or support for concurrent AST mutation.

## Representation and lookup

I follow the existing `module_bindings.nano` use of `HashMap<string,int>`. A map stores the first ordinal for each name; an integer array links later equal names in ascending declaration order. I build in reverse order. Counts beyond the native array endpoint range retain the scan.

`nisa_fn_index` returns the first declaration or `-1`. Its indexed path still performs the original empty print once per skipped declaration, including every declaration on a miss. I preserve that print/flush schedule.

`nisa_call_index` retains indirect-call checks and name/binding resolution in their original order. Its indexed owner scan visits only equal names. I read each candidate's line and the current owner map when queried. A failed owner search still falls back to the first name match, and still refuses an unmatched external declaration even if a later definition exists. Body and type metadata remain in the AST and existing checking paths.

I add no Parser field, schema, ABI, C product change or user option. C ownership remains unchanged.

## Comparison and qualification

My baseline retains the original four production lookup/emission bodies. Both roles contain identical support, new assertions and the private reset cleanup. Removing those common additions restores every original source byte. The candidate alone wires emission lifetimes and indexed lookup branches. I record all changed regions and file hashes.

My comparison fixture embeds each exact implementation file and a common harness. It retains the original lowering-report body, all 24 original source/expected-report pairs and existing assertions. The harness also executes the same new context/owner/alias assertions in its production entry, so native sanitizer routes exercise them without adding a public testing API to shipped code. I retain and hash the generated fixture separately; I do not call it an unchanged invocation of the permanent unittest.

I require both roles through C-seed native, C-producer VM and sanitized native, and NanoLang-producer VM and sanitized native at the existing 600/60/default10 limits. I check empty function lists, duplicate names, owner-specific selection, first-external/defined fallback, binding changes, indirect calls, standalone aliases and growth, changed equal-size contexts, failure recovery, reset and both emission boundaries. I inspect retained C call paths and preserve all source/tool/host/product guards.

Only a complete independently checked and pushed comparison permits the next exact-source original bootstrap and native36/emitter12 gates. Their original 1800/30/10 and 300/default10 limits remain. Paired qualification, pure runtime emission/cutover and all 69 release requirements remain separate.

My [Linux component receipt](evidence/darwin-current-source-20261124/diagnostics/emission-function-index-five-linux-20261126/README.md) passes all270 commands, ten route controls and240 complete reports with both implementations. Its independent reader verifies11046 records, original limits, exact source reversals, common assertions, fixture reconstruction, copied bodies/cleanup wrappers, sixteen retained generated C bodies and all final guards. This accepts the component comparison. My [fresh exact-source bootstrap receipt](evidence/darwin-current-source-20261124/diagnostics/emission-function-index-cold-bootstrap-linux-20261126/README.md) passes all17 original steps, raw Stage1/Stage2 equality, native hellos/install/no-C-seed and native36. The original emitter helper still exceeds300 seconds, with zero methods run. Native acceptance remains false; this component has not repaired that gate.
