# My private callable collector and linker

I implement the logical collector/linker from the pinned
`src/nanoisa/nvm2c_callables.c/h` in `src_nano/nvm2c/callables.nano` under #1142.
My three-route semantic diagnostic passes. My complete five-route gate fails
native translation under #1262. I have not accepted this component for cutover.

```nano
module "src_nano/nvm2c/callables.nano" as callable

let result: callable.Analysis = (callable.analyze input)
let shape_id: int = (callable.callee result function_index byte_offset)
let producer_mask: int = (callable.projection_tags result function_index byte_offset)
```

I take raw code, explicit function ranges/signatures/capture counts, import
metadata and operation-name bytes. Each analysis constructs fresh state. My
managed values own its lifetime; I do not expose the C allocation/destroy ABI.
The module conversion and executable-admission layers still own validation of
the surrounding container, layouts, ownership, imports and profile authority.

I allocate independent result, local and flattened capture shapes for every
body, then collect all referenced globals, aggregate field widths and variant
identities before function flow. Capture slots merge provenance across instances
of one body. They do not imply a shared runtime closure environment.

I decode every body, including dead instructions. Within a function I create
fresh stack shapes at each first-reached frame, require equal incoming heights,
and add every incoming directed flow. I preserve dynamic call/literal/capture/
perform/PICK/return effects, stack permutations, checked local/capture accesses,
producer-tag masks and the original terminal/branch order. I reuse my complete
[effect-arm collector](NANOLANG_EFFECT_FLOW.md) for operation aliases, handler
parameters and resume-result provenance.

I keep constructor-specific variant fields separate from ordinary positional
fields. Only the uninterrupted DUP/AGG_TAG/PUSH_I64/EQ-or-I64_EQ/JMP_FALSE pattern
refines its fallthrough view. Entry into the middle prevents that refinement.
A later match retains every constructor's tag map.

I link direct calls immediately and retain indirect constraints until solver
facts converge. After each graph solve I join array/map element identity through
copied handles and nested record fields, then connect each newly discovered
module-local target exactly once. New links can supply arguments, returned
functions or captures for another pass. I stop only when neither alias identity
nor indirect links change. An unresolved callee remains unresolved.

`callee` returns zero outside a valid function/byte range, on failure, or when
the selected value lacks an exact function shape. A target value of zero remains
a valid module-local function index. `projection_tags` returns producer masks,
with 32768 as UNKNOWN for failed/out-of-range/nonprojection queries. A real
projection can have mask zero. Neither zero nor UNKNOWN proves a scalar type.
I preserve the three signature-qualified empty-library host special cases from
the reference. They grant no general foreign-call authority.

My graph node IDs belong to one analysis. I do not require C and NanoLang to
allocate equal IDs: C leaves some argument evaluation order unspecified. My
comparison retains complete raw diagnostics and normalizes only private node/
conversion IDs in three graph-error templates. Function indexes, byte offsets,
stack counts, all other diagnostic bytes, ordered target sets and projection
masks remain exact.

My [diagnostic receipt](evidence/darwin-current-source-20261124/diagnostics/callables-three-route-diagnostic-linux-20261126/README.md)
retains 978 matrix cases, 424 explicit expectations, all505 unchanged original C
assertions and all26 captured original module inputs. Ordinary/sanitized actual
C agrees with C-seed native and both producer VM routes. Generated native routes
remain refused. This is not five-route or paired-platform acceptance.

Run the full permanent gate with an explicitly qualified producer:

```sh
NANOLANG_TRANSLATOR_SOURCE_COMPILER=/absolute/path/to/nanoc_stage2 \
  make test-nanolang-translator-callables
```

That gate still attempts native generation. I do not select the external
three-route diagnostic as its replacement. The C owner's reviewed #1262 repair
or explicit handoff, a passing full gate, representation/module/body/runtime/
profile/ownership composition, paired platforms, installed cutover and all69
release requirements remain open.
