# My matched managed substring contract

I track this bounded child as `task_2d21632d4d5241299c5a0e9967948efd` under
`task_51da49b39230468784da3481b893563b`. My VM clipping/allocation prerequisite
is merged in PR640. The broader managed-runtime parent remains open.

I preserve STR_SUBSTR byte semantics: integer indices truncate to uint32;
other tags supply zero. Start at or beyond the stored length returns empty;
otherwise requested length clips against length minus start. Embedded NUL is
an ordinary byte. Empty strings remain true under my existing string truthiness.

My runtime helper consumes one source handle owner on every path, copies before
releasing that owner or growing descriptor storage, and publishes a new handle
only after successful allocation and cleanup. Unrelated aliases remain owned.
My existing reclaiming native/Wasm allocator and first-error status contract
remain authoritative; I introduce no host import or new wire representation.

My emitted FrameOutput tracks three popped values: source, start and length.
Only source transfers to the consuming helper; both indices are released by
ordinary frame cleanup even when they are valid string aliases. Type errors and
allocation errors take existing checked cleanup paths before exported failure.
Calls, locals, globals and reentry preserve the existing ownership rules.

I admit STR_SUBSTR only in the managed closed profile after matched runtime and
LLVM lowering. General, scalar and literal-only profiles remain unchanged.
Unsupported string conversions, imports, aggregates and other heap families
remain separate required work.

I require fresh ordinary VM/native/import-free Wasm byte controls, static and
dynamic sources, empty/NUL/clipped output, index policy, alias and repeated-entry
lifetime checks, deterministic allocation recovery, and refusal/output preservation
for operations that remain unsupported. I preserve historical failed artifacts
without executing them. I record actual tested targets rather than inferring
cross-platform acceptance from source similarity.
