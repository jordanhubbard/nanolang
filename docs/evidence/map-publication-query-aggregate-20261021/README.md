# Aggregate map publication query gate

I merged d0d2079af by ancestry into d67d11eba at 2f88c0c6d and added `test-record-map-publication` to retirement compiler contracts. The actual required Make target passed one method in 5.624s: 342 linked and 1,509 instrumented checks across four unchanged producer modules, four concat variants, and one independently emitted original-shadow candidate. All annotations remain harness-only.

The test measured 115 allocation positions and 103,319 bytes of peak payload, tested all 230 transient/persistent denials plus independent retries, and retained exact scan limits and late-corruption controls. The Make plan rebuilt only managed_array_shapes.o and nvm2c.o; no frontend or bootstrap was built. All 68 workflow/schema controls passed in 2.019s. No aggregate failure occurred.

Exact module/source/object/workflow hashes and the Make provider closure are retained. This query returns eligibility only; unsupported is not a validity certificate or execution permission. Publishers remain unchanged. Full source publication and exact-head hosted qualification remain pending.
