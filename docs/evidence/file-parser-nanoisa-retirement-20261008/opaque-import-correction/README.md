# Canonical opaque declaration correction

My import merger removed every `opaque type` line, including declarations in
the root source. I now preserve those lines with their source mapping. The
emitter already recognizes declared opaque types; I do not reinterpret Json
as an integer or discard its nominal declaration.

My canonical checker accepted literal zero at opaque call sites but refused
it as a local initializer. I apply the same literal-zero rule to checked local
initialization. Nonzero integers and undeclared names remain refused.

I retain the original Stage1 failure for root and imported declarations, then
the declarations-only canonical component's initializer failure. After the
checker correction, both test methods pass in 0.256 seconds: root/imported
compile, VM verification/execution, nonzero refusal, prior-output preservation,
recovery and unknown-type refusal. The component is the actual canonical
`src_nano/nanoc_v06.nano` compiled by my C seed, not a fresh installed Stage1/2.
Its builds also execute the adjacent checker shadows.

Compiling the real `scripts/gen_compiler_schema.nano` now advances beyond the
unsupported Json local and refuses missing `list_string_free` lowering. I
retain both schema probes. List cleanup, Json foreign-call qualification and
the full unchanged schema/parser corpus remain required under issue #978.
My new Make gate requires fresh bootstrap and both installed compiler stages.
