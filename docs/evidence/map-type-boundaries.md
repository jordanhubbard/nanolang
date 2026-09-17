# My map type boundaries

I compare both key and value tags when an existing map crosses a declared
boundary. I previously compared only the `HashMap` base type, accepting an
integer-valued map where a string-valued map was declared. The shared C checker
now rejects those changes before either C-native or NanoVirt output publication.

`make test-map-type-boundaries` checks all twelve unequal pairs among my four
supported scalar map types at eight boundaries: direct returns, function-result
returns, bindings, arguments, assignments, globals, record construction and
conditional arms. Both compiler drivers must report the map mismatch and preserve
an existing output artifact: 192 rejection decisions.

I also passed the two constructor-context methods covering all four valid map
types and the component build gate. The retained logs are
`/tmp/nanolang-map-boundaries.log`, `/tmp/nanolang-map-boundary-positive.log` and
`/tmp/nanolang-map-boundary-bootstrap.log`. Independent review found no blocker
within this scope.

This closes `task_d0438e26b84147cdb9fd16b654c44a6a`. Record projection metadata and
map operations remain `task_160826784e8a4aa4ac9d5e589a54c814`. Native lexical map
cleanup remains `task_1edadd5eb33a445d9bf6516744bc405e`. These checks do not establish
complete generic type equivalence or the bytecode compiler fixed point.

The subsequent integrated `make bootstrap` check, including generated
`nanoc_stage1` and `nanoc_stage2`, also passes with the selected-array context
repair. Its log is `/tmp/nanolang-selected-array-full-bootstrap.log`. This is
stronger evidence than the earlier `make build` component check.
