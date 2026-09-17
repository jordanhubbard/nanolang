# My compiler map subset

I lower `HashMap<string,int>` parameters, locals and results, with `map_new`,
`map_get`, `map_put`/`map_set`, `map_has`, and `map_length`/`map_size`. I retain
zero-result mutation calls and shared map identity across calls and returns.
I require exact map/key/value argument types; other map shapes remain refused.
Constructors in my source fixtures use explicit typed locals, as required by
my C frontend.

On Linux ARM64, `make -j8 test-nanoisa-src-nano` passes 86 baseline comparisons
and 12 focused Python cases. The map fixture adds 16 bytecode/function checks
and executes both C-seed and self-hosted modules in NanoVM and strict C11 native
output. It checks direct and tail returns, updates through a void helper,
shared updates through a returned alias, existing zero values versus missing
keys, and size. Seven malformed map types/calls fail without output.

I reran real `src_nano/nanoc_v06.nano` emission. It passes the map boundary and
first refuses `undefined function string_to_int`. I recorded that continuation
as `task_62d9f8ab389e4299b1b16a33ed591630`. This slice is tracked by
`task_2c74662f99d44f7faac1fe7427e44325`. Full compiler emission and bootstrap
equality remain open; map iteration, removal, clearing and other shapes are
outside this bounded lowering slice.
