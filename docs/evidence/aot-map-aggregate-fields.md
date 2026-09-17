# My native hashmap fields

I retain a hashmap handle in a dedicated native aggregate field. Packing and
extracting that field preserve map identity; copying a record does not clone
its map. My existing map arena owns the handle until program cleanup. Nested
record snapshots do not take a second ownership of it.

I retain recursive shape constraints when extracting a map field. The flat
field vector records that the field is a map, not the key and value types
inside it. Those types remain constrained by the shape graph. I reject an
integer-valued map receiving a string after aggregate extraction.

My regression matrix covers integer and string values, records and variants,
direct and nested packing, both function orders, returned aggregates, empty
maps and mutation observed through both the original map and the aggregate.
I compile generated C with warnings as errors and execute the fixtures.

Normal and fresh ASan/UBSan runs pass 1,509 AOT and 994 shape checks. Leak
freedom is not established by these runs; sanitizer leak detection is disabled.

I run `make -j1 test-nvm2c`, `make test-nvm2c-sanitizers` and
`make -j1 test-one-ir-compiler`. Full compiler acceptance remains incomplete:
map packing in `collect_files_dfs` no longer stops classification, but a later
inference pass finds string versus tagged-value facts in `check_function`
(function 333, offset 351), at field 3 of parameter 5 of `symbol_new` (301).
I track that separately as `task_ad09fe6dbbb24d5781e970dca9db299a`.
