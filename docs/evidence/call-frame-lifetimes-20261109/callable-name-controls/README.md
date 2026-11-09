# My separate callable allocation witness

I forward actual `strdup`, `calloc` and `free` operations in freshly compiled
current env/eval translation units. I count names duplicated by `create_function`
for the scalar function `clean`, and annotation allocations made by `payload_alloc`.
My fixed pointer table matches each observed free to a still-live allocation;
forwarding functions use the real allocator and do not allocate tracker storage.

Two5000-call indirect batches create10000 callable names and40000 annotation
allocations, with no matching frees observed during the batches or interpreter
teardown. Two interleaved5000-call direct batches add none. Every scalar result
and restored symbol count passes. My real2MiB allocation/free control passes;
the indirect live-byte deltas are652960 and653104. Counts describe the observed
allocation sites, not every allocator in the process or a complete leak scan.

The actual source path reconstructs an owned signature and name when reading a
function declaration. A local binding keeps that value, then `eval_scoped_block`
pops local symbols without releasing callable values. My record/union constructors
can retain borrowed callable fields, so blanket freeing local functions would
need separate escaped-value ownership qualification. I have not changed cleanup.

I retain the complete unit, forwarding header, commands, streams and stable
selected source/object/native-image hashes in the compressed report. These
controls use cached dependency objects; they do not establish complete fresh
provider, sanitizer, nested-signature, escaped-record or NativeSDK acceptance.
My underlying compiler executable was not measured before this diagnostic;
I do not claim a complete toolchain identity receipt. #1140 remains open.
