# My field metadata lookup checkpoint

My sampled native full-source compiler spends time in `lookup_field_type`, which
previously built a complete `HashMap<string,string>` of field metadata on every
field access. That repeated key construction, type-string conversion and map
allocation for fields that the lookup never returned.

I now scan the metadata array from newest declaration to oldest, first for the
exact struct name and then for the existing capitalization fallback. I convert
only the selected field through the same `field_metadata_type_string` and
`type_from_string` functions. I do not cache mutable metadata or add global
checker state. Reverse order retains the old map's last-declaration precedence.
My metadata comes from declared identifier fields; qualified struct names retain
their complete spelling.

My changed shadow checks builtin metadata, missing and empty metadata, duplicate
declarations, exact lowercase names before uppercase fallback, array element
names and qualified records with List fields. C-seed bytecode publication runs
these dependency shadows successfully.

My persistent VM/native fixture includes 256 unrelated fields, a duplicate field,
capitalization fallback and missing reads. Its 200 lookups previously allocated
200 complete maps. The new regression requires zero map creations while retaining
the same observed results. It passes NanoVM execution and strict Clang C11 `-O0`
with ASan, UBSan and leak detection enabled. The baseline native run is ordinary
`-O0` measurement, not a sanitizer qualification.

I add this regression to `test-one-ir-compiler`. The complete 86-method compiler
product suite and structured-C gate are running with the collector and metadata
corrections. Their completion and a fresh release-pin fixed point remain required;
the earlier clean VM fixed point predates this source change. MAC task creation
returns `Operation not permitted`, so hub synchronization remains unavailable.
