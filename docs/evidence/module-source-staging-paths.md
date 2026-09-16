# My generated module source paths

I reproduce GCC 13's `invalid built-in macro "__FILE__"` failure when a generated
C source path inherits quotes, backslashes and control characters from an
otherwise valid NanoLang module filename. An assertion in generated runtime
helpers expands that physical path before source `#line` directives apply.

I now create private `.nano-module-XXXXXX` directories beside the destination
object. I retain `mkdtemp` uniqueness and mode 0700, same-filesystem atomic
publication, the original destination name, source module identity and original
path diagnostics. I do not rename or restrict the user's source file. Only the
compiler's private staging basename changes.

On Linux ARM64 with GCC 13.3.0:

- All 24 module compile invocation cases pass, including overlapping builds,
  failed partial outputs, missing outputs, diagnostic draining and arbitrary
  filename escapes.
- The escaped-filename case now also executes an assertion inside the module
  function, after its source `#line` directive. Its generated source path is
  checked independently for an ASCII staging stem without quotes or backslashes.
- The cache publication suite passes 47 cases. Its existing seven Darwin-only
  cases and one Clang-only PCH case remain conditional skips on this GCC host.

I ran `python3 tests/test_module_compile_invocation.py` and
`python3 -m unittest tests.test_module_cache_publication` after building
`stage1`, `nano_virt`, `nano_vm` and `obj/test_module_generation_probe`.
