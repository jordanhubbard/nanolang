# Complete native module link closure

I track this correction as `task_7c505ffcb1f3464aa8cda8fb461dc4ee`.

I reproduced the ARM64 launcher link failure on puck at `767a7fce` by giving
`NANO_BUILD_CACHE` a longer absolute root. My 2048-byte module argument buffer
silently omitted later foreign objects, including `std_collections.o`; the linker
then reported missing `nl_sb_*` symbols. The short cache root passed. This was a
path-length-dependent argument loss, not a missing platform library.

I now grow the module object and linker fragment dynamically. I keep object
quoting, duplicate suppression, immutable generation selection, and the checked
final compiler-command limit. I free the owned fragment on all post-build exits.
My remaining bounded compile-flag buffers fail explicitly if their complete
contents do not fit.

I verified:

- Linux and Darwin ARM64: eight real foreign modules with more than 2048 bytes
  of object paths compile and execute, both after publication and from cache.
  The same regression fails on the preceding compiler with missing foreign
  function symbols.
- Linux and Darwin ARM64: multiple NanoLang imports still select one foreign
  generation per module directory and native link.
- Linux and Darwin ARM64: oversized module compile flags produce my explicit
  diagnostic and leave no executable.
- Linux: all 24 module compiler-invocation and path-transport tests pass.
- Darwin ARM64: `sdl_example_launcher.nano` compiles with the same long cache
  root that previously failed. I did not launch the interactive SDL application.

My final compiler command still has its existing checked 16384-byte limit;
exceeding that limit is an explicit error rather than an incomplete link.
