# Artifact fixture sanitizer acceptance

I track this correction as `task_3f7613b3c2d7453aa2d365bb5eb8d8b3`.

My artifact handle test loaded two globally visible shared libraries with the
same scalar function name. Both also exported the first fixture's array ABI
data markers, although every array test used only the first image. ASan rejected
those duplicate data definitions while loading the second image.

I now build the second image from a dedicated scalar identity fixture. I keep
all array ABI and copyback cases in the first image and keep the shared scalar
function name with distinct answers (42 and 43). My direct loader probe now
also holds both images open with `RTLD_GLOBAL` and checks each handle's answer.
I did not change production symbol lookup or suppress sanitizer diagnostics.

I passed all 27 VM FFI tests in a normal build and again after `make clean` with
my CI flags:

```sh
ASAN_OPTIONS=detect_leaks=0 make -j8 test-vm-ffi \
  CFLAGS='-Wall -Wextra -Werror -std=c99 -g -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDFLAGS='-lm -lcrypto -fsanitize=address,undefined'
```

I left ASan ODR checking at its default. These checks cover the shared-library
reload probe, simultaneous handle isolation, array ABI compatibility, array
copyback, retained callbacks, and all VM FFI unit cases. They do not establish
that arbitrary third-party libraries with duplicate data exports are isolated.
