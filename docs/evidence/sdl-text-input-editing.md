# My SDL text-input editing contract

I qualified this bounded change on macOS at production source
`5cfe2a9e53d7e40cbbcfb02f865eb1be1deb5684`.

## Contract

My text-input widget edits a caller-owned `array<u8>`. I do not write through
an immutable string or accept an invented capacity. I require valid UTF-8 with
no embedded NUL and a declared byte limit. I append complete valid
`SDL_TEXTINPUT` payloads that fit, remove one complete code point on Backspace,
and report Return or keypad Enter without changing the bytes. Invalid input is
not consumed or drawn.

I mirror editing events while draining SDL's shared queue. Generic key and text
polling may consume its own buffered copies without stealing or reordering the
widget's stream. Starting or stopping focus clears the bounded mirror queue.
Only the active buffer owns SDL text mode.

At immutable boundaries, `bytes_from_string` and `file_read_bytes` now have the
same parsed `array<u8>` identity as annotations. The self-hosted C runtime emits
the C-seed byte conversion helpers: it constructs `ELEM_U8`, scans input with
the existing 64 MiB bound, rejects non-U8 arrays on conversion back, and uses
the managed string allocator for the result.

## Preserved first failure

The first fresh bootstrap at production checkpoint
`cf4140d9d913a8225b889feee8ac33cd9ba08469` passed both stages in 315.18
seconds. The unchanged UI example then failed through both self-hosted stages:
their generated C called undeclared `nl_bytes_from_string` and
`nl_string_from_bytes`. C-seed had already compiled the same source. I recorded
MAC `task_72cc251cbf894ec5f4e7b0adf85192f3` before adding the missing self-hosted
runtime definitions and exact shadows.

The baseline bootstrap log is
`/private/tmp/nanolang-ui-text-bootstrap.log` (SHA-256
`7c01300c481c6556f09a734f2ccd7ab033035e03c60ea82c6bbdeb5c7b005e31`).
The first failing example log is
`/private/tmp/nanolang-ui-installed-example.log` (SHA-256
`87aecd917752f51109030792b9319ac92cba02986bffd596ee5ccaca6036d5b1`).

## Final qualification

- `make -j8 bootstrap` passed Stage 1, Stage 2, both smoke tests, installation,
  and the no-C-seed independence check in 298.61 seconds. I record the native
  Stage 1/Stage 2 size difference and do not call this a fixed point. Log:
  `/private/tmp/nanolang-ui-text-bootstrap-corrected.log`, SHA-256
  `428264288ea7652606ebf1a6f4e950b4b659c84e162293b700f077d986497c8b`.
- The unchanged `examples/graphics/sdl_ui_widgets_extended.nano` compiled
  through `nanoc_c`, Stage 1 and Stage 2 in 2.71, 3.57 and 4.28 seconds.
  The three output hashes are respectively `30bf65b1341ff2ddba86c96efa34c7e8a492299200fcef91d38832a1416aecdb`,
  `9bb9f10d9e5287ae321b6900a6d29ee58055d2826ceca8e6fc79da3e098c90da`,
  and `61739fe572e9f053daa8ab21c29432a15f5b8c36c88576c1bed9c96b5b74769a`.
  Log: `/private/tmp/nanolang-ui-cross-stage-example.log`, SHA-256
  `839c8d48a926d156229594893acdb7fb3b4a61497b6a1e35a6cfa32488f2a88c`.
- Three focused methods passed in 129.655 seconds. They cover the mutable byte
  ABI, focus start/stop, ordered text/Backspace/Enter, capacity, multibyte
  deletion, invalid UTF-8, invalid array shape, generic-poller isolation, and
  C-seed/Stage 1/Stage 2 `array<u8>` acceptance plus `array<int>` rejection.
  Log: `/private/tmp/nanolang-ui-text-focused.log`, SHA-256
  `6ee89aa924325df3c2f2e10611bee113747852905f847a256e4a18ca3159b7c3`.
- The widget mutation/render method passed with Homebrew Clang 23.1.1
  ASan, UBSan and LSan (`detect_leaks=1`) in 1.589 seconds. Log:
  `/private/tmp/nanolang-ui-text-sanitized.log`, SHA-256
  `564f3c7c8a791016efe806055df0eefdb2faf82bd0b28fb67fe6ac40f603f8dc`.
- The existing SDL array-adapter syntax and upload method passed in 0.257
  seconds. Log: `/private/tmp/nanolang-ui-sdl-adjacent.log`, SHA-256
  `0ac3ad2d4b5989bedde6ccc7df289d4c397b36f2e7891f5e3a297b859d8fe07a`.

I do not claim sanitizer coverage for the mocked mirrored-queue translation
unit. Loading Homebrew's SDL2 compatibility dylib under the sanitizer opened a
host AppKit error before test `main`; the deterministic no-allocation mirror
test passes normally. The widget's allocation, mutation, UTF-8 and rendering
paths do carry ASan/UBSan/LSan coverage.

## Identities

Apple Clang 21.0.0 resolved to
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`
with SHA-256
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.
Homebrew Clang 23.1.1 had SHA-256
`570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888`.
The selected compiler hashes after bootstrap were:

- C-seed: `e230094f886c417ac70606fc107c2940169602fc8f4f6cd4895347bff32f57f3`
- Stage 1: `f4e955b4224bb0fa61c228dde01a48c221c7876d9082502616c61d81be79c374`
- Stage 2: `204e1a0adea0e90264fd90f7981af709aa56abfbd58004a7405172a6a3600fce`

The complete selected source/tool inventory is retained at
`/private/tmp/nanolang-ui-text-inventory.log`, SHA-256
`c6ce35e19ec98008be65204cce5337200f72c142180dcd53af586787fe64e233`.
It recorded a clean source tree at the production checkpoint.

This evidence closes only the two bounded byte prerequisites and the SDL
text-input row. Recursive/nominal array identity MAC
`task_3c5d8625cec144ba87efd9695239275a`, product PR #522, and release
publication remain open.
