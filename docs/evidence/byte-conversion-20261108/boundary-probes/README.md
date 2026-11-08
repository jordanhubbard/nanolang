# My observed byte-to-string boundaries

Before implementation under #1110, I run seven small independent source programs through my older local C creation path and C-created VM provider. Each declares exact `array<u8>` values and calls `string_from_bytes` in production main. All creation and runtime commands exit zero, and provider hashes remain unchanged. I retain the complete byte stdout/stderr as hex in `results.json` inside my exact text evidence, avoiding text decoding of invalid bytes.

| Input bytes | Observed visible bytes | Observed `str_length` |
| --- | --- | --- |
| empty | empty | 0 |
| C3 A9 | C3 A9 | 2 |
| 41 00 42 | 41 | 1 |
| FF | FF | 1 |
| C3 | C3 | 1 |
| C0 80 | C0 80 | 2 |
| 80 | 80 | 1 |

Both routes agree for these inputs. Existing conversion does not validate UTF-8, and embedded NUL truncates subsequent C-string observations. I preserve those facts rather than inventing a stricter native conversion policy. This does not qualify arbitrary binary STRING use across every other host boundary, whose checks remain separate.

A fourth production-main test retains NUL truncation, unchanged source-array length, invalid-byte encoding/decoding and overlong-byte indexing without printing invalid text. It passes old source creation, verification and VM execution, then refuses the missing exact `vm_string_from_bytes` native ABI. Its native conversion behavior remains unqualified.

My private ZIP passes CRC verification and contains 97 independently hashed files, 74 exact text files, and seven actual older-C-created arm64 Mach-O programs. `native-products.json` independently verifies those program bytes against the captured hashes. The private binaries occupy about 59 MiB before compression; these small programs do not rebuild my compiler. This is baseline evidence from older providers, not changed-source, nvm2c, fresh-stage, Apple hosted or release qualification.
