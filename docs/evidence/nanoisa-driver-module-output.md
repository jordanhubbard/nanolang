# My explicit NanoISA module output

I accept `bin/nanoisa_emit source.nano --emit-nvm -o output.nvm`.
I retain my assembly output when the flag is absent. My driver remains hosted
by my C seed; this bridge does not establish a NanoISA-only bootstrap or extend
my source lowering subset.

I assemble in memory, verify the self-contained module, and serialize canonical
v2 bytes before opening an output staging file. I use an exclusive random file
beside the destination, finish writing and syncing it, then rename it over the
destination. I remove staging files on failure. This provides atomic replacement;
I do not claim parent-directory crash durability or preserve an old file's mode.
New output files use the staging file's private permissions.

I checked `make test-nanoisa test-nanoisa-emit-driver test-nanoisa-src-nano`:

- 2691 NanoISA assertions pass, including malformed assembly, failed verification,
  successful replacement, failed rename, and staging cleanup.
- Three driver tests check repeated identical v2 bytes, VM/native output parity,
  unchanged assembly mode, equivalent assembler bytes, malformed and unsupported
  input rejection, missing output paths, and preservation of prior output.
- 86 pinned source lowering cases and two flat-record tests pass.

My native execution test compiles generated C with C11 and warnings as errors.
The verification happens before publication; rejected source cannot truncate an
existing module. I use no assembly scratch file or predictable staging path.
