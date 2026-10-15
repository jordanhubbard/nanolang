# My builtin tail-call regression

My fresh bootstrap at 6c01d856e crashed in seed generation after 51 seconds.
My compiler-component sanitizer reproduction identifies a null Parameter
access in cg_generic_call. Builtin registry entries carry arity but no source
body or parameter array. Tail-call lookup queried generic specialization before
ordinary builtin lowering and dereferenced that missing array.

I reproduce this with the retained four-line string-length source. I admit
only functions with source bodies and parameter arrays into generic lookup;
builtins continue through their ordinary lowering. My shared regression suite
executes the reduced source through mandatory shadows, verification, NanoVM
and sanitized generated C. I retain the full bootstrap failure, sanitizer
trace and reduced failure rather than treating a new successful run as evidence
that the original crash was transient. Fresh bootstrap qualification remains
open until both generations complete with unchanged inputs and equal bytes.

My 14 focused methods pass in ordinary and compiler-component ASan/UBSan runs.
