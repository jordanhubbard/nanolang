# My raw C substring overflow baseline and correction

At `b7473bb09`, I compile only the standalone bounds.c probe, nl_string.c and utf8.c with ASan/UBSan. The original call nl_string_substring(source, 1, SIZE_MAX) wraps the unchecked addition, attempts a SIZE_MAX allocation and aborts with ASan status -6. I retain the complete error, exact command and source/product hashes; the native executable remains in my private audit directory.

After changing the raw clamp to length > source_length - start, the same input builds and executes with status zero under the same sanitizer flags. I retain both command sets, hashes and outputs. The original full C string unit suite also passes locally under ASan/UBSan, including new tail/SIZE_MAX, beyond-SIZE_MAX, end and zero-length assertions. I disable leak detection on Darwin as in my existing sanitizer targets.

These small standalone checks do not rebuild my providers or establish SDK, source-stage or release qualification. Hosted Linux/Apple gates still must qualify this correction. The separately measured generated C signed-overflow baseline remains in ../e9-gcc-early and has its own O0/O2 production-native gate.
