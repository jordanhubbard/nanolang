# My C-seed module-global identity checkpoint

My baseline imports two modules with private globals named answer and distinct int/string types. Both public accessors retain their source names. C-seed shadow bytecode returns the second module string from the first module integer accessor, aborting before publication.

I key bytecode global storage by cached module AST and declaration identity. Plain lookup uses the current declaring program, which I preserve through module initializers, function bodies and shadows alongside the source-file/module context. Repeated imports register the cached declaration once.

The unchanged fixture then passes checked shadows and NanoVM, exposing a separate native failure: nvm2c gives both accessors the C name nl_read_answer. I retain that strict-C terminal. I use function-index names for duplicate, invalid, reserved-prefix or overlong debug names, retaining ordinary unique names. Generated fallback names cannot collide with source names using that reserved prefix.

My three expanded source methods pass checked shadows, verified VM and strict C11 ASan/UBSan/LSan native products in 1.112 seconds. They cover distinct module/root types, module-local initializer references, repeated imports, reserved-looking source names and long distinct function names. All 90 bytecode-generation checks and 2,435 native checks pass. I retain the complete logs.

Public C-seed global import aliases remain unimplemented under #986. This prerequisite does not establish all initialization effects, public global mutation/callable/aggregate behavior, the final compiler fixed point or Linux qualification.
