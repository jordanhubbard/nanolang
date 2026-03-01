#ifndef FORTH_SEE_H
#define FORTH_SEE_H

/* Decompile the NanoISA implementation of a Forth built-in word.
 *
 * word_name  - the Forth word to look up (e.g. "dup", "+", "if")
 * nvm_path   - path to the compiled interpreter bytecode
 *              (e.g. "bin/nl_forth_interpreter_vm")
 *
 * Returns a pointer to a static buffer containing a human-readable
 * NanoISA listing.  The buffer is overwritten on each call.
 * Never returns NULL. */
const char *nl_forth_see(const char *word_name, const char *nvm_path);

#endif /* FORTH_SEE_H */
