/*
 * wrapper_gen.h - Generate native binary wrappers for .nvm bytecode
 *
 * Produces a standalone native executable that embeds the .nvm bytecode
 * blob and a minimal VM runtime, similar to how nanoc produces native
 * binaries from transpiled C code.
 */

#ifndef NANOVIRT_WRAPPER_GEN_H
#define NANOVIRT_WRAPPER_GEN_H

#include "../nanoisa/nvm_format.h"
#include "../nanolang.h"
#include <stdbool.h>
#include <stdint.h>

/*
 * Generate a native executable that embeds the given .nvm bytecode blob.
 *
 * module:      The compiled NVM module (for import table inspection)
 * blob:        The serialized .nvm bytecode
 * blob_size:   Size of the blob in bytes
 * output_path: Path for the output native binary
 * source_path: Original .nano source path (for error messages)
 * program:     The AST (for scanning AST_IMPORT nodes)
 * verbose:     Print compilation commands
 *
 * Returns true on success.
 */
bool wrapper_generate(const NvmModule *module, const uint8_t *blob, uint32_t blob_size,
                      const char *output_path, const char *source_path,
                      const ASTNode *program, bool verbose);

/*
 * Generate a daemon-mode native executable.
 * The resulting binary embeds the .nvm blob but is much smaller because it
 * only links the VMD client library (vmd_protocol.o + vmd_client.o).
 * At runtime it connects to the nano_vmd daemon (lazy-launching if needed)
 * and sends the blob for execution.
 *
 * Returns true on success.
 */
bool wrapper_generate_daemon(const uint8_t *blob, uint32_t blob_size,
                              const char *output_path, bool verbose);

#endif /* NANOVIRT_WRAPPER_GEN_H */
