#ifndef NANOLANG_MODULE_LINK_RESPONSE_H
#define NANOLANG_MODULE_LINK_RESPONSE_H

#include "module_builder.h"

/* Internal graph-capture boundary. A caller must identify its selected linker
 * before choosing a grammar; capture alone does not authorize cache reuse. */
typedef enum { MODULE_LINK_RESPONSE_GNU = 1, MODULE_LINK_RESPONSE_APPLE = 2 } ModuleLinkResponseGrammar;

/* I capture 1..64 ordered roots in one transaction, sharing identity and byte
 * budgets. I return all owned path strings or NULL, never a partial set. The
 * caller frees each of count strings and the array. Input strings stay owned
 * by the caller. I freeze at most 128 source spellings per transaction, each
 * at most 4095 bytes. */
char **module_capture_link_responses(const ModuleBuildMetadata *meta, const char *const *sources,
                                     size_t count, ModuleLinkResponseGrammar grammar);

/* I return count owned literal shell fragments of ordered -Xlinker/value
 * pairs, or NULL with no partial result. I expand directly from captured
 * bytes, without publishing or reopening response sidecars. Input bounds
 * match graph capture; quoted output totals at most 64 KiB across all roots.
 * GNU repeats expand in order; Apple repeated resolved identities fail with
 * ELOOP, as do cycles. The caller frees each fragment and the array. This
 * representation does not admit arbitrary linker options or authorize reuse. */
char **module_capture_link_arguments(const char *const *sources, size_t count,
                                     ModuleLinkResponseGrammar grammar);

/* I return an owned path to a retained response graph, or NULL on failure.
 * The caller frees the path string; retained files live with the module cache. */
char *module_capture_link_response(const ModuleBuildMetadata *meta, const char *source,
                                   ModuleLinkResponseGrammar grammar);

/* I query a complete literal command with a disposable primary output beneath
 * parent. I pin driver and linker output, then remove the private directory.
 * I admit explicit literal option forms and reject unresolved responses,
 * indirect controls, plugins, auxiliary outputs and unknown switches before
 * starting the tool. I accept canonical separate -l/value pairs and a small
 * set of joined library forms; this is not a general compiler option parser.
 * The caller still owns native-input admission and trust in the configured
 * compiler/wrappers. This is not filesystem isolation or a contents check.
 * Zero means unrecognized, invalid, failed, oversized or timed out. This
 * identifies a supported tool contract, not the authenticity of a toolchain. */
ModuleLinkResponseGrammar module_query_link_response_grammar(const char *command, const char *parent);

#endif
