#ifndef NANOLANG_MODULE_LINK_RESPONSE_H
#define NANOLANG_MODULE_LINK_RESPONSE_H

#include "module_builder.h"

/* Internal graph-capture boundary. A caller must identify its selected linker
 * before choosing a grammar; capture alone does not authorize cache reuse. */
typedef enum { MODULE_LINK_RESPONSE_GNU = 1, MODULE_LINK_RESPONSE_APPLE = 2 } ModuleLinkResponseGrammar;

/* I return an owned path to a retained response graph, or NULL on failure.
 * The caller frees the path string; retained files live with the module cache. */
char *module_capture_link_response(const ModuleBuildMetadata *meta, const char *source,
                                   ModuleLinkResponseGrammar grammar);

/* I query the selected linker through a complete literal compiler command.
 * The caller must supply disposable inputs/outputs: a version request can
 * still execute compilation or linking. Never pass a published artifact path.
 * Zero means unrecognized, invalid, failed, oversized or timed out. This
 * identifies a supported tool contract, not the authenticity of a toolchain. */
ModuleLinkResponseGrammar module_query_link_response_grammar(const char *command);

#endif
