#ifndef NANOLANG_DOCGEN_H
#define NANOLANG_DOCGEN_H

#include "nanolang.h"
#include <stdbool.h>

/*
 * emit_doc_html — generate HTML API documentation for a nanolang module.
 *
 * output_path  path to write the .html file
 * program      parsed + type-checked AST (AST_PROGRAM node)
 * source_text  raw source text (used for /// comment extraction)
 * module_name  module name used in the <title> and heading
 *
 * Returns true on success, false on I/O error.
 */
bool emit_doc_html(const char *output_path, ASTNode *program,
                   const char *source_text, const char *module_name);

#endif /* NANOLANG_DOCGEN_H */
