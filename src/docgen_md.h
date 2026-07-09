#ifndef NANOLANG_DOCGEN_MD_H
#define NANOLANG_DOCGEN_MD_H

#include "nanolang.h"
#include <stdbool.h>

/*
 * emit_doc_md — generate GitHub-flavored Markdown API docs for a nanolang module.
 *
 * output_path  path to write the .md file
 * program      parsed + type-checked AST (AST_PROGRAM node)
 * source_text  raw source text (used for /// comment extraction)
 * module_name  module name used in the top-level heading
 *
 * Returns true on success, false on I/O error.
 */
bool emit_doc_md(const char *output_path, ASTNode *program,
                 const char *source_text, const char *module_name);

#endif /* NANOLANG_DOCGEN_MD_H */
