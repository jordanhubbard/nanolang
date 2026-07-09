/*
 * llvm_backend.h — LLVM IR emitter for nanolang
 *
 * Emits LLVM IR (.ll) from a nanolang AST.
 * The resulting .ll file can be compiled to native code with:
 *   llc -march=aarch64 prog.ll -o prog.s && clang prog.s -o prog
 *   llc -march=x86-64  prog.ll -o prog.s && clang prog.s -o prog
 *   clang -O2 prog.ll -o prog   (direct compilation)
 *
 * Compared to the C transpiler path, this emits structured SSA form
 * without an intermediate C compiler step — enabling direct LLVM
 * optimization passes (mem2reg, inlining, vectorization).
 *
 * Usage:
 *   int rc = llvm_backend_emit(program, "output.ll", "source.nano", verbose);
 */

#ifndef NANOLANG_LLVM_BACKEND_H
#define NANOLANG_LLVM_BACKEND_H

#include "nanolang.h"

/*
 * llvm_backend_emit — generate LLVM IR from a nanolang AST.
 *
 * program     — parsed + type-checked AST
 * out_path    — output .ll file path
 * source_file — original .nano path (for !DIFile metadata)
 * verbose     — print progress
 * debug       — emit DWARF v4 debug metadata (!DICompileUnit, !DISubprogram)
 *
 * Returns 0 on success, non-zero on error.
 */
int llvm_backend_emit(ASTNode *program, const char *out_path,
                       const char *source_file, bool verbose, bool debug);

#endif /* NANOLANG_LLVM_BACKEND_H */
