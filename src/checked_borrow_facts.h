#ifndef NANOLANG_CHECKED_BORROW_FACTS_H
#define NANOLANG_CHECKED_BORROW_FACTS_H
#include "nanolang.h"

/* I own this scope only through one Environment, never through AST copies. */
typedef struct CheckedBorrowLedger CheckedBorrowLedger;
typedef struct {
    const ASTNode *function, *call, *argument, *place, *callee_expression;
    ASTNodeType callee_kind;
    uint64_t epoch;
    uint32_t ordinal, call_line, call_column, argument_line, argument_column;
    unsigned mode;
    const char *source_path, *root, *projection, *callee_name, *callee_owner;
} CheckedBorrowEvent;

typedef struct {
    const ASTNode *function, *field, *root_node, *nodes[32];
    uint64_t epoch;
    uint32_t line, column;
    unsigned category, depth;
    const char *source_path, *root, *projection;
} CheckedFieldReadEvent;
bool checked_field_read_capture(Environment *env, ASTNode *function, ASTNode *field,
    ASTNode *root_node, unsigned category, const char *root, const char *projection);
const CheckedFieldReadEvent *checked_field_read_lookup(const Environment *env,
    const ASTNode *function, const ASTNode *field);

typedef struct {
    const ASTNode *function, *assignment, *rhs;
    ASTNodeType rhs_kind;
    uint64_t epoch;
    uint32_t line, column;
    const char *source_path, *root, *projection;
} CheckedFieldWriteEvent;
bool checked_field_write_capture(Environment *env, ASTNode *function,
    ASTNode *assignment, const char *root, const char *projection);
const CheckedFieldWriteEvent *checked_field_write_lookup(const Environment *env,
    const ASTNode *function, const ASTNode *assignment);

bool checked_borrow_begin(Environment *env);
void checked_borrow_finish_capture(Environment *env);
bool checked_borrow_seal(Environment *env);
bool checked_borrow_is_sealed(const Environment *env);
void checked_borrow_end(Environment *env);
/* I use these only at actual ownership-check boundaries. */
void checked_borrow_function_begin(Environment *env, ASTNode *function);
bool checked_borrow_capture(Environment *env, ASTNode *function, ASTNode *call,
    ASTNode *argument, unsigned ordinal, const char *root, const char *projection);
/* I return only an exact sealed successful source event; absent is not authority. */
const CheckedBorrowEvent *checked_borrow_lookup(const Environment *env,
    const ASTNode *function, const ASTNode *call, const ASTNode *argument, unsigned ordinal);
#endif
