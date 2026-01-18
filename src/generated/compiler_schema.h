/* AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY. */

#ifndef NANOLANG_GENERATED_COMPILER_SCHEMA_H
#define NANOLANG_GENERATED_COMPILER_SCHEMA_H

#include <stdint.h>
#include <stdbool.h>
#include "runtime/dyn_array.h"

/* Forward declare List types */
#ifndef FORWARD_DEFINED_List_ASTArrayLiteral
#define FORWARD_DEFINED_List_ASTArrayLiteral
typedef struct List_ASTArrayLiteral List_ASTArrayLiteral;
#endif
#ifndef FORWARD_DEFINED_List_ASTAssert
#define FORWARD_DEFINED_List_ASTAssert
typedef struct List_ASTAssert List_ASTAssert;
#endif
#ifndef FORWARD_DEFINED_List_ASTBinaryOp
#define FORWARD_DEFINED_List_ASTBinaryOp
typedef struct List_ASTBinaryOp List_ASTBinaryOp;
#endif
#ifndef FORWARD_DEFINED_List_ASTBlock
#define FORWARD_DEFINED_List_ASTBlock
typedef struct List_ASTBlock List_ASTBlock;
#endif
#ifndef FORWARD_DEFINED_List_ASTBool
#define FORWARD_DEFINED_List_ASTBool
typedef struct List_ASTBool List_ASTBool;
#endif
#ifndef FORWARD_DEFINED_List_ASTCall
#define FORWARD_DEFINED_List_ASTCall
typedef struct List_ASTCall List_ASTCall;
#endif
#ifndef FORWARD_DEFINED_List_ASTEnum
#define FORWARD_DEFINED_List_ASTEnum
typedef struct List_ASTEnum List_ASTEnum;
#endif
#ifndef FORWARD_DEFINED_List_ASTFieldAccess
#define FORWARD_DEFINED_List_ASTFieldAccess
typedef struct List_ASTFieldAccess List_ASTFieldAccess;
#endif
#ifndef FORWARD_DEFINED_List_ASTFloat
#define FORWARD_DEFINED_List_ASTFloat
typedef struct List_ASTFloat List_ASTFloat;
#endif
#ifndef FORWARD_DEFINED_List_ASTFor
#define FORWARD_DEFINED_List_ASTFor
typedef struct List_ASTFor List_ASTFor;
#endif
#ifndef FORWARD_DEFINED_List_ASTFunction
#define FORWARD_DEFINED_List_ASTFunction
typedef struct List_ASTFunction List_ASTFunction;
#endif
#ifndef FORWARD_DEFINED_List_ASTIdentifier
#define FORWARD_DEFINED_List_ASTIdentifier
typedef struct List_ASTIdentifier List_ASTIdentifier;
#endif
#ifndef FORWARD_DEFINED_List_ASTIf
#define FORWARD_DEFINED_List_ASTIf
typedef struct List_ASTIf List_ASTIf;
#endif
#ifndef FORWARD_DEFINED_List_ASTImport
#define FORWARD_DEFINED_List_ASTImport
typedef struct List_ASTImport List_ASTImport;
#endif
#ifndef FORWARD_DEFINED_List_ASTLet
#define FORWARD_DEFINED_List_ASTLet
typedef struct List_ASTLet List_ASTLet;
#endif
#ifndef FORWARD_DEFINED_List_ASTMatch
#define FORWARD_DEFINED_List_ASTMatch
typedef struct List_ASTMatch List_ASTMatch;
#endif
#ifndef FORWARD_DEFINED_List_ASTNumber
#define FORWARD_DEFINED_List_ASTNumber
typedef struct List_ASTNumber List_ASTNumber;
#endif
#ifndef FORWARD_DEFINED_List_ASTOpaqueType
#define FORWARD_DEFINED_List_ASTOpaqueType
typedef struct List_ASTOpaqueType List_ASTOpaqueType;
#endif
#ifndef FORWARD_DEFINED_List_ASTPrint
#define FORWARD_DEFINED_List_ASTPrint
typedef struct List_ASTPrint List_ASTPrint;
#endif
#ifndef FORWARD_DEFINED_List_ASTReturn
#define FORWARD_DEFINED_List_ASTReturn
typedef struct List_ASTReturn List_ASTReturn;
#endif
#ifndef FORWARD_DEFINED_List_ASTSet
#define FORWARD_DEFINED_List_ASTSet
typedef struct List_ASTSet List_ASTSet;
#endif
#ifndef FORWARD_DEFINED_List_ASTShadow
#define FORWARD_DEFINED_List_ASTShadow
typedef struct List_ASTShadow List_ASTShadow;
#endif
#ifndef FORWARD_DEFINED_List_ASTStmtRef
#define FORWARD_DEFINED_List_ASTStmtRef
typedef struct List_ASTStmtRef List_ASTStmtRef;
#endif
#ifndef FORWARD_DEFINED_List_ASTString
#define FORWARD_DEFINED_List_ASTString
typedef struct List_ASTString List_ASTString;
#endif
#ifndef FORWARD_DEFINED_List_ASTStruct
#define FORWARD_DEFINED_List_ASTStruct
typedef struct List_ASTStruct List_ASTStruct;
#endif
#ifndef FORWARD_DEFINED_List_ASTStructLiteral
#define FORWARD_DEFINED_List_ASTStructLiteral
typedef struct List_ASTStructLiteral List_ASTStructLiteral;
#endif
#ifndef FORWARD_DEFINED_List_ASTTupleIndex
#define FORWARD_DEFINED_List_ASTTupleIndex
typedef struct List_ASTTupleIndex List_ASTTupleIndex;
#endif
#ifndef FORWARD_DEFINED_List_ASTTupleLiteral
#define FORWARD_DEFINED_List_ASTTupleLiteral
typedef struct List_ASTTupleLiteral List_ASTTupleLiteral;
#endif
#ifndef FORWARD_DEFINED_List_ASTUnion
#define FORWARD_DEFINED_List_ASTUnion
typedef struct List_ASTUnion List_ASTUnion;
#endif
#ifndef FORWARD_DEFINED_List_ASTUnionConstruct
#define FORWARD_DEFINED_List_ASTUnionConstruct
typedef struct List_ASTUnionConstruct List_ASTUnionConstruct;
#endif
#ifndef FORWARD_DEFINED_List_ASTUnsafeBlock
#define FORWARD_DEFINED_List_ASTUnsafeBlock
typedef struct List_ASTUnsafeBlock List_ASTUnsafeBlock;
#endif
#ifndef FORWARD_DEFINED_List_ASTWhile
#define FORWARD_DEFINED_List_ASTWhile
typedef struct List_ASTWhile List_ASTWhile;
#endif
#ifndef FORWARD_DEFINED_List_CompilerDiagnostic
#define FORWARD_DEFINED_List_CompilerDiagnostic
typedef struct List_CompilerDiagnostic List_CompilerDiagnostic;
#endif
#ifndef FORWARD_DEFINED_List_LexerToken
#define FORWARD_DEFINED_List_LexerToken
typedef struct List_LexerToken List_LexerToken;
#endif

#ifndef DEFINED_DiagnosticSeverity
#define DEFINED_DiagnosticSeverity
typedef enum DiagnosticSeverity {
    DiagnosticSeverity_DIAG_INFO,
    DiagnosticSeverity_DIAG_WARNING,
    DiagnosticSeverity_DIAG_ERROR
} DiagnosticSeverity;
#endif

#ifndef DEFINED_CompilerPhase
#define DEFINED_CompilerPhase
typedef enum CompilerPhase {
    CompilerPhase_PHASE_LEXER,
    CompilerPhase_PHASE_PARSER,
    CompilerPhase_PHASE_TYPECHECK,
    CompilerPhase_PHASE_TRANSPILER,
    CompilerPhase_PHASE_RUNTIME
} CompilerPhase;
#endif

typedef enum {
    TOKEN_EOF = 0,
    TOKEN_NUMBER = 1,
    TOKEN_FLOAT = 2,
    TOKEN_STRING = 3,
    TOKEN_IDENTIFIER = 4,
    TOKEN_TRUE = 5,
    TOKEN_FALSE = 6,
    TOKEN_LPAREN = 7,
    TOKEN_RPAREN = 8,
    TOKEN_LBRACE = 9,
    TOKEN_RBRACE = 10,
    TOKEN_LBRACKET = 11,
    TOKEN_RBRACKET = 12,
    TOKEN_COMMA = 13,
    TOKEN_COLON = 14,
    TOKEN_DOUBLE_COLON = 15,
    TOKEN_ARROW = 16,
    TOKEN_ASSIGN = 17,
    TOKEN_DOT = 18,
    TOKEN_MODULE = 19,
    TOKEN_PUB = 20,
    TOKEN_FROM = 21,
    TOKEN_USE = 22,
    TOKEN_EXTERN = 23,
    TOKEN_FN = 24,
    TOKEN_LET = 25,
    TOKEN_MUT = 26,
    TOKEN_SET = 27,
    TOKEN_IF = 28,
    TOKEN_ELSE = 29,
    TOKEN_COND = 30,
    TOKEN_WHILE = 31,
    TOKEN_FOR = 32,
    TOKEN_IN = 33,
    TOKEN_RETURN = 34,
    TOKEN_BREAK = 35,
    TOKEN_CONTINUE = 36,
    TOKEN_ASSERT = 37,
    TOKEN_SHADOW = 38,
    TOKEN_REQUIRES = 39,
    TOKEN_ENSURES = 40,
    TOKEN_PRINT = 41,
    TOKEN_ARRAY = 42,
    TOKEN_STRUCT = 43,
    TOKEN_ENUM = 44,
    TOKEN_UNION = 45,
    TOKEN_MATCH = 46,
    TOKEN_IMPORT = 47,
    TOKEN_AS = 48,
    TOKEN_OPAQUE = 49,
    TOKEN_TYPE_INT = 50,
    TOKEN_TYPE_U8 = 51,
    TOKEN_TYPE_FLOAT = 52,
    TOKEN_TYPE_BOOL = 53,
    TOKEN_TYPE_STRING = 54,
    TOKEN_TYPE_BSTRING = 55,
    TOKEN_TYPE_VOID = 56,
    TOKEN_PLUS = 57,
    TOKEN_MINUS = 58,
    TOKEN_STAR = 59,
    TOKEN_SLASH = 60,
    TOKEN_PERCENT = 61,
    TOKEN_EQ = 62,
    TOKEN_NE = 63,
    TOKEN_LT = 64,
    TOKEN_LE = 65,
    TOKEN_GT = 66,
    TOKEN_GE = 67,
    TOKEN_AND = 68,
    TOKEN_OR = 69,
    TOKEN_NOT = 70,
    TOKEN_RANGE = 71,
    TOKEN_UNSAFE = 72,
    TOKEN_RESOURCE = 73
} TokenType;

typedef enum {
    PNODE_NUMBER = 0,
    PNODE_FLOAT = 1,
    PNODE_STRING = 2,
    PNODE_BOOL = 3,
    PNODE_IDENTIFIER = 4,
    PNODE_BINARY_OP = 5,
    PNODE_CALL = 6,
    PNODE_ARRAY_LITERAL = 7,
    PNODE_LET = 8,
    PNODE_SET = 9,
    PNODE_IF = 10,
    PNODE_COND = 11,
    PNODE_WHILE = 12,
    PNODE_FOR = 13,
    PNODE_RETURN = 14,
    PNODE_BREAK = 15,
    PNODE_CONTINUE = 16,
    PNODE_BLOCK = 17,
    PNODE_PRINT = 18,
    PNODE_ASSERT = 19,
    PNODE_PROGRAM = 20,
    PNODE_FUNCTION = 21,
    PNODE_SHADOW = 22,
    PNODE_STRUCT_DEF = 23,
    PNODE_STRUCT_LITERAL = 24,
    PNODE_FIELD_ACCESS = 25,
    PNODE_ENUM_DEF = 26,
    PNODE_UNION_DEF = 27,
    PNODE_UNION_CONSTRUCT = 28,
    PNODE_MATCH = 29,
    PNODE_IMPORT = 30,
    PNODE_OPAQUE_TYPE = 31,
    PNODE_TUPLE_LITERAL = 32,
    PNODE_TUPLE_INDEX = 33,
    PNODE_STRUCT = 34,
    PNODE_ENUM = 35,
    PNODE_UNION = 36,
    PNODE_UNSAFE_BLOCK = 37
} ParseNodeType;

#ifndef DEFINED_nl_LexerToken
#define DEFINED_nl_LexerToken
typedef struct nl_LexerToken {
    int token_type;
    const char * value;
    int line;
    int column;
} nl_LexerToken;
typedef nl_LexerToken LexerToken;
typedef nl_LexerToken Token;
#endif

#ifndef DEFINED_nl_ParseNode
#define DEFINED_nl_ParseNode
typedef struct nl_ParseNode {
    int node_type;
    int line;
    int column;
} nl_ParseNode;
typedef nl_ParseNode ParseNode;
#endif

#ifndef DEFINED_nl_ASTNumber
#define DEFINED_nl_ASTNumber
typedef struct nl_ASTNumber {
    int node_type;
    int line;
    int column;
    const char * value;
} nl_ASTNumber;
typedef nl_ASTNumber ASTNumber;
#endif

#ifndef DEFINED_nl_ASTFloat
#define DEFINED_nl_ASTFloat
typedef struct nl_ASTFloat {
    int node_type;
    int line;
    int column;
    const char * value;
} nl_ASTFloat;
typedef nl_ASTFloat ASTFloat;
#endif

#ifndef DEFINED_nl_ASTString
#define DEFINED_nl_ASTString
typedef struct nl_ASTString {
    int node_type;
    int line;
    int column;
    const char * value;
} nl_ASTString;
typedef nl_ASTString ASTString;
#endif

#ifndef DEFINED_nl_ASTBool
#define DEFINED_nl_ASTBool
typedef struct nl_ASTBool {
    int node_type;
    int line;
    int column;
    bool value;
} nl_ASTBool;
typedef nl_ASTBool ASTBool;
#endif

#ifndef DEFINED_nl_ASTIdentifier
#define DEFINED_nl_ASTIdentifier
typedef struct nl_ASTIdentifier {
    int node_type;
    int line;
    int column;
    const char * name;
} nl_ASTIdentifier;
typedef nl_ASTIdentifier ASTIdentifier;
#endif

#ifndef DEFINED_nl_ASTBinaryOp
#define DEFINED_nl_ASTBinaryOp
typedef struct nl_ASTBinaryOp {
    int node_type;
    int line;
    int column;
    int op;
    int left;
    int right;
    int left_type;
    int right_type;
} nl_ASTBinaryOp;
typedef nl_ASTBinaryOp ASTBinaryOp;
#endif

#ifndef DEFINED_nl_ASTCall
#define DEFINED_nl_ASTCall
typedef struct nl_ASTCall {
    int node_type;
    int line;
    int column;
    int function;
    int arg_start;
    int arg_count;
} nl_ASTCall;
typedef nl_ASTCall ASTCall;
#endif

#ifndef DEFINED_nl_ASTArrayLiteral
#define DEFINED_nl_ASTArrayLiteral
typedef struct nl_ASTArrayLiteral {
    int node_type;
    int line;
    int column;
    const char * element_type;
    int element_start;
    int element_count;
} nl_ASTArrayLiteral;
typedef nl_ASTArrayLiteral ASTArrayLiteral;
#endif

#ifndef DEFINED_nl_ASTLet
#define DEFINED_nl_ASTLet
typedef struct nl_ASTLet {
    int node_type;
    int line;
    int column;
    const char * name;
    const char * var_type;
    int value;
    int value_type;
    bool is_mut;
} nl_ASTLet;
typedef nl_ASTLet ASTLet;
#endif

#ifndef DEFINED_nl_ASTSet
#define DEFINED_nl_ASTSet
typedef struct nl_ASTSet {
    int node_type;
    int line;
    int column;
    const char * target;
    int value;
    int value_type;
} nl_ASTSet;
typedef nl_ASTSet ASTSet;
#endif

#ifndef DEFINED_nl_ASTStmtRef
#define DEFINED_nl_ASTStmtRef
typedef struct nl_ASTStmtRef {
    int node_id;
    int node_type;
} nl_ASTStmtRef;
typedef nl_ASTStmtRef ASTStmtRef;
#endif

#ifndef DEFINED_nl_ASTIf
#define DEFINED_nl_ASTIf
typedef struct nl_ASTIf {
    int node_type;
    int line;
    int column;
    int condition;
    int condition_type;
    int then_body;
    int else_body;
} nl_ASTIf;
typedef nl_ASTIf ASTIf;
#endif

#ifndef DEFINED_nl_ASTWhile
#define DEFINED_nl_ASTWhile
typedef struct nl_ASTWhile {
    int node_type;
    int line;
    int column;
    int condition;
    int condition_type;
    int body;
} nl_ASTWhile;
typedef nl_ASTWhile ASTWhile;
#endif

#ifndef DEFINED_nl_ASTFor
#define DEFINED_nl_ASTFor
typedef struct nl_ASTFor {
    int node_type;
    int line;
    int column;
    const char * var_name;
    int iterable;
    int iterable_type;
    int body;
} nl_ASTFor;
typedef nl_ASTFor ASTFor;
#endif

#ifndef DEFINED_nl_ASTReturn
#define DEFINED_nl_ASTReturn
typedef struct nl_ASTReturn {
    int node_type;
    int line;
    int column;
    int value;
    int value_type;
} nl_ASTReturn;
typedef nl_ASTReturn ASTReturn;
#endif

#ifndef DEFINED_nl_ASTBlock
#define DEFINED_nl_ASTBlock
typedef struct nl_ASTBlock {
    int node_type;
    int line;
    int column;
    List_ASTStmtRef * statements;
} nl_ASTBlock;
typedef nl_ASTBlock ASTBlock;
#endif

#ifndef DEFINED_nl_ASTUnsafeBlock
#define DEFINED_nl_ASTUnsafeBlock
typedef struct nl_ASTUnsafeBlock {
    int node_type;
    int line;
    int column;
    List_ASTStmtRef * statements;
} nl_ASTUnsafeBlock;
typedef nl_ASTUnsafeBlock ASTUnsafeBlock;
#endif

#ifndef DEFINED_nl_ASTPrint
#define DEFINED_nl_ASTPrint
typedef struct nl_ASTPrint {
    int node_type;
    int line;
    int column;
    int value;
    int value_type;
} nl_ASTPrint;
typedef nl_ASTPrint ASTPrint;
#endif

#ifndef DEFINED_nl_ASTAssert
#define DEFINED_nl_ASTAssert
typedef struct nl_ASTAssert {
    int node_type;
    int line;
    int column;
    int condition;
    int condition_type;
} nl_ASTAssert;
typedef nl_ASTAssert ASTAssert;
#endif

#ifndef DEFINED_nl_ASTFunction
#define DEFINED_nl_ASTFunction
typedef struct nl_ASTFunction {
    int node_type;
    int line;
    int column;
    const char * name;
    int param_start;
    int param_count;
    const char * return_type;
    int body;
} nl_ASTFunction;
typedef nl_ASTFunction ASTFunction;
#endif

#ifndef DEFINED_nl_ASTShadow
#define DEFINED_nl_ASTShadow
typedef struct nl_ASTShadow {
    int node_type;
    int line;
    int column;
    const char * target_name;
    int body;
} nl_ASTShadow;
typedef nl_ASTShadow ASTShadow;
#endif

#ifndef DEFINED_nl_ASTStruct
#define DEFINED_nl_ASTStruct
typedef struct nl_ASTStruct {
    int node_type;
    int line;
    int column;
    const char * name;
    int field_start;
    int field_count;
    DynArray * field_names;
    DynArray * field_types;
} nl_ASTStruct;
typedef nl_ASTStruct ASTStruct;
#endif

#ifndef DEFINED_nl_ASTStructLiteral
#define DEFINED_nl_ASTStructLiteral
typedef struct nl_ASTStructLiteral {
    int node_type;
    int line;
    int column;
    const char * struct_name;
    DynArray * field_names;
    DynArray * field_value_ids;
    DynArray * field_value_types;
    int field_count;
} nl_ASTStructLiteral;
typedef nl_ASTStructLiteral ASTStructLiteral;
#endif

#ifndef DEFINED_nl_ASTFieldAccess
#define DEFINED_nl_ASTFieldAccess
typedef struct nl_ASTFieldAccess {
    int node_type;
    int line;
    int column;
    int object;
    int object_type;
    const char * field_name;
} nl_ASTFieldAccess;
typedef nl_ASTFieldAccess ASTFieldAccess;
#endif

#ifndef DEFINED_nl_ASTEnum
#define DEFINED_nl_ASTEnum
typedef struct nl_ASTEnum {
    int node_type;
    int line;
    int column;
    const char * name;
    int variant_count;
} nl_ASTEnum;
typedef nl_ASTEnum ASTEnum;
#endif

#ifndef DEFINED_nl_ASTUnion
#define DEFINED_nl_ASTUnion
typedef struct nl_ASTUnion {
    int node_type;
    int line;
    int column;
    const char * name;
    int variant_count;
} nl_ASTUnion;
typedef nl_ASTUnion ASTUnion;
#endif

#ifndef DEFINED_nl_ASTUnionConstruct
#define DEFINED_nl_ASTUnionConstruct
typedef struct nl_ASTUnionConstruct {
    int node_type;
    int line;
    int column;
    const char * union_name;
    const char * variant_name;
    DynArray * field_names;
    DynArray * field_value_ids;
    DynArray * field_value_types;
    int field_count;
} nl_ASTUnionConstruct;
typedef nl_ASTUnionConstruct ASTUnionConstruct;
#endif

#ifndef DEFINED_nl_ASTMatchArm
#define DEFINED_nl_ASTMatchArm
typedef struct nl_ASTMatchArm {
    const char * variant_name;
    const char * binding_name;
    int body_id;
    int body_type;
} nl_ASTMatchArm;
typedef nl_ASTMatchArm ASTMatchArm;
#endif

#ifndef DEFINED_nl_ASTMatch
#define DEFINED_nl_ASTMatch
typedef struct nl_ASTMatch {
    int node_type;
    int line;
    int column;
    int scrutinee;
    int scrutinee_type;
    DynArray * arm_variants;
    DynArray * arm_bindings;
    DynArray * arm_body_ids;
    DynArray * arm_body_types;
    int arm_count;
} nl_ASTMatch;
typedef nl_ASTMatch ASTMatch;
#endif

#ifndef DEFINED_nl_ASTImport
#define DEFINED_nl_ASTImport
typedef struct nl_ASTImport {
    int node_type;
    int line;
    int column;
    const char * module_path;
    const char * module_name;
    bool is_unsafe;
    bool is_selective;
    bool is_wildcard;
    bool is_pub_use;
    DynArray * import_symbols;
    DynArray * import_aliases;
    int import_symbol_count;
} nl_ASTImport;
typedef nl_ASTImport ASTImport;
#endif

#ifndef DEFINED_nl_ASTOpaqueType
#define DEFINED_nl_ASTOpaqueType
typedef struct nl_ASTOpaqueType {
    int node_type;
    int line;
    int column;
    const char * type_name;
} nl_ASTOpaqueType;
typedef nl_ASTOpaqueType ASTOpaqueType;
#endif

#ifndef DEFINED_nl_ASTTupleLiteral
#define DEFINED_nl_ASTTupleLiteral
typedef struct nl_ASTTupleLiteral {
    int node_type;
    int line;
    int column;
    DynArray * element_ids;
    DynArray * element_types;
    int element_count;
} nl_ASTTupleLiteral;
typedef nl_ASTTupleLiteral ASTTupleLiteral;
#endif

#ifndef DEFINED_nl_ASTTupleIndex
#define DEFINED_nl_ASTTupleIndex
typedef struct nl_ASTTupleIndex {
    int node_type;
    int line;
    int column;
    int tuple;
    int tuple_type;
    int index;
} nl_ASTTupleIndex;
typedef nl_ASTTupleIndex ASTTupleIndex;
#endif

#ifndef DEFINED_nl_Parser
#define DEFINED_nl_Parser
typedef struct nl_Parser {
    List_LexerToken * tokens;
    const char * file_name;
    int position;
    int token_count;
    bool has_error;
    List_CompilerDiagnostic * diagnostics;
    List_ASTNumber * numbers;
    List_ASTFloat * floats;
    List_ASTString * strings;
    List_ASTBool * bools;
    List_ASTIdentifier * identifiers;
    List_ASTBinaryOp * binary_ops;
    List_ASTCall * calls;
    List_ASTStmtRef * call_args;
    List_ASTStmtRef * array_elements;
    List_ASTArrayLiteral * array_literals;
    List_ASTLet * lets;
    List_ASTSet * sets;
    List_ASTIf * ifs;
    List_ASTWhile * whiles;
    List_ASTFor * fors;
    List_ASTReturn * returns;
    List_ASTBlock * blocks;
    List_ASTUnsafeBlock * unsafe_blocks;
    List_ASTPrint * prints;
    List_ASTAssert * asserts;
    List_ASTFunction * functions;
    List_ASTShadow * shadows;
    List_ASTStruct * structs;
    List_ASTStructLiteral * struct_literals;
    List_ASTFieldAccess * field_accesses;
    List_ASTEnum * enums;
    List_ASTUnion * unions;
    List_ASTUnionConstruct * union_constructs;
    List_ASTMatch * matches;
    List_ASTImport * imports;
    List_ASTOpaqueType * opaque_types;
    List_ASTTupleLiteral * tuple_literals;
    List_ASTTupleIndex * tuple_indices;
    int numbers_count;
    int floats_count;
    int strings_count;
    int bools_count;
    int identifiers_count;
    int binary_ops_count;
    int calls_count;
    int array_literals_count;
    int lets_count;
    int sets_count;
    int ifs_count;
    int whiles_count;
    int fors_count;
    int returns_count;
    int blocks_count;
    int unsafe_blocks_count;
    int prints_count;
    int asserts_count;
    int functions_count;
    int shadows_count;
    int structs_count;
    int struct_literals_count;
    int field_accesses_count;
    int enums_count;
    int unions_count;
    int union_constructs_count;
    int matches_count;
    int imports_count;
    int opaque_types_count;
    int tuple_literals_count;
    int tuple_indices_count;
    int next_node_id;
    int last_expr_node_id;
    int last_expr_node_type;
} nl_Parser;
typedef nl_Parser Parser;
#endif

#ifndef DEFINED_nl_NSType
#define DEFINED_nl_NSType
typedef struct nl_NSType {
    int kind;
    const char * name;
    int element_type_kind;
    const char * element_type_name;
} nl_NSType;
typedef nl_NSType NSType;
#endif

#ifndef DEFINED_nl_OptionType
#define DEFINED_nl_OptionType
typedef struct nl_OptionType {
    bool has_value;
    NSType value;
} nl_OptionType;
typedef nl_OptionType OptionType;
#endif

#ifndef DEFINED_nl_CompilerSourceLocation
#define DEFINED_nl_CompilerSourceLocation
typedef struct nl_CompilerSourceLocation {
    const char * file;
    int line;
    int column;
} nl_CompilerSourceLocation;
typedef nl_CompilerSourceLocation CompilerSourceLocation;
#endif

#ifndef DEFINED_nl_CompilerDiagnostic
#define DEFINED_nl_CompilerDiagnostic
typedef struct nl_CompilerDiagnostic {
    int phase;
    int severity;
    const char * code;
    const char * message;
    CompilerSourceLocation location;
} nl_CompilerDiagnostic;
typedef nl_CompilerDiagnostic CompilerDiagnostic;
#endif

#ifndef DEFINED_nl_LexPhaseOutput
#define DEFINED_nl_LexPhaseOutput
typedef struct nl_LexPhaseOutput {
    List_LexerToken * tokens;
    int token_count;
    List_CompilerDiagnostic * diagnostics;
    bool had_error;
} nl_LexPhaseOutput;
typedef nl_LexPhaseOutput LexPhaseOutput;
#endif

#ifndef DEFINED_nl_ParsePhaseOutput
#define DEFINED_nl_ParsePhaseOutput
typedef struct nl_ParsePhaseOutput {
    Parser parser;
    List_CompilerDiagnostic * diagnostics;
    bool had_error;
} nl_ParsePhaseOutput;
typedef nl_ParsePhaseOutput ParsePhaseOutput;
#endif

#ifndef DEFINED_nl_TypeEnvironment
#define DEFINED_nl_TypeEnvironment
typedef struct nl_TypeEnvironment {
    int error_count;
    bool has_error;
    List_CompilerDiagnostic * diagnostics;
} nl_TypeEnvironment;
typedef nl_TypeEnvironment TypeEnvironment;
#endif

#ifndef DEFINED_nl_TypecheckPhaseOutput
#define DEFINED_nl_TypecheckPhaseOutput
typedef struct nl_TypecheckPhaseOutput {
    TypeEnvironment environment;
    List_CompilerDiagnostic * diagnostics;
    bool had_error;
} nl_TypecheckPhaseOutput;
typedef nl_TypecheckPhaseOutput TypecheckPhaseOutput;
#endif

#ifndef DEFINED_nl_TranspilePhaseOutput
#define DEFINED_nl_TranspilePhaseOutput
typedef struct nl_TranspilePhaseOutput {
    const char * c_source;
    List_CompilerDiagnostic * diagnostics;
    bool had_error;
    const char * output_path;
} nl_TranspilePhaseOutput;
typedef nl_TranspilePhaseOutput TranspilePhaseOutput;
#endif

#endif /* NANOLANG_GENERATED_COMPILER_SCHEMA_H */
