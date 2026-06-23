# Nanolang Language Server Protocol (LSP) Implementation

## Overview

A Language Server Protocol implementation for nanolang, providing IDE features like code completion, go-to-definition, error diagnostics, and more.

## Goals

1. **IDE Integration**: Work with VS Code, Vim, Emacs, and other LSP clients
2. **Rich Features**: Completion, diagnostics, navigation, refactoring
3. **Performance**: Fast response times, incremental updates
4. **Extensibility**: Easy to add new features

## Architecture

```
┌──────────────┐                    ┌──────────────┐
│              │                    │              │
│  LSP Client  │◄──────JSON────────►│  LSP Server  │
│  (VS Code,   │     over stdio     │  (nanolang)  │
│   Vim, etc.) │                    │              │
└──────────────┘                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │   Document   │
                                    │   Manager    │
                                    └──────┬───────┘
                                           │
                   ┌───────────────────────┼───────────────────────┐
                   │                       │                       │
                   ▼                       ▼                       ▼
            ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
            │   Parser    │        │    Type     │        │  Analyzer   │
            │             │        │   Checker   │        │             │
            └─────────────┘        └─────────────┘        └─────────────┘
```

## Core Components

### 1. LSP Server

```c
// src/lsp/server.c

typedef struct {
    int input_fd;
    int output_fd;
    DocumentManager* doc_manager;
    CompletionEngine* completion;
    DiagnosticEngine* diagnostics;
} LSPServer;

void lsp_server_run(LSPServer* server) {
    while (true) {
        LSPMessage* msg = lsp_read_message(server->input_fd);
        if (!msg) break;
        
        LSPResponse* resp = lsp_handle_message(server, msg);
        lsp_send_response(server->output_fd, resp);
        
        lsp_free_message(msg);
        lsp_free_response(resp);
    }
}
```

### 2. Document Manager

Tracks open documents and their state.

```c
// src/lsp/document.c

typedef struct {
    char* uri;
    char* text;
    int version;
    Program* ast;           // Parsed AST
    TypeInfo* type_info;    // Type information
    Symbol* symbols;        // Symbol table
} Document;

typedef struct {
    Document** documents;
    int count;
} DocumentManager;

Document* doc_manager_get(DocumentManager* mgr, const char* uri);
void doc_manager_update(DocumentManager* mgr, const char* uri, 
                       const char* text, int version);
```

### 3. Completion Engine

Provides autocomplete suggestions.

```c
// src/lsp/completion.c

typedef struct {
    char* label;           // "factorial"
    char* kind;            // "function"
    char* detail;          // "(int) -> int"
    char* documentation;   // "Computes factorial of n"
    char* insert_text;     // "factorial($1)"
} CompletionItem;

CompletionItem** get_completions(Document* doc, int line, int column) {
    // 1. Find current scope
    Scope* scope = find_scope_at_position(doc, line, column);
    
    // 2. Get visible symbols
    Symbol* symbols = scope_get_visible_symbols(scope);
    
    // 3. Filter by prefix
    const char* prefix = get_prefix_at_cursor(doc, line, column);
    CompletionItem** items = filter_completions(symbols, prefix);
    
    // 4. Rank by relevance
    sort_by_relevance(items, prefix);
    
    return items;
}
```

### 4. Diagnostic Engine

Provides error and warning messages.

```c
// src/lsp/diagnostics.c

typedef struct {
    char* message;         // "Undefined variable 'x'"
    char* severity;        // "error" | "warning" | "info"
    int start_line;
    int start_column;
    int end_line;
    int end_column;
} Diagnostic;

Diagnostic** get_diagnostics(Document* doc) {
    Diagnostic** diags = NULL;
    int count = 0;
    
    // 1. Parse errors
    if (!doc->ast) {
        diags = add_parse_errors(diags, &count, doc);
    }
    
    // 2. Type errors
    TypeCheckResult* result = typecheck(doc->ast);
    if (result->has_errors) {
        diags = add_type_errors(diags, &count, result);
    }
    
    // 3. Linter warnings
    LintResult* lint = lint_ast(doc->ast);
    diags = add_lint_warnings(diags, &count, lint);
    
    return diags;
}
```

## Implemented Features

### Phase 1: Core Features

#### textDocument/didOpen
- Track opened documents
- Parse and analyze
- Send initial diagnostics

#### textDocument/didChange
- Update document content
- Incremental parsing
- Re-analyze and send diagnostics

#### textDocument/didClose
- Clean up document state

#### textDocument/completion
- Keyword completion
- Function name completion
- Variable name completion
- Type name completion
- Module member completion

#### textDocument/hover
- Show type information
- Show function signatures
- Show documentation

#### textDocument/definition
- Go to function definition
- Go to variable declaration
- Go to type definition

### Phase 2: Advanced Features

#### textDocument/references
- Find all references to symbol
- Highlight in editor

#### textDocument/rename
- Rename symbol across project
- Preview changes

#### textDocument/formatting
- Auto-format code
- Consistent style

#### textDocument/codeAction
- Quick fixes
- Refactoring suggestions

#### textDocument/documentSymbol
- Outline view
- Navigate by symbols

### Phase 3: Extended Features

#### workspace/symbol
- Project-wide symbol search

#### textDocument/signatureHelp
- Function parameter hints

#### textDocument/semanticTokens
- Rich syntax highlighting

#### textDocument/inlayHint
- Type hints
- Parameter names

## Implementation Details

### JSON-RPC Protocol

```c
// src/lsp/protocol.c

typedef struct {
    char* jsonrpc;  // "2.0"
    int id;
    char* method;
    json_t* params;
} LSPRequest;

typedef struct {
    char* jsonrpc;  // "2.0"
    int id;
    json_t* result;
    json_t* error;
} LSPResponse;

LSPRequest* lsp_parse_request(const char* json_str) {
    json_t* root = json_loads(json_str, 0, NULL);
    
    LSPRequest* req = malloc(sizeof(LSPRequest));
    req->jsonrpc = json_string_value(json_object_get(root, "jsonrpc"));
    req->id = json_integer_value(json_object_get(root, "id"));
    req->method = json_string_value(json_object_get(root, "method"));
    req->params = json_object_get(root, "params");
    
    return req;
}
```

### Incremental Parsing

```c
// src/lsp/incremental.c

typedef struct {
    int start_line;
    int start_col;
    int end_line;
    int end_col;
    char* new_text;
} TextEdit;

void apply_incremental_change(Document* doc, TextEdit* edit) {
    // 1. Apply text change
    doc->text = apply_text_edit(doc->text, edit);
    
    // 2. Find affected AST nodes
    ASTNode* affected = find_affected_nodes(doc->ast, edit);
    
    // 3. Re-parse only affected region
    ASTNode* new_nodes = parse_region(doc->text, 
                                      edit->start_line, 
                                      edit->end_line);
    
    // 4. Splice into AST
    replace_ast_nodes(doc->ast, affected, new_nodes);
    
    // 5. Re-typecheck affected region
    typecheck_region(doc->type_info, new_nodes);
}
```

### Symbol Index

```c
// src/lsp/index.c

typedef struct {
    char* name;
    char* file_uri;
    int line;
    int column;
    SymbolKind kind;  // function, variable, type, etc.
} SymbolInfo;

typedef struct {
    SymbolInfo** symbols;
    int count;
    HashMap* name_index;
} SymbolIndex;

void index_document(SymbolIndex* index, Document* doc) {
    // Walk AST and extract all symbols
    ast_walk(doc->ast, &extract_symbol, index);
    
    // Build fast lookup tables
    rebuild_indices(index);
}

SymbolInfo** find_symbols(SymbolIndex* index, const char* query) {
    // Fast prefix search
    return hashmap_prefix_search(index->name_index, query);
}
```

## VS Code Extension

```typescript
// editors/vscode/nanolang/src/extension.ts

import * as vscode from 'vscode';
import { LanguageClient } from 'vscode-languageclient/node';

export function activate(context: vscode.ExtensionContext) {
    const serverExecutable = {
        command: 'nanolanglsp',
        args: []
    };
    
    const client = new LanguageClient(
        'nanolang',
        'Nanolang Language Server',
        serverExecutable,
        {
            documentSelector: [{ scheme: 'file', language: 'nanolang' }]
        }
    );
    
    client.start();
}
```

## Testing

### Unit Tests

```c
// Test completion
void test_completion_basic() {
    Document* doc = create_test_document("let x: int = 42\nx");
    CompletionItem** items = get_completions(doc, 1, 1);
    
    assert(contains_completion(items, "x"));
}

// Test go-to-definition
void test_goto_definition() {
    Document* doc = create_test_document(
        "fn foo() -> int { return 42 }\n"
        "(foo)"
    );
    
    Location* loc = get_definition(doc, 1, 2);  // cursor on 'foo'
    
    assert(loc->line == 0);
    assert(loc->column == 3);
}
```

### Integration Tests

```bash
# Test with VS Code
cd editors/vscode/nanolang
npm test

# Test with vim
vim test.nano
# :LspInfo should show nanolang server running
# :LspDefinition should jump to definition
```

## Performance Targets

- **Completion**: < 50ms response time
- **Diagnostics**: < 200ms for typical file
- **Go-to-definition**: < 10ms
- **Hover**: < 20ms
- **Incremental parse**: < 100ms for typical edit

## Dependencies

- **cJSON**: JSON parsing
- **libjansson**: Alternative JSON library
- **libuv** (optional): Async I/O

## Implementation Roadmap

### Milestone 1: Basic LSP (4 weeks)
- JSON-RPC protocol handling
- Document lifecycle (open/change/close)
- Basic diagnostics
- Simple completion

### Milestone 2: Navigation (2 weeks)
- Go-to-definition
- Find references
- Hover information
- Document symbols

### Milestone 3: Advanced Features (4 weeks)
- Rename refactoring
- Code actions
- Signature help
- Incremental parsing

### Milestone 4: Polish & Performance (2 weeks)
- Performance optimization
- Incremental type checking
- Symbol indexing
- VS Code extension

## References

- [LSP Specification](https://microsoft.github.io/language-server-protocol/)
- [VS Code Extension API](https://code.visualstudio.com/api)
- [clangd LSP Implementation](https://github.com/clangd/clangd)
- [rust-analyzer Architecture](https://github.com/rust-lang/rust-analyzer/blob/master/docs/dev/architecture.md)

## Related Issues

- `nanolang-cxk`: REPL (shares completion logic)
- `nanolang-c2v`: Syntax highlighting (enhanced by LSP)
- Symbol resolution improvements
- Incremental type checking

