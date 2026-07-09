# ERRORS.md — Canonical Error Codes

All errors share shape:
```json
{
  "code": "E_SOMETHING",
  "msg": "Human-readable message",
  "loc": { "line": 12, "col": 5 },
  "path": "/nodes/3/body/1"
}
```

## Codes

### Parsing / Lexing
- E_LEX_INVALID_CHAR
- E_LEX_UNTERMINATED_STRING
- E_PARSE_UNEXPECTED_TOKEN
- E_PARSE_EXPECTED_TOKEN
- E_PARSE_UNTERMINATED_BLOCK

### Validation
- E_DUPLICATE_NAME
- E_DECL_OUTSIDE_BLOCK
- E_CALL_NOT_ALLOWED
- E_UNRESOLVED_ID
- E_TYPE_MISMATCH
- E_DUPLICATE_KEY
- E_RULE_EMPTY_ACTIONS
- E_ASSERT_NOT_BOOLEAN
- E_CALL_EXPR_NOT_ALLOWED
- E_FEATURE_NOT_IMPLEMENTED

### Compilation
- E_COMPILE_INTERNAL  (compiler bug)

## Message guidelines
Keep messages short and specific:
- "Unresolved identifier: foo"
- "Call not allowed: web.search (missing uses web.search)"
