# Transpiler Enum Redefinition Issue

**Date:** November 15, 2025  
**Status:** Known Issue - Workaround Applied  
**Priority:** Medium (fix in transpiler improvements)

## Issue

The transpiler generates enum typedef definitions multiple times when:
1. An enum is defined
2. Multiple structs reference that enum type

### Example

```nano
enum NodeType {
    NUMBER = 0,
    STRING = 1
}

struct Node1 {
    type: int,  /* Works - uses int */
    value: string
}

struct Node2 {
    type: NodeType,  /* Problematic - causes redefinition */
    data: string
}
```

### Generated C (Incorrect)

```c
/* First generation - correct */
typedef enum {
    NodeType_NUMBER = 0,
    NodeType_STRING = 1
} NodeType;

/* Second generation - ERROR! */
typedef enum {
    NodeType_NUMBER = 0,  /* redefinition error */
    NodeType_STRING = 1   /* redefinition error */
} NodeType;  /* typedef redefinition error */
```

## Root Cause

The transpiler generates enum definitions:
1. Once in the enum definitions section
2. Again for each struct that uses the enum type as a field

The transpiler doesn't track which enums have already been generated.

## Workaround

**Use `int` for enum-typed struct fields:**

```nano
enum NodeType {
    NUMBER = 0,
    STRING = 1
}

struct Node {
    type: int,  /* ✓ Works - comment indicates it's NodeType */
    value: string
}

/* Can still use enum values */
fn create_number_node() -> Node {
    return Node {
        type: NodeType.NUMBER,  /* ✓ Enum value converts to int */
        value: "42"
    }
}
```

**This is a common C pattern and works perfectly!**

## Proper Fix

### Solution: Track Generated Types

```c
/* In transpiler.c */
typedef struct {
    char **generated_enums;
    int generated_enum_count;
    int generated_enum_capacity;
} GeneratedTypes;

/* Before generating enum */
bool already_generated(GeneratedTypes *gen, const char *name) {
    for (int i = 0; i < gen->generated_enum_count; i++) {
        if (strcmp(gen->generated_enums[i], name) == 0) {
            return true;
        }
    }
    return false;
}

/* When generating enum */
if (already_generated(&generated, edef->name)) {
    /* Skip - already generated */
    continue;
}

/* Generate enum and mark as generated */
generate_enum(edef);
mark_generated(&generated, edef->name);
```

### Implementation Effort

**Estimated:** 2-3 hours to implement properly

**Steps:**
1. Add `GeneratedTypes` struct to transpiler
2. Initialize tracking in `transpile_program`
3. Check before generating each enum/struct/union
4. Mark as generated after outputting
5. Test with parser_mvp.nano

## Impact

### Current Impact
- **Parser MVP:** Can't compile to C (shadow tests still pass)
- **Workaround:** Use `int` for enum fields (standard C pattern)
- **Code Quality:** Minor - requires comments to indicate enum type

### After Fix
- **Clean Code:** Can use enum types in struct fields
- **Type Safety:** Enum types explicit in structs
- **C Generation:** Clean, no duplicates

## Status

**Current:** Using workaround (int fields with comments)  
**Priority:** Medium (not blocking progress)  
**Fix When:** During transpiler improvements batch

## Related Files

- `src/transpiler.c` - Enum generation code
- `src_nano/parser_mvp.nano` - Uses workaround pattern
- `planning/PHASE2_PARSER_STATUS.md` - Documents issue

---

*Issue Documented: November 15, 2025*  
*Workaround: Use int for enum fields*  
*Proper Fix: Track generated types in transpiler*

