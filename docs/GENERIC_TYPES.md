# Generic Types in Nanolang

## Overview

Nanolang supports generic types through compile-time name mangling. Generic types are instantiated for each concrete type used, following a simple and predictable pattern.

## Syntax

```nano
struct Point {
    x: int,
    y: int
}

fn example() -> int {
    let points: List<Point> = (list_Point_new)
    
    let p1: Point = Point { x: 10, y: 20 }
    let p2: Point = Point { x: 30, y: 40 }
    
    (list_Point_push points p1)
    (list_Point_push points p2)
    
    let len: int = (list_Point_length points)
    let first: Point = (list_Point_get points 0)
    
    return first.x  # Returns 10
}
```

## Generic List Operations

For a generic type `List<T>`, the following operations are available:

| Operation | Function Name | Description |
|-----------|---------------|-------------|
| Create | `list_T_new()` | Create empty list |
| Create with capacity | `list_T_with_capacity(cap)` | Create with initial capacity |
| Add element | `list_T_push(list, elem)` | Append element to end |
| Remove last | `list_T_pop(list)` | Remove and return last element |
| Get element | `list_T_get(list, index)` | Get element at index |
| Set element | `list_T_set(list, index, elem)` | Set element at index |
| Insert | `list_T_insert(list, index, elem)` | Insert at index |
| Remove | `list_T_remove(list, index)` | Remove at index |
| Length | `list_T_length(list)` | Get number of elements |
| Capacity | `list_T_capacity(list)` | Get current capacity |
| Check empty | `list_T_is_empty(list)` | Returns true if empty |
| Clear | `list_T_clear(list)` | Remove all elements |
| Free | `list_T_free(list)` | Free memory |

## Supported Types

### Primitive Types
- `List<int>` → `list_int_*` functions
- `List<string>` → `list_string_*` functions  
- `List<float>` → `list_float_*` functions
- `List<bool>` → `list_bool_*` functions

### User-Defined Types
- `List<Point>` → `list_Point_*` functions
- `List<Player>` → `list_Player_*` functions
- `List<Token>` → `list_Token_*` functions
- Any struct type defined in your program

## Implementation Details

### Name Mangling
Generic types use a simple name mangling scheme:
- `List<Point>` becomes `list_Point_*` in function names
- The transpiler generates `typedef struct List_Point List_Point;`

### Storage Strategy
- **Primitive types**: Stored directly (int, float, bool, string)
- **Struct types**: Stored as heap-allocated pointers
- The interpreter automatically handles conversion between values and pointers

### Type Safety
The type checker validates:
- Generic type declarations: `List<T>` where T is a defined type
- Function calls match the correct type: `list_Point_push` expects Point structs
- Field access on returned structs: `(list_Point_get points 0).x` works correctly

## Examples

### Example 1: List of Points
```nano
struct Point { x: int, y: int }

fn process_points() -> int {
    let points: List<Point> = (list_Point_new)
    
    let p1: Point = Point { x: 1, y: 2 }
    let p2: Point = Point { x: 3, y: 4 }
    
    (list_Point_push points p1)
    (list_Point_push points p2)
    
    let sum: int = 0
    for i in (range 0 (list_Point_length points)) {
        let p: Point = (list_Point_get points i)
        set sum (+ sum (+ p.x p.y))
    }
    
    return sum  # Returns 10
}
```

### Example 2: List of Game Entities
```nano
struct Entity {
    id: int,
    health: int,
    x: float,
    y: float
}

fn spawn_enemies() -> List<Entity> {
    let enemies: List<Entity> = (list_Entity_with_capacity 10)
    
    for i in (range 0 10) {
        let enemy: Entity = Entity {
            id: i,
            health: 100,
            x: (* (cast_float i) 50.0),
            y: 100.0
        }
        (list_Entity_push enemies enemy)
    }
    
    return enemies
}
```

### Example 3: Multiple Generic Types
```nano
struct Player { name: string, score: int }
struct Item { name: string, value: int }

fn game_state() -> int {
    let players: List<Player> = (list_Player_new)
    let items: List<Item> = (list_Item_new)
    
    (list_Player_push players Player { name: "Alice", score: 100 })
    (list_Item_push items Item { name: "Sword", value: 50 })
    
    return (+ (list_Player_length players) (list_Item_length items))
}
```

## Limitations

1. **No type parameters in function definitions**: You cannot write generic functions. Each generic type must be instantiated explicitly.

2. **Name mangling visible to user**: Users must write `list_Point_new` rather than `List.new<Point>()`.

3. **Transpiler limitations**: Some edge cases in struct list transpilation may need manual workarounds.

## Best Practices

1. **Define structs before using in generics**:
   ```nano
   struct Point { x: int, y: int }  # Define first
   let points: List<Point> = ...     # Then use
   ```

2. **Use clear naming conventions**:
   - Struct names: PascalCase (Point, Player, GameEntity)
   - This makes list functions readable: `list_Point_*`, `list_Player_*`

3. **Initialize with capacity for performance**:
   ```nano
   let items: List<Item> = (list_Item_with_capacity 100)
   ```

4. **Clean up lists when done**:
   ```nano
   (list_Point_free points)
   ```

## Future Enhancements

Potential improvements (not currently implemented):
- Generic function definitions
- Type inference for generics
- Syntax sugar: `List.new<Point>()` instead of `list_Point_new`
- Generic structs: `struct Pair<T, U> { first: T, second: U }`

## See Also

- [Dynamic Arrays](./DYNAMIC_ARRAYS.md) - Built-in array operations
- [Type System](./TYPE_SYSTEM.md) - Complete type system documentation
- [Examples](../examples/) - See examples using generic lists
