# Universal GC for All Object Types

**Goal**: Extend GC to manage structs, unions, lists, and nested objects

---

## Architecture: Reference-Counted Heap Objects

### Core Principle
**Any object that needs dynamic allocation should go through the GC.**

### Object Hierarchy
```
GC Object (has GCHeader)
├── Dynamic Array
├── GC Struct (NEW)
├── GC Union (NEW)
├── GC List (NEW)
└── GC Closure (FUTURE - for first-class functions)
```

---

## Implementation Strategy

### 1. Add New GC Object Types

**Update `src/runtime/gc.h`**:
```c
typedef enum {
    GC_TYPE_ARRAY = 1,
    GC_TYPE_STRING = 2,
    GC_TYPE_STRUCT = 3,     // NEW
    GC_TYPE_UNION = 4,      // NEW
    GC_TYPE_LIST = 5,       // NEW
    GC_TYPE_CLOSURE = 6     // FUTURE
} GCObjectType;
```

### 2. Add Struct Constructor

**Create `src/runtime/gc_struct.h`**:
```c
#ifndef NANOLANG_GC_STRUCT_H
#define NANOLANG_GC_STRUCT_H

#include "gc.h"
#include <stdint.h>

/* GC-managed struct */
typedef struct {
    char* struct_name;       /* Type name (e.g., "Player", "Enemy") */
    int field_count;         /* Number of fields */
    char** field_names;      /* Field names */
    void** field_values;     /* Field values (GC-aware) */
    uint8_t* field_gc_flags; /* Which fields are GC objects */
} GCStruct;

/* Create new GC struct */
GCStruct* gc_struct_new(const char* struct_name, int field_count);

/* Set field value */
void gc_struct_set_field(GCStruct* s, int field_index, 
                         const char* field_name, void* value, bool is_gc_object);

/* Get field value */
void* gc_struct_get_field(GCStruct* s, int field_index);

/* Get field by name */
void* gc_struct_get_field_by_name(GCStruct* s, const char* field_name);

#endif
```

**Create `src/runtime/gc_struct.c`**:
```c
#include "gc_struct.h"
#include <stdlib.h>
#include <string.h>

GCStruct* gc_struct_new(const char* struct_name, int field_count) {
    GCStruct* s = (GCStruct*)gc_alloc(sizeof(GCStruct), GC_TYPE_STRUCT);
    if (!s) return NULL;
    
    s->struct_name = strdup(struct_name);
    s->field_count = field_count;
    s->field_names = calloc(field_count, sizeof(char*));
    s->field_values = calloc(field_count, sizeof(void*));
    s->field_gc_flags = calloc(field_count, sizeof(uint8_t));
    
    return s;
}

void gc_struct_set_field(GCStruct* s, int field_index, 
                         const char* field_name, void* value, bool is_gc_object) {
    if (field_index < 0 || field_index >= s->field_count) return;
    
    /* If old field was a GC object, release it */
    if (s->field_gc_flags[field_index] && s->field_values[field_index]) {
        gc_release(s->field_values[field_index]);
    }
    
    /* Set new field */
    if (s->field_names[field_index]) {
        free(s->field_names[field_index]);
    }
    s->field_names[field_index] = strdup(field_name);
    s->field_values[field_index] = value;
    s->field_gc_flags[field_index] = is_gc_object ? 1 : 0;
    
    /* If new field is a GC object, retain it */
    if (is_gc_object && value) {
        gc_retain(value);
    }
}

void* gc_struct_get_field(GCStruct* s, int field_index) {
    if (field_index < 0 || field_index >= s->field_count) return NULL;
    return s->field_values[field_index];
}

void* gc_struct_get_field_by_name(GCStruct* s, const char* field_name) {
    for (int i = 0; i < s->field_count; i++) {
        if (s->field_names[i] && strcmp(s->field_names[i], field_name) == 0) {
            return s->field_values[i];
        }
    }
    return NULL;
}
```

### 3. Update GC Mark Phase for Nested Objects

**Update `src/runtime/gc.c`**:
```c
/* Mark phase - now handles nested objects */
static void gc_mark(GCHeader* header) {
    if (header == NULL || header->marked) {
        return;
    }
    
    header->marked = 1;
    
    /* Get object pointer */
    void* obj = gc_header_to_ptr(header);
    
    /* Recursively mark nested GC objects */
    switch (header->type) {
        case GC_TYPE_ARRAY: {
            DynArray* arr = (DynArray*)obj;
            /* If array contains GC objects, mark them */
            if (arr->elem_type == ELEM_ARRAY || arr->elem_type == ELEM_STRUCT) {
                for (int64_t i = 0; i < arr->length; i++) {
                    void* elem = ((void**)arr->data)[i];
                    if (elem && gc_is_managed(elem)) {
                        gc_mark(gc_get_header(elem));
                    }
                }
            }
            break;
        }
        
        case GC_TYPE_STRUCT: {
            GCStruct* s = (GCStruct*)obj;
            /* Mark all GC object fields */
            for (int i = 0; i < s->field_count; i++) {
                if (s->field_gc_flags[i] && s->field_values[i]) {
                    if (gc_is_managed(s->field_values[i])) {
                        gc_mark(gc_get_header(s->field_values[i]));
                    }
                }
            }
            break;
        }
        
        case GC_TYPE_LIST: {
            /* TODO: Mark list nodes */
            break;
        }
        
        default:
            break;
    }
}
```

---

## Usage Examples

### Example 1: Dynamic Struct Creation

```nano
# Define struct type
struct Player {
    name: string,
    health: int,
    position_x: float,
    position_y: float,
    inventory: array<string>
}

fn spawn_player(name: string, x: float, y: float) -> Player {
    # Create struct on heap (GC-managed)
    let inventory: array<string> = []
    
    let player: Player = Player {
        name: name,
        health: 100,
        position_x: x,
        position_y: y,
        inventory: inventory
    }
    
    return player
    # When this function returns, if no one holds a reference,
    # GC will eventually free it!
}

fn main() -> int {
    let mut players: array<Player> = []
    
    # Spawn 10 players dynamically
    let mut i: int = 0
    while (< i 10) {
        let player: Player = (spawn_player "Player" 
                              (int_to_float i) 
                              (int_to_float (* i 10)))
        set players (array_push players player)
        set i (+ i 1)
    }
    
    # Kill a player (remove from array)
    set players (array_remove_at players 5)
    # GC automatically frees the removed player!
    
    return 0
}
```

### Example 2: Nested Objects

```nano
struct Inventory {
    items: array<string>,
    capacity: int
}

struct Enemy {
    name: string,
    health: int,
    loot: Inventory  # Nested struct!
}

fn spawn_enemy() -> Enemy {
    # Create nested inventory
    let mut items: array<string> = []
    set items (array_push items "Gold Coin")
    set items (array_push items "Health Potion")
    
    let loot: Inventory = Inventory {
        items: items,
        capacity: 10
    }
    
    # Create enemy with nested inventory
    let enemy: Enemy = Enemy {
        name: "Goblin",
        health: 50,
        loot: loot
    }
    
    return enemy
    # When enemy goes out of scope, GC frees:
    # 1. The enemy struct
    # 2. The nested loot struct
    # 3. The dynamic items array
    # All automatically!
}
```

### Example 3: Polymorphic Lists

```nano
# Generic list of any type
struct Node<T> {
    value: T,
    next: Node<T>  # Recursive reference
}

fn create_linked_list() -> Node<int> {
    let node3: Node<int> = Node { value: 3, next: void }
    let node2: Node<int> = Node { value: 2, next: node3 }
    let node1: Node<int> = Node { value: 1, next: node2 }
    
    return node1
    # GC handles the whole chain!
}
```

---

## Syntax for Dynamic Object Creation

### Current (Static)
```nano
# Static allocation - on stack
let player: Player = Player { name: "Alice", health: 100 }
```

### Proposed (Dynamic - GC)
```nano
# Option 1: Explicit "new" keyword
let player: Player = new Player { name: "Alice", health: 100 }

# Option 2: Type annotation implies allocation
let player: Player = Player { name: "Alice", health: 100 }
# (Compiler decides: if struct contains GC types, allocate on heap)

# Option 3: Builder function
let player: Player = (create_player "Alice" 100)
```

**Recommendation**: Option 2 - Compiler infers allocation based on:
- Does struct contain arrays/strings/other structs?
- Is struct returned from function?
- If YES → heap allocate via GC
- If NO → stack allocate (no GC overhead)

---

## Reference Management Rules

### Automatic Reference Counting

**Rule 1**: Assignment increments ref count
```nano
let p1: Player = player  # player.ref_count++
```

**Rule 2**: Variable going out of scope decrements ref count
```nano
fn example() -> void {
    let player: Player = (spawn_player)  # ref_count = 1
    # ... use player ...
}  # player.ref_count--, freed if 0
```

**Rule 3**: Storing in collection increments ref count
```nano
let mut players: array<Player> = []
set players (array_push players p)  # p.ref_count++
```

**Rule 4**: Removing from collection decrements ref count
```nano
set players (array_remove_at players 0)  # removed player.ref_count--, may free
```

**Rule 5**: Field assignment manages refs
```nano
struct Game {
    player: Player
}

let game: Game = Game { player: p }  # p.ref_count++
# When game is freed, p.ref_count--
```

---

## Cycle Detection for Circular References

### Problem: Cycles
```nano
struct Node {
    value: int,
    next: Node,    # Can create cycle!
    prev: Node     # Doubly-linked list
}

# Create cycle
let node1: Node = Node { value: 1, next: void, prev: void }
let node2: Node = Node { value: 2, next: node1, prev: void }
set node1.next node2  # CYCLE: node1 -> node2 -> node1
# Reference counting alone won't free these!
```

### Solution: Periodic Mark-and-Sweep
Our GC already has `gc_collect_cycles()` that runs periodically to detect and break cycles.

**Mark Phase**:
1. Start from roots (variables in scope)
2. Mark all reachable objects
3. Recursively follow references

**Sweep Phase**:
1. Any unmarked object with ref_count > 0 is in a cycle
2. Break the cycle and free

---

## Implementation Roadmap

### Phase 1: GC Structs (1 week)
- [ ] Add `GC_TYPE_STRUCT` to GC
- [ ] Implement `gc_struct_new()`, `gc_struct_set_field()`
- [ ] Update mark phase for nested objects
- [ ] Add `VAL_GC_STRUCT` to Value system
- [ ] Integrate with evaluator

### Phase 2: Language Syntax (1 week)
- [ ] Decide on allocation syntax (new vs implicit)
- [ ] Update parser for heap-allocated structs
- [ ] Update type checker for GC inference
- [ ] Generate proper retain/release calls

### Phase 3: Nested Objects (1 week)
- [ ] Arrays of structs
- [ ] Structs with arrays
- [ ] Structs with struct fields
- [ ] Test complex nesting

### Phase 4: Lists & Collections (1 week)
- [ ] GC-managed lists
- [ ] GC-managed maps/dictionaries
- [ ] Generic collection support

---

## Performance Considerations

### Stack vs Heap Allocation

**Stack (Fast)**:
- Small structs with no GC fields
- Short-lived objects
- Known lifetime

**Heap/GC (Flexible)**:
- Large objects
- Contains arrays/strings/other GC objects
- Unknown lifetime
- Returned from functions

**Compiler Heuristics**:
```nano
# Stack (no GC)
struct Point { x: int, y: int }
let p: Point = Point { x: 10, y: 20 }

# Heap (GC) - contains array
struct Enemy { name: string, inventory: array<int> }
let e: Enemy = Enemy { name: "Orc", inventory: [] }

# Heap (GC) - returned from function
fn create() -> Point {
    return Point { x: 10, y: 20 }  # Must heap allocate!
}
```

### Reference Counting Overhead

**Per operation**: ~2 instructions (inc/dec counter)  
**Per struct**: 24 bytes header  
**Per field update**: Check if GC object, retain/release

**Optimization**: Compiler can elide redundant retain/release pairs.

---

## Example: Full Game with GC Objects

```nano
struct Vector2 {
    x: float,
    y: float
}

struct Asteroid {
    position: Vector2,
    velocity: Vector2,
    size: int,
    health: int
}

struct Bullet {
    position: Vector2,
    velocity: Vector2,
    damage: int
}

struct Game {
    asteroids: array<Asteroid>,
    bullets: array<Bullet>,
    score: int
}

fn spawn_asteroid(x: float, y: float, vx: float, vy: float) -> Asteroid {
    return Asteroid {
        position: Vector2 { x: x, y: y },
        velocity: Vector2 { x: vx, y: vy },
        size: 3,
        health: 100
    }
}

fn fire_bullet(x: float, y: float) -> Bullet {
    return Bullet {
        position: Vector2 { x: x, y: y },
        velocity: Vector2 { x: 0.0, y: -5.0 },
        damage: 25
    }
}

fn update_game(game: Game) -> Game {
    # Spawn new asteroid
    let new_ast: Asteroid = (spawn_asteroid 100.0 200.0 1.0 2.0)
    set game.asteroids (array_push game.asteroids new_ast)
    
    # Fire bullet
    let bullet: Bullet = (fire_bullet 50.0 50.0)
    set game.bullets (array_push game.bullets bullet)
    
    # Collision detection - destroy asteroid
    set game.asteroids (array_remove_at game.asteroids 0)
    # GC automatically frees the destroyed asteroid!
    
    return game
}

fn main() -> int {
    let mut game: Game = Game {
        asteroids: [],
        bullets: [],
        score: 0
    }
    
    # Game loop
    let mut frame: int = 0
    while (< frame 100) {
        set game (update_game game)
        set frame (+ frame 1)
    }
    
    (println "Game over!")
    (print "Final score: ")
    (println game.score)
    
    return 0
    # GC automatically cleans up entire game state!
}
```

---

## Summary

### Key Points

1. **Universal Allocation**: Any heap object goes through GC
2. **Automatic Ref Counting**: Retain/release managed automatically
3. **Nested Objects**: Mark phase traverses object graphs
4. **Cycle Detection**: Periodic mark-and-sweep handles cycles
5. **Zero Manual Management**: User never calls free()

### What User Writes
```nano
let mut enemies: array<Enemy> = []
set enemies (array_push enemies (spawn_enemy))
set enemies (array_remove_at enemies 0)
# That's it! GC handles everything.
```

### What Compiler Generates
```c
// Allocate
Enemy* e = gc_alloc(sizeof(Enemy), GC_TYPE_STRUCT);
gc_retain(e);

// Store in array
dyn_array_push(enemies, e);
gc_retain(e);  // Array holds reference

// Remove from array
Enemy* removed = dyn_array_pop(enemies);
gc_release(removed);  // Array no longer holds reference

// Variable goes out of scope
gc_release(e);  // May free if ref_count == 0
```

### Benefits

✅ **No pointers in user code**  
✅ **Automatic memory management**  
✅ **Type-safe**  
✅ **Handles complex object graphs**  
✅ **Deterministic performance**  
✅ **No GC pauses**

---

## Next Steps

1. Implement `gc_struct.c` for dynamic struct allocation
2. Update mark phase for nested traversal
3. Add language syntax for heap allocation
4. Test with complex game examples
5. Extend to unions and lists

**This makes nanolang suitable for AAA game development!** 🚀

