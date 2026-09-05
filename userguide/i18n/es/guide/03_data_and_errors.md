---
title: Datos y errores
machine_generated: true
reviewed: false
lang: es
---

> Esta página es un borrador generado por máquina y aún no tiene revisión humana. Los ejemplos de código, las órdenes y los identificadores siguen en inglés.

# Datos y errores

Represento colecciones y estados de dominio de forma explícita. Elige un tipo que haga difíciles de expresar los estados inválidos.

## Arrays y tuplas

```nano
let values: array<int> = [10, 20, 30]
let first = (at values 0)
(array_set values 1 25)

let pair: (int, string) = (7, "seven")
let number = pair.0
```

Los arrays tienen comprobación de límites. Algunas funciones de array mutan el almacenamiento y devuelven `void`; otras devuelven un array nuevo. Consulta [Builtins](../generated/builtins.md) en lugar de inferir la propiedad desde un nombre familiar.

## Structs y enums

```nano
struct Point {
    x: int
    y: int
}

enum Direction {
    North
    South
    East
    West
}

let origin = Point { x: 0, y: 0 }
```

Los tipos con nombre usan `UpperCamelCase`. Los valores y las funciones usan `snake_case`.

## Uniones y match

```nano
union ParseResult {
    Value(int)
    Error(string)
}

fn unwrap_or(result: ParseResult, fallback: int) -> int {
    return (match result {
        Value(value) => value
        Error(_) => fallback
    })
}

shadow unwrap_or {
    assert (== (unwrap_or Value(7) 0) 7)
    assert (== (unwrap_or Error("bad") 3) 3)
}
```

Usa `Result<T, E>` u otra unión explícita para fallos que el llamador puede manejar. Empareja variantes donde puedas recuperarte o añadir contexto. El operador posfijo `?` propaga errores de resultado compatibles; usa un `match` explícito cuando importen la limpieza o el contexto.

## Cadenas y cadenas binarias

`string` es texto. `bstring` es datos binarios de longitud explícita y puede contener bytes cero. La conversión y el comportamiento Unicode se documentan por builtin o módulo. No asumas que los índices de byte son índices de carácter Unicode.

## Colecciones

Los arrays y los mapas hash incorporados están separados de los módulos de colección con respaldo C. La página generada [Builtins](../generated/builtins.md) lista las grafías exactas de los builtins. La página generada [Modules](../generated/modules.md) lista declaraciones de módulo y límites de compilación nativa.

