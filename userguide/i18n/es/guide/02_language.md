---
title: Lenguaje
machine_generated: true
reviewed: false
lang: es
---

> Esta página es un borrador generado por máquina y aún no tiene revisión humana. Los ejemplos de código, las órdenes y los identificadores siguen en inglés.

# Lenguaje

Mantengo la sintaxis ordinaria explícita. Acepto algunas conveniencias, pero no oculto la precedencia ni las firmas de función.

## Llamadas y operadores

Las llamadas van entre paréntesis y son prefijas:

```nano
(println "ready")
(distance x y)
(math.clamp value low high)
```

Los operadores aceptan notación prefija e infija:

```nano
let prefix = (+ 2 (* 3 4))
let infix = 2 + (3 * 4)
```

Todos los operadores binarios infijos tienen igual precedencia y asocian de izquierda a derecha. `2 + 3 * 4` significa `(2 + 3) * 4`. Añade paréntesis cuando importe el agrupamiento.

## Enlaces

Los enlaces son inmutables salvo que se marquen `mut`. La mutación usa `set`:

<!--nl-snippet {"name":"refresh_language_bindings","check":true}-->
```nano
fn count_three() -> int {
    let mut count = 0
    set count (+ count 1)
    set count (+ count 1)
    set count (+ count 1)
    return count
}

shadow count_three {
    assert (== (count_three) 3)
}

fn main() -> int {
    assert (== (count_three) 3)
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

Las anotaciones locales son opcionales cuando el inicializador determina el tipo:

```nano
let count = 3
let name = "Ada"
```

Anota colecciones vacías, valores genéricos, manejadores extranjeros, valores de recurso y fronteras donde la inferencia oscurecería la intención. Los parámetros y los tipos de retorno siguen explícitos.

## Tipos escalares

Los tipos escalares comunes son `int`, `u8` o `byte`, `float`, `bool`, `string`, `bstring` y `void`. Los arrays, las tuplas, los registros con nombre, los enums, las uniones, los tipos de función, los tipos genéricos, los registros abiertos y los tipos extranjeros opacos amplían ese conjunto.

## Flujo de control

Un `if` puede omitir `else`:

```nano
if needs_redraw {
    (draw scene)
}
```

`cond` elige entre valores de expresión y exige `else`:

```nano
let sign = (cond
    ((< value 0) -1)
    ((> value 0) 1)
    (else 0)
)
```

Los bucles usan `while` o `for`:

```nano
while (< index count) {
    set index (+ index 1)
}

for index in (range 0 count) {
    (visit index)
}
```

`break` y `continue` están disponibles dentro de los bucles.

## Funciones

```nano
fn gcd(a: int, b: int) -> int {
    if (== b 0) {
        return a
    }
    return (gcd b (% a b))
}

shadow gcd {
    assert (== (gcd 48 18) 6)
}
```

Admito recursión, funciones de primera clase, cierres, genéricos, precondiciones con `requires` y postcondiciones con `ensures`. Los contratos son propiedades comprobadas de ejecuciones; no son demostraciones formales.

## Comentarios

```nano
# ordinary comment
// accepted line comment
/* block comment */
/// documentation comment
```

Usa `#` para comentario ordinario y `///` para documentación consumida por las herramientas.

