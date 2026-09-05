---
title: Inicio
machine_generated: true
reviewed: false
lang: es
---

> Esta página es un borrador generado por máquina y aún no tiene revisión humana. Los ejemplos de código, las órdenes y los identificadores siguen en inglés.

# Guía de usuario de NanoLang

Soy NanoLang. Uso límites explícitos, llamadas prefijas, operadores de igual precedencia y pruebas junto a las funciones que ejercitan.
```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    }
    return (* n (factorial (- n 1)))
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
}
```
## Empieza aquí

1. [Constrúyeme y ejecuta un primer programa](guide/01_getting_started.md).
2. [Aprende mi lenguaje](guide/02_language.md).
3. [Usa datos estructurados y errores tipados](guide/03_data_and_errors.md).
4. [Trabaja con módulos, código extranjero y recursos](guide/04_modules_and_ffi.md).
5. [Comprende sombras, pruebas y mi frontera verificada](guide/05_testing_and_trust.md).
6. [Elige una herramienta o un backend](guide/06_tools_and_backends.md).
7. [Mide el rendimiento nativo y ajusta con evidencia](guide/07_performance_profiling.md).

## Referencia

- [Ejemplos](generated/examples.md) se generan de cada `.nano` bajo `examples/`.
- [Builtins](generated/builtins.md) vienen de la referencia de biblioteca estándar comprobada mecánicamente.
- [Módulos](generated/modules.md) se generan del árbol de módulos y los manifiestos.
- [CLI del compilador](generated/cli.md) viene del compilador usado para construir esta guía.
- [NanoISA](https://github.com/jordanhubbard/nanolang/blob/main/docs/NANOISA.md) es mi frontera de VM tipada compartida.

## Lo que prometo

- Las llamadas a función usan la sintaxis `(function argument)`.
- Los parámetros y los tipos de retorno son explícitos.
- Los enlaces locales pueden usar inferencia cuando el inicializador deja el tipo claro.
- Todos los operadores infijos tienen igual precedencia y asocian de izquierda a derecha.
- La política del proyecto exige sombras útiles para las funciones con nombre cambiadas. La imposición del compilador tiene exenciones documentadas.
- Una sombra que pasa es una prueba, no una demostración.
- Mi backend C es la ruta de compilación de producción. Otros backends tienen subconjuntos documentados más estrechos.

El analizador, el verificador de tipos, el registro de builtins y las pruebas son la autoridad cuando un documento viejo discrepa. Prefiero corregir una guía a conservar un error seguro de sí mismo.

