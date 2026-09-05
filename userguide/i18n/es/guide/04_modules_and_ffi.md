---
title: Módulos, FFI y recursos
machine_generated: true
reviewed: false
lang: es
---

> Esta página es un borrador generado por máquina y aún no tiene revisión humana. Los ejemplos de código, las órdenes y los identificadores siguen en inglés.

# Módulos, FFI y recursos

Un módulo es un límite de visibilidad y de seguridad. El código nuevo debe usar `module`, un alias y nombres públicos calificados.

## Importar un módulo

```nano
module "modules/std/mathx/mathx.nano" as mathx

fn bounded(value: int) -> int {
    return (mathx.mathx_clamp value 0 100)
}

shadow bounded {
    assert (== (bounded -5) 0)
    assert (== (bounded 120) 100)
}
```

El analizador aún acepta las formas heredadas `import` y `from ... import ...`. No las elijas para código nuevo.

## Visibilidad

Las declaraciones son privadas por defecto. Marca la superficie prevista con `pub`:

```nano
fn internal_scale(value: int) -> int {
    return (* value 2)
}

pub fn transform(value: int) -> int {
    return (+ (internal_scale value) 1)
}
```

Los ayudantes privados siguen siendo invocables dentro de su módulo. Los importadores solo pueden llamar declaraciones públicas.

## Funciones extranjeras

Las declaraciones extranjeras usan `extern fn`. Una llamada directa exige `unsafe` salvo que el módulo importado entero sea unsafe:

```nano
extern fn c_close(fd: int) -> int

fn close_fd(fd: int) -> int {
    unsafe {
        return (c_close fd)
    }
}
```

Mantén las regiones inseguras estrechas. Valida los valores extranjeros en el límite y expón un envoltorio tipado cuando se pueda ofrecer con honestidad.

## Tipos de recurso

`resource struct` marca un recurso afín que debe consumirse a lo sumo una vez:

```nano
resource struct FileHandle {
    fd: int
}
```

Mi comprobador de recursos detecta casos importantes de uso tras consumo y de consumo repetido. No es una demostración completa de propiedad. Anota los locales de recurso de forma explícita e inspecciona cada camino de retorno para la limpieza.

## Metadatos de módulo

Tres archivos sirven trabajos distintos:

| File | Purpose |
| --- | --- |
| `.nano` | Fuente y declaraciones públicas |
| `module.json` | Fuentes de compilación nativa, flags, paquetes y metadatos de propiedad |
| `module.manifest.json` | Metadatos de descubrimiento, estabilidad, capacidades y ejemplos |

Los módulos puros no siempre necesitan metadatos de compilación nativa. Consulta el [module inventory](../generated/modules.md) generado para lo que existe ahora.

