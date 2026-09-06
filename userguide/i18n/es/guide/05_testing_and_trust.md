---
title: Pruebas y confianza
machine_generated: true
reviewed: false
lang: es
---

> Esta página es un borrador generado por máquina y aún no tiene revisión humana. Los ejemplos de código, las órdenes y los identificadores siguen en inglés.

# Pruebas y confianza

Distingo pruebas, comprobaciones estáticas y demostraciones. Responden preguntas distintas.

## Sombras

Una sombra es una prueba ejecutable unida a una función:

```nano
fn double(value: int) -> int {
    return (* value 2)
}

shadow double {
    assert (== (double 0) 0)
    assert (== (double 3) 6)
}
```

Durante la compilación ordinaria ejecuto las sombras en el intérprete anfitrión. Los programas nativos generados también contienen su arnés de sombras. Una sombra prueba los casos que ejecuta; no demuestra la función para cada entrada.

La imposición del compilador y la política del proyecto difieren:

- El compilador normalmente avisa de una sombra ausente.
- Exime las funciones extern, `main`, las lambdas generadas, las funciones GPU y las funciones que llaman externs.
- La política del repositorio exige una sombra útil para cada función con nombre no-extern añadida o cambiada cuando se pueda probar.

## Pruebas de propiedades y cobertura

Las pruebas de propiedades muestrean valores generados y pueden reducir fallos. El muestreo no es verificación exhaustiva. La cobertura informa qué código se ejecutó; no establece corrección.

## Contratos

`requires` comprueba precondiciones y `ensures` comprueba postcondiciones en su frontera implementada. Una comprobación en tiempo de ejecución exitosa dice que esa condición valió para esa ejecución.

## Verificación formal

Mi desarrollo Coq demuestra la metateoría enunciada para un modelo nuclear definido. No demuestra automáticamente cada backend, llamada extranjera, asignador, módulo o función de usuario.

Usa el informe de confianza para inspeccionar la frontera:

```bash
./bin/nanoc program.nano --trust-report
```

Di **proved** solo para un teorema comprobado en su modelo enunciado, **tested** solo para el comportamiento ejercido por una prueba con nombre, y **assumed** para comportamiento de plataforma o extranjero no comprobado.

## Comprobaciones útiles

```bash
make test
make userguide-check
make check-stdlib-docs
python3 scripts/check_markdown_links.py
```

