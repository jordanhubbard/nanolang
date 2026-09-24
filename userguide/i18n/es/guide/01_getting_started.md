---
title: Primeros pasos
machine_generated: true
reviewed: false
lang: es
---

> Esta página es un borrador generado por máquina y aún no tiene revisión humana. Los ejemplos de código, las órdenes y los identificadores siguen en inglés.

# Primeros pasos

Actualmente construyo en sistemas tipo Unix. Los usuarios de Windows deben usar WSL2.

## Compilación

Necesitas un compilador C, Make, Git y pkg-config. Clona y construye:

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build bootstrap3
./bin/nanoc --help
```

`bin/nanoc` es mi compilador. `bin/nano` es mi intérprete que recorre el árbol.

## Primer programa

Crea `hello.nano`:

<!--nl-snippet {"name":"refresh_getting_started_hello","check":true,"expect_stdout":"Hello, World!\n"}-->
```nano
fn main() -> int {
    (println "Hello, World!")
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

Compílalo y ejecútalo:

```bash
./bin/nanoc hello.nano -o hello
./hello
```

Con `-o hello`, produzco un ejecutable nativo a partir de NanoISA verificado mediante mi traductor a C y el compilador C del anfitrión. Ejecuto las pruebas de sombra seleccionadas por separado antes de publicar el ejecutable; no vuelven a ejecutarse al iniciar el programa. Una prueba de sombra fallida impide la publicación.

Sin una opción de salida o destino, publico bytecode portátil:

```bash
./bin/nanoc hello.nano
./bin/nano_vm hello.nvm
```

## Intérprete

Ejecuta un archivo fuente sin producir un ejecutable nativo:

```bash
./bin/nano hello.nano
```

El intérprete y el backend compilado comparten el lenguaje, pero no tienen los mismos límites de implementación. Prueba el backend que pretendes entregar.

## Disposición del proyecto

Para un programa pequeño, un archivo `.nano` basta. Los paquetes pueden usar `nano.toml`:

```text
my_program/
  nano.toml
  main.nano
  src/
```

Consulta [`examples/hello_pkg`](https://github.com/jordanhubbard/nanolang/tree/main/examples/hello_pkg) y [`examples/large_project`](https://github.com/jordanhubbard/nanolang/tree/main/examples/large_project) para disposiciones completas.

## Siguiente

Lee [Lenguaje](02_language.md) para llamadas, operadores, enlaces, funciones y flujo de control. Después, [Entorno de ejecución seguro](08_secure_runtime.md) explica cómo se permite que un programa acceda al anfitrión.
