---
title: Perfilado de rendimiento
machine_generated: true
reviewed: false
lang: es
---

> Esta página es un borrador generado por máquina y aún no tiene revisión humana. Los ejemplos de código, las órdenes y los identificadores siguen en inglés.

# Perfilado de rendimiento

Envuelvo un binario nativo compilado con `-pg` para que al ejecutarlo se recoja
un perfil del anfitrión y se imprima JSON que un LLM puede leer. No garantizo
una aceleración. Conservo una optimización cuando un segundo perfil y mis
pruebas muestran que ayudó.

La lista canónica de campos, las variables de entorno de reentrada y los
nombres de artefactos en `/tmp` están en
[docs/PERFORMANCE_MONITORING.md](https://github.com/jordanhubbard/nanolang/blob/main/docs/PERFORMANCE_MONITORING.md).

## Capturar un perfil

```bash
./bin/nanoc program.nano -o bin/program -pg --profile-output profile.json
./bin/program
```

El primer proceso es un coordinador (`_nl_run_with_profiling`). Crea un
artefacto de nombre único a partir de su ID de proceso, lanza la carga medida
en un hijo, espera, convierte la salida del recolector a JSON y borra los
archivos temporales. Las ejecuciones concurrentes no comparten la misma ruta
`/tmp`.

El JSON se escribe en `profile.json` y también en **stdout** (entre dos líneas
de banner). Redirigir stderr no lo capturará.

## Flags que no son `-pg`

| Flag | Role |
| --- | --- |
| `-pg` / `--profile-output` | Recolector del OS + JSON. El `main` nativo pasa a `_nl_run_with_profiling`. |
| `--profile` | Ganchos de tiempo en C generado. Tabla en stderr. |
| `--trace` | Ganchos de rastreo de llamadas en C generado. `-> fn` / `<- fn` indentados en stderr. |
| `--profile-runtime` | También escribe pilas colapsadas `.nano.prof`. Solo backend nativo. |
| `--pgo <file>` | Inline usando un `.nano.prof`. No usa JSON de `-pg`. |

Flags C añadidos con `-pg`: `-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls`.

## Frontera de plataforma

| Platform | Collector | Measurement |
| --- | --- | --- |
| macOS with full Xcode | `xctrace` Time Profiler | muestreo de un hijo lanzado |
| macOS fallback | `sample` | muestreo periódico de un hijo sincronizado |
| Linux | `gprofng collect app` | recolección instrumentada en un proceso hijo |
| Other OS | none | Imprimo que el perfilado no está admitido y ejecuto el programa |

Trato xctrace como disponible solo cuando `which xctrace` y `xctrace version`
tienen éxito ambos. Command Line Tools solo no basta.

En macOS, el respaldo `sample` crea primero el hijo de carga y lo retiene
en una tubería (`_NL_PROFILING_ACTIVE`, `_NL_PROFILING_CWD`). Un hijo muestreador
se adhiere a ese PID (`sample <pid> 60 -f /tmp/nanolang_sample_<pid>.txt -mayDie`),
luego el coordinador suelta la carga. Las ejecuciones cortas aún pueden terminar
antes de que `sample` se adhiera.

En Linux, la reentrada usa `_NL_PROFILING_CHILD` o `LD_PRELOAD` que contiene
`libgp-collector`. Si falta `gprofng`, el hijo corre sin recolector.
gprofng es recolección instrumentada. El campo JSON `profile_type` sigue siendo
la cadena `"sampling"` en cada OS; es una etiqueta histórica, no una afirmación
de que Linux muestrea.

## JSON que emito de verdad

Cada punto caliente tiene solo `function`, `samples` y `pct_time`. No emito
ubicaciones de fuente ni microsegundos por llamada.

```json
{
  "profile_type": "sampling",
  "platform": "macOS",
  "tool": "xctrace",
  "binary": "./bin/program",
  "hotspots": [
    {"function": "nl_hot_function", "samples": 120, "pct_time": 12.0}
  ],
  "analysis_hints": [
    "Functions with high sample counts are hot spots",
    "Look for nl_ prefixed functions (NanoLang generated)",
    "str_ and array_ functions often indicate algorithmic issues",
    "Deep call stacks may indicate recursion or callback chains"
  ]
}
```

`tool` es `"gprofng"`, `"xctrace"` o `"sample"`. En Linux, `samples` es
el porcentaje exclusivo por diez, no un recuento real de muestras. En xctrace
emito los 20 primeros nombres que empiezan por `nl_`.

Trata un perfil como medición de una carga en una máquina.

## Ajustar con un LLM

1. Compila con `-pg --profile-output`.
2. Ejecuta una carga representativa, no una sombra de una línea.
3. Dale a un LLM el JSON, la fuente pertinente y la orden que produjo
   la carga. Pide un cambio ligado a un punto caliente medido.
4. Ejecuta las pruebas, luego vuelve a perfilar la misma carga.
5. Conserva el cambio solo si las pruebas siguen pasando y los números mejoran.

Un LLM puede sugerir una optimización; el perfil puede probar su efecto. Ninguno
sustituye al otro. Compara el reloj de pared **sin** `-pg` cuando informes
velocidad.

El rastreo y el perfilado comparten un gancho de arranque. Los binarios
`--profile` honran `NANO_PROFILE`, y los `--trace` honran `NANO_TRACE`; poner
cualquiera a `0` desactiva ese gancho en tiempo de ejecución sin recompilar, y
un gancho desactivado no hace trabajo por evento.

`--pgo` lee `.nano.prof` de `--profile-runtime`. Ese archivo no es una entrada
PGO del JSON de `-pg`.

