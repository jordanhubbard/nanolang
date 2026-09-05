---
title: Données et erreurs
machine_generated: true
reviewed: false
lang: fr
---

> Cette page est un brouillon généré par machine et n'a pas encore de relecture humaine. Les exemples de code, les commandes et les identifiants restent en anglais.

# Données et erreurs

Je représente les collections et les états de domaine de façon explicite. Choisis un type qui rend les états invalides difficiles à exprimer.

## Tableaux et tuples

```nano
let values: array<int> = [10, 20, 30]
let first = (at values 0)
(array_set values 1 25)

let pair: (int, string) = (7, "seven")
let number = pair.0
```

Les tableaux sont vérifiés aux bornes. Certaines fonctions de tableau mutent le stockage et renvoient `void` ; d'autres renvoient un nouveau tableau. Consulte [Builtins](../generated/builtins.md) plutôt que d'inférer la propriété d'après un nom familier.

## Structs et enums

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

Les types nommés utilisent `UpperCamelCase`. Les valeurs et les fonctions utilisent `snake_case`.

## Unions et match

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

Utilise `Result<T, E>` ou une autre union explicite pour les échecs que l'appelant peut traiter. Apparie les variantes là où tu peux récupérer ou ajouter du contexte. L'opérateur postfixe `?` propage les erreurs de résultat compatibles ; utilise un `match` explicite quand le nettoyage ou le contexte compte.

## Chaînes et chaînes binaires

`string` est du texte. `bstring` est une donnée binaire à longueur explicite et peut contenir des octets nuls. La conversion et le comportement Unicode sont documentés par builtin ou module. N'assume pas que les indices d'octet sont des indices de caractère Unicode.

## Collections

Les tableaux et les tables de hachage intégrés sont distincts des modules de collection adossés à C. La page générée [Builtins](../generated/builtins.md) liste les graphies exactes des builtins. La page générée [Modules](../generated/modules.md) liste les déclarations de module et les frontières de compilation native.

