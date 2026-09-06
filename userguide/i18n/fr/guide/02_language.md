---
title: Langage
machine_generated: true
reviewed: false
lang: fr
---

> Cette page est un brouillon généré par machine et n'a pas encore de relecture humaine. Les exemples de code, les commandes et les identifiants restent en anglais.

# Langage

Je garde la syntaxe ordinaire explicite. J'accepte quelques commodités, mais je ne cache ni la précédence ni les signatures de fonction.

## Appels et opérateurs

Les appels sont entre parenthèses et préfixes :

```nano
(println "ready")
(distance x y)
(math.clamp value low high)
```

Les opérateurs acceptent les notations préfixe et infixe :

```nano
let prefix = (+ 2 (* 3 4))
let infix = 2 + (3 * 4)
```

Tous les opérateurs binaires infixes ont la même précédence et s'associent de gauche à droite. `2 + 3 * 4` signifie `(2 + 3) * 4`. Ajoute des parenthèses quand le groupement compte.

## Liaisons

Les liaisons sont immuables sauf si elles sont marquées `mut`. La mutation utilise `set` :

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

Les annotations locales sont facultatives quand l'initialiseur détermine le type :

```nano
let count = 3
let name = "Ada"
```

Annote les collections vides, les valeurs génériques, les poignées étrangères, les valeurs de ressource, et les frontières où l'inférence obscurcirait l'intention. Les paramètres et les types de retour restent explicites.

## Types scalaires

Les types scalaires courants sont `int`, `u8` ou `byte`, `float`, `bool`, `string`, `bstring` et `void`. Les tableaux, les tuples, les enregistrements nommés, les enums, les unions, les types de fonction, les types génériques, les enregistrements ouverts et les types étrangers opaques étendent cet ensemble.

## Flot de contrôle

Un `if` peut omettre `else` :

```nano
if needs_redraw {
    (draw scene)
}
```

`cond` choisit parmi des valeurs d'expression et exige `else` :

```nano
let sign = (cond
    ((< value 0) -1)
    ((> value 0) 1)
    (else 0)
)
```

Les boucles utilisent `while` ou `for` :

```nano
while (< index count) {
    set index (+ index 1)
}

for index in (range 0 count) {
    (visit index)
}
```

`break` et `continue` sont disponibles à l'intérieur des boucles.

## Fonctions

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

Je prends en charge la récursion, les fonctions de première classe, les fermetures, les génériques, les préconditions avec `requires`, et les postconditions avec `ensures`. Les contrats sont des propriétés vérifiées d'exécutions ; ce ne sont pas des preuves formelles.

## Commentaires

```nano
# ordinary comment
// accepted line comment
/* block comment */
/// documentation comment
```

Utilise `#` pour le commentaire ordinaire et `///` pour la documentation consommée par les outils.

