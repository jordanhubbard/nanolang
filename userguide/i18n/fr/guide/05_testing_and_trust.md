---
title: Tests et confiance
machine_generated: true
reviewed: false
lang: fr
---

> Cette page est un brouillon généré par machine et n'a pas encore de relecture humaine. Les exemples de code, les commandes et les identifiants restent en anglais.

# Tests et confiance

Je distingue les tests, les vérifications statiques et les preuves. Ils répondent à des questions différentes.

## Ombres

Une ombre est un test exécutable attaché à une fonction :

```nano
fn double(value: int) -> int {
    return (* value 2)
}

shadow double {
    assert (== (double 0) 0)
    assert (== (double 3) 6)
}
```

Pendant la compilation ordinaire j'exécute les ombres dans l'interpréteur hôte. Les programmes natifs générés contiennent aussi leur harnais d'ombres. Une ombre teste les cas qu'elle exécute ; elle ne prouve pas la fonction pour chaque entrée.

L'application du compilateur et la politique du projet diffèrent :

- Le compilateur avertit normalement d'une ombre manquante.
- Il exempte les fonctions extern, `main`, les lambdas générées, les fonctions GPU, et les fonctions qui appellent des externs.
- La politique du dépôt exige une ombre utile pour chaque fonction nommée non-extern ajoutée ou changée lorsqu'elle peut être testée.

## Tests de propriétés et couverture

Les tests de propriétés échantillonnent des valeurs générées et peuvent réduire les échecs. L'échantillonnage n'est pas une vérification exhaustive. La couverture rapporte quel code s'est exécuté ; elle n'établit pas la correction.

## Contrats

`requires` vérifie les préconditions et `ensures` vérifie les postconditions à leur frontière implémentée. Une vérification d'exécution réussie dit que cette condition a tenu pour cette exécution.

## Vérification formelle

Mon développement Coq prouve la métathéorie énoncée pour un modèle de noyau défini. Il ne prouve pas automatiquement chaque backend, appel étranger, allocateur, module ou fonction utilisateur.

Utilise le rapport de confiance pour inspecter la frontière :

```bash
./bin/nanoc program.nano --trust-report
```

Dis **proved** seulement pour un théorème vérifié dans son modèle énoncé, **tested** seulement pour un comportement exercé par un test nommé, et **assumed** pour un comportement de plateforme ou étranger non vérifié.

## Vérifications utiles

```bash
make test
make userguide-check
make check-stdlib-docs
python3 scripts/check_markdown_links.py
```

