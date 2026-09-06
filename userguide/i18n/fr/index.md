---
title: Accueil
machine_generated: true
reviewed: false
lang: fr
---

> Cette page est un brouillon généré par machine et n'a pas encore de relecture humaine. Les exemples de code, les commandes et les identifiants restent en anglais.

# Guide utilisateur NanoLang

Je suis NanoLang. J'utilise des frontières explicites, des appels préfixes, des opérateurs de même précédence, et des tests à côté des fonctions qu'ils exercent.
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
## Commencer ici

1. [Construis-moi et exécute un premier programme](guide/01_getting_started.md).
2. [Apprends mon langage](guide/02_language.md).
3. [Utilise des données structurées et des erreurs typées](guide/03_data_and_errors.md).
4. [Travaille avec les modules, le code étranger et les ressources](guide/04_modules_and_ffi.md).
5. [Comprends les ombres, les tests et ma frontière vérifiée](guide/05_testing_and_trust.md).
6. [Choisis un outil ou un backend](guide/06_tools_and_backends.md).
7. [Mesure la performance native et ajuste d'après l'évidence](guide/07_performance_profiling.md).

## Référence

- Les [exemples](generated/examples.md) sont générés depuis chaque `.nano` sous `examples/`.
- Les [builtins](generated/builtins.md) viennent de la référence de bibliothèque standard vérifiée mécaniquement.
- Les [modules](generated/modules.md) sont générés depuis l'arbre des modules et les manifestes.
- Le [CLI du compilateur](generated/cli.md) vient du compilateur utilisé pour construire ce guide.
- [NanoISA](https://github.com/jordanhubbard/nanolang/blob/main/docs/NANOISA.md) est ma frontière de VM typée partagée.

## Ce que je promets

- Les appels de fonction utilisent la syntaxe `(function argument)`.
- Les paramètres et les types de retour sont explicites.
- Les liaisons locales peuvent utiliser l'inférence quand l'initialiseur rend le type évident.
- Tous les opérateurs infixes ont la même précédence et s'associent de gauche à droite.
- La politique du projet exige des ombres utiles pour les fonctions nommées modifiées. L'application du compilateur a des exemptions documentées.
- Une ombre qui passe est un test, pas une preuve.
- Mon backend C est le chemin de compilation de production. Les autres backends ont des sous-ensembles documentés plus étroits.

L'analyseur, le vérificateur de types, le registre des builtins et les tests font autorité quand un ancien document contredit. Je préfère corriger un guide plutôt que conserver une erreur confiante.

