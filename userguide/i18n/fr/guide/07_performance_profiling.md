---
title: Profilage de performance
machine_generated: true
reviewed: false
lang: fr
---

> Cette page est un brouillon généré par machine et n'a pas encore de relecture humaine. Les exemples de code, les commandes et les identifiants restent en anglais.

# Profilage de performance

J'enveloppe un binaire natif compilé avec `-pg` pour que son exécution collecte
un profil hôte et imprime du JSON qu'un LLM peut lire. Je ne garantis pas un
gain de vitesse. Je garde une optimisation quand un second profil et mes tests
montrent qu'elle a aidé.

La liste canonique des champs, les variables d'environnement de réentrée et les
noms d'artefacts `/tmp` sont dans
[docs/PERFORMANCE_MONITORING.md](https://github.com/jordanhubbard/nanolang/blob/main/docs/PERFORMANCE_MONITORING.md).

## Capturer un profil

```bash
./bin/nanoc program.nano -o bin/program -pg --profile-output profile.json
./bin/program
```

Le premier processus est un coordinateur (`_nl_run_with_profiling`). Il crée un
artefact nommé de façon unique à partir de son identifiant de processus, lance
la charge mesurée dans un enfant, attend, convertit la sortie du collecteur en
JSON, et supprime les fichiers temporaires. Les exécutions concurrentes ne
partagent pas le même chemin `/tmp`.

Le JSON est écrit dans `profile.json` et aussi sur **stdout** (entre deux
lignes de bannière). Rediriger stderr ne le capturera pas.

## Drapeaux qui ne sont pas `-pg`

| Flag | Role |
| --- | --- |
| `-pg` / `--profile-output` | Collecteur OS + JSON. Le `main` natif devient `_nl_run_with_profiling`. |
| `--profile` | Crochets de temps dans le C généré. Tableau sur stderr. |
| `--trace` | Crochets de trace d'appels dans le C généré. `-> fn` / `<- fn` indentés sur stderr. |
| `--profile-runtime` | Écrit aussi des piles repliées `.nano.prof`. Backend natif seulement. |
| `--pgo <file>` | Inline avec un `.nano.prof`. N'utilise pas le JSON de `-pg`. |

Drapeaux C ajoutés avec `-pg` : `-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls`.

## Frontière de plateforme

| Platform | Collector | Measurement |
| --- | --- | --- |
| macOS with full Xcode | `xctrace` Time Profiler | échantillonnage d'un enfant lancé |
| macOS fallback | `sample` | échantillonnage périodique d'un enfant synchronisé |
| Linux | `gprofng collect app` | collecte instrumentée dans un processus enfant |
| Other OS | none | J'imprime que le profilage n'est pas pris en charge et j'exécute le programme |

Je traite xctrace comme disponible seulement quand `which xctrace` et
`xctrace version` réussissent tous les deux. Command Line Tools seul ne suffit pas.

Sur macOS, le repli `sample` crée d'abord l'enfant de charge et le retient
sur un tube (`_NL_PROFILING_ACTIVE`, `_NL_PROFILING_CWD`). Un enfant échantillonneur
s'attache à ce PID (`sample <pid> 60 -f /tmp/nanolang_sample_<pid>.txt -mayDie`),
puis le coordinateur relâche la charge. Les exécutions courtes peuvent encore
finir avant que `sample` s'attache.

Sur Linux, la réentrée utilise `_NL_PROFILING_CHILD` ou `LD_PRELOAD` contenant
`libgp-collector`. Si `gprofng` manque, l'enfant s'exécute sans collecteur.
gprofng est une collecte instrumentée. Le champ JSON `profile_type` reste la
chaîne `"sampling"` sur chaque OS ; c'est une étiquette historique, pas une
affirmation que Linux échantillonne.

## JSON que j'émets vraiment

Chaque point chaud n'a que `function`, `samples` et `pct_time`. Je n'émets pas
de positions source ni de microsecondes par appel.

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

`tool` est `"gprofng"`, `"xctrace"` ou `"sample"`. Sur Linux, `samples` est
le pourcentage exclusif fois dix, pas un vrai compte d'échantillons. Sur
xctrace j'émets les 20 premiers noms qui commencent par `nl_`.

Traite un profil comme la mesure d'une charge sur une machine.

## Régler avec un LLM

1. Compile avec `-pg --profile-output`.
2. Exécute une charge représentative, pas une ombre d'une ligne.
3. Donne à un LLM le JSON, la source pertinente, et la commande qui a produit
   la charge. Demande un changement lié à un point chaud mesuré.
4. Lance les tests, puis reprofile la même charge.
5. Garde le changement seulement si les tests passent encore et que les
   nombres s'améliorent.

Un LLM peut suggérer une optimisation ; le profil peut tester son effet. Ni l'un
ni l'autre ne remplace l'autre. Compare l'horloge murale **sans** `-pg` quand tu
rapportes la vitesse.

Le traçage et le profilage partagent un crochet de démarrage. Les binaires
`--profile` honorent `NANO_PROFILE`, et les `--trace` honorent `NANO_TRACE` ;
mettre l'un ou l'autre à `0` désactive ce crochet à l'exécution sans recompiler,
et un crochet désactivé ne fait aucun travail par événement.

`--pgo` lit `.nano.prof` depuis `--profile-runtime`. Ce fichier n'est pas une
entrée PGO du JSON de `-pg`.

