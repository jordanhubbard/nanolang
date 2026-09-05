---
title: Premiers pas
machine_generated: true
reviewed: false
lang: fr
---

> Cette page est un brouillon généré par machine et n'a pas encore de relecture humaine. Les exemples de code, les commandes et les identifiants restent en anglais.

# Premiers pas

Je construis actuellement sur des systèmes de type Unix. Les utilisateurs Windows doivent utiliser WSL2.

## Compilation

Il te faut un compilateur C, Make, Git et pkg-config. Clone et construis :

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build
./bin/nanoc --version
```

`bin/nanoc` est mon compilateur. `bin/nano` est mon interpréteur qui parcourt l'arbre.

## Premier programme

Crée `hello.nano` :

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

Compile-le et exécute-le :

```bash
./bin/nanoc hello.nano -o hello
./hello
```

Je transpile ce programme vers C et j'invoque le compilateur C hôte. Les ombres s'exécutent pendant que je compile ; le programme résultant exécute ensuite son harnais d'ombres généré au démarrage. Une ombre en échec arrête le processus.

## Interpréteur

Exécute un fichier source sans produire d'exécutable natif :

```bash
./bin/nano hello.nano
```

L'interpréteur et le backend compilé partagent le langage, mais n'ont pas les mêmes frontières d'implémentation. Teste le backend que tu comptes livrer.

## Disposition du projet

Pour un petit programme, un fichier `.nano` suffit. Les paquets peuvent utiliser `nano.toml` :

```text
my_program/
  nano.toml
  main.nano
  src/
```

Voir [`examples/hello_pkg`](https://github.com/jordanhubbard/nanolang/tree/main/examples/hello_pkg) et [`examples/large_project`](https://github.com/jordanhubbard/nanolang/tree/main/examples/large_project) pour des dispositions complètes.

## Suite

Lis [Langage](02_language.md) pour les appels, les opérateurs, les liaisons, les fonctions et le flot de contrôle.

