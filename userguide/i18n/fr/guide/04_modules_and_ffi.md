---
title: Modules, FFI et ressources
machine_generated: true
reviewed: false
lang: fr
---

> Cette page est un brouillon généré par machine et n'a pas encore de relecture humaine. Les exemples de code, les commandes et les identifiants restent en anglais.

# Modules, FFI et ressources

Un module est une frontière de visibilité et de sûreté. Le code nouveau doit utiliser `module`, un alias, et des noms publics qualifiés.

## Importer un module

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

L'analyseur accepte encore les formes héritées `import` et `from ... import ...`. Ne les choisis pas pour du code nouveau.

## Visibilité

Les déclarations sont privées par défaut. Marque la surface prévue avec `pub` :

```nano
fn internal_scale(value: int) -> int {
    return (* value 2)
}

pub fn transform(value: int) -> int {
    return (+ (internal_scale value) 1)
}
```

Les aides privées restent appelables dans leur module. Les importateurs ne peuvent appeler que les déclarations publiques.

## Fonctions étrangères

Les déclarations étrangères utilisent `extern fn`. Un appel direct exige `unsafe` sauf si le module importé entier est unsafe :

```nano
extern fn c_close(fd: int) -> int

fn close_fd(fd: int) -> int {
    unsafe {
        return (c_close fd)
    }
}
```

Garde les régions non sûres étroites. Valide les valeurs étrangères à la frontière et expose un enrobage typé lorsqu'on peut honnêtement en fournir un.

## Types de ressource

`resource struct` marque une ressource affine qui doit être consommée au plus une fois :

```nano
resource struct FileHandle {
    fd: int
}
```

Mon vérificateur de ressources détecte des cas importants d'usage après consommation et de consommation répétée. Ce n'est pas une preuve complète de propriété. Annote les locales de ressource explicitement et inspecte chaque chemin de retour pour le nettoyage.

## Métadonnées de module

Trois fichiers servent des emplois différents :

| File | Purpose |
| --- | --- |
| `.nano` | Source et déclarations publiques |
| `module.json` | Sources de compilation native, drapeaux, paquets et métadonnées de propriété |
| `module.manifest.json` | Métadonnées de découverte, stabilité, capacités et exemples |

Les modules purs n'ont pas toujours besoin de métadonnées de compilation native. Voir l'[module inventory](../generated/modules.md) généré pour ce qui existe maintenant.

