---
layout: post
title: "Un tokenizer BPE from scratch"
date: 2026-10-23
categories: module-1
---

Le module LLM de mon programme a vraiment commencé, par la brique la plus
basse de l'édifice : le tokenizer. Au programme : la vidéo d'Andrej Karpathy
[*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE),
puis une réimplémentation complète de l'algorithme **Byte Pair Encoding**
en Python pur — entraînement, encodage, décodage.

## L'algorithme tient en cinq fonctions

BPE part d'un constat simple : tout texte est une suite de bytes (848
caractères de mon corpus → 868 bytes, merci les accents), donc un
vocabulaire de départ de 256 tokens suffit à représenter *n'importe quoi*.
L'entraînement consiste ensuite à répéter une seule opération : trouver la
paire de tokens adjacents la plus fréquente, et la fusionner en un
nouveau token.

```python
def train(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges = {}                        # {paire: nouvel id}, ordre d'insertion = ordre d'apprentissage
    for i in range(256, vocab_size):
        stats = get_stats(ids)         # compter les paires adjacentes
        pair = max(stats, key=stats.get)
        merges[pair] = i
        ids = merge(ids, pair, i)      # remplacer chaque occurrence par le nouvel id
    return merges
```

Avec `get_stats`, `merge`, `encode` et `decode`, l'ensemble fait une
quarantaine de lignes. C'est toute la magie du truc : le mécanisme qui
découpe le texte pour GPT ou Claude n'a rien d'ésotérique — c'est de la
compression gloutonne, codable en une soirée.

## La subtilité qui fait tout : deux critères différents

Le piège conceptuel de BPE : pendant `train`, on choisit la paire **la
plus fréquente**. Pendant `encode`, on choisit la paire **apprise en
premier**. Ce n'est pas le même critère, et
c'est voulu : à l'encodage, il faut rejouer les fusions exactement dans
l'ordre où elles ont été apprises, sinon on fabrique des tokens que
l'entraînement n'a jamais produits.

Petit détail qui a son importance dans cette implémentation précise : la
structure qui porte cette garantie, c'est l'ordre d'insertion d'un `dict`
Python.

```python
pieges = {(115, 116): 256, (101, 115): 257}   # (s,t) appris AVANT (e,s)
assert encode("est", pieges) == [101, 256]     # le merge le plus ancien gagne
```

Un balayage naïf de gauche à droite fusionnerait `(e,s)` en premier et
échouerait.

## Ce que le tokenizer sait, et ce qu'il ignore

Mon corpus d'entraînement était un texte sur les flux de paiement
(déformation professionnelle assumée : fraude, cantonnement,
authentification forte). Avec 44 fusions apprises, le résultat est déjà
parlant :

```text
"authentification"  →  8 tokens    (le mot est dans le corpus)
"blockchain"        →  9 tokens    (jamais vu : quasi un token par lettre)
```

Un mot fréquent dans le corpus se compresse bien, un mot absent reste en
miettes. C'est exactement pourquoi les tokenizers des vrais modèles
s'entraînent sur des corpus gigantesques, et pourquoi un modèle peut
« voir » certains mots comme des unités et d'autres comme des suites de
fragments. La tokenization est déjà une représentation du monde.

## Prochaine étape

Finir la semaine 2 du cours 2 d'Andrew Ng (entraînement des réseaux de
neurones, avec au passage une belle histoire de stabilité numérique
autour de `from_logits=True`, qui mériterait son billet), puis le gros
morceau du module 1 : le nanoGPT de Karpathy.
