---
layout: post
title: "Un GPT from scratch, et le bug que mes tests ne pouvaient pas voir"
date: 2026-11-27
categories: module-1
---

Suite du module 1 : après [le tokenizer
BPE](/journal/2026/10/23/tokenizer-bpe-from-scratch/), le gros morceau.
Reconstruire un GPT entier, en suivant la vidéo d'Andrej Karpathy
[*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY). Le
code est là : [nano-gpt-fr](https://github.com/Berlimioz/nano-gpt-fr).

Pour être exacte sur ce que j'ai fait moi-même : c'était un notebook à
trous. Les signatures, les shapes attendues et les asserts étaient
fournis ; la logique de chaque brique était à moi, et la vidéo servait de
correction après coup, pas de source pendant.

## Un modèle à l'échelle d'un Mac

Config réduite pour que ça tourne sur `mps` : 128 canaux, 4 têtes
d'attention, 3 couches, une fenêtre de 64 caractères. **618 817
paramètres**, soit 0,62 M — Karpathy, sur sa config, en a 17 fois plus
(10,79 M), et les vrais modèles sont cinq à six ordres de grandeur
au-dessus.

Le corpus est `tinyshakespeare`, un million de caractères, avec un
tokenizer caractère : 65 tokens, un par caractère distinct du texte.

La loss initiale est le premier moment satisfaisant. Avant tout
entraînement, un modèle qui répondrait « les 65 caractères sont
équiprobables » obtiendrait `ln(65) = 4,174`. Le modèle non entraîné
affiche **4,32**, et c'est normal : à l'initialisation ses scores ne sont
pas exactement nuls, et toute déviation de l'uniforme *augmente* la
cross-entropy. Autrement dit, ce 4,17 est un plancher d'ignorance qu'on ne
peut pas améliorer par hasard. 3000 steps plus tard, la loss de
validation est à **1,86**.

Ce qui en sort :

```text
CATINIUS:
For did be with fear rauy oun;
Shall can I' to sabefell he rence with dence,
Sowas drobecite: yet, that whis whid, cose heyelZus,
```

Du charabia, mais du charabia **structuré** : des noms en majuscules
suivis de deux-points, des répliques de longueur plausible, de la
ponctuation aux bons endroits. La forme est acquise, le sens non — et
c'est exactement le partage attendu ici. La structure théâtre est une
régularité courte, elle tient dans la fenêtre de 64 caractères. La
cohérence d'une phrase à l'autre, non : elle n'est pas ratée, elle est
impossible par construction. S'ajoute le coût du tokenizer caractère, qui
oblige le modèle à apprendre à *épeler* avant de pouvoir apprendre à dire
quoi que ce soit — `drobecite`, `conquingfy't`.

## La subtilité qui fait tout : un bug que les tests ne pouvaient pas voir

Voilà ce que j'avais écrit dans le bloc, à côté de la recette :

```python
x = self.sa(x) + self.ln1(x)     # ce que j'avais écrit
x = x + self.sa(self.ln1(x))     # la recette
```

Deux choses sont déplacées. L'attention reçoit `x` brut au lieu de la
version normalisée, et la voie de contournement — celle qui doit porter
`x` intact d'un bout à l'autre du bloc — porte une version normalisée de
`x`. Ce n'est pas la même machine.

**Tous les tests passaient.** Parce qu'ils vérifiaient des shapes, et
qu'une shape ne voit pas ça : la sortie reste `(batch, positions,
canaux)` quel que soit le branchement interne. Une erreur de structure
peut traverser une suite d'asserts sans en déclencher un seul.

J'ai relancé l'entraînement avec les deux versions. Ma version fausse
finit à **1,8171** de validation, la recette à **1,8628**. La version
fausse fait mieux.

Ce qu'on peut en conclure est très limité, et c'est le vrai apprentissage
de la journée. Un run chacun, une seule graine aléatoire : 0,046 d'écart
ne se compare à rien, je n'ai aucune mesure du bruit entre deux runs
identiques. Et surtout, ma variante gardait une voie de contournement
quasi intacte — j'ai comparé deux câblages résiduels, pas « avec » contre
« sans ». Les connexions résiduelles servent à ce que le signal
d'apprentissage atteigne les premières couches malgré l'empilement ; à
trois couches, il n'y a rien à sauver. C'est un argument de profondeur, et
mon modèle n'en a pas.

Le résultat intéressant n'est donc pas le chiffre. C'est le trou de test.

## Quatre tests pour ce que les shapes ne voient pas

J'ai cherché ce qui pouvait être cassé sans qu'aucun assert existant ne le
voie. J'en ai trouvé quatre familles.

**Un sous-module mal réglé.** `nn.Dropout()` sans argument vaut 0,5, pas
la valeur de ma config. Rien ne le testait → un audit qui parcourt tout le
modèle et vérifie que les 18 modules de dropout sont bien à 0,1.

**Un câblage faux.** Celui ci-dessus. Le test met à zéro les sorties des
deux sous-modules du bloc et vérifie qu'il devient alors l'**identité
parfaite** : `bloc(x) == x`. Sans vraie voie de contournement, cette
égalité est fausse.

**Une tranche mal prise.** Tous mes tests tournaient avec des séquences de
64 caractères, la longueur maximale. Le masque qui empêche un token de
regarder la suite du texte doit être tronqué à la longueur courante — si
je l'avais oublié, aucun test ne l'aurait vu, et ça aurait cassé à la
première génération, où la séquence fait un caractère.

**Un facteur scalaire manquant.** Les scores d'attention sont divisés par
la racine de la taille de tête. L'oublier ne change aucune shape et ne
lève aucune erreur. Le test recalcule la sortie attendue une fois **avec**
la division et une fois **sans**, et vérifie laquelle le modèle reproduit.

Ce dernier test porte un second assert, qui teste le test :

```python
assert not torch.allclose(ref_avec, ref_sans, atol=1e-3), "test sans dents"
```

Sans lui, un test qui ne distingue rien resterait vert éternellement.
J'ai vérifié les deux derniers en réintroduisant volontairement leur bug —
ils tombent bien.

Et ce test affiche un chiffre qui m'a fait plaisir : écart-type des scores
`1,334` avant division, `0,334` après. Or `1,334 = 4/3`, et `4` est la
racine de 16, la taille de mes têtes. La division ne cherche pas à obtenir
un écart-type de 1 : elle rend l'échelle des scores **indépendante de la
taille des têtes**. Avec des têtes quatre fois plus grandes, les scores
bruts seraient quatre fois plus gros et la division les ramènerait à la
même valeur. C'était la partie de la vidéo que j'avais le moins comprise ;
la voir tomber juste sur mon propre modèle valait toutes les
réexplications du monde.

## Ce que ce n'est pas

Ce n'est pas un ChatGPT. C'est uniquement la phase de pré-entraînement :
le modèle continue du texte, il ne répond pas à des questions. Aucune des
phases suivantes n'est là. Le corpus est anglais. Le tokenizer caractère
était fourni — le seul que j'aie écrit moi-même est le BPE du billet
précédent. Le tirage des caractères est brut, sans température ni top-k.

Et une limite que je note parce qu'elle est instructive : **je n'ai jamais
sauvegardé les poids.** Pas de `torch.save` dans le notebook, kernel mort
depuis, modèle disparu. Les chiffres de ce billet viennent des sorties
enregistrées dans le fichier `.ipynb`, pas d'un modèle rejouable. Première
leçon de MLOps, apprise par la méthode la plus directe.

## Prochaine étape

Réentraîner en sauvegardant, cette fois. La validation descendait encore
au step 3000, donc il y a de la marge à gratter simplement en continuant.
Puis la température et le top-k à la génération, pour voir ce que ça change
sur le texte. Et ensuite la suite du programme : le pré-entraînement n'est
que la première phase.
