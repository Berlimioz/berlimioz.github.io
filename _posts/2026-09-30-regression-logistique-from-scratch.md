---
layout: post
title: "Cours 1 terminé — coder la régression logistique pour enfin comprendre la régularisation"
date: 2026-09-30
categories: module-1
---

Le premier cours de la [spécialisation Machine Learning](https://www.coursera.org/specializations/machine-learning-introduction)
d'Andrew Ng (Stanford/DeepLearning.AI) est terminé. Les concepts sont en
place — mais si je suis honnête, ce n'est pas en regardant les vidéos
qu'ils se sont ancrés. C'est ce qui s'est passé *autour* du cours qui a
fait la différence.

## Le cours pose les concepts, le code les ancre

La dernière ligne droite (régression logistique, overfitting,
régularisation) était la plus dense conceptuellement. Les vidéos font le
travail — Andrew Ng explique bien, et j'avais besoin de ces explications
mathématiques avant de pouvoir écrire quoi que ce soit. Mais entre
« hocher la tête devant la formule » et « savoir la refaire », il y a un
fossé que le visionnage seul ne comble pas : une notion comprise à
l'écran restait abstraite, et ma concentration s'effritait à mesure que
les formules s'empilaient.

Ce qui a comblé le fossé : **coder chaque notion**. Le cours s'y prête,
il faut le dire — les labs fournissent tout l'échafaudage (données,
graphes, structure), et les labs notés font écrire soi-même le cœur des
méthodes : la sigmoïde, le coût, les gradients. Puis, pour vérifier que
ça tenait sans filet, un notebook personnel en complément : refaire le
pipeline entier en numpy, sans échafaudage cette fois, sur un dataset
synthétique de détection de fraude (clin d'œil à mon ancien métier). Une
formule qu'on vient de coder n'est plus une formule — c'est trois lignes
de Python dont on a vu les asserts passer.

L'autre ingrédient, il faut le dire parce que c'est un vrai changement
par rapport à mes apprentissages d'avant : **un tuteur IA à la demande**,
sollicité dans les deux directions. Côté maths, quand une notion du cours
ne rentrait pas — pourquoi ajoute-t-on ce terme en λ à la fonction de
coût ? — une reformulation en langage de dev (« une taxe sur les gros
poids : coller à chaque point de donnée doit maintenant valoir son
prix ») débloquait en cinq secondes ce qui serait resté flou. Côté code, pour les
micro-frictions de qui a vingt ans de Ruby et trois semaines de numpy :
c'est quoi déjà un logarithme, pourquoi `-log(0.5)` ne donne pas 0.693
(calculatrice = base 10, `np.log` = log naturel), comment marche `X @ w`.
Avant, chaque accroc c'était vingt minutes de forums ; là, la friction ne
s'accumule jamais — et pour quelqu'un qui apprend précisément à
construire ces outils, la boucle est plutôt savoureuse.

## L'algorithme tient dans une quinzaine de lignes

Quatre fonctions numpy, aucune magie :

```python
def sigmoid(z):                      # écrase un score dans ]0,1[ → une proba
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b):          # score linéaire, puis écrasement
    return sigmoid(X @ w + b)

def compute_cost(X, y, w, b):        # log-loss : punir la confiance dans le faux
    p = np.clip(predict_proba(X, w, b), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def compute_gradients(X, y, w, b):   # la pente de chaque paramètre
    err = predict_proba(X, w, b) - y
    return X.T @ err / len(y), err.mean()
```

Note pour les rubyistes : pas un seul `map`. En numpy, les opérations
s'appliquent élément par élément toutes seules (*broadcasting*) — le `map`
est implicite et compilé en C.

Trouvaille au passage : le **gradient checking**, ou comment tester une
dérivée comme on teste du code. On bouge un paramètre d'un chouïa
(`1e-6`), on mesure de combien bouge le coût, et le ratio doit coller à ce
que renvoie la formule. C'est littéralement la définition de la dérivée,
déguisée en test unitaire — deux implémentations indépendantes qui doivent
tomber d'accord.

## L'overfitting, il faut le voir une fois

Le clou du notebook. Recette : seulement 30 exemples d'entraînement (dont
quelques étiquettes volontairement fausses, comme dans la vraie vie), et un
modèle rendu très flexible avec des features polynomiales de degré 6. Puis
on entraîne, avec et sans régularisation L2, et on mesure sur 200 exemples
jamais vus :

```text
λ = 0     →  train 100 %   réalité 59 %    frontière contorsionnée
λ = 0.3   →  train  87 %   réalité 81 %    frontière lisse
λ = 5     →  train  77 %   réalité 68 %    modèle écrasé (underfitting)
```

La ligne du milieu est la plus instructive : le modèle régularisé fait
**moins bien sur ses données d'entraînement** (il refuse d'apprendre par
cœur les étiquettes fausses) et **beaucoup mieux dans la réalité**. C'est
exactement le deal de la régularisation : une taxe sur les gros
coefficients — deux lignes de code — qui force le modèle à arbitrer entre
coller à chaque point et rester simple.

La leçon qui reste, formulable en termes de management : **un modèle
optimise sa note, pas ton intention**. La descente de gradient minimise la
fonction de coût, point. Si la note ne mesure que l'erreur d'entraînement,
il apprendra le bruit par cœur. La régularisation, c'est mettre la
simplicité *dans la note* — l'équivalent ML de « les gens optimisent ce
qu'on mesure ».

## Prochaine étape

Les deux premières semaines du cours 2 (réseaux de neurones) en parallèle
du vrai début du module 1 : la vidéo de Karpathy sur le tokenizer BPE, à
réimplémenter en Python.
