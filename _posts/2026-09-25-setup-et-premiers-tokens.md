---
layout: post
title: "Jour 1 — Setup du labo et premiers tokens"
date: 2026-09-25
categories: module-1
---

Premier jour officiel du programme LLM. Au menu : monter l'environnement
de travail complet, et premières manipulations de tokenization.

## Le labo

- **Python 3.12 via [uv](https://docs.astral.sh/uv/)** — venant de Ruby,
  uv c'est bundler + rbenv en un seul outil, et rapide. Projet `labo-llm`
  initialisé avec JupyterLab et numpy.
- **[Ollama](https://ollama.com)** pour faire tourner des modèles en local :
  un `llama3.2:3b` (2 Go) tourne à **~127 tokens/s** sur mon Mac (Apple
  Silicon). Largement de quoi expérimenter sans GPU cloud.
- Premier script : interroger le modèle local via l'API HTTP d'Ollama en
  Python — l'équivalent d'un `Net::HTTP.post` avec un payload JSON, rien
  d'exotique pour qui vient du web.

## Premiers tokens : le français coûte cher

Un LLM ne voit jamais de texte : il voit des **tokens**, des sous-mots
découpés par un algorithme (BPE) dont le vocabulaire a été appris
majoritairement sur de l'anglais. Vérification avec `tiktoken` :

```text
"The quick brown fox jumps over the lazy dog."          → 10 tokens
"Le rapide renard brun saute par-dessus le chien
 paresseux."                                            → 18 tokens (+80 %)
```

Même sens, presque le double de tokens. Sur cette phrase l'écart est
extrême, mais la tendance est générale : **le français coûte 20 à 30 % plus
cher** en tokens que l'anglais — donc en facturation API, et en place dans
la fenêtre de contexte.

Autre découverte parlante : `strawberry` est découpé en `str` + `aw` +
`berry`. Le modèle ne voit pas les lettres — voilà pourquoi « combien de r
dans strawberry ? » a piégé tant de modèles.

## Prochaine étape

Finir la semaine 3 du cours d'Andrew Ng (régression logistique,
régularisation), puis la vidéo de Karpathy sur le tokenizer BPE, à
réimplémenter en Python.
