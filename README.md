# RAG and Agentic AI

Study and practice repository for an IBM course on Retrieval-Augmented Generation and agentic AI, taken as part of a certification track.

*[Version française ci-dessous](#rag-et-ia-agentique)*

---

## Purpose

This repository is a learning workspace, not a product. It holds the code written while working through the course: notebooks, experiments, and small implementations built to understand how retrieval and agent systems actually behave rather than to ship anything.

Expect the code to favour clarity over robustness, and expect things to be rewritten as the material progresses.

## Topics covered

The course material spans two connected areas:

- **RAG** — document loading and chunking, embeddings, vector stores, retrievers, and grounding a model's answers in retrieved context.
- **Agentic AI** — tool calling, reasoning loops, state and memory, multi-step orchestration, and agents that decide their own next action.

## Stack

- Python 3.12
- [LangChain](https://python.langchain.com/) v1 and [LangGraph](https://langchain-ai.github.io/langgraph/)

## Getting started

Create and activate the virtual environment:

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install langchain
```

> Model provider API keys are read from a `.env` file, which is git-ignored. Never commit it.

## Repository conventions

Vector indexes, embeddings, model weights, and raw or processed datasets are excluded from version control (see [.gitignore](.gitignore)). Everything indexable should be reproducible by re-running the code — the artefacts themselves are not part of the history.

---

# RAG et IA agentique

Dépôt d'étude et de pratique d'un cours IBM consacré au Retrieval-Augmented Generation et à l'IA agentique, suivi dans le cadre d'un parcours certifiant.

## Objectif

Ce dépôt est un espace d'apprentissage, pas un produit. Il rassemble le code écrit au fil du cours : notebooks, expérimentations et petites implémentations destinées à comprendre le comportement réel des systèmes de recherche et d'agents, plutôt qu'à livrer quoi que ce soit.

Le code privilégie donc la lisibilité à la robustesse, et sera réécrit à mesure que le cours avance.

## Thèmes abordés

Le cours couvre deux domaines liés :

- **RAG** — chargement et découpage de documents, embeddings, bases vectorielles, retrievers, et ancrage des réponses du modèle dans le contexte récupéré.
- **IA agentique** — appel d'outils, boucles de raisonnement, état et mémoire, orchestration multi-étapes, et agents qui décident eux-mêmes de leur prochaine action.

## Stack technique

- Python 3.12
- [LangChain](https://python.langchain.com/) v1 et [LangGraph](https://langchain-ai.github.io/langgraph/)

## Démarrage

Créer puis activer l'environnement virtuel :

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
pip install langchain
```

> Les clés d'API des fournisseurs de modèles sont lues depuis un fichier `.env`, ignoré par git. Ne jamais le committer.

## Conventions du dépôt

Les index vectoriels, embeddings, poids de modèles et jeux de données (bruts ou traités) sont exclus du versionnement (voir [.gitignore](.gitignore)). Tout ce qui est indexable doit pouvoir être reconstruit en réexécutant le code : les artefacts eux-mêmes ne font pas partie de l'historique.
