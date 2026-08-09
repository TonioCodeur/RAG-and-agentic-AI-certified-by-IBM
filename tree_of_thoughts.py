"""Arbre de pensées (Tree of Thoughts) appliqué au modèle défini dans le `.env`.

Au lieu de demander une réponse d'un seul jet, le script fait explorer au
modèle plusieurs raisonnements en parallèle :

    1. EXPANSION  — pour chaque piste retenue, le modèle propose N pensées
       suivantes (une étape de raisonnement, pas la réponse finale).
    2. ÉVALUATION — le modèle note toutes les pensées d'un même niveau, en une
       seule requête, ce qui lui permet de les comparer entre elles.
    3. SÉLECTION  — seules les K meilleures survivent (recherche en faisceau).
    4. Retour en 1 jusqu'à la profondeur voulue, puis SYNTHÈSE de la réponse
       finale à partir du meilleur chemin.

L'arbre complet est affiché avec les notes, la réponse finale est diffusée en
flux, et l'ensemble est ajouté à la fin de `output_tree.md`.

Usage :
    .\\.venv\\Scripts\\python.exe tree_of_thoughts.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

# La console Windows utilise cp1252 par défaut : les caractères de dessin de
# l'arbre la feraient planter. On force UTF-8 sur la sortie standard.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Le SDK OpenAI lit lui-même ces variables : vides, elles écrasent ses valeurs
# par défaut et l'appel échoue sur un « Connection error. » trompeur.
for _name in ("OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
    if not (os.environ.get(_name) or "").strip():
        os.environ.pop(_name, None)

DEFAULT_MODEL = "gpt-5.4-mini"
QUIT_WORDS = {"quit", "exit", "q", "quitter"}
OUTPUT_FILE = ROOT / "output_tree.md"

# --- Forme de l'arbre -------------------------------------------------------
# Coût : 1 expansion par piste retenue + 1 évaluation par étape + 1 synthèse,
# soit 9 appels avec ces valeurs (1+2+2 expansions, 3 évaluations, 1 synthèse).
TREE_DEPTH = 3  # nombre d'étapes de raisonnement successives
BRANCHING = 3  # pensées proposées par piste et par étape
BEAM_WIDTH = 2  # pistes conservées après chaque évaluation

EXPLORER_SYSTEM = """\
Tu explores un problème par étapes, à la manière d'un arbre de pensées.
On te donne un problème et le chemin de raisonnement déjà parcouru ; tu
proposes UNIQUEMENT la prochaine étape, sous forme de plusieurs pistes
distinctes et mutuellement différentes.

Règles :
- Une pensée = une étape de raisonnement, pas la réponse finale.
- Les pistes doivent diverger réellement (approches, angles ou hypothèses
  différents), pas être des reformulations l'une de l'autre.
- Chaque pensée fait 1 à 3 phrases, concrète et vérifiable.

Réponds en JSON, au format exact : {"thoughts": ["...", "...", "..."]}
"""

CRITIC_SYSTEM = """\
Tu évalues des pistes de raisonnement concurrentes pour un même problème.
Note chaque piste de 0 à 10 selon sa capacité à mener à une bonne réponse :
pertinence, solidité, potentiel de progression, absence d'impasse.

Sois discriminant : n'attribue pas la même note à tout le monde.

Réponds en JSON, au format exact :
{"evaluations": [{"id": 1, "score": 7.5, "reason": "..."}]}
Le champ "reason" fait une phrase au maximum.
"""

SYNTHESIS_SYSTEM = """\
On te donne un problème et le meilleur chemin de raisonnement retenu après
exploration d'un arbre de pensées. Rédige la réponse finale en t'appuyant sur
ce chemin.

Ne décris pas l'exploration et ne mentionne pas les notes : produis la réponse
utile, structurée en Markdown, directement exploitable.
"""


# --- Lecture du .env --------------------------------------------------------


def env_str(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def env_float(name: str) -> float | None:
    value = env_str(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        sys.exit(f"[config] {name} doit être un nombre, reçu : {value!r}")


def env_int(name: str) -> int | None:
    value = env_str(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        sys.exit(f"[config] {name} doit être un entier, reçu : {value!r}")


def build_client() -> OpenAI:
    api_key = env_str("OPENAI_API_KEY")
    if not api_key:
        sys.exit("[config] OPENAI_API_KEY manquante. Copie .env.example en .env et renseigne-la.")

    kwargs: dict[str, Any] = {"api_key": api_key}
    for key, value in (
        ("base_url", env_str("OPENAI_BASE_URL")),
        ("organization", env_str("OPENAI_ORG_ID")),
        ("timeout", env_float("OPENAI_TIMEOUT")),
        ("max_retries", env_int("OPENAI_MAX_RETRIES")),
    ):
        if value is not None:
            kwargs[key] = value
    return OpenAI(**kwargs)


def build_request_params() -> dict[str, Any]:
    """Paramètres d'échantillonnage, envoyés seulement s'ils sont définis."""
    optional: dict[str, float | int | None] = {
        "temperature": env_float("OPENAI_TEMPERATURE"),
        "top_p": env_float("OPENAI_TOP_P"),
        "max_completion_tokens": env_int("OPENAI_MAX_TOKENS"),
        "seed": env_int("OPENAI_SEED"),
    }
    return {k: v for k, v in optional.items() if v is not None}


def describe_error(exc: Exception) -> str:
    """Déplie la chaîne des causes : le SDK masque souvent l'erreur réelle."""
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    while cause is not None and len(parts) < 4:
        label = f"{type(cause).__name__}: {cause}"
        if label != parts[-1]:
            parts.append(label)
        cause = cause.__cause__
    return "\n  cause: ".join(parts)


# --- Appels au modèle -------------------------------------------------------


def messages_of(system: str, user: str) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=user),
    ]


def ask_json(client: OpenAI, model: str, system: str, user: str) -> dict[str, Any]:
    """Appelle le modèle en mode JSON strict et renvoie l'objet décodé.

    `response_format` force une sortie JSON valide. Si le point d'entrée ne le
    gère pas, on réessaie sans, puis on récupère le premier objet `{...}` du
    texte. Un échec renvoie un dictionnaire vide : l'appelant décide quoi faire.
    """
    params = build_request_params()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_of(system, user),
            response_format={"type": "json_object"},
            **params,
        )
    except OpenAIError:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages_of(system, user), **params
            )
        except OpenAIError as exc:
            print(f"[erreur] appel au modèle échoué : {describe_error(exc)}")
            return {}

    content = response.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    print("[erreur] réponse illisible : JSON attendu.")
    return {}


def stream_answer(client: OpenAI, model: str, system: str, user: str) -> str | None:
    """Diffuse la réponse au fil des tokens et renvoie le texte complet."""
    chunks: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages_of(system, user),
            stream=True,
            **build_request_params(),
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)
    except OpenAIError as exc:
        print(f"\n[erreur] appel au modèle échoué : {describe_error(exc)}\n")
        return None
    print()
    return "".join(chunks)


# --- Structure de l'arbre ---------------------------------------------------


@dataclass
class Thought:
    """Un nœud de l'arbre : une étape de raisonnement, notée et rattachée."""

    text: str
    depth: int
    parent: Thought | None = None
    score: float = 0.0
    reason: str = ""
    kept: bool = False
    children: list[Thought] = field(default_factory=list)

    def path(self) -> list[Thought]:
        """Le chemin de la racine jusqu'à ce nœud, racine exclue."""
        chain: list[Thought] = []
        node: Thought | None = self
        while node is not None and node.depth > 0:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    def path_text(self) -> str:
        steps = self.path()
        if not steps:
            return "(aucune étape pour l'instant : c'est la première)"
        return "\n".join(f"Étape {i}. {t.text}" for i, t in enumerate(steps, 1))


def expand(client: OpenAI, model: str, problem: str, node: Thought) -> list[Thought]:
    """Demande au modèle BRANCHING pensées suivantes pour ce nœud."""
    user = (
        f"Problème :\n{problem}\n\n"
        f"Chemin de raisonnement déjà parcouru :\n{node.path_text()}\n\n"
        f"Propose {BRANCHING} pistes distinctes pour l'étape suivante, en JSON."
    )
    data = ask_json(client, model, EXPLORER_SYSTEM, user)
    raw = data.get("thoughts")
    if not isinstance(raw, list):
        return []

    children = [
        Thought(text=str(item).strip(), depth=node.depth + 1, parent=node)
        for item in raw[:BRANCHING]
        if str(item).strip()
    ]
    node.children = children
    return children


def evaluate(client: OpenAI, model: str, problem: str, candidates: list[Thought]) -> None:
    """Note toutes les pensées d'un niveau en un seul appel, pour comparaison."""
    listing = "\n\n".join(
        f"[{i}] (issue de : {t.parent.text if t.parent and t.parent.depth else 'la racine'})\n{t.text}"
        for i, t in enumerate(candidates, 1)
    )
    user = (
        f"Problème :\n{problem}\n\n"
        f"Pistes à évaluer :\n{listing}\n\n"
        f"Note les {len(candidates)} pistes, en JSON."
    )
    data = ask_json(client, model, CRITIC_SYSTEM, user)

    scores: dict[int, tuple[float, str]] = {}
    for item in data.get("evaluations", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("id", 0))
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            continue
        scores[index] = (score, str(item.get("reason", "")).strip())

    # Une piste que le modèle a oublié de noter reste à 0 et sera écartée.
    for i, thought in enumerate(candidates, 1):
        thought.score, thought.reason = scores.get(i, (0.0, "non évaluée"))


def select(candidates: list[Thought]) -> list[Thought]:
    """Recherche en faisceau : on ne garde que les BEAM_WIDTH meilleures."""
    ranked = sorted(candidates, key=lambda t: t.score, reverse=True)
    beam = ranked[:BEAM_WIDTH]
    for thought in beam:
        thought.kept = True
    return beam


# --- Affichage et journal ---------------------------------------------------


def render_tree(levels: list[list[Thought]]) -> str:
    """Représentation texte de l'arbre, notes comprises."""
    lines = ["Arbre exploré :"]
    for depth, level in enumerate(levels, 1):
        indent = "  " * depth
        lines.append(f"{indent}Étape {depth}")
        for thought in level:
            marker = "[+]" if thought.kept else "[-]"
            lines.append(f"{indent}  {marker} {thought.score:>4.1f}  {thought.text}")
            if thought.reason:
                lines.append(f"{indent}        ({thought.reason})")
    return "\n".join(lines)


def append_to_output(model: str, problem: str, tree: str, path: str, answer: str) -> None:
    """Ajoute la session à la fin de `output_tree.md`, sans jamais l'écraser."""
    entry = (
        f"\n---\n\n"
        f"## Problème (modèle : {model}, profondeur {TREE_DEPTH}, "
        f"branchement {BRANCHING}, faisceau {BEAM_WIDTH})\n\n{problem}\n\n"
        f"## Arbre de pensées\n\n```text\n{tree}\n```\n\n"
        f"## Chemin retenu\n\n{path}\n\n"
        f"## Réponse finale\n\n{answer.strip()}\n"
    )
    try:
        with OUTPUT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError as exc:
        print(f"[erreur] écriture dans {OUTPUT_FILE.name} impossible : {exc}")


# --- Orchestration ----------------------------------------------------------


def solve(client: OpenAI, model: str, problem: str) -> None:
    """Construit l'arbre, sélectionne le meilleur chemin, synthétise la réponse."""
    root = Thought(text=problem, depth=0, kept=True)
    beam = [root]
    levels: list[list[Thought]] = []

    for depth in range(1, TREE_DEPTH + 1):
        print(f"\n[étape {depth}/{TREE_DEPTH}] génération de {BRANCHING} pistes "
              f"par piste retenue ({len(beam)})...")
        candidates: list[Thought] = []
        for node in beam:
            candidates.extend(expand(client, model, problem, node))

        if not candidates:
            print("[arrêt] le modèle n'a proposé aucune piste exploitable.")
            return

        print(f"[étape {depth}/{TREE_DEPTH}] évaluation de {len(candidates)} pistes...")
        evaluate(client, model, problem, candidates)
        beam = select(candidates)
        levels.append(candidates)
        print(f"[étape {depth}/{TREE_DEPTH}] retenu : "
              + ", ".join(f"{t.score:.1f}" for t in beam))

    tree = render_tree(levels)
    print(f"\n{tree}\n")

    best = beam[0]
    path = "\n".join(f"{i}. {t.text}" for i, t in enumerate(best.path(), 1))
    print(f"--- Chemin retenu (note finale {best.score:.1f}) ---\n{path}\n")

    user = (
        f"Problème :\n{problem}\n\n"
        f"Meilleur chemin de raisonnement retenu :\n{path}\n\n"
        f"Rédige maintenant la réponse finale."
    )
    print(f"--- Réponse de {model} ---")
    answer = stream_answer(client, model, SYNTHESIS_SYSTEM, user)
    if answer is None:
        return
    print("--- fin de la réponse ---")

    append_to_output(model, problem, tree, path, answer)
    print(f"Session ajoutée à la fin de {OUTPUT_FILE.name}.\n")


def main() -> None:
    model = env_str("OPENAI_MODEL") or DEFAULT_MODEL
    client = build_client()

    print(f"Arbre de pensées — modèle : {model}")
    print(f"Profondeur {TREE_DEPTH}, branchement {BRANCHING}, faisceau {BEAM_WIDTH} "
          f"— journal : {OUTPUT_FILE.name}")
    print("Entre un problème à résoudre. Ligne vide ou 'quit' pour sortir.\n")

    while True:
        try:
            problem = input("Problème > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            return

        if not problem or problem.lower() in QUIT_WORDS:
            print("Au revoir.")
            return

        print(f"\nProblème envoyé à {model}.")
        solve(client, model, problem)


if __name__ == "__main__":
    main()
