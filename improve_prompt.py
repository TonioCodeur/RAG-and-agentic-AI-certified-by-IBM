"""Boucle interactive d'amélioration de prompts.

L'utilisateur saisit un prompt au clavier ; le script y joint les consignes de
`prompt.md`, envoie le tout au LLM, affiche la réponse et l'ajoute à la suite
de `output.md` (le fichier n'est jamais écrasé). On peut enchaîner autant de
prompts que l'on veut ; ligne vide, `quit` ou Ctrl+C pour sortir.

Usage :
    .\\.venv\\Scripts\\python.exe improve_prompt.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Le SDK OpenAI lit lui-même certaines variables d'environnement. Une variable
# déclarée mais vide dans le .env (`OPENAI_BASE_URL=`) n'est pas « absente » :
# elle écrase la valeur par défaut, l'URL part sans `https://` et l'appel échoue
# sur un laconique « Connection error. ». On purge donc les valeurs vides.
for _name in ("OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
    if not (os.environ.get(_name) or "").strip():
        os.environ.pop(_name, None)

DEFAULT_MODEL = "gpt-5.4-mini"
QUIT_WORDS = {"quit", "exit", "q", "quitter"}

# Consignes jointes à chaque prompt, et journal des réponses (jamais écrasé).
PROMPT_FILE = ROOT / "prompt.md"
OUTPUT_FILE = ROOT / "output.md"

# Consignes utilisées si `prompt.md` est vide/absent et que le .env n'en fournit aucune.
DEFAULT_SYSTEM_PROMPT = """\
Tu es un expert en prompt engineering. On te fournit un prompt brut rédigé par \
un utilisateur ; ta mission est de le réécrire pour qu'il produise de bien \
meilleurs résultats avec un LLM.

Méthode :
1. Identifie l'intention réelle, les ambiguïtés et les informations manquantes.
2. Réécris le prompt en appliquant les techniques pertinentes : rôle explicite, \
contexte, tâche précise, contraintes, format de sortie attendu, critères de \
qualité, exemples (few-shot) si utile, découpage en étapes si la tâche est \
complexe. La sobriété prime : pas de technique inutile.
3. N'invente pas de faits : si une information manque, laisse un emplacement \
explicite entre accolades, par exemple {public_cible}.

Réponds STRICTEMENT dans ce format Markdown :

## Analyse
- (3 à 6 puces : faiblesses du prompt d'origine)

## Prompt amélioré
```text
(le prompt réécrit, prêt à copier-coller, et rien d'autre dans ce bloc)
```

## Choix appliqués
- (2 à 5 puces : techniques utilisées et pourquoi)

## Questions ouvertes
- (informations manquantes ; "aucune" si tout est clair)
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


def load_system_prompt() -> str:
    """Le texte joint à chaque prompt.

    Priorité : `prompt.md`, puis PE_SYSTEM_PROMPT_FILE, PE_SYSTEM_PROMPT, et
    enfin le méta-prompt intégré.
    """
    if PROMPT_FILE.is_file():
        content = PROMPT_FILE.read_text(encoding="utf-8").strip()
        if content:
            return content
        print(f"[config] {PROMPT_FILE.name} est vide, consignes par défaut utilisées.")

    prompt_file = env_str("PE_SYSTEM_PROMPT_FILE")
    if prompt_file:
        path = Path(prompt_file)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            sys.exit(f"[config] PE_SYSTEM_PROMPT_FILE introuvable : {path}")
        return path.read_text(encoding="utf-8")

    return env_str("PE_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT


def append_to_output(model: str, user_prompt: str, answer: str) -> None:
    """Ajoute l'échange à la fin de `output.md` sans jamais écraser l'existant."""
    entry = (
        f"\n---\n\n"
        f"## Prompt d'origine (modèle : {model})\n\n"
        f"{user_prompt}\n\n"
        f"## Réponse\n\n"
        f"{answer.strip()}\n"
    )
    try:
        with OUTPUT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError as exc:
        print(f"[erreur] écriture dans {OUTPUT_FILE.name} impossible : {exc}")


# --- Client et appel du modèle ---------------------------------------------


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
    """Paramètres d'échantillonnage : envoyés seulement s'ils sont définis.

    Certains modèles (familles de raisonnement) rejettent temperature/top_p ;
    laisser la variable vide dans le .env suffit à ne pas les envoyer.
    """
    optional: dict[str, float | int | None] = {
        "temperature": env_float("OPENAI_TEMPERATURE"),
        "top_p": env_float("OPENAI_TOP_P"),
        "presence_penalty": env_float("OPENAI_PRESENCE_PENALTY"),
        "frequency_penalty": env_float("OPENAI_FREQUENCY_PENALTY"),
        "max_completion_tokens": env_int("OPENAI_MAX_TOKENS"),
        "seed": env_int("OPENAI_SEED"),
    }
    return {k: v for k, v in optional.items() if v is not None}


def describe_error(exc: Exception) -> str:
    """Déplie la chaîne des causes : le SDK masque souvent l'erreur réelle.

    Un `APIConnectionError` s'affiche par exemple « Connection error. » alors
    que la cause exacte (DNS, TLS, URL invalide) n'est visible que dans
    `__cause__`.
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    while cause is not None and len(parts) < 4:
        label = f"{type(cause).__name__}: {cause}"
        if label != parts[-1]:
            parts.append(label)
        cause = cause.__cause__
    return "\n  cause → ".join(parts)


def ask_model(client: OpenAI, model: str, system: str, user_prompt: str) -> str | None:
    """Envoie le prompt et affiche la réponse en flux, au fil des tokens.

    Renvoie le texte complet, ou None si l'appel échoue.
    """
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Voici le prompt à améliorer :\n\n<prompt>\n{user_prompt}\n</prompt>",
        ),
    ]
    chunks: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, **build_request_params()
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


# --- Boucle interactive -----------------------------------------------------


def main() -> None:
    system = load_system_prompt()
    model = env_str("OPENAI_MODEL") or DEFAULT_MODEL
    client = build_client()

    print(f"Améliorateur de prompts — modèle : {model}")
    print(f"Consignes : {PROMPT_FILE.name} — journal : {OUTPUT_FILE.name} (ajout en fin de fichier)")
    print("Entre un prompt à améliorer. Ligne vide ou 'quit' pour sortir.\n")

    while True:
        try:
            user_prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            return

        if not user_prompt or user_prompt.lower() in QUIT_WORDS:
            print("Au revoir.")
            return

        print(f"\nMessage envoyé à {model}.")
        print(f"Attente de la réponse de {model}...\n")
        print(f"--- Réponse de {model} ---")

        answer = ask_model(client, model, system, user_prompt)
        if answer is None:
            continue

        print("--- fin de la réponse ---")
        append_to_output(model, user_prompt, answer)
        print(f"Réponse ajoutée à la fin de {OUTPUT_FILE.name}.\n")


if __name__ == "__main__":
    main()
