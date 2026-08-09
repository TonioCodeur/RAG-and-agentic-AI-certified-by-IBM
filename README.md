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

## `improve_prompt.py` — walkthrough

An interactive command-line tool that rewrites the prompts you type. You enter a rough prompt, the script prepends the instructions held in `prompt.md`, sends both to an OpenAI model, streams the answer to the terminal, and appends the exchange to `output.md`. The loop then asks for the next prompt.

Run it with:

```bash
.\.venv\Scripts\python.exe improve_prompt.py
```

### Overall flow

1. Load configuration from `.env` and clean up empty variables.
2. Read the instruction text (`prompt.md`) once, at startup.
3. Open an OpenAI client and print a banner.
4. Loop: read a prompt → announce the send → stream the answer → append it to `output.md`.
5. Exit on an empty line, a quit word, `Ctrl+C`, or end of input.

### Line by line

**Lines 1–10 — Module docstring.** Describes the behaviour and the run command. Written before any import, so `python -c "help(improve_prompt)"` and IDE tooltips pick it up.

**Line 12 — `from __future__ import annotations`.** Defers the evaluation of type annotations, which is what lets `str | None` be written without importing `Optional`.

**Lines 14–25 — Imports.** `os` and `sys` for environment access and early exits; `Path` for filesystem paths that behave the same on Windows and Unix; `Any` for the loosely-typed keyword dictionaries. `load_dotenv` reads the `.env` file. `OpenAI` is the API client and `OpenAIError` the base class of every SDK error. The three `ChatCompletion*Param` types are `TypedDict`s: they let the type checker validate the message list instead of accepting a bare `dict`.

**Line 27 — `ROOT`.** `Path(__file__).resolve().parent` is the folder containing the script. Every other path derives from it, so the script works regardless of the current working directory.

**Line 28 — `load_dotenv(ROOT / ".env")`.** Loads the `.env` file into `os.environ`. By default it does **not** override variables already set in the real environment — a shell-exported key wins over the file.

**Lines 30–36 — Purging empty variables.** The critical fix. The OpenAI SDK reads `OPENAI_BASE_URL`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` from the environment on its own. A variable declared but left empty in `.env` (`OPENAI_BASE_URL=`) is *present*, not absent: the SDK takes the empty string as the API base URL, the request goes out without `https://`, and it fails with an unhelpful `Connection error.` The loop deletes any such empty variable so the SDK falls back to its defaults.

**Lines 38–39 — Constants.** `DEFAULT_MODEL` is used when `OPENAI_MODEL` is absent. `QUIT_WORDS` is a `set` — membership tests are O(1) and the intent reads better than a chain of `or`.

**Lines 41–43 — File paths.** `PROMPT_FILE` is the instruction file joined to every prompt; `OUTPUT_FILE` is the answer log.

**Lines 46–75 — `DEFAULT_SYSTEM_PROMPT`.** The built-in meta-prompt, used only if no instructions are found elsewhere. It assigns a role, states a three-step method, forbids invention (missing facts become `{placeholders}`), and imposes a strict four-section Markdown output format. The trailing backslashes join lines without inserting a newline into the string.

**Lines 81–83 — `env_str`.** Reads a variable and returns `None` when it is missing, empty, or whitespace only. This is the reason a blank line in `.env` behaves like an absent one everywhere in the script.

**Lines 86–93 — `env_float`.** Same, converted to `float`. An unparsable value calls `sys.exit` with an explanatory message: a typo in the configuration stops the script immediately instead of surfacing later as an obscure API error.

**Lines 96–103 — `env_int`.** Identical, for integers.

**Lines 106–127 — `load_system_prompt`.** Resolves the instruction text in priority order: `prompt.md` if it exists and is non-empty; otherwise the file named by `PE_SYSTEM_PROMPT_FILE` (relative paths resolved against `ROOT`, a missing file being fatal); otherwise the inline `PE_SYSTEM_PROMPT`; otherwise `DEFAULT_SYSTEM_PROMPT`. An empty `prompt.md` prints a notice rather than silently sending nothing.

**Lines 130–143 — `append_to_output`.** Formats one exchange as a Markdown block — a `---` separator, the original prompt, then the answer — and writes it with `open("a")`. Append mode creates the file if needed and **never** truncates it, which is what makes `output.md` a cumulative log. `OSError` is caught and reported without interrupting the session: a locked file should not lose you a session's worth of work.

**Lines 149–163 — `build_client`.** Missing `OPENAI_API_KEY` is fatal, with a message pointing to `.env.example`. Optional settings — alternate endpoint, organisation, timeout, retry count — are added to `kwargs` only when defined, so anything left blank keeps the SDK's own default.

**Lines 166–180 — `build_request_params`.** Builds the sampling parameters the same way: only defined values are sent. This matters because reasoning models reject `temperature` and `top_p` — leaving them blank in `.env` is enough to omit them. Note `max_completion_tokens`, the parameter that replaces the deprecated `max_tokens` on recent models.

**Lines 183–197 — `describe_error`.** Unrolls the `__cause__` chain of an exception. The SDK wraps low-level failures: `APIConnectionError` prints `Connection error.` while the real cause (DNS, TLS, invalid URL) sits in `__cause__`. The walk stops at four entries and skips consecutive duplicates.

**Lines 200–228 — `ask_model`.** Builds a two-message conversation: the instructions as the `system` message, and the user's prompt as the `user` message, wrapped in `<prompt>` tags so the model can tell the text to rewrite apart from the text telling it what to do. `stream=True` returns an iterator of chunks instead of one complete response. Chunks with an empty `choices` list are skipped — the API sends usage-only chunks. Each `delta` is printed immediately with `flush=True`, since Python otherwise buffers output and the streaming effect would be lost, and is also accumulated in `chunks`. The function returns the full text, or `None` if the call failed.

**Lines 234–264 — `main`.** Startup: read the instructions, resolve the model, build the client, print the banner. Then the loop: `input()` reads a prompt; `EOFError` (piped input running out) and `KeyboardInterrupt` (`Ctrl+C`) exit cleanly instead of dumping a traceback. An empty line or a quit word ends the session. Otherwise the script announces the send and the wait — both naming the model — prints the header *before* calling `ask_model`, since the text arrives during the call, and finally logs the answer. A failed call skips straight to the next iteration: an API error does not end the session.

**Lines 267–268 — Entry point.** `main()` runs only when the file is executed directly, so importing the module for testing does not start the loop.

### Configuration

`.env` variables the script actually reads:

| Variable | Role |
| --- | --- |
| `OPENAI_API_KEY` | API key — the only mandatory one |
| `OPENAI_MODEL` | Model id (default `gpt-5.4-mini`) |
| `OPENAI_BASE_URL`, `OPENAI_ORG_ID` | Alternate endpoint and organisation |
| `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES` | Client timeout and retry count |
| `OPENAI_TEMPERATURE`, `OPENAI_TOP_P`, `OPENAI_PRESENCE_PENALTY`, `OPENAI_FREQUENCY_PENALTY`, `OPENAI_MAX_TOKENS`, `OPENAI_SEED` | Sampling — sent only when set |
| `PE_SYSTEM_PROMPT_FILE`, `PE_SYSTEM_PROMPT` | Instruction fallbacks if `prompt.md` is empty |

Streaming is not configurable: it is always on, hardcoded at line 215.

## `tree_of_thoughts.py` — walkthrough

An implementation of **Tree of Thoughts** (Yao et al., 2023) on top of the model named in `.env`. Where chain-of-thought asks a model to reason in a single straight line, tree of thoughts makes it explore several lines in parallel, judge them against each other, and keep only the most promising ones.

You type a problem, and the script runs a **beam search over reasoning steps**:

1. **Expansion** — for each surviving branch, the model proposes N distinct next thoughts (one reasoning step, not the answer).
2. **Evaluation** — the model scores every thought of a level in a single request, which lets it compare them rather than judge each in isolation.
3. **Selection** — only the K best survive.
4. Back to step 1 until the target depth, then **synthesis** of the final answer from the winning path.

The full tree is printed with its scores, the final answer is streamed, and the whole session is appended to `output_tree.md`.

```bash
.\.venv\Scripts\python.exe tree_of_thoughts.py
```

The three roles the model plays — explorer, critic, writer — are three different system prompts sent to the same model. Nothing but the instructions changes.

### Line by line

**Lines 1–19 — Module docstring.** States the four phases of the algorithm and the run command.

**Line 21 — `from __future__ import annotations`.** Beyond the `str | None` syntax, this is what allows the `Thought` dataclass to reference its own type (`parent: Thought | None`) inside its own definition, at a point where the class does not exist yet.

**Lines 23–37 — Imports.** Compared with `improve_prompt.py`, three additions: `json` to decode the model's structured answers, `re` to recover a JSON object embedded in prose, and `dataclass`/`field` to build the tree nodes.

**Lines 39–42 — Forcing UTF-8 on stdout.** The Windows console defaults to cp1252, in which the accented characters and box glyphs of the tree raise a `UnicodeEncodeError`. `reconfigure` switches the stream to UTF-8. The call is guarded by `hasattr` so the script does not break on an exotic stdout (a pipe, a captured stream).

**Lines 44–45 — `ROOT` and `load_dotenv`.** The script folder, then loading `.env` into the environment.

**Lines 47–51 — Purging empty variables.** Same fix as in the other script: an `OPENAI_BASE_URL=` left empty in `.env` is read by the SDK and breaks the API URL.

**Lines 53–55 — Basic constants.** Fallback model, quit words, and the log file — `output_tree.md`, kept separate from `improve_prompt.py`'s log.

**Lines 57–62 — Tree shape.** The three parameters that define the search. `TREE_DEPTH` is the number of successive reasoning steps, `BRANCHING` the number of thoughts proposed per branch, `BEAM_WIDTH` the number of branches kept after each evaluation. With 3/3/2 the run costs 9 API calls: 1 + 2 + 2 expansions, 3 evaluations, 1 synthesis. Raising `BRANCHING` broadens the exploration, raising `BEAM_WIDTH` multiplies the calls — the cost grows fast in both directions, which is why the values live in one visible place.

**Lines 64–77 — `EXPLORER_SYSTEM`.** The generator's instructions. Three constraints do the real work: a thought is *a step, not the answer* (otherwise the model answers immediately and the tree is pointless); branches must *genuinely diverge* (otherwise you get three rephrasings of the same idea and the search explores nothing); each thought stays short. The imposed format is `{"thoughts": [...]}`.

**Lines 79–89 — `CRITIC_SYSTEM`.** The judge's instructions: score 0–10 on relevance, soundness, and room to progress. The explicit "be discriminating" instruction matters — left alone, a model tends to hand out the same 8/10 to everything, which would turn the selection into an arbitrary draw.

**Lines 91–98 — `SYNTHESIS_SYSTEM`.** The writer's instructions. It is explicitly told *not* to describe the exploration: the user wants the answer, not a report on how it was found.

**Lines 104–127 — `env_str`, `env_float`, `env_int`.** Identical to `improve_prompt.py`: `None` for anything missing or blank, immediate exit on an unparsable value.

**Lines 129–144 — `build_client`.** Missing API key is fatal; optional settings are passed only when defined.

**Lines 146–155 — `build_request_params`.** Sampling parameters. Shorter than in the other script: only `temperature`, `top_p`, `max_completion_tokens` and `seed` are kept, the penalties being irrelevant here.

**Lines 157–167 — `describe_error`.** Unrolls the `__cause__` chain to expose the error the SDK hides behind `Connection error.`

**Lines 172–177 — `messages_of`.** A one-line helper building the system + user message pair. The three phases differ only by their two strings, so the construction is factored out once.

**Lines 179–215 — `ask_json`.** The structured-output layer, with three levels of fallback. First attempt: `response_format={"type": "json_object"}`, which constrains the model to emit syntactically valid JSON. If the endpoint rejects the parameter, second attempt without it. If the returned text still is not JSON, a regex grabs the first `{...}` block — this catches the classic case of a model wrapping its JSON in a Markdown fence. Total failure returns an empty dict; the caller decides what to do rather than the program crashing.

**Lines 217–239 — `stream_answer`.** Used for the final synthesis only. Same streaming as `improve_prompt.py`: `stream=True`, immediate `print(flush=True)`, text accumulated and returned.

**Lines 244–254 — The `Thought` dataclass.** One tree node. `text` is the reasoning step, `depth` its level, `parent` the node it descends from — that back-reference is what makes reconstructing a path possible without storing the whole tree. `score` and `reason` come from the evaluation, `kept` records whether the node survived selection (used for display), and `children` keeps the descendants. `field(default_factory=list)` is mandatory: a bare `[]` as a default would be shared by every instance.

**Lines 256–263 — `Thought.path`.** Walks back up the `parent` chain to the root, stopping at `depth > 0` to exclude the root (which holds the problem, not a thought), then reverses the list to get the path in reading order.

**Lines 265–269 — `Thought.path_text`.** Formats that path as numbered steps, to be injected into the next prompt. An empty path — the first expansion — returns an explicit sentence rather than an empty string, which would leave the model guessing.

**Lines 272–290 — `expand`.** Asks the model for `BRANCHING` next thoughts for a given node. The prompt contains the problem *and* the path already travelled: the model needs the context of the branch it is extending. The answer is validated (`isinstance(raw, list)`), truncated to `BRANCHING`, stripped of blanks, and turned into child `Thought`s.

**Lines 293–319 — `evaluate`.** Numbers every candidate of the level, mentions the parent each descends from, and asks for all scores in **one** call — that is what lets the model rank them relative to one another. Parsing is defensive: a malformed entry is skipped, and any candidate the model forgot to score stays at 0 and gets eliminated. Scores and justifications are written back onto the nodes.

**Lines 322–328 — `select`.** The beam search itself: sort by descending score, keep the first `BEAM_WIDTH`, mark them `kept`. Everything else is abandoned — that is the pruning without which the tree would grow exponentially.

**Lines 334–346 — `render_tree`.** Text rendering of the exploration, one indentation level per step, `[+]` for kept and `[-]` for discarded, the score, the thought, and the justification underneath. This is what makes the run auditable: you can see what the model rejected and why.

**Lines 348–363 — `append_to_output`.** Appends the session to `output_tree.md` in four Markdown sections: the problem with its parameters, the tree, the winning path, and the final answer. Opened in `"a"` mode, so the file is never truncated.

**Lines 368–411 — `solve`.** The orchestrator. It creates a root node holding the problem, initialises the beam with it, then loops `TREE_DEPTH` times: expand every branch of the beam, stop cleanly if nothing usable came back, evaluate all candidates in one go, select, store the level for display. After the loop it prints the tree, rebuilds the winning path from the best node, sends it to the synthesis prompt and streams the answer, then logs everything.

**Lines 414–435 — `main`.** Resolves the model, builds the client, prints the banner with the tree parameters, then loops: read a problem, exit on an empty line, a quit word, `Ctrl+C` or end of input, otherwise call `solve`.

**Lines 438–439 — Entry point.** `main()` runs only on direct execution.

### Reading the output

```text
[+]  9.2  Reduce the weight of critical assets...
      (directly actionable, targets the most common front-end levers)
[-]  8.7  Measure the bottlenecks first...
```

`[+]` marks a branch kept for the next step, `[-]` one abandoned. The number is the critic's score, the line underneath its justification. A run where every score sits within a few tenths of a point is a sign that the branches were not diverse enough: raise `BRANCHING`, or make `EXPLORER_SYSTEM` more demanding about divergence.

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

## `improve_prompt.py` — explication détaillée

Un outil en ligne de commande interactif qui réécrit les prompts que tu saisis. Tu entres un prompt brut, le script y joint les consignes contenues dans `prompt.md`, envoie le tout à un modèle OpenAI, affiche la réponse en flux dans le terminal et ajoute l'échange à la fin de `output.md`. La boucle redemande ensuite un prompt.

Pour le lancer :

```bash
.\.venv\Scripts\python.exe improve_prompt.py
```

### Déroulement général

1. Charger la configuration depuis `.env` et nettoyer les variables vides.
2. Lire une seule fois, au démarrage, le texte de consignes (`prompt.md`).
3. Ouvrir un client OpenAI et afficher le bandeau d'accueil.
4. Boucler : lire un prompt → annoncer l'envoi → afficher la réponse en flux → l'ajouter à `output.md`.
5. Sortir sur une ligne vide, un mot de sortie, `Ctrl+C`, ou une fin d'entrée.

### Ligne par ligne

**Lignes 1–10 — Docstring du module.** Décrit le comportement et la commande de lancement. Placée avant tout import, elle est reprise par `python -c "help(improve_prompt)"` et par les infobulles de l'éditeur.

**Ligne 12 — `from __future__ import annotations`.** Diffère l'évaluation des annotations de type ; c'est ce qui permet d'écrire `str | None` sans importer `Optional`.

**Lignes 14–25 — Imports.** `os` et `sys` pour l'environnement et les sorties anticipées ; `Path` pour des chemins qui se comportent pareil sous Windows et sous Unix ; `Any` pour les dictionnaires d'arguments faiblement typés. `load_dotenv` lit le fichier `.env`. `OpenAI` est le client d'API et `OpenAIError` la classe de base de toutes les erreurs du SDK. Les trois types `ChatCompletion*Param` sont des `TypedDict` : ils permettent au vérificateur de types de valider la liste de messages au lieu d'accepter un `dict` quelconque.

**Ligne 27 — `ROOT`.** `Path(__file__).resolve().parent` désigne le dossier du script. Tous les autres chemins en dérivent, si bien que le script fonctionne quel que soit le répertoire courant.

**Ligne 28 — `load_dotenv(ROOT / ".env")`.** Charge le `.env` dans `os.environ`. Par défaut, il n'écrase **pas** les variables déjà définies dans l'environnement réel : une clé exportée dans le shell l'emporte sur le fichier.

**Lignes 30–36 — Purge des variables vides.** La correction déterminante. Le SDK OpenAI lit tout seul `OPENAI_BASE_URL`, `OPENAI_ORG_ID` et `OPENAI_PROJECT_ID` dans l'environnement. Une variable déclarée mais laissée vide dans le `.env` (`OPENAI_BASE_URL=`) est *présente*, pas absente : le SDK prend la chaîne vide comme URL de base, la requête part sans `https://` et échoue sur un laconique `Connection error.` La boucle supprime ces variables vides pour que le SDK retrouve ses valeurs par défaut.

**Lignes 38–39 — Constantes.** `DEFAULT_MODEL` sert quand `OPENAI_MODEL` est absent. `QUIT_WORDS` est un `set` : le test d'appartenance est en temps constant et l'intention se lit mieux qu'une suite de `or`.

**Lignes 41–43 — Chemins des fichiers.** `PROMPT_FILE` est le fichier de consignes joint à chaque prompt ; `OUTPUT_FILE` est le journal des réponses.

**Lignes 46–75 — `DEFAULT_SYSTEM_PROMPT`.** Le méta-prompt intégré, utilisé seulement si aucune consigne n'est trouvée ailleurs. Il attribue un rôle, énonce une méthode en trois étapes, interdit d'inventer (une information manquante devient un `{emplacement}`) et impose un format de sortie Markdown strict en quatre sections. Les antislashs en fin de ligne joignent les lignes sans insérer de retour à la ligne dans la chaîne.

**Lignes 81–83 — `env_str`.** Lit une variable et renvoie `None` si elle est absente, vide ou composée uniquement d'espaces. C'est la raison pour laquelle, partout dans le script, une ligne vide du `.env` se comporte comme une ligne absente.

**Lignes 86–93 — `env_float`.** Idem, converti en `float`. Une valeur non convertible déclenche `sys.exit` avec un message explicite : une faute de frappe dans la configuration arrête le script tout de suite, au lieu de ressortir plus tard sous forme d'erreur d'API obscure.

**Lignes 96–103 — `env_int`.** Identique, pour les entiers.

**Lignes 106–127 — `load_system_prompt`.** Résout le texte de consignes par ordre de priorité : `prompt.md` s'il existe et n'est pas vide ; sinon le fichier désigné par `PE_SYSTEM_PROMPT_FILE` (chemin relatif résolu depuis `ROOT`, fichier introuvable = arrêt) ; sinon le texte `PE_SYSTEM_PROMPT` ; sinon `DEFAULT_SYSTEM_PROMPT`. Un `prompt.md` vide déclenche un avertissement plutôt qu'un envoi silencieusement vide.

**Lignes 130–143 — `append_to_output`.** Met en forme un échange sous forme de bloc Markdown — séparateur `---`, prompt d'origine, puis réponse — et l'écrit via `open("a")`. Le mode ajout crée le fichier au besoin et ne le tronque **jamais** : c'est ce qui fait de `output.md` un journal cumulatif. Une `OSError` est attrapée et signalée sans interrompre la session : un fichier verrouillé ne doit pas te coûter une session de travail.

**Lignes 149–163 — `build_client`.** Une `OPENAI_API_KEY` manquante est fatale, avec un message qui renvoie vers `.env.example`. Les réglages facultatifs — point d'entrée alternatif, organisation, timeout, nombre de tentatives — ne sont ajoutés à `kwargs` que s'ils sont définis : tout ce qui est laissé vide conserve la valeur par défaut du SDK.

**Lignes 166–180 — `build_request_params`.** Construit les paramètres d'échantillonnage sur le même principe : seules les valeurs définies sont envoyées. C'est important car les modèles de raisonnement refusent `temperature` et `top_p` — les laisser vides dans le `.env` suffit à ne pas les transmettre. Noter `max_completion_tokens`, le paramètre qui remplace `max_tokens` (déprécié) sur les modèles récents.

**Lignes 183–197 — `describe_error`.** Déroule la chaîne des `__cause__` d'une exception. Le SDK emballe les erreurs de bas niveau : `APIConnectionError` affiche `Connection error.` alors que la cause réelle (DNS, TLS, URL invalide) se trouve dans `__cause__`. Le parcours s'arrête à quatre entrées et ignore les doublons consécutifs.

**Lignes 200–228 — `ask_model`.** Construit une conversation de deux messages : les consignes en message `system`, et le prompt de l'utilisateur en message `user`, encadré par des balises `<prompt>` pour que le modèle distingue le texte à réécrire du texte qui lui dit quoi faire. `stream=True` renvoie un itérateur de fragments au lieu d'une réponse complète. Les fragments dont la liste `choices` est vide sont ignorés — l'API envoie des fragments ne portant que des statistiques d'usage. Chaque `delta` est affiché immédiatement avec `flush=True`, car Python bufferise sinon la sortie et l'effet de flux serait perdu, et il est aussi accumulé dans `chunks`. La fonction renvoie le texte complet, ou `None` si l'appel a échoué.

**Lignes 234–264 — `main`.** Démarrage : lire les consignes, résoudre le modèle, construire le client, afficher le bandeau. Puis la boucle : `input()` lit un prompt ; `EOFError` (entrée redirigée épuisée) et `KeyboardInterrupt` (`Ctrl+C`) provoquent une sortie propre au lieu d'une trace d'erreur. Une ligne vide ou un mot de sortie termine la session. Sinon le script annonce l'envoi puis l'attente — en nommant le modèle dans les deux cas —, affiche l'en-tête *avant* d'appeler `ask_model` puisque le texte arrive pendant l'appel, et journalise enfin la réponse. Un appel échoué passe directement à l'itération suivante : une erreur d'API ne met pas fin à la session.

**Lignes 267–268 — Point d'entrée.** `main()` ne s'exécute que si le fichier est lancé directement ; importer le module pour le tester ne démarre donc pas la boucle.

### Configuration

Variables du `.env` réellement lues par le script :

| Variable | Rôle |
| --- | --- |
| `OPENAI_API_KEY` | Clé d'API — la seule obligatoire |
| `OPENAI_MODEL` | Identifiant du modèle (défaut `gpt-5.4-mini`) |
| `OPENAI_BASE_URL`, `OPENAI_ORG_ID` | Point d'entrée alternatif et organisation |
| `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES` | Timeout et nombre de tentatives du client |
| `OPENAI_TEMPERATURE`, `OPENAI_TOP_P`, `OPENAI_PRESENCE_PENALTY`, `OPENAI_FREQUENCY_PENALTY`, `OPENAI_MAX_TOKENS`, `OPENAI_SEED` | Échantillonnage — envoyés seulement si définis |
| `PE_SYSTEM_PROMPT_FILE`, `PE_SYSTEM_PROMPT` | Consignes de repli si `prompt.md` est vide |

Le streaming n'est pas configurable : il est toujours actif, codé en dur à la ligne 215.

## `tree_of_thoughts.py` — explication détaillée

Une implémentation de l'**arbre de pensées** (Tree of Thoughts, Yao et al., 2023) appliquée au modèle désigné dans le `.env`. Là où le chain-of-thought demande au modèle de raisonner en une seule ligne droite, l'arbre de pensées lui fait explorer plusieurs pistes en parallèle, les juger les unes contre les autres, et ne conserver que les plus prometteuses.

Tu saisis un problème, et le script exécute une **recherche en faisceau sur les étapes de raisonnement** :

1. **Expansion** — pour chaque piste survivante, le modèle propose N pensées suivantes distinctes (une étape de raisonnement, pas la réponse).
2. **Évaluation** — le modèle note toutes les pensées d'un même niveau en une seule requête, ce qui lui permet de les comparer plutôt que de juger chacune isolément.
3. **Sélection** — seules les K meilleures survivent.
4. Retour à l'étape 1 jusqu'à la profondeur voulue, puis **synthèse** de la réponse finale à partir du chemin gagnant.

L'arbre complet est affiché avec ses notes, la réponse finale est diffusée en flux, et la session entière est ajoutée à la fin de `output_tree.md`.

```bash
.\.venv\Scripts\python.exe tree_of_thoughts.py
```

Les trois rôles que joue le modèle — explorateur, critique, rédacteur — sont trois prompts système différents envoyés au même modèle. Rien d'autre que les consignes ne change.

### Ligne par ligne

**Lignes 1–19 — Docstring du module.** Énonce les quatre phases de l'algorithme et la commande de lancement.

**Ligne 21 — `from __future__ import annotations`.** Au-delà de la syntaxe `str | None`, c'est ce qui permet à la dataclass `Thought` de référencer son propre type (`parent: Thought | None`) à l'intérieur de sa définition, à un moment où la classe n'existe pas encore.

**Lignes 23–37 — Imports.** Par rapport à `improve_prompt.py`, trois ajouts : `json` pour décoder les réponses structurées du modèle, `re` pour récupérer un objet JSON noyé dans du texte, et `dataclass`/`field` pour construire les nœuds de l'arbre.

**Lignes 39–42 — Forçage de l'UTF-8 sur la sortie.** La console Windows utilise cp1252 par défaut, encodage dans lequel les caractères accentués et les glyphes de l'arbre lèvent une `UnicodeEncodeError`. `reconfigure` bascule le flux en UTF-8. L'appel est protégé par `hasattr` pour que le script ne casse pas sur une sortie standard exotique (un tube, un flux capturé).

**Lignes 44–45 — `ROOT` et `load_dotenv`.** Le dossier du script, puis le chargement du `.env` dans l'environnement.

**Lignes 47–51 — Purge des variables vides.** Même correctif que dans l'autre script : une `OPENAI_BASE_URL=` laissée vide dans le `.env` est lue par le SDK et casse l'URL de l'API.

**Lignes 53–55 — Constantes de base.** Modèle de repli, mots de sortie, et le fichier journal — `output_tree.md`, distinct de celui de `improve_prompt.py`.

**Lignes 57–62 — Forme de l'arbre.** Les trois paramètres qui définissent la recherche. `TREE_DEPTH` est le nombre d'étapes de raisonnement successives, `BRANCHING` le nombre de pensées proposées par piste, `BEAM_WIDTH` le nombre de pistes conservées après chaque évaluation. Avec 3/3/2, une session coûte 9 appels d'API : 1 + 2 + 2 expansions, 3 évaluations, 1 synthèse. Augmenter `BRANCHING` élargit l'exploration, augmenter `BEAM_WIDTH` multiplie les appels — le coût grimpe vite dans les deux sens, d'où le regroupement de ces valeurs à un seul endroit visible.

**Lignes 64–77 — `EXPLORER_SYSTEM`.** Les consignes du générateur. Trois contraintes font tout le travail : une pensée est *une étape, pas la réponse* (sinon le modèle répond tout de suite et l'arbre ne sert à rien) ; les pistes doivent *réellement diverger* (sinon on obtient trois reformulations de la même idée et la recherche n'explore rien) ; chaque pensée reste courte. Le format imposé est `{"thoughts": [...]}`.

**Lignes 79–89 — `CRITIC_SYSTEM`.** Les consignes du juge : noter de 0 à 10 sur la pertinence, la solidité et le potentiel de progression. L'instruction explicite « sois discriminant » est importante : livré à lui-même, un modèle a tendance à distribuer le même 8/10 à tout le monde, ce qui transformerait la sélection en tirage au sort.

**Lignes 91–98 — `SYNTHESIS_SYSTEM`.** Les consignes du rédacteur. On lui interdit explicitement de décrire l'exploration : l'utilisateur veut la réponse, pas un compte rendu de la manière dont elle a été trouvée.

**Lignes 104–127 — `env_str`, `env_float`, `env_int`.** Identiques à `improve_prompt.py` : `None` pour tout ce qui est absent ou vide, sortie immédiate sur une valeur non convertible.

**Lignes 129–144 — `build_client`.** Clé d'API manquante = arrêt ; les réglages facultatifs ne sont transmis que s'ils sont définis.

**Lignes 146–155 — `build_request_params`.** Les paramètres d'échantillonnage. Plus court que dans l'autre script : seuls `temperature`, `top_p`, `max_completion_tokens` et `seed` sont conservés, les pénalités n'ayant pas d'intérêt ici.

**Lignes 157–167 — `describe_error`.** Déroule la chaîne des `__cause__` pour exposer l'erreur que le SDK masque derrière `Connection error.`

**Lignes 172–177 — `messages_of`.** Un utilitaire d'une ligne qui construit le couple de messages système + utilisateur. Les trois phases ne diffèrent que par leurs deux chaînes, la construction est donc factorisée une fois pour toutes.

**Lignes 179–215 — `ask_json`.** La couche de sortie structurée, avec trois niveaux de repli. Première tentative : `response_format={"type": "json_object"}`, qui contraint le modèle à produire un JSON syntaxiquement valide. Si le point d'entrée refuse ce paramètre, deuxième tentative sans lui. Si le texte renvoyé n'est toujours pas du JSON, une expression régulière récupère le premier bloc `{...}` — cela rattrape le cas classique du modèle qui encadre son JSON dans un bloc de code Markdown. Un échec total renvoie un dictionnaire vide : c'est l'appelant qui décide quoi faire, le programme ne plante pas.

**Lignes 217–239 — `stream_answer`.** Utilisée pour la seule synthèse finale. Même diffusion que dans `improve_prompt.py` : `stream=True`, `print(flush=True)` immédiat, texte accumulé puis renvoyé.

**Lignes 244–254 — La dataclass `Thought`.** Un nœud de l'arbre. `text` est l'étape de raisonnement, `depth` son niveau, `parent` le nœud dont elle descend — cette référence arrière est ce qui permet de reconstituer un chemin sans stocker l'arbre entier. `score` et `reason` proviennent de l'évaluation, `kept` mémorise si le nœud a survécu à la sélection (utilisé pour l'affichage), et `children` conserve les descendants. `field(default_factory=list)` est obligatoire : un `[]` en valeur par défaut serait partagé par toutes les instances.

**Lignes 256–263 — `Thought.path`.** Remonte la chaîne des `parent` jusqu'à la racine, s'arrête à `depth > 0` pour exclure la racine (qui porte le problème, pas une pensée), puis inverse la liste pour obtenir le chemin dans l'ordre de lecture.

**Lignes 265–269 — `Thought.path_text`.** Met ce chemin en forme sous forme d'étapes numérotées, destinées à être injectées dans le prompt suivant. Un chemin vide — la première expansion — renvoie une phrase explicite plutôt qu'une chaîne vide, qui laisserait le modèle deviner.

**Lignes 272–290 — `expand`.** Demande au modèle `BRANCHING` pensées suivantes pour un nœud donné. Le prompt contient le problème *et* le chemin déjà parcouru : le modèle a besoin du contexte de la branche qu'il prolonge. La réponse est validée (`isinstance(raw, list)`), tronquée à `BRANCHING`, débarrassée des entrées vides, puis convertie en `Thought` enfants.

**Lignes 293–319 — `evaluate`.** Numérote tous les candidats du niveau, mentionne le parent dont chacun descend, et demande toutes les notes en **un seul** appel — c'est ce qui permet au modèle de les classer les uns par rapport aux autres. L'analyse est défensive : une entrée mal formée est ignorée, et un candidat que le modèle a oublié de noter reste à 0 et se fait éliminer. Notes et justifications sont ensuite réécrites sur les nœuds.

**Lignes 322–328 — `select`.** La recherche en faisceau proprement dite : trier par note décroissante, garder les `BEAM_WIDTH` premiers, les marquer `kept`. Tout le reste est abandonné — c'est l'élagage sans lequel l'arbre croîtrait de façon exponentielle.

**Lignes 334–346 — `render_tree`.** Rendu texte de l'exploration, un niveau d'indentation par étape, `[+]` pour gardé et `[-]` pour écarté, la note, la pensée, et la justification en dessous. C'est ce qui rend la session auditable : on voit ce que le modèle a rejeté, et pourquoi.

**Lignes 348–363 — `append_to_output`.** Ajoute la session à `output_tree.md` en quatre sections Markdown : le problème avec ses paramètres, l'arbre, le chemin retenu, et la réponse finale. Ouverture en mode `"a"`, le fichier n'est donc jamais tronqué.

**Lignes 368–411 — `solve`.** L'orchestrateur. Il crée un nœud racine portant le problème, initialise le faisceau avec lui, puis boucle `TREE_DEPTH` fois : étendre chaque piste du faisceau, s'arrêter proprement si rien d'exploitable n'est revenu, évaluer tous les candidats d'un coup, sélectionner, mémoriser le niveau pour l'affichage. Après la boucle, il affiche l'arbre, reconstitue le chemin gagnant depuis le meilleur nœud, l'envoie au prompt de synthèse et diffuse la réponse, puis journalise l'ensemble.

**Lignes 414–435 — `main`.** Résout le modèle, construit le client, affiche le bandeau avec les paramètres de l'arbre, puis boucle : lire un problème, sortir sur une ligne vide, un mot de sortie, `Ctrl+C` ou une fin d'entrée, sinon appeler `solve`.

**Lignes 438–439 — Point d'entrée.** `main()` ne s'exécute qu'en lancement direct.

### Lire la sortie

```text
[+]  9.2  Réduire le poids des assets critiques...
      (directement actionnable, cible les leviers front-end les plus fréquents)
[-]  8.7  Mesurer d'abord les goulots d'étranglement...
```

`[+]` marque une piste conservée pour l'étape suivante, `[-]` une piste abandonnée. Le nombre est la note du critique, la ligne en dessous sa justification. Une session où toutes les notes tiennent en quelques dixièmes signale que les pistes n'étaient pas assez diverses : augmente `BRANCHING`, ou durcis les exigences de divergence dans `EXPLORER_SYSTEM`.

## Conventions du dépôt

Les index vectoriels, embeddings, poids de modèles et jeux de données (bruts ou traités) sont exclus du versionnement (voir [.gitignore](.gitignore)). Tout ce qui est indexable doit pouvoir être reconstruit en réexécutant le code : les artefacts eux-mêmes ne font pas partie de l'historique.
