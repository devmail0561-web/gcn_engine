# Plan Phase 16 — Spécialisation GCN sur ses cas d'usage réels

**Date :** 2026-09-17  
**Statut :** ✅ Terminé — 2026-09-17  
**Objectif :** Retirer tous les dérapages LLM du moteur et le recentrer sur
ses vrais cas d'usage : cybersécurité, audit, investigation, analyse de code,
recherche scientifique.

---

## Contexte

L'audit de Phase 16 a identifié 11 dérapages dans le code — des éléments
ajoutés pour imiter les LLMs (ChatGPT, Claude) alors que GCN est un moteur
de **raisonnement causal vérifiable sur corpus**, pas un assistant conversationnel.

**GCN répond à des questions sur ce qu'il a extrait — pas à des instructions.**

La valeur de GCN est la **vérifiabilité et la traçabilité** des réponses
causales, pas la génération de texte plausible.

---

## Dérapages identifiés (11)

| # | Sévérité | Fichier | Dérapage |
|---|----------|---------|---------|
| D1 | **CRITIQUE** | `engine.py:2` | Docstring "même usage qu'un LLM" |
| D2 | **CRITIQUE** | `engine.py:152` | `_split_sentences()` dans le moteur ML |
| D3 | **CRITIQUE** | `chat.py` entier | REPL conversationnel style chatbot |
| D4 | **HAUTE** | `engine.py:27` | Comparaison HuggingFace pipeline |
| D5 | **HAUTE** | `engine.py:203` | `analyze_document()` — découpe texte dans moteur |
| D6 | **HAUTE** | `instructions.py` | `CausalGraph` volatile en mémoire (session) |
| D7 | **MOYENNE** | `engine.py:274` | `stream_documents()` — dépend de D2/D5 |
| D8 | **MOYENNE** | `instructions.py` | `verbalize` mal nommé — c'est un `dump` structuré |
| D9 | **MOYENNE** | `cli.py:13` | `--text` promu comme mode primaire |
| D10 | **FAIBLE** | `chat.py:82` | Bannière ASCII chatbot |
| D11 | **FAIBLE** | `engine.py:34` | Exemples docstring orientés usage LLM |

---

## Corrections

### C1 — Réécrire `engine.py` — docstrings et suppression des méthodes hors scope

**Fichier :** `gcn-python/src/gcn_python/engine.py`

**Supprimer :**
- `_split_sentences()` (lignes ~152-170) — appartient aux frontends
- `analyze_document()` (lignes ~203-236) — construit sur `_split_sentences`
- `stream_documents()` (lignes ~274-289) — construit sur `analyze_document`

**Réécrire la docstring du module :**
```python
"""
GCNEngine — moteur de requêtes causales sur corpus.

Extrait des structures causales depuis des textes ou du code et
répond à des requêtes formelles sur ces structures avec traçabilité
jusqu'aux sources.

Ce que GCN fait :
  - Analyser un corpus → graphe causal vérifiable
  - Répondre à des requêtes causales (causes, effets, chemins, contrefactuels)
  - Tracer chaque réponse jusqu'au document source

Ce que GCN ne fait PAS :
  - Générer du texte libre (→ LLM)
  - Suivre des instructions générales (→ LLM)
  - Segmenter du texte brut (→ frontends Rust)
"""
```

**Réécrire la docstring de `GCNEngine` :**
```python
"""
Interface du moteur GCN Causal Engine.

Charge un checkpoint et expose l'extraction causale sur un corpus
de textes ou de code. Chaque résultat est tracé jusqu'à sa source.

Exemples
--------
>>> # Analyser un corpus de rapports CTI
>>> from gcn_python import GCNEngine
>>> engine = GCNEngine.from_pretrained("model.npz")
>>> cirs = engine.analyze_batch(open("report.txt").readlines())
>>> # Interroger le graphe
>>> graph = CausalGraph.from_cirs(cirs)
>>> graph.causes_of("data_exfiltration")
"""
```

**Garder :**
- `analyze(text)` — pour analyser une phrase pré-segmentée
- `analyze_batch(texts)` — pour analyser un corpus pré-segmenté
- `stream(source)` — itérateur sur corpus pré-segmenté (une phrase par ligne)
- `from_pretrained()` — chargement checkpoint
- `node_types`, `relation_types` — introspection

---

### C2 — Supprimer `chat.py` et créer `gcn-discuss`

**Fichier actuel :** `gcn-python/src/gcn_python/chat.py` → **supprimer**

**Principe :** une seule session qui fait les deux — analyse le texte donné
ET répond aux questions sur la structure causale extraite. Pas besoin de
pré-annoter ou d'indexer séparément avant de pouvoir discuter.

#### `gcn-discuss` — analyser et questionner dans la même session

```bash
gcn-discuss --checkpoint model.npz [--graph session_precedente.json]
```

**Deux modes d'entrée dans la session :**
- `analyze <fichier_ou_texte>` → GCN analyse et enrichit le graphe
- Question en langage naturel → GCN répond sur le graphe accumulé

**Session type (cybersécurité) :**
```
  GCN Causal Engine
  Checkpoint : model.npz  |  Corpus : vide (nouvelle session)
  ─────────────────────────────────────────────────────

  > analyze rapport_incident_2024.txt
    → 47 relations causales extraites et indexées.

  > analyze "La vulnérabilité CVE-2024-1234 dans le module SSH
     permet une exécution de code à distance."
    → 1 relation causale extraite et indexée.

  > What causes remote code execution?

  causes_of: remote code execution
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. CVE-2024-1234 (SSH vulnerability)  --[enable]-->  remote_code_execution
       source: texte analysé, confidence: 0.91

    2. authentication_bypass  --[enable]-->  remote_code_execution
       source: rapport_incident_2024.txt:34, confidence: 0.87
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  > analyze rapport_incident_2025.txt
    → 31 relations causales extraites. Total : 79 relations dans la session.

  > Any contradictions?
    ⚠ firewall_policy --[prevent]--> data_exfiltration (rapport_2024.txt:12)
      MAIS 8 autres sources affirment data_exfiltration se produit quand même.

  > save cti_session.json
    → Graphe sauvegardé (79 relations, 3 sources).

  > quit

# Session suivante — reprendre sans tout réanalyser
gcn-discuss --checkpoint model.npz --graph cti_session.json
```

**Différences fondamentales avec `gcn-chat` supprimé :**
- Analyse des **fichiers entiers** (pas phrase par phrase)
- Graphe **persistant entre sessions** (`save`/`load`)
- Répond uniquement sur la **causalité extraite** du corpus soumis
- Rapport **multi-lignes** avec sources, fréquences, contradictions
- Si hors corpus : "aucune structure causale sur ce sujet"

**Nouveau fichier :** `gcn-python/src/gcn_python/discuss.py`

La session comprend :
- `analyze <path_ou_texte>` → `GCNEngine.analyze()` + `CausalGraph.add_cir()`
- Questions → détection du type + `CausalGraph.*` + `QueryVerbalizer.format_*()`
- `save <path>` → `CausalGraph.save()`
- `load <path>` → `CausalGraph.load()` (ou `--graph` au démarrage)
- `summarize` → `QueryVerbalizer.format_summary()`
- Hors scope → "aucune structure causale sur ce sujet dans le corpus soumis"

**`gcn-index` devient optionnel** — utile pour indexer un grand corpus en
batch avant une session, mais plus obligatoire pour commencer à discuter.

---

### C3 — Rendre `CausalGraph` persistable

**Fichier :** `gcn-python/src/gcn_python/verbalizer/instructions.py`

**Problème :** `CausalGraph` est volatile — disparaît à la fermeture.
Pour les vrais usages (audit d'un corpus de 1000 rapports), le graphe
doit être construit une fois et réutilisé.

**Changements :**
```python
class CausalGraph:
    def save(self, path: Path) -> None:
        """Persiste le graphe en JSON."""

    @classmethod
    def load(cls, path: Path) -> "CausalGraph":
        """Charge un graphe depuis un fichier JSON."""

    @classmethod
    def from_cirs(cls, cirs: list[dict]) -> "CausalGraph":
        """Construit depuis une liste de CIR."""
```

---

### C4 — Intégrer le verbalizer dans la sortie de `gcn-query`

**Problème :** Le plan initial a `gcn-query` qui retourne des résultats JSON bruts.
Ce n'est pas suffisant. Une requête sur un corpus de 1000 documents peut trouver
47 relations causales avec des sources, des niveaux de confiance et des contradictions
différents. Le verbalizer doit formater ce rapport complet.

**Architecture de la sortie :**

```
gcn-query --graph cti_graph.json --causes-of "data_exfiltration"
```

**Sortie verbalisée (pas une seule phrase) :**
```
causes_of: data_exfiltration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. [authentication_bypass]  --[enable]-->  [data_exfiltration]
     sources: APT28_report.txt:47, Mandiant_2024.pdf:12
     confidence: 0.91 (23 occurrences)

  2. [credential_theft]  --[cause]-->  [data_exfiltration]
     sources: CrowdStrike_Q3.txt:89
     confidence: 0.87 (8 occurrences)

  3. [sql_injection]  --[enable]-->  [data_exfiltration]
     sources: OWASP_report.txt:34, NVD_CVE_2024.txt:5
     confidence: 0.79 (5 occurrences)

  ⚠ CONTRADICTION : [firewall_rule]  --[prevent]-->  [data_exfiltration]
     sources: SecPolicy_v2.txt:12   (mais 23 sources affirment le contraire)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total : 3 causes, 1 contradiction, 30+ sources analysées
```

**Ce que le verbalizer produit ici :**
- Toutes les relations causales trouvées (pas une seule)
- Chaque relation avec ses sources, sa fréquence, sa confiance
- Les contradictions flaggées explicitement
- Un résumé quantitatif en bas

**Rôle de chaque composant :**
```
CausalGraph.causes_of("data_exfiltration")
    → liste de (src_node, dst_node, attrs, sources[])

QueryVerbalizer.format_causes_report(results)
    → rapport multi-lignes structuré ci-dessus
```

**Fichier à créer :** `gcn-python/src/gcn_python/verbalizer/query_report.py`

```python
class QueryVerbalizer:
    """
    Formate les résultats de requêtes CausalGraph en rapports lisibles.
    Pas de langage naturel généré — structure + données du CIR.
    """
    def format_causes(self, results, subject: str) -> str: ...
    def format_effects(self, results, subject: str) -> str: ...
    def format_path(self, path, kw_from: str, kw_to: str) -> str: ...
    def format_contradictions(self, results) -> str: ...
    def format_summary(self, graph) -> str: ...
    def format_counterfactual(self, results, subject: str) -> str: ...
```

**Le `ReferenceDecoder._verbalize()` reste** mais renommé en `dump` —
il sérialise un CIR individuel en texte structuré. La vraie verbalisation
en langage naturel fluide est le rôle du `TrainableDecoder` entraîné.

---

### C5 — Repositionner `--text` dans `gcn-forward`

**Fichier :** `gcn-python/src/gcn_python/pipeline/cli.py`

`--text` reste disponible pour les tests unitaires et le debugging,
mais l'aide doit le présenter clairement comme mode de test,
pas comme mode d'usage principal.

```python
@click.option("--text", default=None,
              help="[DEBUG] Analyser une seule phrase (test uniquement). "
                   "Pour les corpus, utiliser --file.")
```

---

### C6 — Mettre à jour le README

**Fichier :** `gcn-python/README.md`

Sections à réécrire :
- Supprimer la section "Session interactive" (gcn-chat)
- Remplacer les exemples `analyze("une phrase")` par des exemples corpus
- Ajouter la section "gcn-index / gcn-query" (nouveau CLI)
- Clarifier le scope : "GCN répond à des questions causales sur votre corpus"

---

### C7 — Mettre à jour `pyproject.toml`

```toml
[project.scripts]
gcn-index   = "gcn_python.query_cli:index_cmd"   # NOUVEAU — indexe un corpus → graphe persistant
gcn-discuss = "gcn_python.discuss:discuss_cmd"   # NOUVEAU — discussion causale sur corpus indexé
gcn-forward = "gcn_python.pipeline.cli:forward_cmd"
gcn-train   = "gcn_python.training.train:train_cmd"
gcn-eval    = "gcn_python.evaluation.eval_runner:eval_cmd"
# gcn-chat supprimé (dérapage LLM)
# gcn-bootstrap conservé (outil de génération de datasets)
# gcn-verbalize conservé (TrainableDecoder : CIR → texte naturel entraîné)
```

---

## Ordre d'exécution

```
C1 — engine.py : supprimer méthodes hors scope + réécrire docstrings
C3 — CausalGraph persistable (save/load/from_cirs)
C4 — verbalizer/query_report.py : QueryVerbalizer (rapports multi-lignes)
     + renommer _verbalize → dump dans ReferenceDecoder
C2 — Supprimer chat.py, créer query_cli.py (gcn-index) + discuss.py (gcn-discuss)
     gcn-discuss charge le graphe persistant + discussion causale + QueryVerbalizer
C5 — Repositionner --text dans gcn-forward
C6 — Mettre à jour README
C7 — Mettre à jour pyproject.toml
```

---

## Interface cible après Phase 16

```bash
# Démarrer une session — analyse et discussion dans la même session
gcn-discuss --checkpoint model.npz

# OU reprendre une session précédente
gcn-discuss --checkpoint model.npz --graph session.json
```

**Dans la session :**
```
  GCN Causal Engine
  ─────────────────────────────────────────────────────

  > /analyze rapport_incident.txt
    → rapport_incident.txt : 47 relations causales extraites.

  > /analyze threat_reports/
    → APT28_2024.txt     : 23 relations
    → Mandiant_Q3.txt    : 31 relations
    → CrowdStrike.pdf    : 18 relations
    → Total : 72 nouvelles relations. Session : 119 relations.

  > What causes data exfiltration?
    [rapport multi-lignes avec sources et niveaux de confiance]

  > Any contradictions?
    [claims contradictoires entre sources]

  > /save session.json
    → 119 relations sauvegardées.

  > /quit
```

**Commandes de session (préfixe /) :**
- `/analyze <fichier_ou_répertoire>` — analyser un fichier ou tous les fichiers d'un répertoire
- `/save <path.json>` — persister le graphe de session
- `/load <path.json>` — charger un graphe existant
- `/summarize` — résumé de la session courante
- `/help` — aide
- `/quit` — quitter

**Questions libres (sans préfixe) :**
Tout ce qui n'est pas une commande `/` est traité comme une question sur le graphe accumulé.

---

## Ce qui NE change PAS

- Pipeline ML (MLP + R-GCN), entraînement, checkpoints : inchangés
- `analyze()` et `analyze_batch()` : conservés (nécessaires pour gcn-index)
- `stream()` : conservé (corpus pré-segmenté)
- `InstructionHandler` et `CausalGraph` : conservés, améliorés (persistance)
- Toute la stack Rust (frontends, GCN-QL, Pearl) : inchangée
- `gcn-forward`, `gcn-train`, `gcn-eval` : conservés
- Tests existants (206) : doivent tous passer

---

## Vérification

```bash
python -m pytest gcn-python/tests/ -q
# ≥ 206 passed

gcn-index --help
gcn-query --help

# Test intégration
gcn-index --corpus gcn-datasets/real/ --checkpoint gcn-datasets/checkpoints/prod_v1.npz \
          --output /tmp/test_graph.json
gcn-query --graph /tmp/test_graph.json --summarize
```
