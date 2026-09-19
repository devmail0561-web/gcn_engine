# gcn-scraper — Scraper multilingue pour le dataset GCN causal

Outil de scraping multi-sources qui collecte du texte brut diversifié et équilibré FR/EN pour le dataset causal GCN.

## Installation

```bash
pip install gcn-scraper
```

## Usage

```bash
# Défaut — FR + EN, 50k phrases
gcn-scrape --output-dir gcn-datasets/raw --target-total 50000

# Contrôle complet
gcn-scrape --output-dir gcn-datasets/raw \
  --langs fr,en,es,de \
  --prog-langs python,rust,javascript,typescript,java,go \
  --target-total 80000

# Texte uniquement (pas de code)
gcn-scrape --output-dir gcn-datasets/raw --prog-langs none

# Tout activer
gcn-scrape --output-dir gcn-datasets/raw --langs all --prog-langs all

# Reproductible (même seed = même ordre)
gcn-scrape --output-dir gcn-datasets/raw --seed 42

# Avec email de contact (améliore le débit OpenAlex/PubMed)
gcn-scrape --output-dir gcn-datasets/raw --contact-email moi@example.org

# Reprendre un run interrompu
gcn-scrape --output-dir gcn-datasets/raw --resume
```

## Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `--output-dir` | *(requis)* | Répertoire de sortie |
| `--target-total` | 50000 | Nombre de phrases cibles |
| `--resume` | false | Reprend depuis le checkpoint |
| `--langs` | fr,en | Langues humaines (ISO, séparées par virgule) |
| `--prog-langs` | python,rust | Langages de programmation |
| `--min-quality` | 0.3 | Score de qualité minimum [0-1] |
| `--seed` | aléatoire | Graine de diversité |
| `--contact-email` | None | Email pour les APIs polies |
| `--hal/--no-hal` | true | Activer/désactiver HAL |
| `--arxiv/--no-arxiv` | true | Activer/désactiver arXiv |
| `--news/--no-news` | true | Activer/désactiver News RSS |
| `--github/--no-github` | false | Activer GitHub (token recommandé) |
| `--doc/--no-doc` | true | Activer Documentation |
| `--web-search/--no-web-search` | false | Activer Recherche web |

## Sources

| Source | Langues | API | Notes |
|--------|---------|-----|-------|
| Wikipedia | FR, EN, ES, DE, PT, IT | MediaWiki | Recherche paginée + expansion catégories |
| HAL Scientifique | FR, EN | REST | Papiers CC-BY |
| arXiv | EN | Atom/REST | Abstracts scientifiques |
| News RSS | FR, EN | RSS | BBC, Guardian, Reuters |
| GitHub Code | tous | REST | Code source (token pour rate-limit) |
| Documentation | tous | Web | Docs officielles |
| Recherche web | FR, EN | DuckDuckGo + OpenAlex + PubMed | Croisement multi-moteurs |

## Sortie

Chaque session génère des fichiers horodatés :
```
gcn-datasets/raw/
├── sentences_20260919_140607.jsonl          # Fusion globale
├── wikipedia_fr_20260919_140607.jsonl       # Par source
├── wikipedia_en_20260919_140607.jsonl
├── hal_20260919_140607.jsonl
├── arxiv_20260919_140607.jsonl
└── .checkpoint.json                          # État du run
```

Format JSONL par ligne :
```json
{"text": "...", "lang": "fr", "source": "wikipedia_fr", "url": "...", "quality": 0.85}
```

## Architecture

```
gcn-scraper/
├── src/gcn_scraper/
│   ├── cli.py              # CLI click : gcn-scrape
│   ├── pipeline.py         # Orchestrateur multi-sources
│   ├── splitter.py         # Découpage en phrases
│   ├── diversity.py        # RNG déterministe, mélange, pagination
│   ├── balance_tracker.py  # Budget par langue
│   ├── checkpoint.py       # Reprise inter-runs
│   ├── filters/
│   │   ├── quality_scorer.py   # Score de qualité
│   │   ├── lang_detector.py    # Détection de langue
│   │   ├── deduplicator.py     # Déduplication
│   │   └── causal_scorer.py    # Scoring causal
│   ├── sources/
│   │   ├── wikipedia.py        # Wikipedia générique (toutes langues)
│   │   ├── hal_scientific.py   # HAL archives ouvertes
│   │   ├── arxiv.py            # arXiv
│   │   ├── news_rss.py         # News RSS
│   │   ├── github_code.py      # GitHub Code Search
│   │   ├── doc_scrape.py       # Documentation
│   │   ├── web_search.py       # DuckDuckGo + OpenAlex + PubMed
│   │   └── _net.py             # Utilitaires réseau (retry, rate-limit)
│   └── config/
│       ├── sources.yaml        # Configuration centralisée
│       └── loader.py           # Chargement YAML
└── pyproject.toml
```

## Développement

```bash
cd gcn-tools/gcn-scraper
pip install -e ".[dev]"
```
