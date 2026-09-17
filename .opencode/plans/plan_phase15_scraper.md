# Plan Phase 15 — Scraper de Données Réelles pour Dataset Causal

**Date** : 2026-09-17
**Objectif** : Créer un outil de scraping pour générer un dataset de 5000+ phrases françaises à partir de sources réelles, annotées par le LLM.

---

## Contexte

Le dataset actuel (990 phrases) est synthétique et permet la memorisation. Il faut des données réelles pour prouver que le modèle apprend véritablement la sémantique causale.

**Principe** : Le scraper ne fait QUE du scraping. Pas de filtre de connecteurs, pas de detection de causalite. Le LLM (gcn-annotate) apprend tout sur les donnees annotees.

---

## Sources sélectionnées

| Source | Volume estimé | Qualité | Accès |
|--------|---------------|---------|-------|
| Wikipedia FR | 100k+ articles | Élevée (texte explicatif) | API officielle |
| HAL (scientifique) | 1.6M+ papiers | Élevée (CC-BY) | REST API |
| Éducation | 10k+ fiches | Moyenne | Web scraping |

---

## Architecture

```
[Wikipedia API] ──┐
[HAL REST API] ───┼──→ Scraper ──→ Sentence Splitter ──→ sentences_raw.txt ──→ gcn-annotate ──→ GCN-NL JSON
[Éducation Web] ──┘    (trafilatura)  (regex longueur)     (1 phrase/ligne)    (LLM Claude/GPT)
```

---

## Étapes d'implémentation

### Étape 1 : `gcn-datasets/scraper/sources/wikipedia_fr.py`

Scraper Wikipedia FR via API.

```python
import wikipediaapi

class WikipediaFRScraper:
    def __init__(self, user_agent: str):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language='fr',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
    
    def scrape_category(self, category: str, max_articles: int = 100) -> list[dict]:
        """Extrait les articles d'une catégorie Wikipedia."""
        pass
    
    def scrape_articles(self, titles: list[str]) ->list[dict]:
        """Extrait des articles spécifiques."""
        pass
```

**Catégories cibles** : Sciences, Économie, Santé, Histoire, Technologie

**Estimation** : ~150 lignes

---

### Étape 2 : `gcn-datasets/scraper/sources/hal_scientific.py`

Scraper HAL via REST API.

```python
import requests

class HALScraper:
    BASE_URL = "https://api.archives-ouvertes.fr/search/"
    
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.session = requests.Session()
    
    def search_papers(self, query: str, max_results: int = 100) -> list[dict]:
        """Recherche des papiers sur HAL."""
        pass
    
    def extract_abstract(self, paper: dict) -> str | None:
        """Extrait l'abstract d'un papier."""
        pass
```

**Estimation** : ~100 lignes

---

### Étape 3 : `gcn-datasets/scraper/sources/education_web.py`

Scraper sites éducatifs avec trafilatura.

```python
import trafilatura

class EducationScraper:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
    
    def scrape_url(self, url: str) -> str | None:
        """Télécharge et nettoie une page web."""
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
        return None
```

**Estimation** : ~80 lignes

---

### Étape 4 : `gcn-datasets/scraper/splitter.py`

Splitter le texte en phrases. **Pas de filtre de connecteurs.**

```python
import re

def split_sentences(text: str, min_length: int = 10, max_length: int = 500) -> list[str]:
    """Découpe un texte en phrases.
    
    Pas de filtre de connecteurs - le LLM determine la causalite.
    Filtrage uniquement par longueur (eviter les fragments trop courts).
    """
    # Split sur les points, points d'interrogation, points d'exclamation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filtrer par longueur
    result = []
    for s in sentences:
        s = s.strip()
        if min_length <= len(s) <= max_length:
            result.append(s)
    
    return result
```

**Estimation** : ~30 lignes

---

### Étape 5 : `gcn-datasets/scraper/pipeline.py`

Pipeline orchestrateur : scraping → split → sauvegarde.

```python
from pathlib import Path
from .sources.wikipedia_fr import WikipediaFRScraper
from .sources.hal_scientific import HALScraper
from .sources.education_web import EducationScraper
from .splitter import split_sentences

class ScrapingPipeline:
    def __init__(self, output_dir: Path, user_agent: str):
        self.output_dir = output_dir
        self.scraper_wiki = WikipediaFRScraper(user_agent)
        self.scraper_hal = HALScraper(user_agent)
        self.scraper_edu = EducationScraper(user_agent)
    
    def run(self, config: dict) -> dict:
        """Lance le pipeline complet."""
        all_sentences = []
        
        # 1. Scraping
        if config.get("wikipedia"):
            texts = self._scrape_wikipedia(config["wikipedia"])
            all_sentences.extend(self._split_texts(texts, "wikipedia"))
        
        if config.get("hal"):
            texts = self._scrape_hal(config["hal"])
            all_sentences.extend(self._split_texts(texts, "hal"))
        
        if config.get("education"):
            texts = self._scrape_education(config["education"])
            all_sentences.extend(self._split_texts(texts, "education"))
        
        # 2. Dédoublonnage
        unique = list(dict.fromkeys(all_sentences))
        
        # 3. Sauvegarde (format gcn-annotate : 1 phrase/ligne)
        self._save(unique)
        
        return {
            "total": len(all_sentences),
            "unique": len(unique),
            "output": self.output_dir / "sentences_raw.txt"
        }
    
    def _split_texts(self, texts: list[dict], source: str) -> list[str]:
        """Découpe les textes en phrases."""
        result = []
        for t in texts:
            sentences = split_sentences(t["text"])
            result.extend(sentences)
        return result
    
    def _save(self, sentences: list[str]):
        """Sauvegarde au format gcn-annotate."""
        output_file = self.output_dir / "sentences_raw.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")
```

**Estimation** : ~100 lignes

---

### Étape 6 : `gcn-datasets/scraper/run_scraping.py`

CLI principal.

```python
import click
from pathlib import Path
from .pipeline import ScrapingPipeline

@click.command()
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--max-wikipedia", default=500, type=int)
@click.option("--max-hal", default=200, type=int)
@click.option("--max-education", default=100, type=int)
@click.option("--user-agent", default="GCN-DatasetBuilder/1.0 (research)")
def scrape(output_dir, max_wikipedia, max_hal, max_education, user_agent):
    """Scrape des données réelles pour le dataset causal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = ScrapingPipeline(output_dir, user_agent)
    config = {
        "wikipedia": {"max_articles": max_wikipedia},
        "hal": {"max_papers": max_hal},
        "education": {"max_pages": max_education},
    }
    
    result = pipeline.run(config)
    click.echo(f"Terminé : {result['unique']} phrases uniques")
    click.echo(f"Fichier : {result['output']}")
```

**Estimation** : ~40 lignes

---

### Étape 7 : `gcn-datasets/scraper/annotate_dataset.py`

Annotation avec gcn-annotate (existant).

```python
import click
from pathlib import Path
import subprocess

@click.command()
@click.option("--input", required=True, type=click.Path(path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--llm-backend", default="anthropic")
@click.option("--model", default=None)
@click.option("--batch-size", default=20, type=int)
@click.option("--max-sentences", default=None, type=int)
def annotate(input, output_dir, llm_backend, model, batch_size, max_sentences):
    """Annotate les phrases avec gcn-annotate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if max_sentences:
        with open(input) as f:
            lines = f.readlines()
        filtered = output_dir / "sentences_filtered.txt"
        with open(filtered, "w") as f:
            f.writelines(lines[:max_sentences])
        input = filtered
    
    cmd = [
        "gcn-annotate", "annotate",
        "--input", str(input),
        "--output", str(output_dir),
        "--llm-backend", llm_backend,
        "--batch-size", str(batch_size),
    ]
    if model:
        cmd.extend(["--model", model])
    
    subprocess.run(cmd, check=True)
```

**Estimation** : ~40 lignes

---

### Étape 8 : `gcn-datasets/scraper/split_dataset.py`

Split train/val/test.

```python
import json
import random
from pathlib import Path

def split_dataset(input_file: Path, output_dir: Path, seed: int = 42):
    """Split le dataset en train/val/test (70/15/15)."""
    with open(input_file) as f:
        data = json.load(f)
    
    sentences = data["document"]["sentences"]
    rng = random.Random(seed)
    rng.shuffle(sentences)
    
    n = len(sentences)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    splits = {
        "train": sentences[:n_train],
        "val": sentences[n_train:n_train + n_val],
        "test": sentences[n_train + n_val:],
    }
    
    for name, data in splits.items():
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump({"document": {"id": f"split_{name}", "lang": "fr", "sentences": data}}, f, indent=2, ensure_ascii=False)
    
    return {k: len(v) for k, v in splits.items()}
```

**Estimation** : ~40 lignes

---

## Structure finale

```
gcn-datasets/scraper/
├── __init__.py
├── run_scraping.py          # CLI scraping
├── annotate_dataset.py      # CLI annotation
├── split_dataset.py         # CLI split
├── pipeline.py              # Orchestrateur
├── splitter.py              # Split phrases
└── sources/
    ├── __init__.py
    ├── wikipedia_fr.py
    ├── hal_scientific.py
    └── education_web.py
```

---

## Commandes

```bash
# 1. Scraping
python -m gcn_datasets.scraper.run_scraping \
  --output-dir gcn-datasets/scraper/output/ \
  --max-wikipedia 500 --max-hal 200 --max-education 100

# 2. Annotation
python -m gcn_datasets.scraper.annotate_dataset \
  --input gcn-datasets/scraper/output/sentences_raw.txt \
  --output-dir gcn-datasets/scraper/annotated/ \
  --llm-backend anthropic --batch-size 20 --max-sentences 5000

# 3. Split
python -m gcn_datasets.scraper.split_dataset \
  --input gcn-datasets/scraper/annotated/sentences_raw_llm.json \
  --output-dir gcn-datasets/real/

# 4. Entraînement
gcn-train --data-dir gcn-datasets/real/ --epochs 50 --lr 0.001 \
  --embedding-dim 50 --embedding-file gcn-python/models/wiki.fr.vec \
  --use-attention --bidirectional \
  --output gcn-datasets/checkpoints/real_data_model.npz
```

---

## Dépendances

```bash
pip install trafilatura newspaper4k beautifulsoup4 requests wikipedia-api
```

---

## Coûts estimés

- ~80,000 phrases extraites → ~5,000 annotées
- Anthropic : ~$2-5 | OpenAI : ~$5-10
