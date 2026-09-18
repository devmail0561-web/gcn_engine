"""Pipeline de scraping — collecte du texte brut diversifié, équilibré FR/EN.

Responsabilité unique : collecter du texte de qualité, pas l'annoter.
L'annotation causale est la responsabilité du moteur GCN.

Sortie JSONL horodatée :
  {"text": "...", "lang": "fr", "source": "wikipedia_fr", "url": "...", "quality": 0.85}
"""
from __future__ import annotations
import contextlib
import json
import warnings
from datetime import datetime
from pathlib import Path

from .splitter import split_sentences
from .filters.quality_scorer import QualityScorer
from .filters.lang_detector import detect_lang
from .filters.deduplicator import Deduplicator
from .balance_tracker import BalanceTracker
from .checkpoint import ScrapingCheckpoint


class ScrapingPipeline:
    """
    Orchestre le scraping multi-sources vers 50k phrases équilibrées FR/EN.

    Sortie horodatée — chaque session génère des fichiers uniques :
      sentences_20260918_143022.jsonl     (fusion globale)
      wikipedia_fr_20260918_143022.jsonl  (par source)
      ...

    En mode --resume, les fichiers de la session précédente sont réutilisés.
    """

    def __init__(self, output_dir: Path, user_agent: str = "GCN-Dataset/2.0"):
        self.output_dir = Path(output_dir)
        self.user_agent = user_agent
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def run(self, config: dict) -> dict:
        """
        Lance le pipeline complet.

        config keys (tous optionnels) :
          wikipedia_fr, wikipedia_en, hal, arxiv, news, github, doc, web_search,
          budget, resume, min_quality, langs
        """
        scorer = QualityScorer()
        dedup = Deduplicator()
        tracker = BalanceTracker(config.get("budget"))
        checkpoint = ScrapingCheckpoint(self.output_dir / ".checkpoint.json")
        min_quality: float = config.get("min_quality", 0.3)
        resume: bool = config.get("resume", False)

        # Timestamp : réutiliser depuis le checkpoint en mode resume, sinon nouveau
        if resume and checkpoint.state.get("session_timestamp"):
            timestamp: str = checkpoint.state["session_timestamp"]
            _mode = "a"
            print(f"=== Reprise session {timestamp} ===")
        else:
            # Pas de session_timestamp (run v2 ou premier run) — démarrer fresh
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _mode = "w"
            # Ne pas accumuler les compteurs d'une ancienne session incompatible
            checkpoint.reset()
            checkpoint.state["session_timestamp"] = timestamp
            checkpoint._save()

        global_file = self.output_dir / f"sentences_{timestamp}.jsonl"

        _source_keys = [
            ("wikipedia_fr", f"wikipedia_fr_{timestamp}.jsonl"),
            ("wikipedia_en", f"wikipedia_en_{timestamp}.jsonl"),
            ("hal",          f"hal_{timestamp}.jsonl"),
            ("arxiv",        f"arxiv_{timestamp}.jsonl"),
            ("news",         f"news_rss_{timestamp}.jsonl"),
            ("github",       f"github_{timestamp}.jsonl"),
            ("doc",          f"doc_{timestamp}.jsonl"),
            ("web_search",   f"web_search_{timestamp}.jsonl"),
        ]

        total_written = 0

        with contextlib.ExitStack() as stack:
            out = stack.enter_context(open(global_file, _mode, encoding="utf-8"))
            _src_files: dict[str, object] = {}
            for cfg_key, fname in _source_keys:
                if config.get(cfg_key):
                    _src_files[cfg_key] = stack.enter_context(
                        open(self.output_dir / fname, _mode, encoding="utf-8")
                    )

            # --- Wikipedia FR ---
            if config.get("wikipedia_fr") and not tracker.is_globally_full():
                key = "wikipedia_fr"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== Wikipedia FR ===")
                    from .sources.wikipedia_fr import WikipediaFRScraper
                    scraper = WikipediaFRScraper(self.user_agent)
                    mpc = config["wikipedia_fr"].get("max_per_category", 50)
                    articles = scraper.scrape(max_per_query=mpc, max_per_category=mpc)
                    n = self._process_texts(articles, out, scorer, dedup, tracker, min_quality,
                                            default_lang="fr", source_out=_src_files.get(key))
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

            # --- Wikipedia EN ---
            if config.get("wikipedia_en") and not tracker.is_globally_full():
                key = "wikipedia_en"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== Wikipedia EN ===")
                    from .sources.wikipedia_en import WikipediaENScraper
                    scraper = WikipediaENScraper(self.user_agent)
                    mpc = config["wikipedia_en"].get("max_per_category", 50)
                    articles = scraper.scrape(max_per_query=mpc, max_per_category=mpc)
                    n = self._process_texts(articles, out, scorer, dedup, tracker, min_quality,
                                            default_lang="en", source_out=_src_files.get(key))
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

            # --- HAL ---
            if config.get("hal") and not tracker.is_globally_full():
                key = "hal"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== HAL Scientifique ===")
                    from .sources.hal_scientific import HALScraper
                    scraper = HALScraper(self.user_agent)
                    papers = scraper.scrape(max_per_query=config["hal"].get("max_per_query", 100))
                    n = self._process_texts(papers, out, scorer, dedup, tracker, min_quality,
                                            source_out=_src_files.get(key))
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

            # --- arXiv ---
            if config.get("arxiv") and not tracker.is_globally_full():
                key = "arxiv"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== arXiv ===")
                    from .sources.arxiv import ArXivScraper
                    scraper = ArXivScraper(self.user_agent)
                    papers = scraper.scrape(max_per_query=config["arxiv"].get("max_per_query", 100))
                    n = self._process_texts(papers, out, scorer, dedup, tracker, min_quality,
                                            default_lang="en", source_out=_src_files.get(key))
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

            # --- News RSS ---
            if config.get("news") and not tracker.is_globally_full():
                key = "news"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== News RSS ===")
                    from .sources.news_rss import NewsRSSScraper
                    scraper = NewsRSSScraper(self.user_agent)
                    langs = config["news"].get("langs", ["fr", "en"])
                    items = scraper.scrape(langs=langs)
                    n = self._process_texts(items, out, scorer, dedup, tracker, min_quality,
                                            source_out=_src_files.get(key))
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

            # --- Recherche web multi-sources ---
            if config.get("web_search") and not tracker.is_globally_full():
                key = "web_search"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== Recherche web (DuckDuckGo + OpenAlex + PubMed) ===")
                    from .sources.web_search import WebSearchScraper
                    scraper = WebSearchScraper(self.user_agent)
                    items = scraper.scrape()
                    n = self._process_texts(
                        items, out, scorer, dedup, tracker, min_quality,
                        source_out=_src_files.get(key),
                        extra_fields=("relevance_score", "found_by"),
                    )
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

            # --- GitHub Code ---
            if config.get("github") and not tracker.is_globally_full():
                key = "github"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== GitHub Code ===")
                    from .sources.github_code import GitHubCodeScraper
                    scraper = GitHubCodeScraper(
                        self.user_agent,
                        github_token=config["github"].get("token"),
                    )
                    languages = config["github"].get("languages", ["python", "rust"])
                    items = scraper.scrape(languages=languages,
                                          max_per_query=config["github"].get("max_per_query", 30))
                    n = self._process_texts(items, out, scorer, dedup, tracker, min_quality,
                                            skip_split=True, default_lang="en",
                                            source_out=_src_files.get(key))
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} extraits retenus | {tracker.progress_bar()}")

            # --- Documentation ---
            if config.get("doc") and not tracker.is_globally_full():
                key = "doc"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== Documentation ===")
                    from .sources.doc_scrape import DocScraper
                    scraper = DocScraper(self.user_agent)
                    docs = scraper.scrape()
                    n = self._process_texts(docs, out, scorer, dedup, tracker, min_quality,
                                            default_lang="en", source_out=_src_files.get(key))
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

        print("\n=== Résumé final ===")
        balance = tracker.summary()
        for lang, stat in balance.items():
            print(f"  {lang:<6} {stat}")
        print(f"\nTotal écrit : {total_written}")
        print(f"Fichier global : {global_file.name}")
        print(tracker.progress_bar())

        return {
            "total_written": total_written,
            "balance": balance,
            "output": global_file,
            "timestamp": timestamp,
        }

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    @staticmethod
    def _warn_zero(key: str, n: int) -> None:
        if n == 0:
            warnings.warn(
                f"Source '{key}' : 0 phrases retenues — "
                "vérifier la connectivité réseau ou les paramètres de la source.",
                UserWarning, stacklevel=3,
            )

    def _process_texts(
        self,
        items: list[dict],
        out,
        scorer: QualityScorer,
        dedup: Deduplicator,
        tracker: BalanceTracker,
        min_quality: float,
        skip_split: bool = False,
        default_lang: str = "auto",
        source_out=None,
        extra_fields: tuple[str, ...] = (),
    ) -> int:
        """
        Pour chaque item : split → qualité → filtre → écrit en JSONL.

        Champs de sortie : text, lang, source, url, quality.
        Pas de relation_hints — l'annotation est faite par le moteur GCN.
        """
        written = 0
        for item in items:
            raw_text = item.get("text", "")
            if not raw_text:
                continue
            source = item.get("source", "unknown")
            url = item.get("url", "")
            item_lang = item.get("lang", default_lang)

            sentences = [raw_text] if skip_split else split_sentences(raw_text)

            for sent in sentences:
                if tracker.is_globally_full():
                    return written

                # Détection de langue
                lang = item_lang if item_lang != "auto" else detect_lang(sent)

                # Filtre budget langue
                if tracker.is_full(lang):
                    continue

                # Score de qualité (longueur, densité, propreté)
                quality = scorer.score(sent)
                if quality < min_quality:
                    continue

                # Déduplication
                if dedup.is_duplicate(sent):
                    continue

                record: dict = {
                    "text": sent,
                    "lang": lang,
                    "source": source,
                    "url": url,
                    "quality": quality,
                }
                for field in extra_fields:
                    if field in item:
                        record[field] = item[field]

                line = json.dumps(record, ensure_ascii=False) + "\n"
                out.write(line)
                if source_out is not None:
                    source_out.write(line)
                tracker.add(lang)
                written += 1
        return written
