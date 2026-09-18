"""Pipeline de scraping avancé : balance, dédup, checkpoint, JSONL."""
from __future__ import annotations
import json
from pathlib import Path

from .splitter import split_sentences
from .filters.causal_scorer import CausalScorer
from .filters.deduplicator import Deduplicator
from .balance_tracker import BalanceTracker
from .checkpoint import ScrapingCheckpoint


class ScrapingPipeline:
    """
    Orchestre le scraping multi-sources vers 50k phrases équilibrées.

    Sortie : sentences.jsonl, une ligne JSON par phrase :
      {"text": "...", "lang": "fr", "source": "...", "url": "...",
       "relation_hints": ["cause"], "score": 0.09}
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
          wikipedia_fr, wikipedia_en, hal, arxiv, news, github, doc,
          budget, resume, min_score, langs
        """
        scorer = CausalScorer()
        dedup = Deduplicator()
        tracker = BalanceTracker(config.get("budget"))
        checkpoint = ScrapingCheckpoint(self.output_dir / ".checkpoint.json")
        output_file = self.output_dir / "sentences.jsonl"
        min_score: float = config.get("min_score", 0.0)
        resume: bool = config.get("resume", False)

        if not resume and output_file.exists():
            output_file.unlink()
        if not resume:
            checkpoint.reset()

        total_written = 0

        # Ouvrir le fichier global + les fichiers par source en mode append
        _src_files: dict[str, object] = {}
        _source_keys = [
            ("wikipedia_fr", "wikipedia_fr"),
            ("wikipedia_en", "wikipedia_en"),
            ("hal",          "hal"),
            ("arxiv",        "arxiv"),
            ("news",         "news_rss"),
            ("github",       "github"),
            ("doc",          "doc"),
        ]
        _mode = "a" if resume else "w"
        for cfg_key, file_key in _source_keys:
            if config.get(cfg_key):
                _src_files[cfg_key] = open(
                    self.output_dir / f"{file_key}.jsonl", _mode, encoding="utf-8"
                )

        with open(output_file, _mode, encoding="utf-8") as out:

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
                    articles = scraper.scrape_all(config["wikipedia_fr"].get("max_per_category", 50))
                    n = self._process_texts(articles, out, scorer, dedup, tracker, min_score,
                                            source_out=_src_files.get(key))
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
                    _mpc = config["wikipedia_en"].get("max_per_category", 50)
                    articles = scraper.scrape(max_per_query=_mpc, max_per_category=_mpc)
                    n = self._process_texts(articles, out, scorer, dedup, tracker, min_score,
                                            source_out=_src_files.get(key))
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
                    n = self._process_texts(papers, out, scorer, dedup, tracker, min_score,
                                            source_out=_src_files.get(key))
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
                    n = self._process_texts(papers, out, scorer, dedup, tracker, min_score,
                                            source_out=_src_files.get(key))
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
                    n = self._process_texts(items, out, scorer, dedup, tracker, min_score,
                                            source_out=_src_files.get(key))
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
                    n = self._process_texts(items, out, scorer, dedup, tracker, min_score,
                                            skip_split=True, source_out=_src_files.get(key))
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
                    n = self._process_texts(docs, out, scorer, dedup, tracker, min_score,
                                            source_out=_src_files.get(key))
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

        # Fermer les fichiers par source
        for f in _src_files.values():
            f.close()

        print("\n=== Résumé final ===")
        balance = tracker.summary()
        for rel, stat in balance.items():
            print(f"  {rel:<22} {stat}")
        print(f"\nTotal écrit : {total_written}")
        print(tracker.progress_bar())

        return {
            "total_written": total_written,
            "balance": balance,
            "output": output_file,
        }

    # ------------------------------------------------------------------
    # Méthode interne
    # ------------------------------------------------------------------

    def _process_texts(
        self,
        items: list[dict],
        out,
        scorer: CausalScorer,
        dedup: Deduplicator,
        tracker: BalanceTracker,
        min_score: float,
        skip_split: bool = False,
        source_out=None,
    ) -> int:
        """
        Pour chaque item, split → score → filtre → écrit en JSONL.

        Retourne le nombre de phrases écrites.
        """
        written = 0
        for item in items:
            raw_text = item.get("text", "")
            if not raw_text:
                continue
            lang = item.get("lang", "auto")
            source = item.get("source", "unknown")
            url = item.get("url", "")

            sentences = [raw_text] if skip_split else split_sentences(raw_text)

            for sent in sentences:
                if tracker.is_globally_full():
                    return written
                score, hints = scorer.score(sent, lang=lang)
                if score < min_score and not hints:
                    continue
                if tracker.is_full(hints):
                    continue
                if dedup.is_duplicate(sent):
                    continue

                record = {
                    "text": sent,
                    "lang": lang if lang != "auto" else scorer.detect_lang(sent),
                    "source": source,
                    "url": url,
                    "relation_hints": hints,
                    "score": round(score, 4),
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                out.write(line)
                if source_out is not None:
                    source_out.write(line)
                tracker.add(hints)
                written += 1
        return written
