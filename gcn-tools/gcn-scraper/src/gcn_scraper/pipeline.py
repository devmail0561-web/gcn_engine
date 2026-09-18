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
        checkpoint = ScrapingCheckpoint(self.output_dir / ".checkpoint.json")
        min_quality: float = config.get("min_quality", 0.3)
        resume: bool = config.get("resume", False)
        selected_langs: list[str] = config.get("langs", ["fr", "en"])
        selected_prog_langs: list[str] = config.get("prog_langs", ["python", "rust"])
        target_total: int = config.get("target_total", 50000)

        # Budget calculé sur les langues sélectionnées uniquement ;
        # bucket "code" seulement si des prog-langs sont actifs (audit-2 Fix 4).
        from .balance_tracker import _compute_budget
        budget = config.get("budget") or _compute_budget(
            selected_langs, target_total, include_code=bool(selected_prog_langs)
        )
        tracker = BalanceTracker(budget)

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

        # Découvrir dynamiquement les sources Wikipedia pour les langues sélectionnées,
        # triées par budget_weight décroissant (audit-2 Fix 5 : fr/en en premier,
        # pas en ordre alphabétique) puis par ordre des --langs pour départager.
        from .config.loader import get_config as _get_cfg
        _all_cfg = _get_cfg()
        _lang_rank = {lang: i for i, lang in enumerate(selected_langs)}
        _wiki_keys = sorted(
            (
                k for k, v in _all_cfg.get("sources", {}).items()
                if k.startswith("wikipedia_")
                and v.get("enabled", True)
                and v.get("lang", k.replace("wikipedia_", "")) in selected_langs
            ),
            key=lambda k: (
                -float(_all_cfg["sources"][k].get("budget_weight", 1.0)),
                _lang_rank.get(
                    _all_cfg["sources"][k].get("lang", k.replace("wikipedia_", "")),
                    len(selected_langs),
                ),
                k,
            ),
        )

        _fixed_source_keys = [
            ("hal",          f"hal_{timestamp}.jsonl"),
            ("arxiv",        f"arxiv_{timestamp}.jsonl"),
            ("news",         f"news_rss_{timestamp}.jsonl"),
            ("github",       f"github_{timestamp}.jsonl"),
            ("doc",          f"doc_{timestamp}.jsonl"),
            ("web_search",   f"web_search_{timestamp}.jsonl"),
        ]
        _wiki_source_keys = [(k, f"{k}_{timestamp}.jsonl") for k in _wiki_keys]
        _source_keys = _wiki_source_keys + _fixed_source_keys

        total_written = 0

        with contextlib.ExitStack() as stack:
            out = stack.enter_context(open(global_file, _mode, encoding="utf-8"))
            _src_files: dict[str, object] = {}
            for cfg_key, fname in _source_keys:
                # NOTE (audit-2 Fix 1) : tester la présence de la clé, pas sa
                # valeur — cli.py assigne config['doc'] = {} / config['web_search'] = {}
                # et bool({}) == False désactiverait ces sources.
                if cfg_key in config or cfg_key in _wiki_keys:
                    _src_files[cfg_key] = stack.enter_context(
                        open(self.output_dir / fname, _mode, encoding="utf-8")
                    )

            # --- Wikipedia (toutes langues, découverte dynamique depuis config) ---
            for key in _wiki_keys:
                if tracker.is_globally_full():
                    break
                wiki_cfg = _all_cfg.get("sources", {}).get(key, {})
                if not wiki_cfg.get("enabled", True):
                    continue
                lang = wiki_cfg.get("lang", key.replace("wikipedia_", ""))
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print(f"=== Wikipedia {lang.upper()} ===")
                    from .sources.wikipedia import WikipediaLangScraper
                    scraper = WikipediaLangScraper(lang, self.user_agent)
                    articles = scraper.scrape(tracker=tracker)
                    n = self._process_texts(articles, out, scorer, dedup, tracker, min_quality,
                                            default_lang=lang, source_out=_src_files.get(key))
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} phrases retenues | {tracker.progress_bar()}")

            # --- HAL (FR + EN) ---
            _hal_langs = ("fr" in selected_langs or "en" in selected_langs)
            if "hal" in config and not tracker.is_globally_full() and _hal_langs:
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

            # --- arXiv (EN uniquement) ---
            if "arxiv" in config and not tracker.is_globally_full() and "en" in selected_langs:
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
            if "news" in config and not tracker.is_globally_full():
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
            if "web_search" in config and not tracker.is_globally_full():
                key = "web_search"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print("=== Recherche web (DuckDuckGo + OpenAlex + PubMed) ===")
                    from .sources.web_search import WebSearchScraper
                    scraper = WebSearchScraper(self.user_agent)
                    items = scraper.scrape(tracker=tracker, langs=selected_langs)
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
            if "github" in config and selected_prog_langs and not tracker.is_globally_full():
                key = "github"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print(f"=== GitHub Code ({', '.join(selected_prog_langs)}) ===")
                    from .sources.github_code import GitHubCodeScraper
                    scraper = GitHubCodeScraper(
                        self.user_agent,
                        github_token=config["github"].get("token"),
                    )
                    items = scraper.scrape(
                        languages=selected_prog_langs,
                        max_per_query=config["github"].get("max_per_query", 30),
                    )
                    n = self._process_texts(items, out, scorer, dedup, tracker, min_quality,
                                            skip_split=True, default_lang="code",
                                            source_out=_src_files.get(key))
                    self._warn_zero(key, n)
                    total_written += n
                    checkpoint.mark_done(key, n)
                    print(f"  → {n} extraits retenus | {tracker.progress_bar()}")

            # --- Documentation ---
            if "doc" in config and selected_prog_langs and not tracker.is_globally_full():
                key = "doc"
                if resume and checkpoint.is_done(key):
                    print(f"=== {key}: déjà fait ({checkpoint.get_count(key)} phrases), skip ===")
                    total_written += checkpoint.get_count(key)
                else:
                    print(f"=== Documentation ({', '.join(selected_prog_langs)}) ===")
                    from .sources.doc_scrape import DocScraper
                    scraper = DocScraper(self.user_agent)
                    docs = scraper.scrape(languages=selected_prog_langs)
                    n = self._process_texts(docs, out, scorer, dedup, tracker, min_quality,
                                            default_lang="code", source_out=_src_files.get(key))
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
