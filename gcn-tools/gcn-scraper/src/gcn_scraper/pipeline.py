# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Pipeline de scraping — collecte du texte brut diversifié, équilibré FR/EN.

Responsabilité unique : collecter du texte de qualité, pas l'annoter.
L'annotation causale est la responsabilité du moteur GCN.

Sortie JSONL horodatée :
  {"text": "...", "lang": "fr", "source": "wikipedia_fr", "url": "...", "quality": 0.85}
"""
from __future__ import annotations
import contextlib
import json
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .splitter import split_sentences
from .filters.quality_scorer import QualityScorer
from .filters.lang_detector import detect_lang
from .filters.deduplicator import Deduplicator
from .balance_tracker import BalanceTracker
from .checkpoint import ScrapingCheckpoint
from .url_registry import URLRegistry


def _close_registry(registry, campaign_id, total_written: int) -> None:
    """Ferme proprement le registre URL (appelé via ExitStack.callback)."""
    try:
        if campaign_id is not None:
            registry.update_campaign_total(campaign_id, total_written)
        registry.close()
    except Exception:
        pass


class ScrapingPipeline:
    """
    Orchestre le scraping multi-sources vers 50k phrases équilibrées FR/EN.

    Sortie horodatée — chaque session génère des fichiers uniques :
      sentences_20260918_143022.jsonl     (fusion globale)
      wikipedia_fr_20260918_143022.jsonl  (par source)
      ...

    En mode --resume, les fichiers de la session précédente sont réutilisés.
    """

    def __init__(self, output_dir: Path, user_agent: str = "GCN-Dataset/2.0",
                 contact_email: str | None = None,
                 registry_db: Path | None = None):
        self.output_dir = Path(output_dir)
        self.user_agent = user_agent
        self.contact_email = contact_email
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # registry_db=None → pas de registre (désactivé par --no-registry).
        # registry_db fourni → chemin explicite.
        # Valeur sentinelle "auto" non utilisée ici : la résolution du chemin
        # par défaut est faite dans cli.py avant d'instancier le pipeline.
        self._registry_db: Path | None = Path(registry_db) if registry_db else None

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def run(self, config: dict) -> dict:
        """
        Lance le pipeline complet.

        config keys (tous optionnels) :
          wikipedia (dict, {"enabled": bool} — kill-switch wiki),
          hal, arxiv, news, github, doc, web_search (dicts d'options ;
            présence de la clé = source active, combinée ET avec
            `enabled:` du YAML),
          langs, prog_langs, target_total, min_quality, resume, seed, budget.
        """
        scorer = QualityScorer()
        dedup = Deduplicator()
        checkpoint = ScrapingCheckpoint(self.output_dir / ".checkpoint.json")

        # Registre URL (SQLite) — None si désactivé via --no-registry.
        registry: URLRegistry | None = None
        campaign_id: int | None = None
        if self._registry_db is not None:
            registry = URLRegistry(self._registry_db)
            reg_stats = registry.stats()
            if reg_stats["urls"]:
                print(f"=== Registre URL : {reg_stats['urls']} URLs connues"
                      f" ({reg_stats['campaigns']} campagnes) ===")

        from .diversity import make_rng as _make_rng
        _, seed_eff = _make_rng(config.get("seed"))
        print(f"=== Diversité: seed={seed_eff} (ordre requêtes/URLs/offsets mélangé) ===")
        checkpoint.state["last_seed"] = seed_eff
        checkpoint.save()
        min_quality: float = config.get("min_quality", 0.3)
        resume: bool = config.get("resume", False)
        selected_langs: list[str] = config.get("langs", ["fr", "en"])
        selected_prog_langs: list[str] = config.get("prog_langs", ["python", "rust"])
        target_total: int = config.get("target_total", 50000)

        # Budget calculé sur les langues sélectionnées uniquement ;
        # bucket "code" seulement si des prog-langs sont actifs (audit-2 Fix 4).
        from .balance_tracker import _compute_budget
        _code_ratio: float = float(config.get("code_ratio", 0.10))
        budget = config.get("budget") or _compute_budget(
            selected_langs, target_total,
            include_code=bool(selected_prog_langs),
            code_ratio=_code_ratio,
            prog_langs=selected_prog_langs if selected_prog_langs else None,
        )
        tracker = BalanceTracker(budget)

        # Timestamp : réutiliser depuis le checkpoint en mode resume, sinon nouveau
        if resume and checkpoint.state.get("session_timestamp"):
            timestamp: str = checkpoint.state["session_timestamp"]
            _mode = "a"
            print(f"=== Reprise session {timestamp} ===")
            # Avertir si la config CLI diffère de la config originale du run.
            # Un changement de --langs ou --target-total peut réduire le budget et
            # déclencher is_globally_full() prématurément, sautant des sources.
            _saved_cfg = checkpoint.state.get("session_config", {})
            if _saved_cfg:
                _warnings = []
                if "langs" in _saved_cfg and sorted(selected_langs) != sorted(_saved_cfg["langs"]):
                    _warnings.append(
                        f"--langs {','.join(selected_langs)} ≠ session originale {','.join(_saved_cfg['langs'])}"
                    )
                if "target_total" in _saved_cfg and target_total != _saved_cfg["target_total"]:
                    _warnings.append(
                        f"--target-total {target_total} ≠ session originale {_saved_cfg['target_total']}"
                    )
                if _warnings:
                    import warnings as _w
                    _w.warn(
                        f"--resume avec config différente du run initial : {'; '.join(_warnings)}. "
                        "Le budget calculé peut être trop petit et sauter des sources. "
                        "Utilise les mêmes --langs et --target-total que le run initial.",
                        UserWarning, stacklevel=3,
                    )
            # Réhydrater les compteurs budgétaires (sinon dépassement target_total).
            for lang, count in checkpoint.get_budget_counts().items():
                if lang in tracker.budget:
                    tracker.counts[lang] = int(count)
            if tracker.counts:
                print(f"=== Budgets restaurés : {tracker.summary()} ===")
        else:
            # Pas de session_timestamp (run v2 ou premier run) — démarrer fresh
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _mode = "w"
            # Ne pas accumuler les compteurs d'une ancienne session incompatible
            checkpoint.reset()
            checkpoint.state["session_timestamp"] = timestamp
            checkpoint.state["session_config"] = {
                "langs": selected_langs,
                "prog_langs": selected_prog_langs,
                "target_total": target_total,
            }
            checkpoint.save()

        # URLs déjà collectées — chargées depuis le registre SQLite (cross-campagnes)
        # ou depuis le checkpoint JSON (fallback si --no-registry).
        # Construit APRÈS le reset : un run fresh ne doit rien skipper (Audit 3 C2).
        if registry is not None:
            # Registre actif : seen_urls = toutes les URLs de toutes les campagnes.
            seen_urls: set[str] = registry.get_all_urls()
            _cfg_summary = json.dumps({
                "langs": selected_langs,
                "prog_langs": selected_prog_langs,
                "target_total": target_total,
            }, ensure_ascii=False)
            campaign_id = registry.new_campaign(
                str(self.output_dir), timestamp, _cfg_summary
            )
        else:
            seen_urls = set(checkpoint.state.get("seen_urls", []))
        if seen_urls:
            print(f"=== Anti doublons: {len(seen_urls)} URLs déjà vues, skippées ===")
        print(f"=== Budget : { {k: v for k, v in tracker.budget.items()} } ===")

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

        # Kill-switch Wikipedia depuis la config runtime (CLI --no-wiki, Audit 3 M1).
        if not config.get("wikipedia", {}).get("enabled", True):
            _wiki_keys = []

        def _yaml_key(key: str) -> str:
            # Clé YAML correspondante (news → news_rss, Audit 3 F40).
            return "news_rss" if key == "news" else key

        def _yaml_enabled(key: str) -> bool:
            # `enabled:` du YAML combiné ET avec les flags CLI (Audit 3 M4).
            return bool(
                _all_cfg.get("sources", {}).get(_yaml_key(key), {}).get("enabled", True)
            )

        def _src_opt(key: str, opt: str, default):
            # Priorité : config runtime (CLI) > sources.yaml > défaut (Audit 3 M3).
            if isinstance(config.get(key), dict) and opt in config[key]:
                return config[key][opt]
            return _all_cfg.get("sources", {}).get(_yaml_key(key), {}).get(opt, default)

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
        _fname_map: dict[str, str] = {k: f for k, f in _source_keys}

        total_written = 0

        with contextlib.ExitStack() as stack:
            # Garantir la fermeture du registre même si une source lève une exception.
            if registry is not None:
                stack.callback(lambda: _close_registry(registry, campaign_id, total_written))
            out = stack.enter_context(open(global_file, _mode, encoding="utf-8"))
            _src_files: dict[str, object] = {}
            for cfg_key, fname in _source_keys:
                # NOTE (audit-2 Fix 1) : tester la présence de la clé, pas sa
                # valeur — cli.py assigne config['doc'] = {} / config['web_search'] = {}
                # et bool({}) == False désactiverait ces sources.
                # Les clés wiki sont déjà filtrées (enabled + langues) ; les
                # sources fixes exigent en plus `enabled:` du YAML (Audit 3 M4).
                if (cfg_key in config or cfg_key in _wiki_keys) and (
                    cfg_key in _wiki_keys or _yaml_enabled(cfg_key)
                ):
                    _src_files[cfg_key] = stack.enter_context(
                        open(self.output_dir / fname, _mode, encoding="utf-8")
                    )

            # Imports scraper (hors threads pour éviter les conflits d'import)
            from .sources.wikipedia import WikipediaLangScraper
            from .sources.hal_scientific import HALScraper
            from .sources.arxiv import ArXivScraper
            from .sources.news_rss import NewsRSSScraper
            from .sources.web_search import WebSearchScraper
            from .sources.github_code import GitHubCodeScraper
            from .sources.doc_scrape import DocScraper

            _hal_langs = "fr" in selected_langs or "en" in selected_langs
            write_lock = threading.Lock()

            def _run_one(key: str, scraper_fn, process_kw: dict) -> tuple:
                """Scrape en parallèle ; écriture/tracking sérialisés par write_lock."""
                if resume and checkpoint.is_done(key):
                    count = checkpoint.get_count(key)
                    print(f"=== {key}: déjà fait ({count} phrases), skip ===")
                    return key, count
                print(f"=== {key} : démarrage ===", flush=True)
                import time as _time
                _t0 = _time.time()
                try:
                    items = scraper_fn()
                except Exception as exc:
                    warnings.warn(
                        f"Source '{key}': erreur scraping ({exc})",
                        UserWarning, stacklevel=2,
                    )
                    items = []
                with write_lock:
                    if tracker.is_globally_full():
                        print(f"=== {key}: budget global atteint à l'arrivée, skip ===")
                        checkpoint.mark_done(key, 0, dict(tracker.counts))
                        return key, 0
                    n = self._process_texts(items, out, scorer, dedup, tracker, min_quality,
                                            **process_kw)
                    self._warn_zero(key, n)
                    checkpoint.mark_done(key, n, dict(tracker.counts))
                    print(f"  [{key}] → {n} phrases | {tracker.progress_bar()} "
                          f"({_time.time() - _t0:.0f}s)")
                return key, n

            # --- Construction de la liste des tâches ---
            _tasks: list = []

            # Wikipedia (toutes langues sélectionnées)
            for key in _wiki_keys:
                wiki_cfg = _all_cfg.get("sources", {}).get(key, {})
                if not wiki_cfg.get("enabled", True):
                    continue
                _wlang = wiki_cfg.get("lang", key.replace("wikipedia_", ""))
                _tasks.append((
                    key,
                    (lambda l=_wlang: WikipediaLangScraper(l, self.user_agent)
                     .scrape(tracker=tracker, seed=seed_eff)),
                    dict(default_lang=_wlang, source_out=_src_files.get(key),
                         seen_urls=seen_urls, registry=registry,
                         campaign_id=campaign_id,
                         output_file=_fname_map.get(key, "")),
                ))

            # HAL (FR + EN)
            if "hal" in config and _yaml_enabled("hal") and _hal_langs:
                _hal_max = _src_opt("hal", "max_per_query", 100)
                _tasks.append((
                    "hal",
                    (lambda mx=_hal_max: HALScraper(self.user_agent)
                     .scrape(max_per_query=mx, seed=seed_eff, tracker=tracker)),
                    dict(source_out=_src_files.get("hal"),
                         seen_urls=seen_urls, registry=registry,
                         campaign_id=campaign_id,
                         output_file=_fname_map.get("hal", "")),
                ))

            # arXiv (EN)
            if "arxiv" in config and _yaml_enabled("arxiv") and "en" in selected_langs:
                _arxiv_max = _src_opt("arxiv", "max_per_query", 100)
                _tasks.append((
                    "arxiv",
                    (lambda mx=_arxiv_max: ArXivScraper(self.user_agent)
                     .scrape(max_per_query=mx, seed=seed_eff, tracker=tracker)),
                    dict(default_lang="en", source_out=_src_files.get("arxiv"),
                         seen_urls=seen_urls, registry=registry,
                         campaign_id=campaign_id,
                         output_file=_fname_map.get("arxiv", "")),
                ))

            # News RSS
            if "news" in config and _yaml_enabled("news"):
                _news_langs = _src_opt("news", "langs", ["fr", "en"])
                _tasks.append((
                    "news",
                    (lambda ll=_news_langs: NewsRSSScraper(self.user_agent)
                     .scrape(langs=ll, seed=seed_eff, tracker=tracker)),
                    dict(source_out=_src_files.get("news"),
                         seen_urls=seen_urls, registry=registry,
                         campaign_id=campaign_id,
                         output_file=_fname_map.get("news", "")),
                ))

            # Recherche web
            if "web_search" in config and _yaml_enabled("web_search"):
                _ws_langs = list(selected_langs)
                _tasks.append((
                    "web_search",
                    (lambda ll=_ws_langs: WebSearchScraper(
                        self.user_agent, contact_email=self.contact_email)
                     .scrape(tracker=tracker, langs=ll, seed=seed_eff)),
                    dict(source_out=_src_files.get("web_search"),
                         extra_fields=("relevance_score", "found_by"),
                         seen_urls=seen_urls, registry=registry,
                         campaign_id=campaign_id,
                         output_file=_fname_map.get("web_search", "")),
                ))

            # GitHub Code
            if "github" in config and _yaml_enabled("github") and selected_prog_langs:
                _gh_token = config["github"].get("token")
                _gh_max = _src_opt("github", "max_per_query", 30)
                _gh_langs = list(selected_prog_langs)
                _tasks.append((
                    "github",
                    (lambda tok=_gh_token, mx=_gh_max, ll=_gh_langs:
                     GitHubCodeScraper(self.user_agent, github_token=tok)
                     .scrape(languages=ll, max_per_query=mx,
                             seed=seed_eff, tracker=tracker)),
                    dict(skip_split=True, default_lang="code",
                         source_out=_src_files.get("github"),
                         seen_urls=seen_urls, registry=registry,
                         campaign_id=campaign_id,
                         output_file=_fname_map.get("github", "")),
                ))

            # Documentation
            if "doc" in config and _yaml_enabled("doc") and selected_prog_langs:
                _doc_langs = list(selected_prog_langs)
                _tasks.append((
                    "doc",
                    (lambda ll=_doc_langs: DocScraper(self.user_agent)
                     .scrape(languages=ll, seed=seed_eff, tracker=tracker)),
                    dict(default_lang="code", source_out=_src_files.get("doc"),
                         seen_urls=seen_urls, registry=registry,
                         campaign_id=campaign_id,
                         output_file=_fname_map.get("doc", "")),
                ))

            # --- Exécution parallèle ---
            _max_workers = max(1, min(len(_tasks), 8))
            print(f"=== Scraping parallèle : {len(_tasks)} sources, {_max_workers} workers ===",
                  flush=True)
            with ThreadPoolExecutor(max_workers=_max_workers) as _pool:
                _futures = {
                    _pool.submit(_run_one, k, fn, kw): k
                    for k, fn, kw in _tasks
                }
                for _future in as_completed(_futures):
                    _key = _futures[_future]
                    try:
                        _, _n = _future.result()
                        total_written += _n
                    except Exception as _exc:
                        warnings.warn(
                            f"Source '{_key}' : exception non gérée ({_exc})",
                            UserWarning, stacklevel=2,
                        )

        # Fallback sans registre : persister seen_urls dans le checkpoint JSON.
        if registry is None:
            try:
                checkpoint.state["seen_urls"] = sorted(seen_urls)[-100000:]
                checkpoint.save()
            except Exception:
                pass

        print("\n=== Résumé final ===")
        print(f"  seed: {seed_eff} | URLs uniques vues (cumul): {len(seen_urls)}")
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
                UserWarning, stacklevel=2,
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
        seen_urls: set[str] | None = None,
        registry=None,
        campaign_id: int | None = None,
        output_file: str = "",
    ) -> int:
        """
        Pour chaque item : skip URL déjà vue → split → qualité → filtre → écrit en JSONL.

        Champs de sortie : text, lang, source, url, quality.
        seen_urls est muté (ajout des URLs) pour l'anti-doublon intra-run.
        registry (URLRegistry) enregistre chaque URL nouvelle en DB (cross-campagnes).
        """
        written = 0
        for item in items:
            raw_text = item.get("text", "")
            if not raw_text:
                continue
            source = item.get("source", "unknown")
            url = item.get("url", "")
            if seen_urls is not None and url and url in seen_urls:
                continue
            item_lang = item.get("lang", default_lang)

            sentences = [raw_text] if skip_split else split_sentences(raw_text)

            url_registered = False
            for sent in sentences:
                if tracker.is_globally_full():
                    return written

                lang = item_lang if item_lang != "auto" else detect_lang(sent)

                if tracker.is_full(lang):
                    continue

                quality = scorer.score(sent)
                if quality < min_quality:
                    continue

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

                # Enregistrer l'URL une seule fois par item (première phrase retenue).
                if url and not url_registered:
                    if seen_urls is not None:
                        seen_urls.add(url)
                    if registry is not None and campaign_id is not None:
                        registry.register(url, campaign_id, source, output_file)
                    url_registered = True

        return written
