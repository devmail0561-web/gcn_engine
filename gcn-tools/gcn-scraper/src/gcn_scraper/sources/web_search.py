# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Recherche web ouverte avec croisement de sources — DuckDuckGo + OpenAlex + PubMed."""
from __future__ import annotations
import time
import requests
import trafilatura
from ._net import retry_get as _retry_get, DEFAULT_USER_AGENT
from ..config.loader import get_source_config, get_config
from ..filters.lang_detector import detect_lang


# Régions DuckDuckGo par langue (sinon les requêtes FR retournent de l'EN).
_DDG_REGIONS = {
    "fr": "fr-fr", "en": "en-us", "de": "de-de",
    "es": "es-es", "it": "it-it", "pt": "pt-pt",
}


def _build_queries(lang: str) -> list[str]:
    """Requêtes de recherche depuis config/sources.yaml — indépendant de la taxonomie causale."""
    return get_config().get("search_queries", {}).get(lang, [])


class WebSearchScraper:
    """
    Scraper multi-moteurs avec croisement de sources.

    Sources croisées (sans clé API) :
      - DuckDuckGo (duckduckgo-search) — web général
      - OpenAlex — académique FR+EN gratuit
      - PubMed E-utilities — médecine/biologie gratuit

    Les URLs trouvées par plusieurs sources sont prioritaires.
    Le texte est extrait avec trafilatura.
    """

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, contact_email: str | None = None):
        self._cfg = get_source_config("web_search")
        self._delay: float = float(self._cfg.get("rate_limit_delay", 2.0))
        self._max_chars: int = int(self._cfg.get("max_chars_per_page", 5000))
        self._max_results: int = int(self._cfg.get("max_results_per_query", 15))
        self._openalex_url: str = self._cfg.get("openalex_url", "https://api.openalex.org/works")
        self._pubmed_search_url: str = self._cfg.get(
            "pubmed_search_url",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        )
        self._pubmed_fetch_url: str = self._cfg.get(
            "pubmed_fetch_url",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        )
        self._engines: list[str] = self._cfg.get("engines", ["duckduckgo", "openalex"])
        self._contact_email: str = contact_email or self._cfg.get("contact_email", "gcn-research@example.org")
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    # ------------------------------------------------------------------
    # Moteurs individuels
    # ------------------------------------------------------------------

    def _search_ddg(self, query: str, max_results: int, lang: str = "auto") -> list[dict]:
        """Recherche DuckDuckGo — tente ddgs (nouveau nom) puis duckduckgo_search (ancien)."""
        DDGS = None
        for mod in ("ddgs", "duckduckgo_search"):
            try:
                import importlib
                m = importlib.import_module(mod)
                DDGS = getattr(m, "DDGS", None)
                if DDGS:
                    break
            except ImportError:
                continue
        if DDGS is None:
            return []
        # Région = langue de la requête, sinon DDG retourne de l'EN générique.
        region = _DDG_REGIONS.get(lang)
        try:
            time.sleep(self._delay)
            ddg = DDGS()
            try:
                raw = ddg.text(query, max_results=max_results, region=region) if region else ddg.text(query, max_results=max_results)
            except TypeError:
                raw = ddg.text(query, max_results=max_results)  # ancienne API sans region
            return list(raw) if raw else []
        except Exception as e:
            # DDG rate-limite agressivement (202/429/Ratelimit) → backoff, pas de boucle.
            msg = str(e).lower()
            if "ratelimit" in msg or "429" in msg or "202" in msg:
                print(f"    DDG rate-limit, pause {self._delay * 2:.0f}s...")
                time.sleep(self._delay * 2)
            return []

    def _search_openalex(self, query: str, max_results: int) -> list[dict]:
        """Recherche OpenAlex — retourne les works avec abstract_inverted_index."""
        params = {
            "search": query,
            "per_page": min(max_results, 25),
            "select": "id,doi,title,abstract_inverted_index,language",
            # Polite pool : sans mailto, OpenAlex throttle agressivement.
            "mailto": self._contact_email,
        }
        resp, self.session = _retry_get(
            self.session, self._openalex_url, params,
            user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp is None:
            return []
        try:
            results = []
            for work in resp.json().get("results", []):
                doi = work.get("doi", "")
                url = doi if doi else work.get("id", "")
                abstract = self._decode_inverted_index(work.get("abstract_inverted_index"))
                if abstract and url:
                    results.append({
                        "url": url,
                        "title": work.get("title", ""),
                        "text": abstract,
                        "lang": work.get("language", "en"),
                    })
            return results
        except Exception:
            return []

    def _decode_inverted_index(self, inv_idx: dict | None) -> str | None:
        """Reconstitue l'abstract depuis l'index inversé OpenAlex."""
        if not inv_idx:
            return None
        try:
            words: list[tuple[int, str]] = []
            for word, positions in inv_idx.items():
                for pos in positions:
                    words.append((pos, word))
            words.sort()
            return " ".join(w for _, w in words)
        except Exception:
            return None

    def _search_pubmed(self, query: str, max_results: int) -> list[dict]:
        """Recherche PubMed + fetch des abstracts."""
        # Étape 1 : recherche → IDs (tool+email requis, sinon throttle à 3 req/s)
        search_params = {
            "db": "pubmed", "term": query,
            "retmax": min(max_results, 20),
            "retmode": "json",
            "tool": "gcn-scraper",
            "email": self._contact_email,
        }
        resp, self.session = _retry_get(
            self.session, self._pubmed_search_url, search_params,
            user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp is None:
            return []
        try:
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception:
            return []
        if not ids:
            return []

        # Étape 2 : fetch abstracts
        fetch_params = {
            "db": "pubmed", "id": ",".join(ids),
            "rettype": "abstract", "retmode": "text",
            "tool": "gcn-scraper",
            "email": self._contact_email,
        }
        time.sleep(self._delay)
        resp2, self.session = _retry_get(
            self.session, self._pubmed_fetch_url, fetch_params,
            user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp2 is None:
            return []

        # Découpe par article (séparateur "PMID:")
        results = []
        for chunk in resp2.text.split("PMID:")[1:]:
            lines = chunk.strip().split("\n")
            text = " ".join(l.strip() for l in lines if l.strip())[:self._max_chars]
            if len(text) > 100:
                pmid = lines[0].strip().split()[0] if lines else ""
                results.append({
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "title": "",
                    "text": text,
                    "lang": "en",
                })
        return results

    # ------------------------------------------------------------------
    # Extraction de texte
    # ------------------------------------------------------------------

    def _extract_text(self, url: str) -> str | None:
        """Scrape la page cible et extrait le texte propre avec trafilatura."""
        resp, self.session = _retry_get(
            self.session, url, {},
            user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp is None:
            return None
        try:
            text = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            return text[:self._max_chars] if text else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Croisement de sources
    # ------------------------------------------------------------------

    def search_multi(
        self, query: str, max_results: int | None = None, lang: str = "auto"
    ) -> list[dict]:
        """
        Cherche sur plusieurs moteurs, croise les résultats par URL.

        Score de pertinence = nb de sources qui ont trouvé cette URL/texte.
        Sources académiques (openalex, pubmed) ont un poids double.
        Retourne les résultats triés par score décroissant.
        """
        n = max_results or self._max_results
        url_scores: dict[str, dict] = {}

        if "duckduckgo" in self._engines:
            for r in self._search_ddg(query, n, lang=lang):
                url = r.get("href", "")
                if not url:
                    continue
                entry = url_scores.setdefault(url, {"score": 0, "sources": [], "title": "", "snippet": ""})
                entry["score"] += 1
                entry["sources"].append("duckduckgo")
                entry["title"] = entry["title"] or r.get("title", "")
                entry["snippet"] = entry["snippet"] or r.get("body", "")

        if "openalex" in self._engines:
            for r in self._search_openalex(query, n):
                url = r.get("url", "")
                if not url:
                    continue
                entry = url_scores.setdefault(url, {"score": 0, "sources": [], "title": "", "snippet": ""})
                entry["score"] += 2  # source académique — poids double
                entry["sources"].append("openalex")
                entry["title"] = entry["title"] or r.get("title", "")
                # Pour OpenAlex, le texte est déjà dans r["text"] (abstract reconstruit)
                if r.get("text"):
                    entry["_text"] = r["text"]
                    entry["_lang"] = r.get("lang", lang)
            time.sleep(self._delay)

        if "pubmed" in self._engines:
            for r in self._search_pubmed(query, n):
                url = r.get("url", "")
                if not url:
                    continue
                entry = url_scores.setdefault(url, {"score": 0, "sources": [], "title": "", "snippet": ""})
                entry["score"] += 2
                entry["sources"].append("pubmed")
                if r.get("text"):
                    entry["_text"] = r["text"]
                    entry["_lang"] = "en"
            time.sleep(self._delay)

        # Trier par score décroissant
        ranked = sorted(url_scores.items(), key=lambda x: x[1]["score"], reverse=True)

        results = []
        for url, meta in ranked[:n]:
            # Principe : scraper le SITE lié, pas la sortie du moteur.
            # On fetch la page cible en priorité ; snippet/abstract en fallback.
            # La langue est détectée sur le texte réel (pas hallucinée depuis
            # la langue de requête pour les hits DDG purs — Audit 3 M11).
            text = self._extract_text(url)
            time.sleep(self._delay)
            if text:
                item_lang = detect_lang(text, default=meta.get("_lang", lang))
            else:
                text = meta.get("_text") or meta.get("snippet")
                item_lang = meta.get("_lang", lang)
            if text and len(text) > 100:
                results.append({
                    "url": url,
                    "title": meta.get("title", ""),
                    "text": text[:self._max_chars],
                    "lang": item_lang,
                    "relevance_score": meta["score"],
                    "found_by": meta["sources"],
                })
        return results

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def scrape(
        self,
        tracker=None,
        langs: list[str] | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """
        Scrape via recherche web multi-sources.

        langs=None → ("fr", "en") historique. Le pipeline passe les langues
        sélectionnées (audit-2 Fix 6 : es/de/it/pt n'étaient jamais cherchés).
        """
        results: list[dict] = []
        seen_urls: set[str] = set()

        for lang in langs or ("fr", "en"):
            if tracker and tracker.is_full(lang):
                print(f"  web/{lang}: budget langue atteint, skip")
                continue
            query_list = _build_queries(lang)
            if seed is not None:
                from ..diversity import shuffled as _shuffled
                query_list = _shuffled(query_list, seed, f"web-{lang}")
            for query in query_list:
                if tracker and tracker.is_full(lang):
                    break
                print(f"  web/{lang}: '{query[:50]}'...", end=" ", flush=True)
                hits = self.search_multi(query, lang=lang)
                count = 0
                for hit in hits:
                    url = hit["url"]
                    if url in seen_urls:
                        continue
                    if tracker and tracker.is_full(lang):
                        break
                    seen_urls.add(url)
                    results.append({
                        "title": hit.get("title", ""),
                        "text": hit["text"],
                        "lang": hit.get("lang", lang),
                        "source": "web_search",
                        "url": url,
                        "relevance_score": hit.get("relevance_score", 1),
                        "found_by": hit.get("found_by", []),
                    })
                    count += 1
                print(f"{count} pages")
                time.sleep(self._delay)

        print(f"  web total: {len(results)} pages")
        return results
