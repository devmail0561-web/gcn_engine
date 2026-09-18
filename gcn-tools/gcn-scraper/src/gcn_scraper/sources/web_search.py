"""Recherche web ouverte avec croisement de sources — DuckDuckGo + OpenAlex + PubMed."""
from __future__ import annotations
import time
import requests
import trafilatura
from ._net import retry_get as _retry_get
from ..config.loader import get_source_config, get_config


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

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self._cfg = get_source_config("web_search")
        self._delay: float = float(self._cfg.get("rate_limit_delay", 1.0))
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
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    # ------------------------------------------------------------------
    # Moteurs individuels
    # ------------------------------------------------------------------

    def _search_ddg(self, query: str, max_results: int) -> list[dict]:
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
        try:
            raw = DDGS().text(query, max_results=max_results)
            return list(raw) if raw else []
        except Exception:
            return []

    def _search_openalex(self, query: str, max_results: int) -> list[dict]:
        """Recherche OpenAlex — retourne les works avec abstract_inverted_index."""
        params = {
            "search": query,
            "per_page": min(max_results, 25),
            "select": "id,doi,title,abstract_inverted_index,language",
        }
        resp, self.session = _retry_get(self.session, self._openalex_url, params)
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
        # Étape 1 : recherche → IDs
        search_params = {
            "db": "pubmed", "term": query,
            "retmax": min(max_results, 20),
            "retmode": "json",
        }
        resp, self.session = _retry_get(self.session, self._pubmed_search_url, search_params)
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
        }
        time.sleep(self._delay)
        resp2, self.session = _retry_get(self.session, self._pubmed_fetch_url, fetch_params)
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
        """Extrait le texte propre d'une URL avec trafilatura."""
        resp, self.session = _retry_get(self.session, url, {})
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
            for r in self._search_ddg(query, n):
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
            # Utiliser le texte pré-extrait (OpenAlex/PubMed) ou fetcher la page
            text = meta.get("_text")
            item_lang = meta.get("_lang", lang)
            if not text:
                text = meta.get("snippet") or self._extract_text(url)
                time.sleep(self._delay)
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

    def scrape(self, tracker=None) -> list[dict]:
        """
        Scrape via recherche web multi-sources pour tous les types de relations FR+EN.
        """
        results: list[dict] = []
        seen_urls: set[str] = set()

        for lang in ("fr", "en"):
            if tracker and tracker.is_full(lang):
                print(f"  web/{lang}: budget langue atteint, skip")
                continue
            query_list = _build_queries(lang)
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
