"""Scraper Wikipedia EN — dynamique par API Search + fallback catégories."""
from __future__ import annotations
import time
import requests


def _retry_get(session, url, params, max_retries=3, base_delay=1.0):
    """GET avec retry exponentiel sur erreur réseau ou rate-limit (429)."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(base_delay * (2 ** attempt) + 1.0)
                continue
            if resp.status_code == 200:
                return resp
            time.sleep(base_delay * (attempt + 1))
        except requests.RequestException:
            time.sleep(base_delay * (2 ** attempt))
    return None


# Requêtes ciblées par type de relation (EN)
RELATION_QUERIES: dict[str, list[str]] = {
    "cause": [
        "disease causation epidemiology", "climate effects consequences",
        "economic crisis causes", "pathological mechanism disease",
        "risk factors trigger", "environmental impact pollution",
    ],
    "enable": [
        "enabling factors growth biology", "conditions allowing development",
        "catalytic mechanism chemistry", "technology enabling innovation",
        "facilitating factors social change",
    ],
    "prevent": [
        "prevention treatment medical", "inhibition bacterial growth",
        "protective factors immunity", "preventive measures public health",
        "safety mechanisms engineering",
    ],
    "condition": [
        "necessary conditions logic mathematics", "legal requirements contract",
        "conditional probability statistics", "eligibility criteria law",
        "prerequisite formal methods",
    ],
    "concession": [
        "although results show contradiction", "despite progress limitations remain",
        "scientific controversy debate", "paradox counterintuitive result",
        "limitations methodology research",
    ],
    "sequence": [
        "step by step experimental protocol", "chronological historical process",
        "sequential algorithm computation", "procedural steps synthesis",
        "pipeline stages workflow",
    ],
    "motivation": [
        "strategic objectives policy decision", "behavioral motivation psychology",
        "economic incentives theory", "goals driven approach research",
        "rationale behind decision",
    ],
    "opposition": [
        "opposing theories scientific debate", "contradictory findings research",
        "controversy historical interpretation", "conflicting evidence results",
    ],
    "data_dependency": [
        "machine learning training dataset", "neural network data driven",
        "statistical model parameters", "deep learning data preprocessing",
        "dataset benchmark evaluation",
    ],
    "control_dependency": [
        "software architecture control flow", "operating system process management",
        "distributed computing orchestration", "program execution control",
        "microservices coordination",
    ],
}

# Catégories de secours (fallback phase 2)
FALLBACK_CATEGORIES: list[str] = [
    "Epidemiology", "Medicine", "Chemistry", "Computer_science",
    "Economics", "Psychology", "Algorithms", "Law",
    "Ecology", "Physics",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"


class WikipediaENScraper:
    """Scraper Wikipedia EN — dynamique par API Search + fallback catégories."""

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def search_articles(self, query: str, max_results: int = 30) -> list[str]:
        """Recherche d'articles via l'API search Wikipedia EN."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": min(max_results, 50),
            "srnamespace": 0,
            "format": "json",
        }
        resp = _retry_get(self.session, WIKI_API, params)
        if resp is None:
            return []
        try:
            return [h["title"] for h in resp.json().get("query", {}).get("search", [])]
        except Exception:
            return []

    def get_article_text(self, title: str, max_chars: int = 5000) -> str | None:
        """Récupère le texte extrait d'un article Wikipedia EN."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "format": "json",
        }
        resp = _retry_get(self.session, WIKI_API, params)
        if resp is None:
            return None
        try:
            for page in resp.json().get("query", {}).get("pages", {}).values():
                text = page.get("extract", "")
                if text and len(text) > 100:
                    return text[:max_chars]
        except Exception:
            pass
        return None

    def get_category_articles(self, category: str, max_articles: int = 30) -> list[str]:
        """Articles d'une catégorie ; descend dans les sous-catégories (niveau 1) si vide."""
        titles = self._category_members(category, max_articles, ns=0)
        if not titles:
            subcats = self._category_members(category, 20, ns=14)
            for subcat in subcats[:5]:
                name = subcat.replace("Category:", "")
                titles.extend(self._category_members(name, 10, ns=0))
                if len(titles) >= max_articles:
                    break
        return titles[:max_articles]

    def _category_members(self, category: str, limit: int, ns: int) -> list[str]:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(limit, 500),
            "cmnamespace": ns,
            "format": "json",
        }
        resp = _retry_get(self.session, WIKI_API, params)
        if resp is None:
            return []
        try:
            return [
                m["title"]
                for m in resp.json().get("query", {}).get("categorymembers", [])
                if m.get("ns") == ns
            ]
        except Exception:
            return []

    def scrape(self, max_per_query: int = 20, max_per_category: int = 20) -> list[dict]:
        """Scrape dynamique : API Search par relation type + fallback catégories."""
        seen: set[str] = set()
        results: list[dict] = []

        # Phase 1 — recherche par type de relation
        for relation, queries in RELATION_QUERIES.items():
            for query in queries:
                print(f"  EN/search [{relation}]: '{query[:40]}'...", end=" ", flush=True)
                count = 0
                for title in self.search_articles(query, max_per_query):
                    if title in seen:
                        continue
                    seen.add(title)
                    text = self.get_article_text(title)
                    if text:
                        results.append({
                            "title": title, "text": text, "lang": "en",
                            "source": f"wikipedia_en:{relation}",
                            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            "relation_hint": relation,
                        })
                        count += 1
                    time.sleep(0.3)
                print(f"{count} articles")
                time.sleep(0.5)

        # Phase 2 — fallback catégories
        print("  EN/fallback catégories...")
        for cat in FALLBACK_CATEGORIES:
            for title in self.get_category_articles(cat, max_per_category):
                if title in seen:
                    continue
                seen.add(title)
                text = self.get_article_text(title)
                if text:
                    results.append({
                        "title": title, "text": text, "lang": "en",
                        "source": f"wikipedia_en:cat:{cat}",
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    })
                time.sleep(0.3)
            time.sleep(0.5)

        print(f"  EN total: {len(results)} articles ({len(seen)} uniques)")
        return results
