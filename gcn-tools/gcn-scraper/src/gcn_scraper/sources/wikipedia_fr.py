"""Scraper Wikipedia FR — dynamique par API Search + fallback catégories."""
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


# Requêtes ciblées par type de relation (FR)
RELATION_QUERIES: dict[str, list[str]] = {
    "cause": [
        "causalité épidémiologie maladie", "effets pollution environnement",
        "conséquences économiques crise", "mécanisme pathologique cause",
        "facteurs déclencheurs risque", "impact changement climatique",
    ],
    "enable": [
        "facteurs favorisant croissance", "conditions permettant développement",
        "mécanisme activation cellulaire", "catalyseur réaction chimique",
        "technologies permettant innovation",
    ],
    "prevent": [
        "prévention traitement médical", "inhibition croissance bactérienne",
        "facteurs protection immunité", "mesures préventives risque",
        "barrières protection environnement",
    ],
    "condition": [
        "conditions nécessaires application loi", "règlement juridique condition",
        "critères admissibilité procédure", "exigences réglementaires contrat",
        "dispositions légales françaises",
    ],
    "concession": [
        "bien que résultats contradictoires", "malgré progrès difficultés persistantes",
        "paradoxe économique social", "controverse scientifique débat",
        "limites méthode scientifique",
    ],
    "sequence": [
        "protocole expérimental étapes", "processus chronologique historique",
        "déroulement procédure médicale", "synthèse chimique étapes",
        "algorithme instructions séquence",
    ],
    "motivation": [
        "objectifs politique économique", "raisons décision gouvernement",
        "motivations comportement psychologie", "stratégie entreprise objectifs",
        "enjeux transition écologique",
    ],
    "opposition": [
        "opposition théories scientifiques", "débat philosophique contradictions",
        "controverse historique interprétation", "courants opposés sociologie",
    ],
    "data_dependency": [
        "apprentissage automatique données entraînement", "algorithme réseau neuronal",
        "modèle statistique paramètres", "traitement données informatique",
        "intelligence artificielle jeu de données",
    ],
    "control_dependency": [
        "architecture logicielle systèmes", "gestion processus informatique",
        "contrôle exécution programme", "système embarqué temps réel",
        "orchestration microservices",
    ],
}

# Catégories de secours (fallback phase 2)
FALLBACK_CATEGORIES: list[str] = [
    "Épidémiologie", "Médecine", "Chimie", "Informatique",
    "Économie", "Psychologie", "Algorithme", "Droit_français",
    "Sciences_sociales", "Géologie",
]

WIKI_API = "https://fr.wikipedia.org/w/api.php"


class WikipediaFRScraper:
    """Scraper Wikipedia FR — dynamique par API Search + fallback catégories."""

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def search_articles(self, query: str, max_results: int = 30) -> list[str]:
        """Recherche d'articles via l'API search Wikipedia FR."""
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
        """Récupère le texte extrait d'un article Wikipedia FR."""
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
                name = subcat.replace("Catégorie:", "").replace("Category:", "")
                titles.extend(self._category_members(name, 10, ns=0))
                if len(titles) >= max_articles:
                    break
        return titles[:max_articles]

    def _category_members(self, category: str, limit: int, ns: int) -> list[str]:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Catégorie:{category}",
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
                print(f"  FR/search [{relation}]: '{query[:40]}'...", end=" ", flush=True)
                count = 0
                for title in self.search_articles(query, max_per_query):
                    if title in seen:
                        continue
                    seen.add(title)
                    text = self.get_article_text(title)
                    if text:
                        results.append({
                            "title": title, "text": text, "lang": "fr",
                            "source": f"wikipedia_fr:{relation}",
                            "url": f"https://fr.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            "relation_hint": relation,
                        })
                        count += 1
                    time.sleep(0.3)
                print(f"{count} articles")
                time.sleep(0.5)

        # Phase 2 — fallback catégories
        print("  FR/fallback catégories...")
        for cat in FALLBACK_CATEGORIES:
            for title in self.get_category_articles(cat, max_per_category):
                if title in seen:
                    continue
                seen.add(title)
                text = self.get_article_text(title)
                if text:
                    results.append({
                        "title": title, "text": text, "lang": "fr",
                        "source": f"wikipedia_fr:cat:{cat}",
                        "url": f"https://fr.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    })
                time.sleep(0.3)
            time.sleep(0.5)

        print(f"  FR total: {len(results)} articles ({len(seen)} uniques)")
        return results

    def scrape_all(self, max_per_category: int = 50) -> list[dict]:
        """Rétrocompat — délègue à scrape()."""
        return self.scrape(max_per_query=max_per_category, max_per_category=max_per_category)
