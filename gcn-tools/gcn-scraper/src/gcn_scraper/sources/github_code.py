"""Scraper de code GitHub — Python/Rust/JS/TS/Java/Go/C# (commentaires et docstrings)."""
from __future__ import annotations
import time
from ._net import retry_get
from ..config.loader import get_source_config


class GitHubCodeScraper:
    """Récupère du code commenté depuis GitHub Search API.

    Supporte : Python, Rust, JavaScript, TypeScript, Java, Go, C#.
    URLs et requêtes lues depuis config/sources.yaml.
    Sans token GitHub : 10 req/min. Avec token : 30 req/min.
    """

    def __init__(
        self,
        user_agent: str = "GCN-Dataset/2.0 (research)",
        github_token: str | None = None,
    ):
        import requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        if github_token:
            self.session.headers["Authorization"] = f"token {github_token}"
        cfg = get_source_config("github")
        self._api_url = cfg.get("api_url", "https://api.github.com/search/code")
        self._delay = cfg.get("rate_limit_delay", 0.5)
        self._queries_by_lang: dict = cfg.get("queries", {})

    def search_code(self, query: str, language: str, max_results: int = 30) -> list[dict]:
        """Recherche du code sur GitHub Search API."""
        params = {
            "q": f"{query} language:{language}",
            "per_page": min(max_results, 100),
        }
        resp, self.session = retry_get(self.session, self._api_url, params)
        if resp is None:
            return []
        try:
            data = resp.json()
            return data.get("items", [])
        except Exception:
            return []

    def get_file_content(self, url: str) -> str | None:
        """Récupère le contenu brut d'un fichier GitHub."""
        resp, self.session = retry_get(self.session, url, {})
        if resp is None:
            return None
        return resp.text[:5000]

    def extract_comments(self, content: str, language: str) -> list[str]:
        """Extrait les commentaires/docstrings d'un fichier source."""
        lines = content.split('\n')
        results = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if language == "python":
                if line.startswith('#'):
                    c = line[1:].strip()
                    if len(c) > 20:
                        results.append(c)
                elif '"""' in line or "'''" in line:
                    c = line.strip('"\' ')
                    if len(c) > 20:
                        results.append(c)
            elif language == "rust":
                if line.startswith('//'):
                    c = line.lstrip('/').strip()
                    if len(c) > 20:
                        results.append(c)
            elif language in ("javascript", "typescript", "java", "go", "csharp"):
                if line.startswith('//'):
                    c = line[2:].strip()
                    if len(c) > 20:
                        results.append(c)
                elif line.startswith('*') and not line.startswith('*/'):
                    c = line.lstrip('*').strip()
                    if len(c) > 20:
                        results.append(c)
        return results

    def scrape(
        self,
        languages: list[str] | None = None,
        max_per_query: int | None = None,
    ) -> list[dict]:
        """Scrape du code commenté depuis GitHub."""
        cfg = get_source_config("github")
        if languages is None:
            languages = cfg.get("languages", ["python", "rust", "javascript", "typescript", "java", "go"])
        if max_per_query is None:
            max_per_query = cfg.get("max_per_query", 30)

        results: list[dict] = []
        for lang in languages:
            queries = self._queries_by_lang.get(lang, [])
            if not queries:
                print(f"  GitHub/{lang}: aucune requête configurée, skip")
                continue
            for query in queries:
                print(f"  GitHub/{lang}: '{query[:40]}'...", end=" ", flush=True)
                items = self.search_code(query, lang, max_per_query)
                count = 0
                for item in items:
                    content_url = item.get("url")
                    if content_url:
                        content = self.get_file_content(content_url)
                        if content:
                            comments = self.extract_comments(content, lang)
                            for comment in comments:
                                results.append({
                                    "text": comment,
                                    "lang": "code",
                                    "source": f"github_{lang}",
                                    "url": item.get("html_url", content_url),
                                })
                                count += 1
                    time.sleep(self._delay)
                print(f"{count} extraits")
                time.sleep(self._delay)
        return results
