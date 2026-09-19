"""Scraper de code GitHub — Python/Rust/JS/TS/Java/Go/C# (commentaires et docstrings)."""
from __future__ import annotations
import time
from ._net import retry_get, DEFAULT_USER_AGENT
from ..config.loader import get_source_config


class GitHubCodeScraper:
    """Récupère du code commenté depuis GitHub Search API.

    Supporte : Python, Rust, JavaScript, TypeScript, Java, Go, C#.
    URLs et requêtes lues depuis config/sources.yaml.
    Sans token GitHub : 10 req/min. Avec token : 30 req/min.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        github_token: str | None = None,
    ):
        import requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.session.headers["Accept"] = "application/vnd.github+json"
        self.session.headers["X-GitHub-Api-Version"] = "2022-11-28"
        self._has_token = bool(github_token)
        if github_token:
            self.session.headers["Authorization"] = f"Bearer {github_token}"
        cfg = get_source_config("github")
        self._api_url = cfg.get("api_url", "https://api.github.com/search/code")
        # Sans token : 10 req/min → imposer 6s min. Avec token : 30 req/min → 2s min.
        configured = float(cfg.get("rate_limit_delay", 0.5))
        self._delay = max(configured, 2.0 if self._has_token else 6.0)
        self._queries_by_lang: dict = cfg.get("queries", {})

    def search_code(self, query: str, language: str, max_results: int = 30) -> list[dict]:
        """Recherche du code sur GitHub Search API."""
        params = {
            "q": f"{query} language:{language}",
            "per_page": min(max_results, 100),
        }
        resp, self.session = retry_get(
            self.session, self._api_url, params,
            min_interval=self._delay, base_delay=2.0,
        )
        if resp is None:
            return []
        try:
            data = resp.json()
            return data.get("items", [])
        except Exception:
            return []

    def get_file_content(self, url: str, download_url: str | None = None) -> str | None:
        """Récupère le contenu brut d'un fichier GitHub.

        L'URL contents-API (api.github.com/repos/.../contents/...) retourne
        du JSON avec contenu base64 — pas du code source (audit-2 Fix 2).
        On préfère download_url (raw.githubusercontent.com) ; en repli,
        on demande le format brut via l'en-tête Accept.
        """
        if download_url:
            resp, self.session = retry_get(
                self.session, download_url, {},
                min_interval=self._delay, base_delay=2.0,
            )
            if resp is None:
                return None
            return resp.text[:5000]
        resp, self.session = retry_get(
            self.session, url, {},
            extra_headers={"Accept": "application/vnd.github.raw"},
        )
        if resp is None:
            return None
        text = resp.text
        # Garde-fou : si l'API a ignoré l'en-tête, on reçoit du JSON base64.
        stripped = text.lstrip()
        if stripped.startswith("{"):
            try:
                import base64
                import json as _json
                payload = _json.loads(text)
                b64 = payload.get("content", "")
                if b64:
                    return base64.b64decode(b64).decode("utf-8", errors="replace")[:5000]
            except Exception:
                pass
            return None
        return text[:5000]

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
        seed: int | None = None,
    ) -> list[dict]:
        """Scrape du code commenté depuis GitHub."""
        cfg = get_source_config("github")
        if languages is None:
            languages = cfg.get("languages", ["python", "rust", "javascript", "typescript", "java", "go"])
        if max_per_query is None:
            max_per_query = cfg.get("max_per_query", 30)

        from ..diversity import shuffled as _shuffled
        results: list[dict] = []
        if not self._has_token:
            print("  GitHub: pas de token → quota 10 req/min, délais 6s imposés (export GITHUB_TOKEN recommandé).")
        if seed is not None:
            languages = _shuffled(languages, seed, "github-langs")
        consecutive_failures = 0
        for lang in languages:
            queries = self._queries_by_lang.get(lang, [])
            if not queries:
                print(f"  GitHub/{lang}: aucune requête configurée, skip")
                continue
            if seed is not None:
                queries = _shuffled(queries, seed, f"github-{lang}")
            for query in queries:
                print(f"  GitHub/{lang}: '{query[:40]}'...", end=" ", flush=True)
                items = self.search_code(query, lang, max_per_query)
                if not items:
                    consecutive_failures += 1
                    print("0 extrait (rate-limit/échec)")
                    if consecutive_failures >= 3:
                        print("  GitHub: 3 échecs consécutifs, arrêt (cooldown actif).")
                        return results
                    continue
                consecutive_failures = 0
                count = 0
                for item in items:
                    content_url = item.get("url")
                    if content_url:
                        content = self.get_file_content(
                            content_url, item.get("download_url")
                        )
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
