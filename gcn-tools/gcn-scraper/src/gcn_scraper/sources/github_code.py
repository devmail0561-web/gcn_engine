# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
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
        self.user_agent = user_agent
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
        self._max_chars: int = int(cfg.get("max_chars_per_file", 5000))

    def search_code(self, query: str, language: str, max_results: int = 30) -> list[dict]:
        """Recherche du code sur GitHub Search API, avec pagination."""
        items: list[dict] = []
        page = 1
        remaining = max(0, max_results)
        while remaining > 0:
            per_page = min(remaining, 100)
            params = {
                "q": f"{query} language:{language}",
                "per_page": per_page,
                "page": page,
            }
            resp, self.session = retry_get(
                self.session, self._api_url, params,
                user_agent=self.user_agent,
                min_interval=self._delay, base_delay=2.0,
            )
            if resp is None:
                break
            try:
                page_items = resp.json().get("items", [])
            except Exception:
                break
            if not page_items:
                break
            items.extend(page_items)
            if len(page_items) < per_page:
                break  # dernière page
            page += 1
            remaining -= len(page_items)
        return items[:max_results]

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
                user_agent=self.user_agent,
                min_interval=self._delay, base_delay=2.0,
            )
            if resp is None:
                return None
            return resp.text[:self._max_chars]
        resp, self.session = retry_get(
            self.session, url, {},
            user_agent=self.user_agent,
            min_interval=self._delay, base_delay=2.0,
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
                    return base64.b64decode(b64).decode("utf-8", errors="replace")[:self._max_chars]
            except Exception:
                pass
            return None
        return text[:self._max_chars]

    def extract_functions(self, content: str, language: str) -> list[str]:
        """Extrait les blocs fonction/méthode réels depuis un fichier source.

        Retourne une liste de chaînes, chacune contenant le source complet d'une
        fonction (signature + corps). Taille minimale : 50 caractères.
        """
        if language == "python":
            return self._extract_python_functions(content)
        return self._extract_brace_functions(content, language)

    @staticmethod
    def _extract_python_functions(content: str) -> list[str]:
        import ast as _ast
        lines = content.splitlines()
        try:
            tree = _ast.parse(content)
        except SyntaxError:
            return []
        results = []
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if not hasattr(node, "end_lineno"):
                    continue
                src = "\n".join(lines[node.lineno - 1:node.end_lineno]).strip()
                if len(src) >= 50:
                    results.append(src)
        return results

    @staticmethod
    def _extract_brace_functions(content: str, language: str) -> list[str]:
        import re as _re
        patterns: dict[str, str] = {
            "rust":       r"\bfn\s+\w[\w<>\']*\s*(?:<[^>]*>)?\s*\([^)]*\)",
            "go":         r"\bfunc\s+(?:\([^)]*\)\s*)?\w+\s*\([^)]*\)",
            "java":       r"(?:public|private|protected|static|final|\s)+\w[\w<>\[\]]*\s+\w+\s*\([^)]*\)",
            "csharp":     r"(?:public|private|protected|static|override|virtual|\s)+\w[\w<>\[\]]*\s+\w+\s*\([^)]*\)",
            "javascript": r"(?:function\s+\w+\s*\([^)]*\)|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
            "typescript": r"(?:function\s+\w+\s*\([^)]*\)|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
        }
        pattern_str = patterns.get(language)
        if pattern_str is None:
            return []
        results = []
        for match in _re.finditer(pattern_str, content):
            brace_start = content.find("{", match.end())
            if brace_start == -1:
                continue
            depth = 0
            i = brace_start
            while i < len(content):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        block = content[match.start():i + 1].strip()
                        if len(block) >= 50:
                            results.append(block)
                        break
                i += 1
        return results

    def scrape(
        self,
        languages: list[str] | None = None,
        max_per_query: int | None = None,
        seed: int | None = None,
        tracker=None,
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
        for lang in languages:
            if tracker is not None and tracker.is_globally_full():
                break
            queries = self._queries_by_lang.get(lang, [])
            if not queries:
                print(f"  GitHub/{lang}: aucune requête configurée, skip")
                continue
            if seed is not None:
                queries = _shuffled(queries, seed, f"github-{lang}")
            consecutive_failures = 0  # par langage : un langage en panne n'annule pas les autres
            for query in queries:
                if tracker is not None and tracker.is_globally_full():
                    break
                print(f"  GitHub/{lang}: '{query[:40]}'...", end=" ", flush=True)
                items = self.search_code(query, lang, max_per_query)
                if not items:
                    consecutive_failures += 1
                    print("0 extrait (rate-limit/échec)")
                    if consecutive_failures >= 3:
                        print(f"  GitHub/{lang}: 3 échecs consécutifs, langage suivant (cooldown actif).")
                        break
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
                            # NOTE : les blocs de code ont un ratio chars/mots
                            # élevé — QualityScorer peut les rejeter avec le
                            # seuil par défaut (0.3). Utiliser --min-quality 0.0
                            # lors d'un run code-only, ou adapter le scorer.
                            functions = self.extract_functions(content, lang)
                            for func in functions:
                                results.append({
                                    "text": func,
                                    "lang": "code",
                                    "source": f"github_{lang}",
                                    "url": item.get("html_url", content_url),
                                })
                                count += 1
                    time.sleep(self._delay)
                print(f"{count} extraits")
                time.sleep(self._delay)
        return results
