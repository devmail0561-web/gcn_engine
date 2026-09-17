"""Scraper de code Python/Rust pour relations causales (docstrings, commentaires)."""

import requests
import re
import time


class GitHubCodeScraper:
    """Recupere du code commente (Python/Rust) contenant des relations causales."""

    # Patterns de relations causales dans le code
    CAUSAL_PATTERNS = [
        r'#.*\b(because|since|therefore|thus|hence|so|if|when|enables?|causes?|prevents?)\b',
        r'""".*?\b(because|since|therefore|thus|hence|so|if|when)\b.*?"""',
        r"'''.*?\b(because|since|therefore|thus|hence|so|if|when)\b.*?'''",
        r'//.*\b(because|since|therefore|thus|hence|so|if|when|enables?|causes?|prevents?)\b',
        r'//!.*\b(because|since|therefore|thus|hence|so|if|when)\b',
        r'///.*\b(because|since|therefore|thus|hence|so|if|when)\b',
    ]

    PYTHON_QUERIES = [
        "if error raise", "if result then", "because error",
        "therefore return", "thus we can", "since the function",
        "when the value", "enables the user", "causes the error",
        "prevents the access", "if condition then",
    ]

    RUST_QUERIES = [
        "if let Some", "match result", "unwrap because",
        "expect because", "therefore we", "thus the compiler",
        "since the trait", "when the error", "enables safety",
        "causes panic", "prevents undefined",
    ]

    def __init__(self, user_agent: str = "GCN-Dataset/1.0 (research)"):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def search_code(self, query: str, language: str, max_results: int = 50) -> list[dict]:
        """Recherche du code sur GitHub."""
        url = "https://api.github.com/search/code"
        params = {
            "q": f"{query} language:{language}",
            "per_page": min(max_results, 100),
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 403:  # Rate limit
                time.sleep(60)
                return []
            data = resp.json()
            return data.get("items", [])
        except Exception:
            return []

    def get_file_content(self, url: str) -> str | None:
        """Recupere le contenu d'un fichier."""
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text[:5000]
        except Exception:
            pass
        return None

    def extract_causal_comments(self, content: str, language: str) -> list[str]:
        """Extrait les commentaires/docstrings contenant des relations causales."""
        lines = content.split('\n')
        causal_texts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Python comments
            if language == "python":
                if line.startswith('#'):
                    comment = line[1:].strip()
                    if any(re.search(p, comment, re.IGNORECASE) for p in self.CAUSAL_PATTERNS):
                        causal_texts.append(comment)
                # Docstrings
                if '"""' in line or "'''" in line:
                    if any(re.search(p, line, re.IGNORECASE) for p in self.CAUSAL_PATTERNS):
                        causal_texts.append(line)
            
            # Rust comments
            elif language == "rust":
                if line.startswith('//'):
                    comment = line[2:].strip()
                    if any(re.search(p, comment, re.IGNORECASE) for p in self.CAUSAL_PATTERNS):
                        causal_texts.append(comment)
        
        return causal_texts

    def scrape(self, languages: list[str] | None = None, max_per_query: int = 30) -> list[dict]:
        """Scrape du code commente de GitHub."""
        if languages is None:
            languages = ["python", "rust"]

        results = []
        for lang in languages:
            queries = self.PYTHON_QUERIES if lang == "python" else self.RUST_QUERIES
            for query in queries:
                print(f"  GitHub/{lang}: '{query}'...", end=" ", flush=True)
                items = self.search_code(query, lang, max_per_query)
                count = 0
                for item in items:
                    content_url = item.get("url")
                    if content_url:
                        content = self.get_file_content(content_url)
                        if content:
                            comments = self.extract_causal_comments(content, lang)
                            for comment in comments:
                                results.append({
                                    "source": f"github_{lang}",
                                    "category": "code",
                                    "title": item.get("path", ""),
                                    "text": comment,
                                })
                                count += 1
                    time.sleep(0.5)
                print(f"{count} extraits")
        return results
