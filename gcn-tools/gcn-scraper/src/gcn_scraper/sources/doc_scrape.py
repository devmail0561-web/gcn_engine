"""Scraper de doc Python/Rust via trafilatura (pas besoin de token GitHub)."""

import trafilatura
import requests
import time


PYTHON_URLS = [
    "https://docs.python.org/3/faq/programming.html",
    "https://docs.python.org/3/faq/design.html",
    "https://docs.python.org/3/howto/functional.html",
    "https://docs.python.org/3/library/exceptions.html",
    "https://realpython.com/python-conditional-structures/",
    "https://realpython.com/python-errors-exceptions/",
    "https://realpython.com/python-thinking-processes/",
    "https://realpython.com/python-data-structures/",
    "https://realpython.com/defining-your-own-python-function/",
    "https://realpython.com/python-inheritance/",
]

RUST_URLS = [
    "https://doc.rust-lang.org/book/ch03-03-how-functions-work.html",
    "https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html",
    "https://doc.rust-lang.org/book/ch13-01-closures.html",
    "https://doc.rust-lang.org/book/ch05-01-defining-structs.html",
    "https://doc.rust-lang.org/book/ch10-01-syntax.html",
    "https://doc.rust-lang.org/book/ch15-01-smart-pointers.html",
    "https://doc.rust-lang.org/book/ch17-01-what-is-oo.html",
    "https://rustbyexample.com/season2/itertools/why_iter",
    "https://doc.rust-lang.org/std/error/trait.Error.html",
]


class DocScraper:
    """Scrape des pages de doc Python/Rust pour en extraire des explications causales."""

    def __init__(self, user_agent: str = "GCN-Dataset/1.0 (research)"):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def scrape_urls(self, urls: list[str], source_label: str) -> list[dict]:
        """Scrape une liste d'URLs."""
        results = []
        for url in urls:
            print(f"  {source_label}: {url.split('/')[-1][:40]}...", end=" ", flush=True)
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    text = trafilatura.extract(resp.text, include_comments=False)
                    if text and len(text) > 500:
                        results.append({
                            "source": source_label,
                            "category": "documentation",
                            "title": url.split("/")[-1],
                            "text": text,
                        })
                        print(f"OK ({len(text)} chars)")
                    else:
                        print("vide")
                else:
                    print(f"HTTP {resp.status_code}")
            except Exception as e:
                print(f"erreur: {e}")
            time.sleep(0.3)
        return results

    def scrape(self) -> list[dict]:
        """Scrape toute la doc Python + Rust."""
        results = []
        results.extend(self.scrape_urls(PYTHON_URLS, "python_doc"))
        results.extend(self.scrape_urls(RUST_URLS, "rust_doc"))
        return results
