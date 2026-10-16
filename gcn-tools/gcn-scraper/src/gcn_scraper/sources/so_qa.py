# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Source StackOverflow Q&A — C1-doc (phase 0, T4).

Questions votées contenant des marqueurs causaux + meilleure réponse :
prose technique causale (« because X, use Y »). 100 % API officielle
(TOS : attribution portée par chaque notice — `url` + `auteur` + `licence`
CC BY-SA fournie par l'API). Sans clé : 300 req/j — suffisant en vagues.
"""
from __future__ import annotations

import html
import re
import time

from ._net import DEFAULT_USER_AGENT

API = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"

#: Tags par langage de programmation (clé interne → tag SO).
TAGS = {"python": "python", "rust": "rust", "javascript": "javascript",
        "typescript": "typescript", "java": "java", "go": "go",
        "csharp": "c#"}

#: Marqueurs causaux cherchés dans titre/corps (prose, pas code).
QUERIES = ["because", "therefore", "in order to", "to prevent", "unless",
           "instead of", "workaround", "race condition", "as a result",
           "only if", "even if", "leads to", "caused by", "to avoid",
           "breaking change", "edge case"]

LICENCE_DEFAUT = "CC-BY-SA-4.0"

_CODE_RE = re.compile(r"<pre.*?</pre>|<code.*?</code>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def nettoie(html_text: str) -> str:
    """HTML SO → texte : blocs code opacifiés, tags retirés, entités décodées."""
    t = _CODE_RE.sub(" [CODE] ", html_text or "")
    t = _TAG_RE.sub(" ", t)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


class StackOverflowScraper:
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        import requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._delay = 1.0

    def _get(self, path: str, params: dict, retries: int = 3) -> dict | None:
        for i in range(retries):
            try:
                r = self.session.get(f"{API}{path}", params=params, timeout=30)
                if r.status_code == 400 and "backoff" in r.text:
                    time.sleep(int(r.json().get("backoff", 10)) + 1)
                    continue
                r.raise_for_status()
                data = r.json()
                if "backoff" in data:
                    time.sleep(int(data["backoff"]) + 1)
                return data
            except Exception:
                time.sleep(2.0 * (i + 1))
        return None

    def search(self, tag: str, query: str, pagesize: int = 50,
               pages: int = 1) -> list[dict]:
        out = []
        for page in range(1, pages + 1):
            d = self._get("/search/advanced", {
                "order": "desc", "sort": "votes", "q": query, "tagged": tag,
                "site": SITE, "pagesize": pagesize, "page": page,
                "filter": "withbody"})
            if not d:
                break
            out.extend(d.get("items", []))
            time.sleep(self._delay)
            if not d.get("has_more"):
                break
        return out

    def top_answers(self, qids: list[int]) -> dict[int, dict]:
        res = {}
        for i in range(0, len(qids), 100):
            chunk = ";".join(str(q) for q in qids[i:i + 100])
            d = self._get(f"/questions/{chunk}/answers", {
                "order": "desc", "sort": "votes", "site": SITE,
                "pagesize": 100, "filter": "withbody"})
            if not d:
                continue
            for a in d.get("items", []):
                qid = a.get("question_id")
                if qid not in res:
                    res[qid] = a
            time.sleep(self._delay)
        return res

    def scrape(self, languages: list[str] | None = None,
               max_per_query: int = 50, seed: int | None = None,
               tracker=None) -> list[dict]:
        from ..diversity import shuffled as _shuffled
        langs = languages or list(TAGS)
        if seed is not None:
            langs = _shuffled(langs, seed, "so-langs")
            queries = _shuffled(list(QUERIES), seed, "so-queries")
        else:
            queries = list(QUERIES)
        results = []
        for lang in langs:
            if tracker is not None and tracker.is_globally_full():
                break
            tag = TAGS.get(lang)
            if not tag:
                print(f"  SO/{lang}: pas de tag configuré, skip")
                continue
            for query in queries:
                if tracker is not None and tracker.is_globally_full():
                    break
                print(f"  SO/{tag}: '{query}'...", end=" ", flush=True)
                qs = self.search(tag, query, pagesize=min(max_per_query, 100))
                if not qs:
                    print("0 question (quota/échec)")
                    continue
                ans = self.top_answers([q["question_id"] for q in qs])
                n = 0
                for q in qs:
                    a = ans.get(q["question_id"])
                    if not a:
                        continue
                    body = nettoie(q.get("body", ""))
                    rep = nettoie(a.get("body", ""))
                    text = f"{q.get('title', '')}\n{body}\nRéponse : {rep}"
                    if len(text) < 200:
                        continue
                    results.append({
                        "text": text[:5000], "lang": "en",
                        "source": "stackoverflow",
                        "url": q.get("link", ""),
                        "licence": q.get("content_license", LICENCE_DEFAUT),
                        "auteur": (q.get("owner") or {}).get("display_name", "?"),
                    })
                    n += 1
                print(f"{n} notices")
        return results
