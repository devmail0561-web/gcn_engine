"""Pipeline de scraping complet : FR/EN/Python/Rust."""

from pathlib import Path
from .sources.wikipedia_fr import WikipediaFRScraper
from .sources.wikipedia_en import WikipediaENScraper
from .sources.hal_scientific import HALScraper
from .sources.education_web import EducationScraper
from .sources.github_code import GitHubCodeScraper
from .sources.doc_scrape import DocScraper
from .splitter import split_sentences


class ScrapingPipeline:
    """Orchestre le scraping multi-sources."""

    def __init__(self, output_dir: Path, user_agent: str = "GCN-Dataset/1.0"):
        self.output_dir = output_dir
        self.user_agent = user_agent

    def run(self, config: dict) -> dict:
        """Lance le pipeline complet."""
        all_sentences = []
        stats = {}

        # Wikipedia FR
        if config.get("wikipedia_fr"):
            print("=== Wikipedia FR ===")
            scraper = WikipediaFRScraper(self.user_agent)
            articles = scraper.scrape_all(config["wikipedia_fr"].get("max_per_category", 30))
            for a in articles:
                all_sentences.extend(split_sentences(a["text"]))
            stats["wikipedia_fr"] = len(articles)
            print(f"  Total: {len(articles)} articles")

        # Wikipedia EN
        if config.get("wikipedia_en"):
            print("=== Wikipedia EN ===")
            scraper = WikipediaENScraper(self.user_agent)
            articles = scraper.scrape(max_per_category=config["wikipedia_en"].get("max_per_category", 30))
            for a in articles:
                all_sentences.extend(split_sentences(a["text"]))
            stats["wikipedia_en"] = len(articles)
            print(f"  Total: {len(articles)} articles")

        # HAL Scientifique
        if config.get("hal"):
            print("=== HAL Scientifique ===")
            scraper = HALScraper(self.user_agent)
            queries = config["hal"].get("queries", ["causalite"])
            papers = scraper.scrape(queries, config["hal"].get("max_per_query", 50))
            for p in papers:
                all_sentences.extend(split_sentences(p["text"]))
            stats["hal"] = len(papers)
            print(f"  Total: {len(papers)} papiers")

        # Education
        if "education" in config:
            print("=== Education ===")
            scraper = EducationScraper(self.user_agent)
            pages = scraper.scrape_all()
            for p in pages:
                all_sentences.extend(split_sentences(p["text"]))
            stats["education"] = len(pages)
            print(f"  Total: {len(pages)} pages")

        # GitHub Code (Python/Rust)
        if "github" in config:
            print("=== GitHub Code (Python/Rust) ===")
            scraper = GitHubCodeScraper(self.user_agent)
            languages = config["github"].get("languages", ["python", "rust"])
            items = scraper.scrape(languages=languages, max_per_query=config["github"].get("max_per_query", 30))
            for item in items:
                all_sentences.append(item["text"])
            stats["github"] = len(items)
            print(f"  Total: {len(items)} extraits de code")

        # Doc Python/Rust (trafilatura)
        if "doc" in config:
            print("=== Documentation Python/Rust ===")
            scraper = DocScraper(self.user_agent)
            docs = scraper.scrape()
            for d in docs:
                all_sentences.extend(split_sentences(d["text"]))
            stats["doc"] = len(docs)
            print(f"  Total: {len(docs)} pages de doc")

        # Dedup
        unique = list(dict.fromkeys(all_sentences))
        print(f"\n=== Resultat ===")
        print(f"Sources: {stats}")
        print(f"Phrases totales: {len(all_sentences)}")
        print(f"Phrases uniques: {len(unique)}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "sentences_raw.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for s in unique:
                f.write(s + "\n")
        print(f"Fichier: {output_file}")

        return {"total": len(all_sentences), "unique": len(unique), "output": output_file, "stats": stats}
