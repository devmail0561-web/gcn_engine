"""CLI pour gcn-scraper."""

import click
from pathlib import Path
from .pipeline import ScrapingPipeline


@click.command("gcn-scrape")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Repertoire de sortie pour les donnees brutes")
@click.option("--max-per-category", default=30, show_default=True, type=int,
              help="Nombre max d'articles par categorie Wikipedia")
@click.option("--max-papers", default=50, show_default=True, type=int,
              help="Nombre max de papiers HAL par requete")
@click.option("--max-code", default=30, show_default=True, type=int,
              help="Nombre max d'extraits de code par requete")
@click.option("--wikipedia-fr/--no-wikipedia-fr", default=True, show_default=True)
@click.option("--wikipedia-en/--no-wikipedia-en", default=True, show_default=True)
@click.option("--hal/--no-hal", default=True, show_default=True)
@click.option("--education/--no-education", default=True, show_default=True)
@click.option("--github/--no-github", default=True, show_default=True)
@click.option("--doc/--no-doc", default=True, show_default=True)
@click.option("--user-agent", default="GCN-Dataset/1.0 (research)")
def scrape_cmd(output_dir, max_per_category, max_papers, max_code,
               wikipedia_fr, wikipedia_en, hal, education, github, doc, user_agent):
    """Scrape des donnees reelles (FR/EN/Python/Rust) pour le dataset causal."""
    config = {}
    if wikipedia_fr:
        config["wikipedia_fr"] = {"max_per_category": max_per_category}
    if wikipedia_en:
        config["wikipedia_en"] = {"max_per_category": max_per_category}
    if hal:
        config["hal"] = {"queries": ["causalite", "relation causale"], "max_per_query": max_papers}
    if education:
        config["education"] = {}
    if github:
        config["github"] = {"languages": ["python", "rust"], "max_per_query": max_code}
    if doc:
        config["doc"] = {}

    pipeline = ScrapingPipeline(output_dir, user_agent)
    result = pipeline.run(config)
    click.echo(f"\nTermine: {result['unique']} phrases uniques -> {result['output']}")


if __name__ == "__main__":
    scrape_cmd()
