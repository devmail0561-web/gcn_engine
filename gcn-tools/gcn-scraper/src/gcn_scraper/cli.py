"""CLI gcn-scrape v3 — scraping avancé bilingue FR+EN, horodaté, multi-sources."""
from __future__ import annotations
import click
from pathlib import Path
from .pipeline import ScrapingPipeline


@click.command("gcn-scrape")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire de sortie (sentences.jsonl)")
@click.option("--target-total", default=50000, show_default=True, type=int,
              help="Nombre de phrases cibles (arrêt dès que le budget global est atteint)")
@click.option("--resume", is_flag=True, default=False,
              help="Reprend depuis le checkpoint existant (skip les sources déjà faites)")
@click.option("--langs", default="fr,en", show_default=True,
              help="Langues à scraper (comma-separated : fr,en)")
@click.option("--min-quality", default=0.3, show_default=True, type=float,
              help="Score de qualité minimum [0-1] pour conserver une phrase")
@click.option("--max-per-category", default=50, show_default=True, type=int,
              help="Nb max d'articles par catégorie Wikipedia")
@click.option("--max-per-query", default=100, show_default=True, type=int,
              help="Nb max de résultats par requête HAL/arXiv")
@click.option("--wikipedia-fr/--no-wikipedia-fr", default=True, show_default=True)
@click.option("--wikipedia-en/--no-wikipedia-en", default=True, show_default=True)
@click.option("--hal/--no-hal", default=True, show_default=True)
@click.option("--arxiv/--no-arxiv", default=True, show_default=True)
@click.option("--news/--no-news", default=True, show_default=True)
@click.option("--github/--no-github", default=False, show_default=True,
              help="Scraping GitHub (nécessite un token pour dépasser le rate limit public)")
@click.option("--github-token", default=None, envvar="GITHUB_TOKEN",
              help="Token GitHub (ou variable GITHUB_TOKEN)")
@click.option("--doc/--no-doc", default=True, show_default=True)
@click.option("--web-search/--no-web-search", default=False, show_default=True,
              help="Recherche web multi-sources (DuckDuckGo + OpenAlex + PubMed)")
@click.option("--user-agent", default="GCN-Dataset/2.0 (research)", show_default=True)
def scrape_cmd(
    output_dir, target_total, resume, langs, min_quality,
    max_per_category, max_per_query,
    wikipedia_fr, wikipedia_en, hal, arxiv, news, github, github_token, doc,
    web_search, user_agent,
):
    """
    Scrape des données bilingues FR/EN pour le dataset causal GCN.

    Produit sentences.jsonl dans --output-dir, avec balance automatique
    sur les 11 types de relations du moteur GCN.

    Exemples :
      gcn-scrape --output-dir ./raw --target-total 50000
      gcn-scrape --output-dir ./raw --resume --no-arxiv
      gcn-scrape --output-dir ./raw --langs fr --no-wikipedia-en --no-arxiv
    """
    lang_list = [l.strip() for l in langs.split(",") if l.strip()]

    config: dict = {
        "resume": resume,
        "min_quality": min_quality,
    }
    if wikipedia_fr:
        config["wikipedia_fr"] = {"max_per_category": max_per_category}
    if wikipedia_en:
        config["wikipedia_en"] = {"max_per_category": max_per_category}
    if hal:
        config["hal"] = {"max_per_query": max_per_query}
    if arxiv:
        config["arxiv"] = {"max_per_query": max_per_query}
    if news:
        config["news"] = {"langs": lang_list}
    if github:
        config["github"] = {
            "languages": ["python", "rust"],
            "max_per_query": 30,
            "token": github_token,
        }
    if doc:
        config["doc"] = {}
    if web_search:
        config["web_search"] = {}

    # Budget proportionnel à target_total (50% fr, 50% en)
    if target_total != 50000:
        per_lang = max(1, target_total // 2)
        config["budget"] = {"fr": per_lang, "en": per_lang}

    pipeline = ScrapingPipeline(output_dir, user_agent)
    result = pipeline.run(config)

    click.echo(f"\nTerminé : {result['total_written']} phrases → {result['output']} (session: {result['timestamp']})")
    click.echo("\nBalance par langue :")
    for rel, stat in result["balance"].items():
        click.echo(f"  {rel:<22} {stat}")


if __name__ == "__main__":
    scrape_cmd()
