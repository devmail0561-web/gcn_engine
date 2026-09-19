"""CLI gcn-scrape v3 — scraping multilingue agnostique, horodaté, multi-sources."""
from __future__ import annotations
import click
from pathlib import Path
from .pipeline import ScrapingPipeline


@click.command("gcn-scrape")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire de sortie (sentences_<timestamp>.jsonl)")
@click.option("--target-total", default=50000, show_default=True, type=int,
              help="Nombre de phrases cibles (arrêt dès que le budget global est atteint)")
@click.option("--resume", is_flag=True, default=False,
              help="Reprend depuis le checkpoint existant (skip les sources déjà faites)")
@click.option("--langs", default="fr,en", show_default=True,
              help="Langues humaines à scraper (codes ISO séparés par virgule). "
                   "Ex: fr,en ou fr,en,es,de. Utiliser 'all' pour toutes les langues configurées.")
@click.option("--prog-langs", default="python,rust", show_default=True,
              help="Langages de programmation pour GitHub/doc. "
                   "Ex: python,rust,javascript,java,go. 'all' = tous configurés, 'none' = désactiver.")
@click.option("--min-quality", default=0.3, show_default=True, type=float,
              help="Score de qualité minimum [0-1] pour conserver une phrase")
@click.option("--max-per-query", default=100, show_default=True, type=int,
              help="Nb max de résultats par requête HAL/arXiv")
@click.option("--hal/--no-hal", default=True, show_default=True)
@click.option("--arxiv/--no-arxiv", default=True, show_default=True)
@click.option("--news/--no-news", default=True, show_default=True)
@click.option("--github/--no-github", default=False, show_default=True,
              help="Scraping GitHub (token recommandé pour éviter le rate-limit public)")
@click.option("--github-token", default=None, envvar="GITHUB_TOKEN",
              help="Token GitHub (ou variable d'env GITHUB_TOKEN)")
@click.option("--doc/--no-doc", default=True, show_default=True)
@click.option("--web-search/--no-web-search", default=False, show_default=True,
              help="Recherche web multi-sources (DuckDuckGo + OpenAlex + PubMed)")
@click.option("--user-agent", default="GCN-Dataset/3.0 (research; contact: gcn-research@example.org)", show_default=True)
@click.option("--contact-email", default="gcn-research@example.org", show_default=True,
              help="Email de contact pour les APIs polies (OpenAlex, PubMed, Wikipedia).")
@click.option("--seed", default=None, type=int,
              help="Graine de diversité : même seed = même ordre (reproductible), "
                   "seeds différents = requêtes/URLs/offsets différents. Défaut: tirage aléatoire.")
def scrape_cmd(
    output_dir, target_total, resume, langs, prog_langs, min_quality,
    max_per_query, hal, arxiv, news, github, github_token, doc,
    web_search, user_agent, contact_email, seed,
):
    """
    Scrape du texte brut multilingue pour le dataset causal GCN.

    Sortie : sentences_<timestamp>.jsonl + un fichier par source.
    L'annotation causale est faite séparément par le moteur GCN.

    Exemples :
      gcn-scrape --output-dir ./raw --target-total 50000
      gcn-scrape --output-dir ./raw --resume
      gcn-scrape --output-dir ./raw --langs fr,en,es --prog-langs python,rust,javascript
      gcn-scrape --output-dir ./raw --langs fr --prog-langs none
      gcn-scrape --output-dir ./raw --langs all --prog-langs all --target-total 80000
    """
    from .config.loader import get_config
    cfg = get_config()

    # Résoudre les langues humaines
    if langs.strip().lower() == "all":
        selected_langs = [
            v["lang"] for k, v in cfg.get("sources", {}).items()
            if k.startswith("wikipedia_") and v.get("enabled", True) and "lang" in v
        ]
    else:
        selected_langs = [l.strip() for l in langs.split(",") if l.strip()]
    if not selected_langs:
        selected_langs = ["fr", "en"]

    # Résoudre les langages de programmation
    if prog_langs.strip().lower() == "none":
        selected_prog_langs: list[str] = []
    elif prog_langs.strip().lower() == "all":
        # Union github.languages + doc.urls : csharp n'existe que côté doc
        # (audit-2 Fix 8 : "all" l'omettait).
        _gh_langs = cfg.get("sources", {}).get("github", {}).get("languages", [])
        _doc_langs = list(cfg.get("sources", {}).get("doc", {}).get("urls", {}).keys())
        selected_prog_langs = list(dict.fromkeys(_gh_langs + _doc_langs)) or ["python", "rust"]
    else:
        selected_prog_langs = [l.strip() for l in prog_langs.split(",") if l.strip()]

    config: dict = {
        "resume": resume,
        "min_quality": min_quality,
        "langs": selected_langs,
        "prog_langs": selected_prog_langs,
        "target_total": target_total,
        "seed": seed,
    }
    if hal:
        config["hal"] = {"max_per_query": max_per_query}
    if arxiv:
        config["arxiv"] = {"max_per_query": max_per_query}
    if news:
        config["news"] = {"langs": [l for l in selected_langs if l in ("fr", "en")]}
    if github and selected_prog_langs:
        config["github"] = {"token": github_token}
    if doc and selected_prog_langs:
        config["doc"] = {}
    if web_search:
        config["web_search"] = {}

    pipeline = ScrapingPipeline(output_dir, user_agent, contact_email=contact_email)
    result = pipeline.run(config)

    click.echo(f"\nTerminé : {result['total_written']} phrases → {result['output']} (session: {result['timestamp']})")
    click.echo("\nBalance par langue :")
    for lang_key, stat in result["balance"].items():
        click.echo(f"  {lang_key:<10} {stat}")


if __name__ == "__main__":
    scrape_cmd()
