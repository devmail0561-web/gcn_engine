"""CLI gcn-scrape v3 — scraping multilingue agnostique, horodaté, multi-sources."""
from __future__ import annotations
import click
from pathlib import Path
from .pipeline import ScrapingPipeline


@click.command("gcn-scrape")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire de sortie (sentences_<timestamp>.jsonl)")
@click.option("--target-total", default=50000, show_default=True,
               type=click.IntRange(min=0),
               help="Nombre de phrases cibles (arrêt dès que le budget global est atteint)")
@click.option("--resume", is_flag=True, default=False,
              help="Reprend depuis le checkpoint existant (skip les sources déjà faites)")
@click.option("--langs", default="fr,en", show_default=True,
              help="Langues humaines à scraper (codes ISO séparés par virgule). "
                   "Ex: fr,en ou fr,en,es,de. Utiliser 'all' pour toutes les langues configurées.")
@click.option("--prog-langs", default="python,rust", show_default=True,
              help="Langages de programmation pour GitHub/doc. "
                   "Ex: python,rust,javascript,java,go. 'all' = tous configurés, 'none' = désactiver.")
@click.option("--min-quality", default=0.3, show_default=True,
               type=click.FloatRange(0.0, 1.0),
               help="Score de qualité minimum [0-1] pour conserver une phrase")
@click.option("--max-per-query", default=100, show_default=True,
               type=click.IntRange(min=1),
               help="Nb max de résultats par requête (HAL, arXiv, GitHub)")
@click.option("--wiki/--no-wiki", default=True, show_default=True,
               help="Activer/désactiver Wikipedia (toutes langues)")
@click.option("--hal/--no-hal", default=True, show_default=True,
               help="Activer/désactiver HAL (archives ouvertes scientifiques)")
@click.option("--arxiv/--no-arxiv", default=True, show_default=True,
               help="Activer/désactiver arXiv (abstracts scientifiques EN)")
@click.option("--news/--no-news", default=True, show_default=True,
               help="Activer/désactiver les flux News RSS (FR/EN)")
@click.option("--github/--no-github", default=False, show_default=True,
              help="Scraping GitHub (token recommandé pour éviter le rate-limit public)")
@click.option("--github-token", default=None, envvar="GITHUB_TOKEN",
              help="Token GitHub (ou variable d'env GITHUB_TOKEN)")
@click.option("--doc/--no-doc", default=True, show_default=True,
               help="Activer/désactiver la documentation officielle des langages")
@click.option("--web-search/--no-web-search", default=False, show_default=True,
              help="Recherche web multi-sources (DuckDuckGo + OpenAlex + PubMed)")
@click.option("--user-agent", default="GCN-Dataset/3.0 (research; contact: gcn-research@example.org)", show_default=True)
@click.option("--contact-email", default="gcn-research@example.org", show_default=True,
              help="Email de contact pour les APIs polies (OpenAlex, PubMed, Wikipedia).")
@click.option("--seed", default=None, type=int,
              help="Graine de diversité : même seed = même ordre (reproductible), "
                   "seeds différents = requêtes/URLs/offsets différents. Défaut: tirage aléatoire.")
@click.option("--registry-db", default=None, type=click.Path(path_type=Path),
              help="Chemin de la DB SQLite des URLs scrapées. "
                   "Défaut: <output-dir>/../url_registry.db")
@click.option("--no-registry", is_flag=True, default=False,
              help="Désactiver le registre URL (pas de déduplication cross-campagnes).")
def scrape_cmd(
    output_dir, target_total, resume, langs, prog_langs, min_quality,
    max_per_query, wiki, hal, arxiv, news, github, github_token, doc,
    web_search, user_agent, contact_email, seed, registry_db, no_registry,
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
        selected_langs = [_lang.strip() for _lang in langs.split(",") if _lang.strip()]
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
        selected_prog_langs = [_lang.strip() for _lang in prog_langs.split(",") if _lang.strip()]

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
        news_langs = [l for l in selected_langs if l in ("fr", "en")]
        dropped = [l for l in selected_langs if l not in ("fr", "en")]
        if dropped:
            click.echo(
                f"Avertissement : pas de flux News pour {dropped} "
                "(flux FR/EN uniquement).",
                err=True,
            )
        config["news"] = {"langs": news_langs}
    if github:
        if not selected_prog_langs:
            click.echo(
                "Avertissement : --github ignoré car --prog-langs none "
                "(aucun langage à scraper).",
                err=True,
            )
        else:
            config["github"] = {"token": github_token, "max_per_query": max_per_query}
    if doc:
        if not selected_prog_langs:
            click.echo(
                "Avertissement : --doc ignoré car --prog-langs none "
                "(aucun langage à scraper).",
                err=True,
            )
        else:
            config["doc"] = {}
    if web_search:
        config["web_search"] = {}
    if not wiki:
        config["wikipedia"] = {"enabled": False}

    # Résoudre le chemin du registre URL.
    if no_registry and registry_db is not None:
        click.echo("Avertissement : --registry-db ignoré car --no-registry est actif.", err=True)
    effective_registry_db: Path | None
    if no_registry:
        effective_registry_db = None
    elif registry_db is not None:
        effective_registry_db = registry_db
    else:
        effective_registry_db = Path(output_dir).parent / "url_registry.db"

    pipeline = ScrapingPipeline(output_dir, user_agent, contact_email=contact_email,
                                registry_db=effective_registry_db)
    result = pipeline.run(config)

    click.echo(f"\nTerminé : {result['total_written']} phrases → {result['output']} (session: {result['timestamp']})")
    click.echo("\nBalance par langue :")
    for lang_key, stat in result["balance"].items():
        click.echo(f"  {lang_key:<10} {stat}")


if __name__ == "__main__":
    scrape_cmd()
