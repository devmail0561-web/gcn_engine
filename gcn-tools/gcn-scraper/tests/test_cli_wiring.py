"""Phase 2 (M1/M20) : options CLI câblées — wiki off, warnings github/doc, bornes."""
from click.testing import CliRunner

from gcn_scraper.cli import scrape_cmd


def _invoke(args):
    runner = CliRunner()
    # Intercepter pipeline.run pour inspecter la config construite, sans réseau.
    import gcn_scraper.cli as cli_mod

    captured = {}

    class _FakePipeline:
        def __init__(self, *a, **k):
            captured["init"] = (a, k)

        def run(self, config):
            captured["config"] = config
            return {"total_written": 0, "output": "x", "timestamp": "t", "balance": {}}

    cli_mod.ScrapingPipeline = _FakePipeline
    try:
        result = runner.invoke(scrape_cmd, args)
    finally:
        import importlib

        importlib.reload(cli_mod)
    assert result.exit_code == 0, result.output
    return captured, result.output


def test_no_wiki_disables_wikipedia(tmp_path):
    captured, _ = _invoke(["--output-dir", str(tmp_path), "--no-wiki"])
    assert captured["config"].get("wikipedia", {}).get("enabled") is False


def test_github_with_prog_langs_none_warns_and_skips(tmp_path):
    captured, output = _invoke(
        ["--output-dir", str(tmp_path), "--github", "--prog-langs", "none"]
    )
    assert "github" not in captured["config"]
    assert "ignoré" in output  # warning explicite, pas de skip silencieux (M20)


def test_github_receives_max_per_query(tmp_path):
    captured, _ = _invoke(
        ["--output-dir", str(tmp_path), "--github", "--max-per-query", "50"]
    )
    assert captured["config"]["github"]["max_per_query"] == 50


def test_min_quality_bounds_rejected(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        scrape_cmd, ["--output-dir", str(tmp_path), "--min-quality", "99"]
    )
    assert result.exit_code != 0


def test_registry_db_default_path(tmp_path):
    """Sans --registry-db, le chemin par défaut est output-dir/../url_registry.db."""
    from pathlib import Path
    captured, _ = _invoke(["--output-dir", str(tmp_path)])
    init_kwargs = captured["init"][1]
    expected = Path(tmp_path).parent / "url_registry.db"
    assert init_kwargs.get("registry_db") == expected


def test_no_registry_disables_db(tmp_path):
    """--no-registry → registry_db=None transmis au pipeline."""
    captured, _ = _invoke(["--output-dir", str(tmp_path), "--no-registry"])
    init_kwargs = captured["init"][1]
    assert init_kwargs.get("registry_db") is None


def test_no_registry_warns_when_registry_db_also_set(tmp_path):
    """--no-registry + --registry-db → warning explicite."""
    _, output = _invoke([
        "--output-dir", str(tmp_path),
        "--no-registry", "--registry-db", str(tmp_path / "custom.db"),
    ])
    assert "ignoré" in output


def test_custom_registry_db_path(tmp_path):
    """--registry-db chemin_personnalisé → chemin transmis tel quel."""
    custom = tmp_path / "custom_registry.db"
    captured, _ = _invoke(["--output-dir", str(tmp_path), "--registry-db", str(custom)])
    init_kwargs = captured["init"][1]
    assert init_kwargs.get("registry_db") == custom
