"""Chargement de la configuration des sources depuis sources.yaml."""
from __future__ import annotations
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent / "sources.yaml"
_config: dict | None = None


def get_config() -> dict:
    """Charge (et met en cache) la configuration complète."""
    global _config
    if _config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def get_source_config(source_name: str) -> dict:
    """Retourne la config d'une source spécifique (dict vide si inconnue)."""
    return get_config().get("sources", {}).get(source_name, {})


def reload_config() -> None:
    """Force le rechargement de la config (utile pour les tests)."""
    global _config
    _config = None
