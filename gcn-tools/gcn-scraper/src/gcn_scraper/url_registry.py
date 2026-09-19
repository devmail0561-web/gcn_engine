"""Registre SQLite des URLs scrapées — déduplication cross-campagnes.

Le registre est partagé entre toutes les campagnes d'un même projet.
Chaque URL est identifiée par son SHA-256 (index unique), ce qui permet
des lookups O(1) sans scanner les URLs complètes.

Fichier DB par défaut : <output_dir>/../url_registry.db
(configurable via --registry-db)
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


class URLRegistry:
    """Registre persistant des URLs scrapées, partagé entre campagnes."""

    def __init__(self, db_path: Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._known: set[str] = self._load_hashes()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at     TEXT    NOT NULL,
                output_dir     TEXT    NOT NULL,
                timestamp      TEXT    NOT NULL,
                config_summary TEXT,
                total_scraped  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS scraped_urls (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                url_hash       TEXT    NOT NULL UNIQUE,
                url            TEXT    NOT NULL,
                campaign_id    INTEGER NOT NULL REFERENCES campaigns(id),
                source_name    TEXT    NOT NULL,
                output_file    TEXT    NOT NULL,
                scraped_at     TEXT    NOT NULL,
                sentence_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_url_hash ON scraped_urls(url_hash);
        """)
        self._conn.commit()

    def _load_hashes(self) -> set[str]:
        cur = self._conn.execute("SELECT url_hash FROM scraped_urls")
        return {row[0] for row in cur.fetchall()}

    def new_campaign(self, output_dir: str, timestamp: str, config_summary: str = "") -> int:
        """Crée une nouvelle entrée campagne et retourne son id."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO campaigns (started_at, output_dir, timestamp, config_summary)"
            " VALUES (?, ?, ?, ?)",
            (now, output_dir, timestamp, config_summary),
        )
        self._conn.commit()
        campaign_id = cur.lastrowid
        if campaign_id is None:
            raise RuntimeError("Impossible de créer la campagne dans url_registry.db")
        return campaign_id

    def is_known(self, url: str) -> bool:
        """True si l'URL a déjà été scrapée dans n'importe quelle campagne."""
        if not url:
            return False
        return _url_hash(url) in self._known

    def register(
        self,
        url: str,
        campaign_id: int,
        source_name: str,
        output_file: str,
        sentence_count: int = 0,
    ) -> None:
        """Enregistre une URL scrapée (idempotent — double insert ignoré)."""
        if not url:
            return
        h = _url_hash(url)
        if h in self._known:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO scraped_urls"
                " (url_hash, url, campaign_id, source_name, output_file, scraped_at, sentence_count)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (h, url, campaign_id, source_name, output_file, now, sentence_count),
            )
            self._conn.commit()
            self._known.add(h)
        except sqlite3.Error as exc:
            import warnings
            warnings.warn(f"url_registry: échec enregistrement URL ({exc})", UserWarning, stacklevel=2)

    def update_campaign_total(self, campaign_id: int, total: int) -> None:
        """Met à jour le compteur total de phrases d'une campagne."""
        self._conn.execute(
            "UPDATE campaigns SET total_scraped = ? WHERE id = ?", (total, campaign_id)
        )
        self._conn.commit()

    def get_all_urls(self) -> set[str]:
        """Retourne toutes les URLs connues (pour initialiser seen_urls en RAM)."""
        cur = self._conn.execute("SELECT url FROM scraped_urls")
        return {row[0] for row in cur.fetchall()}

    def stats(self) -> dict:
        """Retourne des stats rapides pour affichage CLI."""
        n_campaigns = self._conn.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
        n_urls = len(self._known)
        return {"campaigns": n_campaigns, "urls": n_urls}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "URLRegistry":
        return self

    def __exit__(self, *_) -> None:
        self.close()
