"""Utilitaire réseau partagé — retry avec backoff exponentiel et rebuild de session."""
from __future__ import annotations
import time
import requests


def _rebuild_session(session: requests.Session, user_agent: str = "GCN-Dataset/2.0 (research)") -> requests.Session:
    """Recrée une session propre après une coupure TCP."""
    try:
        session.close()
    except Exception:
        pass
    new = requests.Session()
    new.headers["User-Agent"] = user_agent
    return new


def retry_get(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = 5,
    base_delay: float = 2.0,
    user_agent: str = "GCN-Dataset/2.0 (research)",
):
    """
    GET robuste avec retry exponentiel.

    Gère : rate-limit 429, coupure réseau, timeout, reset TCP.
    Après 3 échecs réseau consécutifs : rebuild de session + pause 30s.
    Retourne la Response HTTP 200 ou None après épuisement des tentatives.
    """
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = base_delay * (2 ** attempt)
                print(f"    rate-limit 429, attente {wait:.0f}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp
            # Autres erreurs HTTP non-fatales : attente courte
            time.sleep(base_delay * (attempt + 1))
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.ChunkedEncodingError,
            ConnectionResetError,
            OSError,
        ) as exc:
            wait = base_delay * (2 ** attempt)
            print(f"    réseau {type(exc).__name__} (tentative {attempt + 1}/{max_retries}), attente {wait:.0f}s...")
            time.sleep(wait)
            # Rebuild session après 2 échecs consécutifs
            if attempt >= 2:
                print("    rebuild session TCP...")
                session = _rebuild_session(session, user_agent)
                if attempt == 2:
                    # Pause longue avant les dernières tentatives
                    print("    pause 30s avant dernières tentatives...")
                    time.sleep(30)
    return None
