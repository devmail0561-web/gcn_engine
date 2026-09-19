"""Utilitaire réseau partagé — retry avec backoff, jitter, cooldown par host.

Fix 429-bloquant :
- 429 : honore Retry-After (secondes ou date HTTP), backoff exponentiel +
  jitter, capé à `max_wait`. Après `max_retries` échecs : cooldown du host
  (60s) et retour immédiat (None) — les appels suivants sur le même host
  échouent vite au lieu de bloquer en boucle.
- 403 : NON retryable par défaut (WAF/Cloudflare/ban). Seule exception :
  rate-limit GitHub explicite (X-RateLimit-Remaining: 0 ou body
  "rate limit" / "abuse detection") → traité comme 429.
- 4xx (400/401/404/422) : fatal, retour immédiat sans retry.
- 5xx : retryable avec backoff + jitter.
- Throttle par host (`min_interval`) + jitter pour éviter les rafales.
"""
from __future__ import annotations

import email.utils
import random
import time
from urllib.parse import urlparse

import requests

DEFAULT_USER_AGENT = "GCN-Dataset/2.0 (research; contact: gcn-research@example.org)"

# host -> timestamp jusqu'auquel on refuse d'appeler (fail-fast, sans sleep)
_HOST_COOLDOWN_UNTIL: dict[str, float] = {}
# host -> timestamp du dernier appel (throttle)
_HOST_LAST_CALL: dict[str, float] = {}


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return url


def host_in_cooldown(url: str) -> bool:
    return _HOST_COOLDOWN_UNTIL.get(_host(url), 0.0) > time.time()


def _set_cooldown(url: str, seconds: float) -> None:
    _HOST_COOLDOWN_UNTIL[_host(url)] = time.time() + max(0.0, seconds)


def clear_cooldown(url: str | None = None) -> None:
    """Réinitialise les cooldowns (utile en tests)."""
    if url is None:
        _HOST_COOLDOWN_UNTIL.clear()
        _HOST_LAST_CALL.clear()
    else:
        _HOST_COOLDOWN_UNTIL.pop(_host(url), None)
        _HOST_LAST_CALL.pop(_host(url), None)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
        if dt is not None:
            return max(0.0, (dt.timestamp() - time.time()))
    except Exception:
        pass
    return None


def _backoff(base: float, attempt: int, cap: float) -> float:
    wait = base * (2**attempt) + random.uniform(0, base)
    return min(wait, cap)


def _is_github_rate_limit(resp: requests.Response, url: str = "") -> bool:
    if "github" not in _host(url) and "raw.githubusercontent" not in _host(url):
        return False
    if resp.headers.get("X-RateLimit-Remaining") == "0":
        return True
    try:
        body = resp.text[:500].lower()
    except Exception:
        return False
    return "rate limit" in body or "abuse detection" in body


def _rebuild_session(session: requests.Session, user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
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
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_wait: float = 60.0,
    min_interval: float = 0.0,
    timeout: float = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    extra_headers: dict | None = None,
) -> tuple:
    """GET robuste, non-bloquant sur 429 répétés.

    Retourne (Response, session). (None, session) = échec définitif ou
    host en cooldown — l'appelant doit skipper, PAS réessayer en boucle.
    """
    host = _host(url)

    # Fail-fast : host déjà en cooldown → pas de sleep, retour immédiat.
    now = time.time()
    if _HOST_COOLDOWN_UNTIL.get(host, 0.0) > now:
        return None, session

    # Throttle par host pour éviter de déclencher le rate-limit.
    last = _HOST_LAST_CALL.get(host, 0.0)
    gap = now - last
    if min_interval > 0 and gap < min_interval:
        time.sleep(min_interval - gap)

    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout, headers=extra_headers)
            _HOST_LAST_CALL[host] = time.time()

            if resp.status_code == 200:
                return resp, session

            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                wait = retry_after if retry_after is not None else _backoff(base_delay, attempt, max_wait)
                wait = min(wait, max_wait)
                if attempt >= max_retries - 1:
                    # Dernier échec → cooldown host pour que les requêtes
                    # suivantes échouent vite au lieu de bloquer chacune.
                    _set_cooldown(url, max(wait, 60.0))
                    print(f"    HTTP 429 sur {host}, cooldown 60s (abandon).")
                    return None, session
                print(f"    HTTP 429, attente {wait:.1f}s (tentative {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue

            if resp.status_code == 403:
                if _is_github_rate_limit(resp, url):
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                    wait = retry_after if retry_after is not None else _backoff(base_delay, attempt, max_wait)
                    wait = min(wait, max_wait)
                    if attempt >= max_retries - 1:
                        _set_cooldown(url, max(wait, 60.0))
                        print(f"    HTTP 403 rate-limit sur {host}, cooldown 60s (abandon).")
                        return None, session
                    print(f"    HTTP 403 rate-limit, attente {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                # 403 WAF/ban/auth → fatal, ne pas marteler.
                print(f"    HTTP 403 sur {host} (accès refusé, non retryable), skip.")
                return None, session

            if resp.status_code in (500, 502, 503, 504):
                wait = _backoff(base_delay, attempt, max_wait)
                if attempt >= max_retries - 1:
                    return None, session
                print(f"    HTTP {resp.status_code}, attente {wait:.1f}s...")
                time.sleep(wait)
                continue

            # Autres 4xx : fatal, retour immédiat.
            return None, session

        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.ChunkedEncodingError,
            ConnectionResetError,
            OSError,
        ) as exc:
            wait = _backoff(base_delay, attempt, max_wait)
            print(f"    réseau {type(exc).__name__} (tentative {attempt + 1}/{max_retries}), attente {wait:.1f}s...")
            time.sleep(wait)
            if attempt >= 2:
                print("    rebuild session TCP...")
                session = _rebuild_session(session, user_agent)
    return None, session
