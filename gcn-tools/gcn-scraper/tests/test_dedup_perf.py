"""C5 : la mémoire near-dedup doit être bornée (pas de croissance O(N) liste)."""
from gcn_scraper.filters.deduplicator import Deduplicator


def test_shingle_memory_bounded():
    dedup = Deduplicator()
    for i in range(3000):
        dedup.is_duplicate(
            f"phrase distincte numéro {i} avec plusieurs mots différents pour le shingling"
        )
    assert len(dedup._seen_shingles) <= 2500


def test_exact_duplicates_still_caught():
    dedup = Deduplicator()
    s = "la pluie cause des inondations dans la vallée entière du fleuve"
    assert not dedup.is_duplicate(s)
    assert dedup.is_duplicate(s)
    assert dedup.is_duplicate(s.upper())
