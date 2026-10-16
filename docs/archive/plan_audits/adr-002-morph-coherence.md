# ADR-002 — Cohérence morphologique train/infer (vacuité)

Status : accepté.

État actuel : cohérent par vacuité.
- `data/json_reader.py` → `morph={}` (tokens sans morphologie).
- `frontend/bridge.py` → `root_morph={}` (CIR Rust sans morphologie).
- `layer1/representation.py` → 14 features `_absent` dans les deux chemins.

Règle : ne jamais peupler `morph` d'un seul côté. Si les données
d'entraînement sont enrichies en morphologie, `bridge.py` (inférence)
doit l'être aussi, ou le drift train/infer invalide les poids.

Décision v2.0 : laisser vide jusqu'à extension du frontend Rust
(`gcn analyze` émet la morphologie). Réévaluer avant tout peuplement
sélectif.
