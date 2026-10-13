install:
	cargo build --release --manifest-path gcn-core/Cargo.toml
	pip install ./gcn-python

install-dev:
	cargo build --manifest-path gcn-core/Cargo.toml
	pip install -e ./gcn-python

build-release:
	cargo build --release --manifest-path gcn-core/Cargo.toml

test:
	cd gcn-core && cargo test --workspace
	PYTHONPATH=gcn-python/src python3 -m pytest gcn-python/tests/ -q
	# NOTE : PYTHONPATH obligatoire — sans lui, pytest résout gcn_python via
	# le paquet installé (potentiellement un checkout stale) au lieu de la source locale,
	# et test_engine_b/test_nlu_infer échouent sur des chemins relatifs.

gates: guards
	for f in gcn-datasets/DATA/supervision/v4/annotated/lot*.json; do \
	  python3 scripts/check_v4.py --min-class-count 1 "$$f" || exit 1; \
	done

guards:
	! grep -rq gcn_transformers gcn-python/src/
	! grep -rn "^import torch\|^from torch" gcn-python/src/
	# V5 — subprocess réel : `git rev-parse` (provenance, training/train.py) est le seul
	# usage autorisé ; frontend/bridge.py (lattice gcn-cli) est le reste connu en
	# transition (bascule P1). Tout NOUVEAU fichier avec import/appel subprocess
	# fait échouer (les mentions en docstring ne comptent pas).
	! grep -rn "import subprocess\|subprocess\.run\|subprocess\.Popen" --include="*.py" gcn-python/src/ | grep -v "frontend/bridge.py" | grep -v "training/train.py"
	# Compteur informatif bascule → 0 en fin de refacto (ne fait pas échouer).
	grep -rln 'gcn_bin\|Lattice\|_call_gcn_lattice' --include="*.py" gcn-python/src/ || true
	# R4 — ban final/ : aucun répertoire final/ sous les chemins de modèles
	# (les splits DATA/final/* fuyards sont exclus par ailleurs : voir audit-gold).
	test ! -d gcn-python/models/final
	test ! -d gcn-datasets/checkpoints/final

lint:
	cd gcn-core && cargo clippy -- -D warnings
	ruff check gcn-python/src/ gcn-python/tests/

fmt:
	cd gcn-core && cargo fmt --check
	ruff format --check gcn-python/src/ gcn-python/tests/

audit:
	cd gcn-core && cargo audit

release: test lint
	cargo build --release --manifest-path gcn-core/Cargo.toml
	pip install ./gcn-python
