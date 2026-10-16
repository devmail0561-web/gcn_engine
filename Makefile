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
	# P6 remediation-donnees : 0 junk actif (latex/EN hors tombstones) sinon FAIL.
	python3 scripts/detect_bruit.py --dir gcn-datasets/DATA/supervision/v4/annotated \
	  --out /tmp/opencode/bruit_gate.json --exclude-tombstones --strict

guards:
	! grep -rq gcn_transformers gcn-python/src/
	# Torch cantonné layer3 PT/GAT, jamais runtime tok : aucun import top-level.
	! grep -rn "^import torch\|^from torch" gcn-python/src/
	# V5 — subprocess restreint au chemin tok : `git rev-parse` (provenance,
	# training/train.py et training/train_tok.py via shutil.which + timeout,
	# best-effort) + frontend/bridge.py (lattice gcn-cli, transition bascule P1)
	# sont les seuls usages autorisés ; pipeline/forward_ids.py, nlu/tok.py,
	# layer0/tokenizer.py et tout NOUVEAU fichier hors de cette liste avec
	# import/appel subprocess fait échouer (mentions en docstring exclues).
	! grep -rn "import subprocess\|subprocess\.run\|subprocess\.Popen" --include="*.py" gcn-python/src/ | grep -v "frontend/bridge.py" | grep -v "training/train.py" | grep -v "training/train_tok.py"
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
