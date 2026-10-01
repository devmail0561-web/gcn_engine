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
	python3 -m pytest gcn-python/tests/ -q

gates:
	for f in gcn-datasets/DATA/supervision/v4/annotated/lot*.json; do \
	  python3 scripts/check_v4.py --min-class-count 1 "$$f" || exit 1; \
	done

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
