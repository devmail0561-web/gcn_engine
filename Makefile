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

release: test
	cargo build --release --manifest-path gcn-core/Cargo.toml
	pip install ./gcn-python
