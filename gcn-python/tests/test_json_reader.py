from gcn_python.data.json_reader import load_sentences


def test_load_paper_examples(paper_examples_yaml):
    records = load_sentences(paper_examples_yaml, "fr")
    assert len(records) >= 12
    assert all(r.text for r in records)
    assert all(r.clauses for r in records)


def test_edges_loaded(paper_examples_yaml):
    records = load_sentences(paper_examples_yaml, "fr")
    records_with_edges = [r for r in records if r.edges]
    assert len(records_with_edges) >= 8
