from gcn_python.taxonomy.loader import TaxonomyIndex


def test_load_all_taxonomies(taxonomy_dir):
    tax = TaxonomyIndex.load(taxonomy_dir, "fr")
    assert len(tax) > 0, "No taxonomies loaded"
    assert "verbes.etat" in tax.data
    assert "verbes.action" in tax.data
    assert "conjonctions.cause" in tax.data
    assert "determinants.quantitatif" in tax.data


def test_membership(taxonomy_dir):
    tax = TaxonomyIndex.load(taxonomy_dir, "fr")
    m = tax.membership("être")
    assert m.get("verbes.etat") is True


def test_unknown_lang_fallback(taxonomy_dir):
    tax = TaxonomyIndex.load(taxonomy_dir, "xx")
    assert isinstance(tax, TaxonomyIndex)
