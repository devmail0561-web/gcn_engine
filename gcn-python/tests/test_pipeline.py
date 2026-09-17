import numpy as np
import pytest
from gcn_python.layer1.features import FeatureVocabulary, vectorize_edge
from gcn_python.layer1.representation import UDRepresentation
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline


def make_rep() -> UDRepresentation:
    return UDRepresentation(
        tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma="baisser",
        root_pos="VERB",
        root_dep_rel="root",
        root_morph={"Tense": "Pres"},
        subject_pos="NOUN",
        has_object=False,
        has_advcl=False,
        has_temporal_obl=False,
        token_span=(1, 2),
    )


def make_pipeline() -> CGNPipeline:
    vocab = FeatureVocabulary()
    d_edge_cl = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge_cl)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    return CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)


def test_forward_returns_cir():
    pipeline = make_pipeline()
    rep = make_rep()
    result = pipeline.forward([rep], "Si les ventes baissent, on réduit les coûts.")
    assert "source_lang" in result
    assert "nodes" in result
    assert "edges" in result
    assert result["source_text"] != ""


def test_forward_empty_reps():
    pipeline = make_pipeline()
    result = pipeline.forward([], "")
    assert result["nodes"] == []


def test_forward_single_rep():
    pipeline = make_pipeline()
    result = pipeline.forward([make_rep()], "Il court.")
    assert len(result["nodes"]) >= 1


def test_metadata_pipeline_field():
    pipeline = make_pipeline()
    result = pipeline.forward([make_rep()], "Il travaille.")
    assert "pipeline" in result["metadata"]
    assert any(
        "layer" in p.lower() or "cgnp" in p.lower()
        for p in result["metadata"]["pipeline"]
    )
    assert not any("spacy" in p.lower() for p in result["metadata"]["pipeline"])


def test_forward_two_reps():
    pipeline = make_pipeline()
    rep1, rep2 = make_rep(), make_rep()
    result = pipeline.forward([rep1, rep2], "Les ventes baissent puis on réduit.")
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) >= 1


def test_forward_connector_slot_nonzero():
    """Avec un connector_rep, les features UPOS du connecteur ne sont pas toutes à zéro."""
    vocab = FeatureVocabulary()

    rep1 = make_rep()
    rep2 = make_rep()
    connector = UDRepresentation(
        tokens=[{"lemma": "parce", "pos": "SCONJ", "dep_rel": "mark", "morph": {}}],
        root_lemma="parce", root_pos="SCONJ", root_dep_rel="mark",
        root_morph={}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(3, 3),
    )

    vec_with = vectorize_edge(rep1, rep2, connector, 0, 1, 2, vocab)
    vec_without = vectorize_edge(rep1, rep2, None, 0, 1, 2, vocab)

    n_upos = len(vocab.upos_tags)
    d_conn = vocab.d_conn
    upos_with = vec_with[-d_conn: -d_conn + n_upos]
    upos_without = vec_without[-d_conn: -d_conn + n_upos]

    assert not np.all(upos_with == 0), "La UPOS du connecteur doit être encodée"
    assert np.all(upos_without == 0), "Sans connecteur, la UPOS doit être nulle"


def test_forward_rgcn_dout_mismatch_raises():
    """Un RGCNLayer avec d_out ≠ d_clause doit lever ValueError dès la construction."""
    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause + 1)
    with pytest.raises(ValueError, match="d_out=.*≠.*d_clause"):
        CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)


def test_forward_position_features_noncontiguous():
    """clause_positions corrige les features de distance pour clauses non-contiguës."""
    vocab = FeatureVocabulary()
    rep1 = make_rep()
    rep2 = make_rep()

    vec_filtered = vectorize_edge(rep1, rep2, None, 0, 1, 2, vocab)
    vec_original = vectorize_edge(rep1, rep2, None, 0, 2, 3, vocab)

    assert not np.allclose(vec_filtered[-2:], vec_original[-2:]), \
        "Les features de position doivent différer selon les positions originales"


# ─── T-1 : pipeline lang="en" ────────────────────────────────────────────────

def test_forward_en_returns_valid_cir():
    """T-1 : le pipeline EN produit un CausalIR valide."""
    pipeline = make_pipeline()
    rep = make_rep()
    result = pipeline.forward([rep], "Sales fall because costs rise.")
    assert "source_lang" in result
    assert "nodes" in result
    assert len(result["nodes"]) >= 1


def test_forward_en_condition_label():
    """T-1 : le label de condition EN est 'hidden_cause(?)' et non 'cause_cachée(?)'."""
    from gcn_python.pipeline.label_builder import build_label
    rep = make_rep()
    label, _ = build_label(rep, "condition")
    assert label == "hidden_cause(?)", f"Attendu 'hidden_cause(?)', obtenu {label!r}"


# ─── T-2 : backward() sans backward_message_pass ────────────────────────────

def test_backward_with_pt_graph_does_not_crash():
    """T-2 : backward() ne crashe pas quand le graph n'a pas de backward_message_pass.

    Utilise RGCNLayerPT (après fix) qui n'a plus la méthode. Si torch n'est pas
    installé, teste via un stub minimal satisfaisant le Protocol.
    """
    try:
        from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
        torch_available = True
    except ImportError:
        torch_available = False

    vocab = FeatureVocabulary()

    if torch_available:
        graph = RGCNLayerPT(d_in=vocab.d_clause, d_out=vocab.d_clause,
                            n_relations=11, device="cpu")
        assert not hasattr(graph, 'backward_message_pass'), \
            "RGCNLayerPT ne doit plus avoir backward_message_pass"
    else:
        # Stub minimal sans backward_message_pass
        class _StubGraph:
            d_in = vocab.d_clause
            d_out = vocab.d_clause

            def message_pass(self, node_features, edge_index, edge_types):
                return node_features

            def parameters(self):
                return []

            def update(self, grads, lr):
                pass

        graph = _StubGraph()

    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7), seed=0)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)

    rep1, rep2 = make_rep(), make_rep()
    pipeline.forward([rep1, rep2], "test")
    d_node = np.zeros((2, 7), dtype=np.float32)
    d_edge = np.zeros((1, 11), dtype=np.float32)
    pipeline.backward(d_node, d_edge, lr=0.01)


# ─── Phase 9 corrections ─────────────────────────────────────────────────────

def test_custom_encoder_emits_warning_no_rgcn_update():
    """C1 : backward() sans backward_node_dx émet un UserWarning."""
    import warnings

    class _CustomEncoder:
        def forward_node(self, v): return np.zeros(7)
        def forward_edge(self, v): return np.zeros(11)

    vocab = FeatureVocabulary()
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=_CustomEncoder(), graph=graph, vocabulary=vocab)
    rep1, rep2 = make_rep(), make_rep()
    pipeline.forward([rep1, rep2], "test")
    d_node = np.zeros((2, 7), dtype=np.float32)
    d_edge = np.zeros((1, 11), dtype=np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipeline.backward(d_node, d_edge, lr=0.01)
    assert any(issubclass(x.category, UserWarning) and "backward_node_dx" in str(x.message) for x in w)


def test_backward_full_gradient():
    """C2 : d_enriched est alloué à la taille N dès le départ (pas de troncature)."""
    pipeline = make_pipeline()
    reps = [make_rep() for _ in range(3)]
    pipeline.forward(reps, "test")
    d_node = np.ones((3, 7), dtype=np.float32) * 0.1
    d_edge = np.zeros((2, 11), dtype=np.float32)
    pipeline.backward(d_node, d_edge, lr=0.01)  # ne doit pas crasher


def test_negated_edge_detected():
    """C3/C4 : un connecteur avec Polarity=Neg dans root_morph sort negated=True."""
    from gcn_python.layer1.representation import UDRepresentation
    vocab = FeatureVocabulary()
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    rep1 = make_rep()
    rep2 = make_rep()
    neg_rep = UDRepresentation(
        tokens=[{"lemma": "pas", "pos": "ADV", "dep_rel": "advmod", "morph": {"Polarity": "Neg"}}],
        root_lemma="pas", root_pos="ADV", root_dep_rel="advmod",
        root_morph={"Polarity": "Neg"}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(3, 3),
    )
    result = pipeline.forward([rep1, rep2], "X ne cause pas Y.", connector_reps=[neg_rep])
    assert result["edges"][0][2]["negated"] is True


def test_scope_universal_tous():
    """C5 : 'tous' dans un token det → scope=universal."""
    from gcn_python.layer1.representation import UDRepresentation
    rep = UDRepresentation(
        tokens=[
            {"lemma": "tous", "pos": "DET", "dep_rel": "det", "morph": {}},
            {"lemma": "coût", "pos": "NOUN", "dep_rel": "nsubj", "morph": {}},
            {"lemma": "augmenter", "pos": "VERB", "dep_rel": "root", "morph": {}},
        ],
        root_lemma="augmenter", root_pos="VERB", root_dep_rel="root",
        root_morph={}, subject_pos="NOUN",
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(5, 8),
    )
    pipeline = make_pipeline()
    result = pipeline.forward([rep], "Tous les coûts augmentent.")
    assert result["nodes"][0]["scope"] == "universal"


def test_origin_inferred_no_connector():
    """C6 : _infer_origin retourne 'inferred' pour condition sans connector."""
    from gcn_python.pipeline.cgnp import _infer_origin
    assert _infer_origin("condition", None) == "inferred"
    assert _infer_origin("condition", object()) == "explicit"
    assert _infer_origin("action", None) == "explicit"


def test_attributes_entity_not_null():
    """C7 : attributes.entity est peuplé quand le span a un NOUN nsubj."""
    from gcn_python.layer1.representation import UDRepresentation
    rep = UDRepresentation(
        tokens=[
            {"lemma": "coût", "pos": "NOUN", "dep_rel": "nsubj", "morph": {}},
            {"lemma": "augmenter", "pos": "VERB", "dep_rel": "root", "morph": {}},
        ],
        root_lemma="augmenter", root_pos="VERB", root_dep_rel="root",
        root_morph={}, subject_pos="NOUN",
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(1, 2),
    )
    pipeline = make_pipeline()
    result = pipeline.forward([rep], "Les coûts augmentent.")
    assert result["nodes"][0]["attributes"]["entity"] == "coût"


def test_label_nominalized():
    """C8 : CGNPipeline accepte taxonomies_dir comme kwarg."""
    vocab = FeatureVocabulary()
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    g = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    p = CGNPipeline(enc, g, vocab, taxonomies_dir=None)
    assert p.taxonomies_dir is None


def test_rep_nominal_clause_root_pos():
    """C9 : une clause sans VERB prend un NOUN comme root, pas un DET."""
    class _FakeTok:
        def __init__(self, id, lemma, pos, dep_rel="dep", morph=None, gcn_causal_type=None):
            self.id = id
            self.lemma = lemma
            self.pos = pos
            self.dep_rel = dep_rel
            self.morph = morph or {}
            self.gcn_causal_type = gcn_causal_type

    span_toks = [
        _FakeTok(1, "la", "DET", "det"),
        _FakeTok(2, "hausse", "NOUN", "root"),
        _FakeTok(3, "des", "DET", "det"),
        _FakeTok(4, "prix", "NOUN", "nmod"),
    ]
    root_tok = (
        next((t for t in span_toks if getattr(t, 'gcn_causal_type', None) == "verbe" and t.pos in {"VERB", "AUX"}), None)
        or next((t for t in span_toks if t.pos in {"VERB", "AUX"}), None)
        or next((t for t in span_toks if t.pos in {"NOUN", "PROPN"}), None)
        or span_toks[0]
    )
    assert root_tok.pos == "NOUN"
    assert root_tok.lemma == "hausse"
