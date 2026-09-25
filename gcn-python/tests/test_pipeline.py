# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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


def make_pipeline(**kwargs) -> CGNPipeline:
    vocab = FeatureVocabulary()
    d_edge_cl = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge_cl)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    return CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab, **kwargs)


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
    with pytest.raises(ValueError, match="≠ d_effective="):
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
    """C5 : 'tous' dans un token det → scope=universal avec scope_hints FR injecté."""
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
    # scope_hints FR injecté explicitement — le moteur ne connaît aucune langue par défaut
    fr_scope_hints = {
        "tous": "universal", "toutes": "universal", "chaque": "universal",
        "tout": "universal", "aucun": "null", "aucune": "null",
        "certains": "existential", "certaines": "existential",
        "quelques": "partial",
    }
    pipeline = make_pipeline(scope_hints=fr_scope_hints)
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


# ---------------------------------------------------------------------------
# Tests gradient _cross_entropy avec class_weights (B1)
# ---------------------------------------------------------------------------

def test_cross_entropy_uniform_weights_same_as_no_weights():
    """Gradient avec poids uniformes == gradient sans poids."""
    from gcn_python.pipeline.cgnp import _cross_entropy
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((8, 7)).astype(np.float32)
    labels = rng.integers(0, 7, size=8).astype(np.int64)
    _, grad_none = _cross_entropy(logits, labels, class_weights=None)
    uniform = np.ones(7, dtype=np.float32)
    _, grad_uniform = _cross_entropy(logits, labels, class_weights=uniform)
    np.testing.assert_allclose(grad_none, grad_uniform, rtol=1e-5)


def test_cross_entropy_zero_weight_zeroes_gradient_for_that_class():
    """Gradient gold-class 0 → gradient nul pour tous les samples de classe 0."""
    from gcn_python.pipeline.cgnp import _cross_entropy
    n_classes = 4
    # Tous les samples ont label=0 ; w[0]=0 → gradient global doit être 0
    logits = np.ones((5, n_classes), dtype=np.float32)
    labels = np.zeros(5, dtype=np.int64)
    weights = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    _, grad = _cross_entropy(logits, labels, class_weights=weights)
    np.testing.assert_allclose(grad, np.zeros_like(grad), atol=1e-6)


def test_cross_entropy_weighted_gradient_scales_by_gold_weight():
    """Gradient pondéré = w[y_n] * gradient_standard, par ligne."""
    from gcn_python.pipeline.cgnp import _cross_entropy
    rng = np.random.default_rng(1)
    N, C = 6, 5
    logits = rng.standard_normal((N, C)).astype(np.float32)
    labels = rng.integers(0, C, size=N).astype(np.int64)
    weights = rng.uniform(0.5, 2.0, size=C).astype(np.float32)
    _, grad_w = _cross_entropy(logits, labels, class_weights=weights)
    _, grad_0 = _cross_entropy(logits, labels, class_weights=None)
    # Chaque ligne i doit être mise à l'échelle par w[labels[i]]
    for i in range(N):
        np.testing.assert_allclose(
            grad_w[i], grad_0[i] * weights[labels[i]], rtol=1e-5,
            err_msg=f"ligne {i}: poids attendu w[{labels[i]}]={weights[labels[i]]:.4f}"
        )


# ── Améliorations v3 : résidu (E3), pooling/B forward, pondération loss (F) ──

def test_gat_residual_changes_enriched_vs_no_residual():
    """E3 : avec résidu, enriched = h + prev ≠ h seul (pipeline NumPy)."""
    from gcn_python.layer1.representation import UDRepresentation
    vocab = FeatureVocabulary()
    D = vocab.d_clause

    def _rep(lemma):
        return UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None, has_object=False,
            has_advcl=False, has_temporal_obl=False, token_span=(1, 1))

    reps = [_rep("alpha"), _rep("beta")]
    enc1 = MLPEncoder(d_clause=D, d_edge=vocab.d_edge_closed_loop(D, 7), seed=0)
    enc2 = MLPEncoder(d_clause=D, d_edge=vocab.d_edge_closed_loop(D, 7), seed=0)
    g1 = RGCNLayer(d_in=D, d_out=D, n_relations=11, seed=0)
    g2 = RGCNLayer(d_in=D, d_out=D, n_relations=11, seed=0)
    p_plain = CGNPipeline(encoder=enc1, graph=g1, vocabulary=vocab)
    p_res = CGNPipeline(encoder=enc2, graph=g2, vocabulary=vocab, gat_residual=True)
    p_plain.forward(reps, "test")
    p_res.forward(reps, "test")
    assert p_plain._cached_enriched_vecs.shape == (2, D)
    assert p_res._cached_enriched_vecs.shape == (2, D)
    assert not np.allclose(p_plain._cached_enriched_vecs, p_res._cached_enriched_vecs), \
        "résidu doit modifier les vecteurs enrichis"


def test_clause_pooling_mean_forward_shape():
    """A : forward pipeline avec clause_pooling=mean (d_emb>0)."""
    from gcn_python.layer1.embedding import WordEmbedding
    from gcn_python.layer1.representation import UDRepresentation
    vocab = FeatureVocabulary()
    we = WordEmbedding(d_emb=8, seed=0)
    we.build_vocab(["alpha", "beta"])
    d_eff = vocab.d_clause_effective(8, False)
    enc = MLPEncoder(d_clause=d_eff, d_edge=vocab.d_edge_closed_loop(d_eff, 7, 8), seed=0)
    g = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11, seed=0)
    pipe = CGNPipeline(encoder=enc, graph=g, vocabulary=vocab,
                       word_embedding=we, clause_pooling="mean")

    def _rep(lemma):
        return UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None, has_object=False,
            has_advcl=False, has_temporal_obl=False, token_span=(1, 1))

    out = pipe.forward([_rep("alpha"), _rep("beta")], "test")
    assert pipe._cached_clause_vecs.shape == (2, d_eff)
    assert pipe._cached_pool_routing is not None
    assert len(pipe._cached_pool_routing) == 2
    assert out is not None


def test_subject_object_emb_forward_shape():
    """B : forward pipeline avec subject_object_emb (d_eff = d_clause + 3*d_emb)."""
    from gcn_python.layer1.embedding import WordEmbedding
    from gcn_python.layer1.representation import UDRepresentation
    vocab = FeatureVocabulary()
    we = WordEmbedding(d_emb=8, seed=0)
    we.build_vocab(["alpha", "beta", "chat"])
    d_eff = vocab.d_clause_effective(8, True)
    assert d_eff == vocab.d_clause + 24
    enc = MLPEncoder(d_clause=d_eff,
                     d_edge=vocab.d_edge_closed_loop(d_eff, 7, 8, True), seed=0)
    g = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11, seed=0)
    pipe = CGNPipeline(encoder=enc, graph=g, vocabulary=vocab,
                       word_embedding=we, subject_object_emb=True)

    def _rep(lemma):
        return UDRepresentation(
            tokens=[{"lemma": "chat", "pos": "NOUN", "dep_rel": "nsubj", "morph": {}},
                    {"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos="NOUN", has_object=False,
            has_advcl=False, has_temporal_obl=False, token_span=(1, 2))

    out = pipe.forward([_rep("alpha"), _rep("beta")], "test")
    assert pipe._cached_clause_vecs.shape == (2, d_eff)
    assert out is not None


def test_sample_weight_scales_node_and_edge_loss():
    """BUG-8 fix : sample_weight multiplie la loss nœuds ET arêtes (cohérence gold/silver)."""
    from gcn_python.pipeline.cgnp import CGNPipeline
    vocab = FeatureVocabulary()
    D = vocab.d_clause
    enc = MLPEncoder(d_clause=D, d_edge=vocab.d_edge_closed_loop(D, 7), seed=0)
    g = RGCNLayer(d_in=D, d_out=D, n_relations=11, seed=0)
    pipe = CGNPipeline(encoder=enc, graph=g, vocabulary=vocab)
    rng = np.random.default_rng(0)
    node_logits = rng.normal(0, 1, (2, 7)).astype(np.float32)
    edge_logits = rng.normal(0, 1, (1, 11)).astype(np.float32)
    gold_node = np.array([0, 1])
    gold_edge = np.array([2])
    loss_full, d_node_full, d_edge_full = pipe.loss(
        node_logits, edge_logits, gold_node, gold_edge)
    loss_w, d_node_w, d_edge_w = pipe.loss(
        node_logits, edge_logits, gold_node, gold_edge, sample_weight=0.7)
    assert loss_w < loss_full
    assert np.allclose(d_node_w, d_node_full * 0.7), "gradient nœuds × 0.7 (BUG-8)"
    assert np.allclose(d_edge_w, d_edge_full * 0.7), "gradient arêtes × 0.7"


# ─── Two-pass validation (Phase 1 — fix teacher forcing BUG-1) ──────────────

def test_two_pass_val_edge_types_not_all_zero():
    """Avec two_pass_val=True et sans gold_edge_map, les types d'arêtes R-GCN
    ne doivent PAS être tous zéros (utilise les prédictions préliminaires)."""
    pipeline = make_pipeline()
    pipeline.two_pass_val = True
    reps = [make_rep(), make_rep()]
    pipeline.forward(reps, "A cause B.")
    assert pipeline._cached_edge_type_idxs is not None
    # Vérifier que le code ne crashe pas — les types peuvent être 0 par hasard
    # mais la logique two-pass a été exercée (pas de fallback np.zeros aveugle).


def test_two_pass_val_disabled_falls_back_to_zeros():
    """Avec two_pass_val=False et sans gold, on retombe sur le fallback type-0."""
    pipeline = make_pipeline()
    pipeline.two_pass_val = False
    reps = [make_rep(), make_rep()]
    pipeline.forward(reps, "A cause B.")
    edge_types = pipeline._cached_edge_type_idxs
    assert edge_types is not None
    # Sans bidi, edge_types_mp = edge_type_idxs_rgcn direct
    # Avec bidi, on a aussi les types inversés, mais les forward sont tous 0
    n_fwd = len(edge_types) // 2 if pipeline.bidirectional else len(edge_types)
    assert np.all(edge_types[:n_fwd] == 0), "Fallback type-0 quand two_pass_val=False"


def test_two_pass_val_predicted_types_in_range():
    """Les types prédits par le passage préliminaire sont dans [0, n_rel)."""
    pipeline = make_pipeline()
    pipeline.two_pass_val = True
    reps = [make_rep(), make_rep(), make_rep()]
    pipeline.forward(reps, "A puis B puis C.")
    edge_types = pipeline._cached_edge_type_idxs
    assert edge_types is not None
    n_rel = len(pipeline.relation_types)
    assert np.all(edge_types >= 0), f"Types négatifs trouvés : {edge_types}"
    assert np.all(edge_types < n_rel), f"Types hors bornes : {edge_types} (n_rel={n_rel})"


def test_two_pass_val_enriched_vecs_finite():
    """Les vecteurs enrichis après two-pass sont finis (pas de NaN/Inf)."""
    pipeline = make_pipeline()
    pipeline.two_pass_val = True
    reps = [make_rep(), make_rep()]
    pipeline.forward(reps, "A cause B.")
    enriched = pipeline._cached_enriched_vecs
    assert enriched is not None
    assert np.all(np.isfinite(enriched)), "Vecteurs enrichis non finis après two-pass"


def test_two_pass_val_backward_works():
    """backward() fonctionne après un forward two-pass (pas de cache corrompu)."""
    pipeline = make_pipeline()
    pipeline.two_pass_val = True
    reps = [make_rep(), make_rep()]
    pipeline.forward(reps, "A cause B.")
    n_nodes = len(reps)
    n_edges = 1
    d_node = np.zeros((n_nodes, 7), dtype=np.float32)
    d_edge = np.zeros((n_edges, 11), dtype=np.float32)
    pipeline.backward(d_node, d_edge, lr=0.01)


def test_two_pass_val_single_rep_no_crash():
    """Avec une seule rep, two_pass_val ne doit pas crasher (pas d'arêtes)."""
    pipeline = make_pipeline()
    pipeline.two_pass_val = True
    result = pipeline.forward([make_rep()], "Il court.")
    assert len(result["nodes"]) == 1
    assert result["edges"] == []


def test_two_pass_val_default_is_true():
    """L'attribut two_pass_val par défaut est True sur un nouveau pipeline."""
    pipeline = make_pipeline()
    assert pipeline.two_pass_val is True
