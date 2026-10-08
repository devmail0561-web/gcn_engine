# SLICE -1 — gate validation du signal d'apprentissage AVANT tout socle.
# Plan docs/refactory/plan-remodelage-tout-appris.md §5 étape -1 :
#   4 phrases, ~300 steps. GO ssi loss<0.20 ET acc 4/4 ET Δcos>=0.05.
#   Zone grise -> WARN-CONTINUE (+100 steps, 2 lr), pas STOP sec.
#
# Réutilise le harnais existant (aucune réinvention) :
#   GCNDataLoader + reps_from_sentence (gcn_python.data.loader),
#   CGNPipeline (pipeline/cgnp.py), MLPEncoder, RGCNLayer, WordEmbedding,
#   FeatureVocabulary. Même boucle que train.py (forward teacher-forcing
#   + loss + backward SGD), arch = défauts gcn-train (d_emb=128,
#   mlp 128/64, edge 256/128/64, 1x RGCN), sauf : dropout=0,
#   label_smoothing=0, weight_decay=0, --min-class-count 1 (4 phrases =>
#   seules les classes vues N>=1 survivent, comme train.py les masquerait).
#
# 4 phrases gold (gold_mini.jsonl, lot A) : 2 noeuds + 1 arete, chargement
# verifie 0 warning, relations toutes distinctes, toutes forward (0,1) :
#   s0555 cause | s0801 sequence | f0005 condition | s2028 opposition
#
# Métriques :
#   loss  = loss moyenne (pipeline.loss) sur les 4 phrases en EVAL
#           (forward SANS gold_edge_map, sans oracle = chiffre honnête).
#   acc   = phrases 100% correctes / 4 (tous types noeuds + relations).
#   Dcos  = separation embeddings : (intra-inter)_final - (intra-inter)_init,
#           cosinus sur les 8 vecteurs root_lemma (4 processus / 4 entite).
#           + spread global (distance cos moyenne) rapporté pour contexte.
# Seeds 42 et 123 (supportés : WordEmbedding/MLPEncoder/RGCNLayer).
#
# Rejouable : PYTHONPATH=gcn-python/src python3 scripts/slice_minus1_gate.py
from __future__ import annotations

import ast
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gcn-python" / "src"))

from gcn_python.constants import NODE_TYPES, RELATION_TYPES, rgcn_n_relations  # noqa: E402
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence  # noqa: E402
from gcn_python.layer1.embedding import WordEmbedding  # noqa: E402
from gcn_python.layer1.features import FeatureVocabulary  # noqa: E402
from gcn_python.layer2.reference import MLPEncoder  # noqa: E402
from gcn_python.layer3.reference import RGCNLayer  # noqa: E402
from gcn_python.pipeline.cgnp import CGNPipeline  # noqa: E402

IDS = ["s0555", "s0801", "f0005", "s2028"]  # cause, sequence, condition, opposition
STEPS = 300
LR = 0.05  # memorisation few-shot (tests: 0.01-0.1) ; le gate teste le signal, pas le lr defaut
SEEDS = [42, 123]
D_EMB = 128
EXTRA_STEPS = 100
EXTRA_LRS = (0.05, 0.015)  # "2 lr" zone grise

GO_LOSS, GO_ACC, GO_DCOS = 0.20, 4, 0.05
GRAY_LOSS, GRAY_ACC = 0.35, 3  # en-deca : STOP sec


def load_slice_samples():
    src = REPO / "gcn-datasets" / "DATA" / "supervision" / "gold_mini" / "gold_mini.jsonl"
    sents = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["id"] not in IDS:
                continue
            cir = o["cir"]
            if isinstance(cir, str):
                try:
                    cir = json.loads(cir)
                except json.JSONDecodeError:
                    cir = ast.literal_eval(cir)
            nodes = [{"id": n["id"], "type": n["type"], "label": n.get("label", ""),
                      "token_span": n.get("token_span")} for n in cir["nodes"]]
            edges = [{"sources": (e.get("sources") or [e.get("source")]), "target": e.get("target"),
                      "relation": e.get("relation"), "confidence": e.get("confidence", 1.0),
                      "explicit": True, "negated": False} for e in cir["edges"]]
            sents.append({"id": o["id"], "text": o["text"], "tokens": o["tokens"],
                          "cir": {"nodes": nodes, "edges": edges}})
    assert [s["id"] for s in sents] == IDS or sorted(s["id"] for s in sents) == sorted(IDS), sents
    by_id = {s["id"]: s for s in sents}
    sents = [by_id[i] for i in IDS]
    td = tempfile.mkdtemp(prefix="slice_m1_")
    Path(td, "train.json").write_text(json.dumps({"document": {"lang": "fr", "sentences": sents}},
                                                 ensure_ascii=False), encoding="utf-8")
    loader = GCNDataLoader(Path(td), shuffle=False)
    samples = list(loader)
    assert len(samples) == 4, f"attendu 4 samples, recu {len(samples)}"
    return samples


def build_pipeline(seed: int, samples):
    vocab = FeatureVocabulary()
    we = WordEmbedding(d_emb=D_EMB, seed=seed)
    lemmas = []
    for s in samples:
        reps, _, _ = reps_from_sentence(s.sentence)
        lemmas.extend(r.root_lemma for r in reps)
    we.build_vocab(lemmas)
    d_eff = vocab.d_clause_effective(D_EMB)
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), D_EMB)
    enc = MLPEncoder(d_clause=d_eff, d_edge=d_edge, edge_hidden=(256, 128, 64),
                     edge_dropout=0.0, seed=seed)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff,
                      n_relations=rgcn_n_relations(len(RELATION_TYPES), False),
                      dropout=0.0, seed=seed)
    pipe = CGNPipeline(encoder=enc, graph=graph, vocabulary=vocab, word_embedding=we)
    # masque classes aretes vides (train.py, --min-class-count 1)
    seen = {int(r) for s in samples for r in s.edge_map.values()}
    mask = np.array([i in seen for i in range(len(RELATION_TYPES))], dtype=bool)
    pipe.edge_logit_mask = mask
    return pipe, we


def train_gold(pipe, sample):
    """Un step SGD teacher-forcing (boucle train.py). Retourne loss."""
    reps, valid, conn = reps_from_sentence(sample.sentence)
    pipe.forward(reps, sample.sentence.text, clause_positions=valid,
                 n_total_clauses=len(sample.sentence.clauses),
                 connector_reps=conn, gold_edge_map=sample.edge_map)
    nl, el = pipe._cached_node_logits, pipe._cached_edge_logits
    gn = sample.gold_node_labels[np.array(valid, dtype=np.int64)]
    pairs = [(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]
    gf = np.array([sample.edge_map.get(p, -1) for p in pairs], dtype=np.int64)
    m = gf >= 0
    loss, dn, de = pipe.loss(nl, el[m] if m.any() else el[:0], gn, gf[m] if m.any() else None,
                             edge_logit_mask=pipe.edge_logit_mask)
    pipe.backward(dn, de, lr=train_gold.lr)
    return loss


def evaluate(pipe, samples):
    """Eval SANS oracle. Retourne (loss_moy, n_phrases_ok, details)."""
    losses, ok, det = [], 0, []
    for s in samples:
        reps, valid, conn = reps_from_sentence(s.sentence)
        pipe.forward(reps, s.sentence.text, clause_positions=valid,
                     n_total_clauses=len(s.sentence.clauses), connector_reps=conn)
        nl, el = pipe._cached_node_logits, pipe._cached_edge_logits
        gn = s.gold_node_labels[np.array(valid, dtype=np.int64)]
        pairs = [(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]
        gf = np.array([s.edge_map.get(p, -1) for p in pairs], dtype=np.int64)
        m = gf >= 0
        loss, _, _ = pipe.loss(nl, el[m] if m.any() else el[:0], gn, gf[m] if m.any() else None,
                               edge_logit_mask=pipe.edge_logit_mask)
        node_ok = bool((np.argmax(nl, axis=1) == gn).all())
        edge_ok = bool((np.argmax(el[m], axis=1) == gf[m]).all()) if m.any() else True
        good = node_ok and edge_ok
        ok += int(good)
        losses.append(loss)
        det.append({"id": s.sentence.id, "loss": round(float(loss), 4),
                    "nodes_ok": node_ok, "edge_ok": edge_ok})
    return float(np.mean(losses)), ok, det


def cos_stats(we, lemmas, types):
    idx = np.array([we._vocab[l] for l in lemmas])
    E = we._E[idx].astype(np.float64)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    S = E @ E.T
    n = len(lemmas)
    iu = np.triu_indices(n, 1)
    sims, same = S[iu], np.array(types)[iu[0]] == np.array(types)[iu[1]]
    intra, inter = sims[same].mean(), sims[~same].mean()
    dist = 1.0 - sims.mean()
    return float(intra - inter), float(dist)


def run_seed(seed: int, samples, steps: int, lr: float):
    pipe, we = build_pipeline(seed, samples)
    lemmas, types = [], []
    for s in samples:
        reps, valid, _ = reps_from_sentence(s.sentence)
        gn = s.gold_node_labels[np.array(valid, dtype=np.int64)]
        for r, g in zip(reps, gn, strict=True):
            lemmas.append(r.root_lemma)
            types.append(NODE_TYPES[int(g)])
    sep_init, spread_init = cos_stats(we, lemmas, types)
    train_gold.lr = lr
    rng = np.random.default_rng(seed)
    order = list(range(len(samples)))
    losses = []
    for step in range(steps):
        if step % len(samples) == 0:
            order = list(rng.permutation(len(samples)))
        losses.append(train_gold(pipe, samples[order[step % len(samples)]]))
    loss_ev, acc, det = evaluate(pipe, samples)
    sep_fin, spread_fin = cos_stats(we, lemmas, types)
    return {"seed": seed, "steps": steps, "lr": lr,
            "loss_train_last4": round(float(np.mean(losses[-4:])), 4),
            "loss_eval": round(loss_ev, 4), "acc": acc, "acc_detail": det,
            "dcos": round(sep_fin - sep_init, 4),
            "sep_init": round(sep_init, 4), "sep_final": round(sep_fin, 4),
            "spread_init": round(spread_init, 4), "spread_final": round(spread_fin, 4)}


def verdict(r):
    if r["loss_eval"] < GO_LOSS and r["acc"] == GO_ACC and r["dcos"] >= GO_DCOS:
        return "GO"
    if (r["loss_eval"] < GRAY_LOSS and r["acc"] >= GRAY_ACC) or \
       (r["loss_eval"] < GO_LOSS and r["acc"] == GO_ACC and 0.0 <= r["dcos"] < GO_DCOS):
        return "WARN-CONTINUE"
    return "STOP"


def main():
    t0 = time.time()
    samples = load_slice_samples()
    rels = sorted({e for s in samples for e in s.edge_map.values()})
    print(f"slice: {IDS} | relations={[RELATION_TYPES[i] for i in rels]} | "
          f"steps={STEPS} lr={LR} d_emb={D_EMB} arch=defaut-gcn-train")
    results = [run_seed(sd, samples, STEPS, LR) for sd in SEEDS]
    for r in results:
        r["verdict_300"] = verdict(r)
        print(f"seed={r['seed']} 300steps: loss_eval={r['loss_eval']} "
              f"loss_train_last4={r['loss_train_last4']} acc={r['acc']}/4 "
              f"dcos={r['dcos']} (sep {r['sep_init']}->{r['sep_final']}) "
              f"=> {r['verdict_300']}")
    # Zone grise -> +100 steps x 2 lr (pas STOP sec), meilleur loss_eval garde
    for r in results:
        if r["verdict_300"] == "WARN-CONTINUE":
            cands = [run_seed(r["seed"], samples, STEPS + EXTRA_STEPS, lr) for lr in EXTRA_LRS]
            best = min(cands, key=lambda c: (c["acc"] < 4, c["loss_eval"]))
            best["verdict_300"] = verdict(best)
            best["extension_of"] = f"300->400steps lrs={list(EXTRA_LRS)}"
            print(f"seed={r['seed']} extension: " +
                  " | ".join(f"lr={c['lr']} loss={c['loss_eval']} acc={c['acc']}/4 "
                              f"dcos={c['dcos']} => {c['verdict_300']}" for c in cands))
            r.update({k: best[k] for k in best if k not in ("extension_of",)})
            r["extension_of"] = best["extension_of"]
            r["verdict"] = verdict(r)
        else:
            r["verdict"] = r["verdict_300"]
    finals = [r["verdict"] for r in results]
    if all(v == "GO" for v in finals):
        final = "GO"
    elif any(v == "STOP" for v in finals):
        final = "STOP"
    else:
        final = "WARN-CONTINUE"
    dur = round(time.time() - t0, 1)
    out = {"ids": IDS, "steps": STEPS, "lr": LR, "seeds": SEEDS, "results": results,
           "verdict": final, "duree_s": dur}
    mp = Path(tempfile.gettempdir()) / "slice_minus1" / "metrics.json"
    mp.parent.mkdir(exist_ok=True)
    mp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"VERDICT SLICE -1 : {final} (seeds {dict(zip(SEEDS, finals, strict=True))}) | "
          f"metrics: {mp} | duree: {dur}s")
    return 0 if final == "GO" else (2 if final == "WARN-CONTINUE" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
