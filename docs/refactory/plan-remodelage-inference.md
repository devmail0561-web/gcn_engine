# Plan refactorisation — architecture cible refacto-2026-10 (volet inférence)

> Zoom DÉPLOIEMENT du plan UNIQUE (même architecture `refacto-2026-10`, mêmes R1-R10/T1-T6 que `plan-remodelage-tout-appris.md`). PAS une deuxième lignée ni une version : 11-14j au total partagés (hors BCE opt +1j, hors S4-opt) ; volet inférence seule = 5-6j inclus. Gate slice `-1` partagée (pas de reconstruction sur L1 non validé). Legacy = `.npz` sans `format_refacto` ; refacto = seul format écrit (`format_refacto=20261007`). `gcn-transformers` = HORS-CIBLE (T5) : gelé, 0 import au runtime (`grep gcn_transformers==0`, torch interdit socle s0) ; teacher offline seul autorisé (S4-opt).

## 1. Casse → cible
```text
texte → [binaire gcn + lattice UD] → 106 → CIR (subprocess 30s, None si absent)
texte → split fixe in-process → ids+E (N,d_model) → CIR (8/19, <19) + intent
```

## 2. Supprimer (8 blocs)
[1] `GCNLatticeParser`+subprocess → `tokenizer.split` + offsets.
[2] `FeatureVocabulary/d_clause/d_edge` → `d_model` + garde `:140-152`.
[3] `vectorize_*` → `forward(ids,mask)` (R1).
[4] Flags morts → whitelist `edge_threshold/temperature/theta/bfs_depth/edge_logit_mask` ; ajouter `theta+mask` à `from_pretrained`.
[5] Heuristiques UD → têtes + B, `explicit=False`, `None` si non-adjacents (R2 + fix `discourse.py:96-99`).
[6] `WordEmbedding` lemmes → `Embedding(vocab_tok,d_model)` + R5.
[7] Double split + `_LABEL_RE` + divergence → un seul `tokenizer.split` + unifiés avant bascule. Frontends par langue supprimés (T6) : 1 tokenizer + profils script (latin/CJK/arabe/autres) ; langue non vue au train → E_tok aléatoire (c'est D2 qui définit le support, pas du code).
[8] Raisonnement Python côté discussion → appel Rust [M1] : le raisonneur d'instructions, les rapports de requêtes et le décodeur à règles (chemin de réponse de `discuss.py`) sont supprimés après bascule. Chemin cible : session Python (état du dialogue) → `engine.analyze` → CIR JSON → sous-processus `gcn query --query ... --ir <fichier>` → UserResponse JSON affiché tel quel. Gardés en Python : moteur ML, session, routage des 3 actes, affichage. L'enveloppe always-answer [A2] est produite par le Rust (`respond()` Phase 8), le Python ne fait que l'afficher — jamais deux enveloppes.

## 3. Garder / seuils
Gardé : `trusted/guard`, `emit()`, `ReferenceDecoder`, masks, third dormants. `all_pairs` **non gardé**.
Seuils legacy ref `(0.0,0.65,1.0)` → grid F1, `assert != ref`, gel si <30/classe.

## 4. Reconstruire
[A] `analyze→split→forward_ids`, `training=False` aussi attention/dropout, `None` si vide/OOM. Fenêtre k=2.
[A2] [R-ALWAYS] Jamais de None/vide vers l'utilisateur : tout point d'entrée utilisateur retourne {phrase NL (langue = `cir.intent.lang`, [R-LANG]) + contenu partiel + statut ok|partiel|degrade + caveats + suggestions} — enveloppe PRODUITE par le Rust (`respond()` Phase 8 backend, décision [M1]), le Python ne fait que l'afficher. `analyze_or_skip`/None → enveloppe dégradée (batch : phrases ignorées comptées et rapportées, pas silencieuses) ; NLU/discourse None → « indéterminé » + caveat ; CIR vide → phrase + miettes (tokens, intent) + cause probable. Batch/CI : raise R3 inchangé (frontière interactive seule convertit).
[B] Format s0 D'ABORD (R4) : `format_refacto=20261007`, amendements s1+s2+s3 ensuite. `convert-legacy-UD` + test legacy/refacto (mtime + hash + forward identique).
[C] `stable_key→chunk_hash`, manifest `{tokenizer_hash, d_model}`, `assert vec_dim==manifest`.
[D] `route_intent` consomme `cir.intent` : P1 22 → P2 `expects_*` → P3 `?`. Concepts depuis chunks E. Lazy + no-cycle.
[E] `request→ack+file-action`, `question→execute`, `assert→add_cir`. `cir.intent` par bloc. Ack dans discuss.

## 5. Ordre — ordre unique partagé (même liste que `plan-remodelage-tout-appris.md` §5)
```text
-1. Slice / gate slice (1j, partagé) : 4 phrases, 300 steps. GO ssi loss<0.20
     ET acc 4/4 ET Δcos≥0.05. Si zone grise (ex 0.25) → WARN-CONTINUE
     (+100 steps, 2 lr), pas STOP sec.
0. Socle (3j en parallèle, partagé) : (a) B+format s0, (b) Rust (PR +
     roundtrip, contrat UserResponse Phase 8 backend à synchroniser),
     (c) gold_mini 60-100 + jeux held-out HORS docs.
1. L0 + tokenizer/segment + loader (1j).
2. L1 + forward_ids + d_model (3-5j).
3. Tête act + route intent (1j, après format s0).
4. Convertisseur legacy→refacto (inclus 0).
5. Loss + seuils grid F1 + chunk_hash (1-2j).
6. NLU sans UD + B (1j).
7. BCE opt (+1j, optionnel).
S4-opt. Teacher HF offline (optionnel, après S3, jamais bloquant,
  jamais au runtime) : cf volet training (benchmark GO/NO-GO puis
  dump .npz + distill). Inférence déployée reste NumPy pur CPU.
```

## 6. Acceptation (anti-gaming, alignée)
[A] `grep -rE "subprocess|which|gcn_bin|Lattice"==0` + never-called + analyze sans binaire.
[B] `cos_final<cos_init-0.05` + held-out non divulguées.
[C] 30 or + paraphrases + ablation `?` + coverage tête.
[D] legacy-UD→`exit!=0` convert-or-retrain ; convert dry-run mtime inchangé + sortie==gold hash + `grep final/==0`.
[E] `valid_mask>0`, OOV warn/raise. [F] 3 phrases + 3 chemins en log.
