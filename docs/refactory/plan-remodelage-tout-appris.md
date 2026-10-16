# Plan refactorisation — architecture cible refacto-2026-10 (volet training)

> Plan de refactorisation UNIQUE vers une NOUVELLE ARCHITECTURE (split fixe + E appris + tête act + cir.intent) en partant de la codebase actuelle. PAS une lignée de modèles v1→v2→v3→v4 (les mentions v1..v4 étaient des itérations du plan, purge faite). Les .npz actuels = legacy-UD (convertis one-shot vers format-refacto ou jetés, pas des ancêtres à incrémenter). Volets : ici training ; `plan-remodelage-inference.md` = zoom déploiement du MÊME plan.
> Contraintes : `embeddings + MLP/RGCN NumPy`, pas de Transformer/LLM. Cible : `texte → CIR` sans UD humain. Dims legacy : `106 / 234 / 247 / 1115` (`features.py:75-146`, garde `cgnp.py:140-152` — valeurs calculées, pas des littéraux ; vérifié 2026-10-08).

## 1. Tranchés
- [T1] B PRÉREQUIS : `spacy==3.8 + fr_core_news_sm` pin, lazy+timeout+fallback regex, `source=silver-B|fallback`, loss masquée en fallback, test sans-spacy passe.
- [T2] N-aires dormant coût 0 (grad 0 `train.py:839-848`). BCE seulement si exigés.
- [T3] Flux DÉCOUPLÉ (pas de dx via probs/two_pass/MHA). « E2E » = test intégration `texte→CIR` sans UD/binaire. Slots = pooling fixe + routage explicite.
- [T4] L2 produit `cir.intent` → NLU consomme `cir.intent` (2 temps). 3 actes : `assert→stocker, question→répondre, request→ack+file-action` (0 requête graphe causale). `cir.intent` porte aussi `lang` COMPRISE (pas configurée, pas de flag) : défaut `fr` ; message utilisateur dans une autre langue → `lang` détectée sur le message ; demande explicite (« réponds en wolof ») → bascule. [R-LANG].
- [T5] `gcn-transformers` = SUPPORTÉ MOTEUR (décision 2026-10-11, remplace « HORS-CIBLE gelé ») : HF drop-in (XLM-R 280M/CamemBERT 110M/CodeBERT 125M + torch), même barre que gcn-python (audit + gate). Socle s0 inchangé : `torch`/`transformers` interdits dans gcn-python, garde CI `grep -r gcn_transformers gcn-python/src == 0` maintenue. Usages : encodeur alternatif + teacher S4-opt + témoin baseline.
- [T6] Moteur AGNOSTIQUE langue : 1 seul moteur, 0 frontend par langue (fini lattice fr/en + suffixes + UD). Différences apprises dans `E_tok`, pas codées. L0 = 1 tokenizer + profils par FAMILLE DE SCRIPT (latin / CJK sans espaces / arabe RTL / autres, ~0.5j regex, pas un frontend) + normalisation par script (latin : minuscule+sans-accent ; autres : NFC + règles propres). Vocab tok multilingue unique (ou par script si collision) + `tokenizer_hash`. Ajouter une langue = ajouter des DONNÉES (D2), pas du code. Silver intent par famille de ponctuation (`?` vs `؟` vs `？`, `¿...?` espagnol, inversion) sinon `assert` par défaut.

## 2. Schéma
```text
texte → L0 lex_split (À CRÉER) → ids+offsets → L1 E_*+slots fixes
  → L2 MLP 8/19 + tête act (1,3) → L3 RGCN → loss+B → CIR + intent
```

## 3. Règles
- [R1] `backward_ids` miroir ; caches `.copy()` → `ids→offsets` ; pooling figé, jamais d'argmax différencié. Backward cause `src>tgt` : garder l'arête (inverser le routage, pas `warn+continue` `loader.py:206-222`).
- [R2] Construire `reps/connecteurs` depuis chunks. Non-adjacents : `connecteur=None`. **Fix** `discourse.py:96-99` : émettre `juxtaposition`, jamais `cause` sans preuve. Longue distance (0,2) : paire gold explicite si marqueur, sinon assumée perdue en fenêtre k=2.
- [R3] CI-strict : `raise` remplace `warn` (`loader.py:127`, `train.py:58,879`) pour cut>50% et gold None>20%.
- [R4] Format D'ABORD, minimal puis amendé : socle `s0` (`format_refacto=20261007` seuls) → `s1` +tok_vocab/hash → `s2` +d_model → `s3` +act_types. `convert-legacy-UD --dry-run` + dual-read transitoire + `ban final/` + test legacy/refacto.
- [R5] `tokenizer_hash=sha256(règles+vocab)` ; ids-stables AVANT vocab. Raise train/val si >5% ; prod : warn + taux en metadata + backoff `_unk` (warn 5%/raise 20%).
- [R6] Lazy + test no-cycle. `engine.py:38` étendu à `checkpoint+cgnp+loader`.
- [R7] Fenêtre k=2 + `bfs_depth` ; `all_pairs` interdit.
- [R8] Masks N>=30 ; calibration : gate N>=30/classe (pas 10), sinon gel.
- [R9] Matrice silver 30 ex AVANT tête ; **weight 0.0 par défaut**, 0.3 seulement si erreur≤20%. PR Rust `Option<CirIntent>` + roundtrip AVANT tête. Gold AVANT B : `!` exclamatif≠request, `Si+Pres`→assert assumé, question sans `?`→assert assumé.
- [R10] Grid F1 (pas ECE), `assert != (0.0,0.65,1.0)`, gel si <30/classe.

## 4. Compat legacy-UD → refacto (one-shot, pas de lignée)
- Legacy = tout .npz sans `format_refacto` (dims 106/234/247/1115, vocab lemmes, têtes 22) → refusé en eval (`exit!=0`, message `legacy-UD : convertir ou ré-entraîner`), convertible one-shot (`convert-legacy-UD`, dry-run : mtime/hash inchangés ; réel : sortie==gold hash + forward avant==après).
- Refacto = seul format écrit (`format_refacto=20261007`, `tok_E/vocab/hash`, `d_model`, `act_types:[3]`). Aucun incrément v1→v2 : un seul format cible ; s0..s3 = étapes de socle.

## 5. Étapes — ordre unique partagé (11-14j total hors BCE opt +1j, hors S4-opt ; même liste que `plan-remodelage-inference.md` §5)
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
S4-opt. Teacher HF offline GO/NO-GO (optionnel, après S3, jamais bloquant) : benchmark `benchmark_comparison.py --encoders mlp,xlmroberta` (gold figé, seeds 42/123, eval() prouvé, teacher GPU / élève CPU). GO ssi `ΔF1 val≥+0.05 ET held-out≥+0.03 ET Δnode≥-0.02 ET stable 2 seeds ET coût<4h GPU ET élève≤1.10× CPU` → dump `scripts/dump_teacher_logits.py` (.npz : node/edge logits bruts T=1.0 + valid_idxs/edge_pairs/data_hash/meta, dry-run non-écrivant) → `training/distill.py` (`loss=α·CE+(1-α)·T²·KL`, T∈[2,4], α∈[0.5,0.9], ~30 lignes NumPy, injection après `loss()` avant `backward()`, même mask/weights). Sinon NO-GO = gel pur (sortie normale).
```

## 6. Acceptation (anti-lettre-sans-esprit)
- [A] `grep -rE "subprocess|shutil.which|gcn_bin|Lattice" == 0` + `subprocess.run` jamais appelé (monkeypatch) + `analyze` sans binaire N≥1.
- [B] `cos_final < cos_init-0.05` (seeds 42/123) ET `intra-inter>0.15` sur 6 paires + 6 held-out non divulguées.
- [B2] 3 phrases + 30 or R9 held-out + paraphrases + ablation `?` + coverage tête act + `grep "?"==0` dans route.
- [C] Gold `data/gold_mini/` jamais vu au train.
- [D] legacy-UD→`exit!=0` + message convert-or-retrain ; refacto→charge/sauve ; convert dry-run vert (non-écriture prouvée) ; `grep final/==0`.
- [E] SCHEMA dims + act (1,3).
- [G] Loss décroît + `loss(w-eps·d)<loss(w)` + finite-diff E_tok.
