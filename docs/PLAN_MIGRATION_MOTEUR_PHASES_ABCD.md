# Plan d'action exhaustif — Refonte GCN-Core (D1-D10)

## STATUS — Progression au 2026-09-27

### Implémenté (commit d9ad8ca)

| Tâche | Statut | Notes |
|-------|--------|-------|
| B0.1 bridge.py:187 has_advcl | ✅ Fait | Était déjà implémenté avant cette session |
| A.0 schéma tokens (id, dep_head, form) | ✅ Fait | Déjà dans loader.py/bridge.py avant cette session |
| A.1 sentence_type.py réécriture | ✅ Committé | 309 lignes, 0 lemme dur, LangMarkers, 7 champs |
| A.2 lang_markers.json | ✅ Committé | FR + EN, 9 catégories subordonnants |
| A.3 layer1/__init__.py exports | ✅ Committé | Tous les types exportés |
| A.4 sentence_profile property | ✅ Committé | representation.py |
| B.1 constants.py | ✅ Committé | NODE_TYPES 8, RELATION_TYPES 19, Voice/PronType |
| B.2/B.3 features.py d_clause 79→106 | ✅ Committé | +Voice/PronType/12 positionnels/5 ternaires |
| B.3 flags --no-positional/--no-ternary | ✅ Committé | Ablation par config (gate C.7) |
| label_builder.py accès par nom | ✅ Committé | Bug silencieux corrigé |
| bridge.py NODE_TYPE_TO_POS/DEP par nom | ✅ Committé | Couvre les 8 types v3.0 |
| C-min edge.rs 19 RelationType | ✅ Committé | 8 nouveaux variants, is_joint() |
| C-min edge.rs joint_group_id | ✅ Committé | Option<String> serde(default), déterministe sha256 |
| 7 sites CausalEdge Rust patchés | ✅ Committé | joint_group_id: None ajouté, build propre |
| C.2 edge_norm.py _norm_single + multi-src | ✅ Committé | normalize_edge → list[dict] sur 2 sources |
| D.0-min d0_min_oov_split.py | ✅ Committé | 30 phrases OOV extraites, lexique 3 577 lemmes |
| test_layer1.py hardcode 79 corrigé | ✅ Committé | Assert dynamique vocab.d_clause |

### Restant à implémenter

| Tâche | Priorité | Bloquant pour |
|-------|----------|--------------|
| `training/bootstrap.py:_normalize_edge` patch joint_cause | 🔴 Haute | Gate C : chemin bootstrap contourne le fix |
| `scripts/init_v3_stub.py` He-init checkpoints v2→v3 | 🔴 Haute | Gate B+C : merge bloqué sans |
| Protocole v3.0 : masquer 8 logits vides dans `train.py` | 🔴 Haute | P2 : dilution softmax sinon garantie |
| `gcn-transformers/base.py:66` : guard `== 79` → `== 106` | 🔴 Haute | Crash shape silencieux sinon |
| `tests/test_sentence_type.py` (~60 tests) | 🟠 Moyenne | Gate A (60 tests verts) |
| `tests/test_layer1.py` +5 tests B.5 | 🟠 Moyenne | Gate B (test_d_clause_equals_106, voice, prontype…) |
| `tests/test_ir_emitter.py` +6 tests C.4 | 🟠 Moyenne | Gate C (19 types, joint_group_id…) |
| `tests/test_cross_lingual.py` T1 logique | 🟠 Moyenne | Gate D |
| `tests/test_bridge_fix.py` T4 e2e | 🟠 Moyenne | Gate D (nécessite D.0-min livré ✅) |
| `tests/test_calibration.py` T5 | 🟠 Moyenne | Gate D |
| T5-min : optimiser temperature sur val | 🟡 Basse | §PARIS_HYPOTHESES (avant isotonie) |
| D6-shadow : émettre confidence_ml + confidence_d6 | 🟡 Basse | §PARIS_HYPOTHESES |
| `REGLE_EQUILIBRE_DATASET.md` 8 nœuds/19 relations | 🟡 Basse | Gate B+C |
| Inventaire dims dérivées (grep 207/877/== 79) | 🟡 Basse | Gate B+C |
| A.5-A.6 instructions.py + discuss.py | 🟡 Basse | Seulement si frontend UD disponible |
| D.0-médical 25 phrases OOV (owner Michel) | 🟡 Basse | Si D.0-min F1 < 70% |

---

## Contexte

Le moteur GCN-Core a un écart de ~70% entre la spécification théorique (ETUDE_LINGUISTIQUE_NLU.md, D1-D10 gelées) et le code actuel. La dette est concentrée sur 4 périmètres : NLU/Layer1, FeatureVector, modèle ternaire, et preuves empiriques. L'objectif est une migration strangler incrémentale — le socle Pearl (Rust, 20+ requêtes) reste intact.

**Séquençage à valeur d'information maximale :**

1. **D.0 en premier** (1-2j) — annotation 25 OOV + 20 conditionnelles. Débloque T2+T4 donc toute publication. `gcn-datasets/test/` n'existe pas encore. Chaque semaine de retard sur D.0 retarde D d'autant. SPOF : owner unique Michel.
2. **Phase B0 + T3** (~1j) — `bridge.py:187` (1 ligne, non-breaking, has_advcl) + test T4 unitaire + T3 ablation Mood. Signal bon marché avant tout breaking. **Kill criteria : voir §KILL_CRITERIA.**
3. **Phase A** (indépendante, en parallèle si possible).
4. **Phases B+C** seulement si les signaux de (2) sont non-négatifs — voir §KILL_CRITERIA. Phase E chiffrée avant merge de C (C sans supervision ternaire = actif sans usage).
5. **Phase D** : gate final — T1-T5 verts, kill criteria tous explicitement évalués.

**Deux trajectoires :**
- **Plan-moins (défaut, ~10-14j)** : B0+T3 → D.0-min (split lexique, 0.5j) → B+C-min (Voice/PronType + flags + C-min joint) → T5-température-first → D6-shadow. Chaque extension coûteuse (D.0 médical, A NLU, positionnels, third, isotonie, D6 in-place) ne s'ouvre que sur signal mesuré via K1-K3.
- **Plan-complet (~17-25j)** : toutes les phases telles que décrites. Justifié si les 4 paris documentés en §PARIS_HYPOTHESES s'avèrent vrais.

**La trajectoire par défaut est le plan-moins.** Les annexes conditionnelles de chaque phase indiquent quand basculer vers le complet.

**Décisions ETUDE non encore couvertes (D3, D4, D7) :** voir §DECISIONS_NON_COUVERTES en fin de document.

---

## PHASE B0 — Bridge fix immédiat (non-breaking, avant tout)

**Périmètre :** `frontend/bridge.py:187` uniquement
**Objectif :** corriger has_advcl (aujourd'hui 0.0 systématique en inférence) et mesurer le signal le moins cher du plan
**Durée estimée : 0.5 jour**

### B0.1 — Corriger `bridge.py:187`

```python
# AVANT : has_advcl=False,
# APRÈS :
has_advcl=(node.get("node_type", "") == "condition"),
```

Zéro breaking, zéro schéma changé, mergeable immédiatement. C'est le seul fix qui peut faire bouger edge F1 en **inférence/eval** aujourd'hui.

**Borne d'effet :** `loader.py:302` calcule déjà le vrai `has_advcl` depuis les tokens annotés lors de l'entraînement — B0 ne change donc pas les métriques train, seulement la parité inférence. Effet attendu : réduction du biais bridge→modèle, pas un gain de F1 train. T4 mesurera l'écart.

### B0.2 — T3 : Ablation Mood (ici, pas en Phase D)

Sur corpus annoté gold existant : classifier CONDITION vs CONCESSION avec et sans feature Mood. Mesurer ΔF1. **Kill criteria §KILL_CRITERIA K1.** Résultat documenté dans CHANGELOG.

### B0.3 — T4 unitaire (préliminaire)

Test unitaire direct : `node_type=condition` → `has_advcl=True`. Coût zéro. La partie end-to-end de T4 (20 phrases conditionnelles annotées) reste en Phase D après D.0.

**Expiry B0 :** B0 change l'inférence sans réentraîner — skew train/inférence assumé (modèle entraîné avec valeurs annotées, inférence avec heuristique bridge). Ce skew est acceptable le temps que T4 e2e soit évalué. **Délai max : T4 e2e doit être exécuté dans les 30 jours suivant le merge de B0.** Au-delà sans résultat, revert B0 ou réentraîner.

---

## PHASE A — NLU / Layer1

**Périmètre :** `gcn-python/src/gcn_python/layer1/` + `verbalizer/instructions.py` + `discuss.py`
**Objectif :** classify(tokens, markers) sans lemmes durs, routing NLU automatique (D1)
**Durée estimée : 5-7 jours** (A.0 inclus ; A.5-A.6 sont sur chemin critique uniquement si un frontend UD réel alimente discuss.py)

**A-min (défaut) :** A.0 (0.5j) + A.1-A.4 + A.7 (tests). Soit : schéma tokens, classify() UD pur, LangMarkers, 60 tests. Durée ~3-4j.

**Basculer vers A-complet (A.5-A.6, +1-2j) seulement si :** un pipeline UD réel produit des tokens et peut alimenter `discuss.py` en production — sinon A.5 et A.6 restent inactifs (le plan le dit L174 : le chemin reste `_format_response` sans frontend UD).

### A.0 — Étendre le schéma `UDRepresentation.tokens` (bloquant A.4 et B.3)

**État actuel :** `representation.py:14` — `tokens: list[dict]` = `{lemma, pos, dep_rel, morph}`. Pas de `id`, pas de `dep_head`, pas de `form`.

**Pourquoi bloquant :**
- A.1/A.4 : `_classify_type` utilise `nsubj.id > root.id` (inversion) et `form` (ponctuation `?/!`). Sans `id` et `form`, `sentence_profile` retourne quasi-toujours DECLARATIVE.
- B.3 : 6 des 12 features positionnelles (`subj_before_root`, `relative_depth`, etc.) exigent `id` et `dep_head`. Inimplémentables avec le schéma actuel.

**Découverte :** les données existent et `json_reader.py` les lit déjà. C'est `loader.py:253` et `loader.py:292` qui jettent `id`, `dep_head`, `form` en construisant `{lemma, pos, dep_rel, morph}` uniquement. A.0 = **~3 lignes dans loader.py + bridge.py**, pas une migration dataset.

**Tâche (réévaluée à 0.5j) :**
- `loader.py:253` : `{"lemma": tok.lemma, "pos": tok.pos, "dep_rel": tok.dep_rel, "morph": tok.morph}` → ajouter `"id": getattr(tok, "id", -1), "dep_head": getattr(tok, "dep_head", -1), "form": getattr(tok, "form", "")`.
- Idem `loader.py:292`.
- `frontend/bridge.py` : propager `id` et `dep_head` depuis les nœuds CIR si disponibles (sinon `-1`).
- Gate fonctionnel (remplace le grep vacuitaire) : `assert "id" in reps_from_sentence(sample_sentence)[0].tokens[0]`.

### A.1 — Réécrire `sentence_type.py` (D1)

**Pré-requis :** `sentence_type.py` est actuellement non versionné (`?? gcn-python/src/gcn_python/layer1/sentence_type.py` au git status). Commiter le fichier actuel avant A.1 pour que le diff de la réécriture soit lisible et que le gate `grep _LEMMAS = 0` soit falsifiable par comparaison.

**État actuel :** 198 lignes, hardcoded lemma lists (L70-74, L76-77, L125, L148-149, L170-173), 4 champs SentenceProfile, classify(tokens) sans markers, zéro Complexity/SubordinationType.

**Supprimer :**
- `_INTERROGATIVE_LEMMAS_FR/EN` (L70-74)
- `_ESTCE_QUE` (L76-77)
- Branche lemma dans `_classify_type` (L123-129) — unreachable car L120 capture `?`
- Pattern "est-ce que" par lemma (L146-153)
- Lemmes négatifs dans `_classify_polarity` (L170-173)

**Nouveaux enums :**
```python
class Complexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"

class SubordinationType(str, Enum):
    NONE = "none"
    CONDITION = "condition"    CAUSE = "cause"         CONCESSION = "concession"
    TEMPORAL = "temporal"      MOTIVATION = "motivation"  FILTER = "filter"
    SEQUENCE = "sequence"      OPPOSITION = "opposition"  ENABLE = "enable"
```

**Nouveau dataclass `LangMarkers` (frozen) :**
```python
@dataclass(frozen=True)
class LangMarkers:
    interrogative_lemmas:   frozenset[str]  = frozenset()
    negation_particles:     frozenset[str]  = frozenset()
    restriction_patterns:   tuple           = ()
    subordination_markers:  dict            = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> "LangMarkers": ...
    @classmethod
    def load(cls, path) -> "LangMarkers": ...
```

**SentenceProfile enrichi — 7 champs :**
```python
@dataclass(frozen=True)
class SentenceProfile:
    sentence_type:  SentenceType
    polarity:       Polarity
    voice:          Voice
    modality:       Modality
    complexity:     Complexity           # NOUVEAU
    subordination:  SubordinationType    # NOUVEAU
    has_restriction: bool                # NOUVEAU
    # propriétés dérivées : expects_response, expects_action, asserts_fact,
    # is_negated, is_passive, is_complex, has_condition
```

**Signature classify mise à jour :**
```python
def classify(tokens: list[dict], markers: LangMarkers | None = None) -> SentenceProfile:
```

**Logique couche 1 (UD structurel pur — zéro lemme) :**
- `_classify_type` : `punct=?`, `PronType=Int` dans morph, `nsubj.id > root.id`, `Mood=Imp`, `¬nsubj + Person in {1,2}`
- `_classify_polarity` : `Polarity=Neg` dans root_morph ou token morph (pas de lemme)
- `_classify_voice` : `Voice=Pass` ou `dep_rel in {aux:pass, nsubj:pass}`
- `_classify_modality` : `Mood in {Sub, Cnd, Imp}`
- `_classify_complexity` : `dep_rel in {advcl, ccomp, xcomp, acl}` sur VerbForm=Fin
- `_classify_subordination` : couche 1 = NONE ; couche 2 si markers = matcher SCONJ.lemma
- `_classify_restriction` : couche 1 = False ; couche 2 si markers = pattern ["ne","que"]

**Fallback couche 2 (markers optionnels) :**
- Interrogatif : si `PronType=Int` absent → tester `lemma in markers.interrogative_lemmas`
- Négatif : si `Polarity=Neg` absent → tester `lemma in markers.negation_particles + dep_rel=advmod`

### A.2 — Créer `gcn-datasets/configs/lang_markers.json`

Lexiques FR + EN pour LangMarkers : `interrogative_lemmas`, `negation_particles`, `restriction_patterns`, `subordination_markers` (condition, cause, concession, temporal, motivation, filter, sequence, enable, opposition). Voir ETUDE §11.3 pour la structure complète.

### A.3 — Mettre à jour `layer1/__init__.py`

**État actuel :** 2 lignes, exporte rien.
**Après :** exporter `SentenceProfile`, `SentenceType`, `Complexity`, `SubordinationType`, `LangMarkers`, `classify as classify_sentence`.

### A.4 — Ajouter `sentence_profile` sur `representation.py`

Après la propriété `is_negative` (L39) :
```python
@property
def sentence_profile(self):
    from .sentence_type import classify
    return classify(self.tokens)  # couche 1 seule, sans markers
```

### A.5 — Modifier `execute()` dans `instructions.py`

**État actuel :** `execute(self, text: str) -> str | None` — L395. `instructions.py` opère sur du CIR symbolique (Rust), pas sur des tokens UD bruts.

**Contrainte de séparation des couches :** `tokens` ici doit être fourni par l'appelant à partir d'un vrai parser UD (bridge ou frontend) — jamais reconstruit depuis les nœuds CIR (cf. A.6). La méthode reste un dispatcher : elle ne fait pas d'extraction linguistique.

**Après :**
```python
def execute(self, text: str,
            tokens: list[dict] | None = None,
            markers: "LangMarkers | None" = None) -> str | None:
    cmd, arg = parse_command(text)
    if cmd == "text":
        if tokens is not None:
            from ..layer1.sentence_type import classify
            profile = classify(tokens, markers)
            return self._dispatch_by_profile(profile, text)
        return None   # comportement inchangé sans tokens
    # ... if-chain existant inchangé ...
```

**Nouvelles méthodes privées (après L447) :**
- `_dispatch_by_profile(profile, text)` → router selon expects_response / expects_action
- `_query_from_text(text)` → heuristique sujet + find_causes/find_effects
- `_instruct_from_text(text)` → hook impératif (MVP : return None)

### A.6 — Modifier `discuss.py`

**`_format_response()` (L107) :** Après dispatch DSL (L116-130), avant heuristique sujet :
```python
is_question = question.strip().endswith("?")
```

**Fallback engine (L410-426) :** Passer `tokens` depuis le contexte UD si disponible.
Si tokens UD réels absents, **ne pas construire de tokens synthétiques depuis les nœuds CIR** (tout NOUN + morph vide retournera quasi toujours DECLARATIVE — le routing NLU ne s'activera pas). Appeler `_format_response` directement :
```python
# Si un parser UD a fourni des tokens dans le contexte, les passer ici.
# Sinon, bypasser le routing NLU — il nécessite de vrais tokens UD.
nlu_resp = None
if ud_tokens is not None:
    nlu_resp = tmp_handler.execute(user_input, tokens=ud_tokens, markers=markers)
if nlu_resp is None:
    nlu_resp = _format_response(tmp_handler, user_input)
```

**Note :** Le routing NLU via A.5 ne sera pleinement actif que lorsqu'un frontend UD alimente `discuss.py` en tokens réels. En production actuelle (bridge heuristique), le chemin `_format_response` reste le chemin principal.

### A.7 — Tests Phase A

**Nouveau `tests/test_sentence_type.py` (~60 tests) :**
- TestSentenceProfile (10) : propriétés, frozen, complexity, restriction
- TestClassifyTypeUD (8) : PronType=Int, Mood=Imp, inversion, !
- TestClassifyTypeMarkers (5) : fallback lemme FR/EN avec LangMarkers
- TestClassifyPolarity (4) : Polarity=Neg root/token, défaut
- TestClassifyVoice (4) : Voice=Pass, aux:pass, nsubj:pass
- TestClassifyComplexity (6) : advcl, ccomp, xcomp, acl, SCONJ+mark
- TestClassifySubordination (7) : condition/cause/concession avec markers
- TestClassifyRestriction (4) : ne...que avec markers, sans markers=False
- TestLangMarkers (4) : from_json(), load()
- TestEdgeCases (5) : tokens vides, 4 clés, 7 clés, clés manquantes
- TestNoHardcoding (3) : grep — zéro liste de lemmes dans le module

**`tests/test_instructions.py` (+5 tests) :**
- `test_execute_text_without_tokens_returns_none`
- `test_execute_text_with_interrogative_tokens`
- `test_execute_dsl_ignores_tokens`
- `test_execute_text_with_declarative_returns_none`
- `test_execute_text_with_conditional_tokens`

---

## PHASE B — FeatureVector

**BREAKING — v3.0.** `d_clause` 79→106 (source unique : `FeatureVocabulary.d_clause` — ne jamais recalculer à la main), `NODE_TYPES` 7→8, `RELATION_TYPES` 11→19. Invalide tous les checkpoints v2 et les tests qui assertent les dims actuelles. Ne pas merger sans C ni sans le script d'initialisation (B.6).

**Périmètre :** `constants.py`, `layer1/features.py`, `frontend/bridge.py`
**Objectif :** FeatureVector §11.3 complet — Voice, PronType, positionnels, lemma_emb (D6, D10)
**Durée estimée : 3-4 jours**

### B.1 — Mettre à jour `constants.py`

**Ajouter :**
```python
UD_VOICE_VALUES    = ["Act", "Pass", "_absent"]
UD_PRONTYPE_VALUES = ["Int", "Rel", "Prs", "Dem", "Ind", "Art", "_absent"]
# 6 valeurs UD + _absent — ETUDE §7.2 : Prs/Int/Rel/Dem/Ind/Art
```

**Remplacer `NODE_TYPES` (7→8, D5) :**
```python
NODE_TYPES = ["processus","etat_local","etat_global","entite","condition",
              "concept","evenement","contrainte"]
```
Note : noms ASCII sans accents — cohérence avec Rust `node.rs` (serde `snake_case` : `Etat→etat`, `EtatSystemique→etat_systemique`). Le renommage `etat→etat_local`, `etat_systemique→etat_global` exige :
1. Alias serde Rust : `#[serde(alias = "etat")]` sur `EtatLocal`, `#[serde(alias = "etat_systemique")]` sur `EtatGlobal` — pour lire les CIR existants.
2. **Matchs exhaustifs `node.rs`** : `causal_direction()` (L34-45) + tout match Rust sur `NodeType` doivent être mis à jour pour les 8 variants (y compris `Contrainte` nouveau). `cargo test` détectera les matchs non-exhaustifs.
3. **`bridge.py:48-68`** : `NODE_TYPE_TO_POS` et `NODE_TYPE_TO_DEP` sont indexés par position dans `NODE_TYPES[i]`, pas par nom. Le renommage change l'ordre de la liste → change silencieusement la sémantique des mappings sans erreur visible. **Vérifier et mettre à jour les dicts** après chaque modification de `NODE_TYPES`. Risque : **Fort**, pas Faible.

**Remplacer `RELATION_TYPES` (11→19, D2) :**
```python
RELATION_TYPES = [
    "cause","condition","enable","prevent","concession","motivation",
    "sequence","filter","opposition","analogy","counterfactual",  # 11 directes
    "conditional_cause","mediated_cause","joint_cause",           # 3 ternaires
    "conditional_prevent","mediated_prevent","joint_prevent",     # 3 *_prevent
    "data_dependency","control_dependency",                       # 2 systémiques
]  # 19 total
```

### B.2 — Étendre `FeatureVocabulary` dans `features.py`

**État actuel :** `d_clause` = 79 dims (18+38+5+5+4+5+1+3).
Note : `UPOS_TAGS` contient 18 chaînes + `_unk` déjà inclus dans les 18 (constants.py:19-20 — le commentaire « 18+_unk=19 » est un double-comptage). Vérification : commit db8ed20 « dims 207/877 » → 207−128 = 79.

**Ajouter 2 champs :**
```python
voice_values:    list[str] = field(default_factory=lambda: list(UD_VOICE_VALUES))
prontype_values: list[str] = field(default_factory=lambda: list(UD_PRONTYPE_VALUES))
```

**Mettre à jour `d_clause` :**
```python
@property
def d_clause(self) -> int:
    base = (len(self.upos_tags) + len(self.dep_rels) + len(self.subject_pos_cats)
            + len(self.tense_values) + len(self.aspect_values) + len(self.mood_values)
            + len(self.voice_values) + len(self.prontype_values) + 1 + 3)
    positionals = 12   # Éq.9 — voir B.3
    ternary_flags = 5  # voir B.3
    return base + positionals + ternary_flags
    # = 79 + 3 + 7 + 12 + 5 = 106 dims (sans embeddings)
    # Source unique : FeatureVocabulary.d_clause — ne jamais recalculer à la main
    # Dims dérivées à mettre à jour dans B.6 : 79→106 entraîne 207→234, 877→904, etc.
```

### B.3 — Étendre `vectorize_clause()` dans `features.py`

**Delta réel :** `3×d_emb` (lemma_emb sujet/verbe/objet) existe déjà via `--subject-object-emb` et `d_clause_effective()`. Le vrai nouveau code est Voice, PronType, les 12 positionnels, et les 5 ternaires — soit 27 dims, pas 27+lemma_emb. Ne pas réimplémenter `_pool_tokens` / `_lemma_features` déjà présents.

**Flags d'ablation par config (une ligne d'implémentation, économise un aller-retour) :** dès l'implémentation de B.3, ajouter `--no-positional-features` et `--no-ternary-flags` dans le CLI (train.py). Cela permet de faire les ablations C.7 par config, pas par patch du code. Coût : ~30 lignes. Gain : C.7 ne nécessite pas de re-déploiement.

Après le bloc `[has_object, has_advcl, has_temporal_obl]`, ajouter :
```python
# Voice one-hot (couche 1 UD)
voice_val = rep.root_morph.get("Voice", "_absent")
voice_vec = _one_hot(voice_val, vocab.voice_values)

# PronType — premier token portant PronType dans morph
pron_type_val = "_absent"
for t in rep.tokens:
    pt = t.get("morph", {}).get("PronType", "")
    if pt:
        pron_type_val = pt
        break
pron_vec = _one_hot(pron_type_val, vocab.prontype_values)

# 12 features positionnelles normalisées (Éq.9)
# [0] subj_before_root : 1.0 si le sujet précède le verbe racine, 0.0 sinon
# [1] subj_after_root  : 1.0 si le sujet suit le verbe racine
# [2] advcl_before_root : 1.0 si l'advcl est à gauche du verbe racine
# [3] advcl_after_root  : 1.0 sinon (exclusive avec [2])
# [4] obj_before_root   : 1.0 si l'objet précède le verbe racine
# [5] obj_after_root    : 1.0 sinon
# [6] neg_before_root   : 1.0 si marqueur de négation précède le verbe
# [7] neg_after_root    : 1.0 si marqueur de négation suit le verbe
# [8] connector_before_advcl_head : 1.0 si le SCONJ/mark est avant la tête advcl
# [9] sentence_initial  : 1.0 si token_span[0] == 0 (clause en début de phrase)
# [10] sentence_final   : 1.0 si token_span[1] == len(sentence_tokens)-1
# [11] relative_depth   : profondeur du verbe racine dans l'arbre / longueur max, ∈ [0,1]
span_len = max(1, rep.token_span[1] - rep.token_span[0])
pos_features = _compute_positional_features(rep.tokens, rep.token_span, span_len)  # float[12]

# 5 features ternaires (encodent la présence de structures ternaires dans la clause)
# [0] has_two_sources     : 1.0 si deux nsubj/agent distincts dans la clause
# [1] has_third_cond      : 1.0 si advcl de type conditionnel présent
# [2] has_obl_mediator    : 1.0 si complément obl exprimant un intermédiaire
# [3] has_joint_marker    : 1.0 si connecteur coordinatif (cc) liant deux agents
# [4] has_modal_condition : 1.0 si Mood=Cnd ou Mood=Sub sur le verbe principal
ternary_features = _compute_ternary_features(rep.tokens)  # float[5]
```

Si `word_embedding is not None` (couche 2) :
```python
# lemma_emb[3×d_emb] — D10 ; utiliser word_embedding.d_emb, jamais 128 en dur
d_emb = word_embedding.d_emb
subj_lemma  = next((t["lemma"] for t in rep.tokens if t.get("dep_rel")
                    in ("nsubj","nsubj:pass")), "")
lemma_emb_src  = word_embedding.lookup(subj_lemma)   # float[d_emb]
lemma_emb_verb = word_embedding.lookup(rep.root_lemma)
lemma_emb_dst  = word_embedding.lookup(next(
    (t["lemma"] for t in rep.tokens if t.get("dep_rel")=="obj"), ""))
```

### B.4 — Corriger `bridge.py` lignes 187 et 203

**L187 dans `_rep_from_cir_node()` :**
```python
# AVANT : has_advcl=False,
# APRÈS :
has_advcl=(node.get("node_type", "") == "condition"),
```

**L203 dans `_build_connector_rep()` :** rester `False` — le connecteur est un token isolé, pas une clause.

### B.5 — Tests Phase B

**`tests/test_layer1.py` (+5 tests) :**
- `test_d_clause_equals_106`
- `test_voice_encoded_passive`
- `test_prontype_int_encoded`
- `test_has_advcl_condition_node_true` (bridge fix)
- `test_lemma_emb_shape_uses_d_emb` (vérifier shape = (3, word_embedding.d_emb), pas (3,128))

**Note gate :** Les tests existants qui assertent `d_clause == 80\|== 79` — vérifier si dynamiques (la plupart le sont déjà) ; seuls les asserts statiques sont à adapter pour refléter 106. Ce n'est pas une régression — c'est la migration breaking documentée.

### B.6 — Migration v2→v3 (obligatoire avant merge)

- `scripts/init_v3_stub.py` (pas "migrate" — c'est une initialisation, pas une migration de poids) : convertir checkpoints `.npz` v2 (79 dims) → stub v3 (dims recalculées via `FeatureVocabulary.d_clause`, poids nouveaux initialisés par **He-init** : `np.random.randn(...) * np.sqrt(2 / d_in)`) avec avertissement explicite "stub non-entraîné — réentraînement obligatoire".
- Archiver `model_baseline.npz`, `model_train_v2.npz` dans `checkpoints/archive_v2/`.
- Bump `schema_version` dans le format checkpoint (champ `"schema": "3.0"`).
- **Inventaire dims dérivées et accès positionnels** : avant merge, grep dans tout le repo :
  ```bash
  grep -rn "== 79\b\|== 80\b\|207\b\|877\b\|NODE_TYPES\[\|RELATION_TYPES\[" \
    --include="*.py" --include="*.rs" --include="*.md" --exclude-dir=target
  ```
  Fichiers connus : `gcn-transformers/base.py:66` (`== 79`), `quick_test.py:14-15` (`207`/`877`), `pipeline/label_builder.py` (`NODE_TYPES[`), `bridge.py` (`NODE_TYPES[`). Corriger dans ce même PR.
- Vérifier que les labels gold dans `gcn-datasets/` sont en **strings** (pas en indices numériques) — `grep -r '"node_type": [0-9]' gcn-datasets/` doit retourner 0.
- Ajouter une assertion d'intégration dans `test_layer1.py` : `assert d_effective == vocab.d_clause_effective(d_emb)` pour détecter toute future divergence.
- Mettre à jour `REGLE_EQUILIBRE_DATASET.md` : 8 nœuds, 19 relations, quotas ±10% pour les nouveaux types ternaires, N_min documenté. **Owner : Michel, gate : avant merge B+C.**

---

## PHASE C — Modèle ternaire

**BREAKING — v3.0** (idem B). Ne merger qu'avec B.

**Périmètre :** `gcn-core/crates/gcn-ir/src/edge.rs`, `data/edge_norm.py`, `pipeline/ir_emitter.py`, + fichiers dérivés listés en C.4-bis et C.4-ter.
**Objectif :** CIREdge avec third?, 19 types de relations, JOINT_CAUSE sans perte de nécessité conjointe (D2)
**Durée estimée : 4-5 jours** (+1j pour les 8 sites CausalEdge Rust, cf. C.4-ter)

**C-min (défaut, ~2j) :** 19-enum + `joint_group_id` déterministe + groupement Pearl + 8 sites Rust. Données `hyperedge_map` existantes permettent de tester le joint immédiatement. **Ne pas implémenter `third`, le détecteur C.6, ni les 6 classes CONDITIONAL/MEDIATED/JointPrevent/etc. avant d'avoir les données D4.**

**Basculer vers C-complet (C.1 `TernaryThird`, C.3 9-tuple, C.6 détecteur) seulement si :** Phase E chiffrée avec données annotations CONDITIONAL/MEDIATED estimées. Sans données, `third` = code mort supervisé par rien.

### C.1 — Étendre `edge.rs` (Rust)

**État actuel :** `RelationType` = 11 variants. `CausalEdge` sans `third`.

**`DataDependency` et `ControlDependency` existent déjà** (edge.rs:21-22 dans les 11 variants actuels). Ne pas les ré-ajouter — doublon = erreur de compilation Rust.

**Ajouter les 8 vrais nouveaux variants à `RelationType` :**
```rust
Analogy,
Counterfactual,
ConditionalCause,
MediatedCause,
JointCause,
ConditionalPrevent,
MediatedPrevent,
JointPrevent,
```
Total : 11 existants + 8 nouveaux = **19 variants**.

**Choix JOINT_CAUSE (acté) :** JOINT_CAUSE est encodé comme **2 arêtes séparées portant le même `joint_group_id`** — pas comme une liste `sources[]` dans une seule arête. Raison : `CausalEdge` garde sa structure scalaire (un `source`, un `target`) ; la nécessité conjointe est reconstruite côté requête en groupant par `joint_group_id`. C'est plus simple que modifier le struct Rust pour `Vec<NodeId>`, et lisible dans Pearl.

**Ajouter `joint_group_id` à `CausalEdge` :**
```rust
#[serde(skip_serializing_if="Option::is_none")]
pub joint_group_id: Option<String>,  // sha256(target|sorted_sources)[:16], déterministe — pas uuid
```

**Ajouter `TernaryThird` + `TernaryRole` :**
```rust
#[derive(Debug,Clone,PartialEq,Serialize,Deserialize)]
#[serde(rename_all="snake_case")]
pub enum TernaryRole { Condition, Mediator }

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct TernaryThird {
    pub role: TernaryRole,
    pub node: NodeId,
    #[serde(skip_serializing_if="Option::is_none")]
    pub polarity: Option<String>,   // "negative" → Règle 2 algèbre §9.4
}
```

**Étendre `CausalEdge` :**
```rust
#[serde(skip_serializing_if="Option::is_none")]
pub third: Option<TernaryThird>,
```

### C.2 — Corriger `edge_norm.py` ligne 131

**État actuel :** `src_raw = src_raw[0]` — perd silencieusement toutes les sources après l'index 0.

**Refactoring requis :** `normalize_edge` (L99) est monolithique. Avant d'ajouter la logique multi-sources, la découper :
1. Extraire `_norm_single(e: dict) -> dict | None` — le corps actuel de `normalize_edge`.
2. Réécrire `normalize_edge` comme wrapper qui appelle `_norm_single` ou retourne une liste.

**Après refactoring :**
```python
if len(src_raw) == 1:
    src_raw = src_raw[0]
elif len(src_raw) == 2:
    # JOINT_CAUSE : retourner 2 arêtes normalisées avec joint_group_id commun
    # Déterministe : même phrase normalisée deux fois → même id (reproductibilité, évaluation gold/pred)
    import hashlib
    key = f"{e.get('target','')}|{'|'.join(sorted(str(s) for s in src_raw))}"
    jgid = hashlib.sha256(key.encode()).hexdigest()[:16]
    e1 = dict(e); e1["source"] = src_raw[0]; e1["relation"] = "joint_cause"; e1["joint_group_id"] = jgid
    e2 = dict(e); e2["source"] = src_raw[1]; e2["relation"] = "joint_cause"; e2["joint_group_id"] = jgid
    return [_norm_single(e1), _norm_single(e2)]
else:
    warnings.warn(f"edge_norm: {len(src_raw)} sources, 2 max supportées")
    src_raw = src_raw[0]
```

`normalize_edge` retourne maintenant `list[dict] | dict | None`. Mettre à jour :
- `data/loader.py` : aplatir les listes.
- `training/bootstrap.py:_normalize_edge` (L168) : **fonction distincte**, chemin bootstrap indépendant — patcher séparément, sinon le fix joint_cause est contourné sur le chemin bootstrap.

### C.3 — Mettre à jour `ir_emitter.py`

**État actuel :** 6-tuple `(src, dst, relation, confidence, negated, marker_token)`.

**Après — 9-tuple rétrocompatible (6→9 direct, pas de branche 8-tuple qui n'a jamais existé) :**
```python
if len(tup) == 6:
    src, dst, relation, confidence, negated, marker_token = tup
    third_role = third_node = joint_group_id = None
else:
    # 9-tuple (nouveau format)
    src, dst, relation, confidence, negated, marker_token, third_role, third_node, joint_group_id = tup

attrs = {
    "relation": relation, "confidence": confidence,
    "negated": negated, "marker_token": marker_token,
    "third": {"role": third_role, "node": third_node} if third_role else None,
    "joint_group_id": joint_group_id,
}
```

### C.4 — Tests Phase C

**`tests/test_ir_emitter.py` (+6 tests) :**
- `test_ternary_edge_conditional_cause` — third dans le CIR émis
- `test_ternary_edge_mediated_cause`
- `test_joint_cause_two_edges_from_two_sources` — joint_group_id identique sur les 2 arêtes
- `test_edge_norm_joint_cause_group_id_preserved`
- `test_19_relation_types_in_constants`
- `test_8_node_types_in_constants`

### C.6 — Détecteur ternaire dans le pipeline (bloquant : production des 9-tuples)

**Problème :** C.1-C.3 créent le stockage ternaire, mais personne ne produit les 9-tuples. `pipeline/cgnp.py` émet aujourd'hui des 6-tuples ; sans détecteur, `third` et `joint_group_id` restent toujours `None`.

**Tâche :**
- Ajouter dans `pipeline/cgnp.py` un détecteur `_detect_ternary(edge, tokens) -> (third_role, third_node, joint_group_id)` basé sur les heuristiques UD :
  - `third_role=Condition` si l'arête a un `advcl` SCONJ conditionnel dans les tokens source.
  - `third_role=Mediator` si un `obl` avec fonction instrumentale est présent.
  - `joint_group_id` : si deux arêtes distinctes partagent la même cible et que leurs sources sont liées par `cc`, leur assigner un id déterministe `sha256(target|sorted(sources))[:16]` — pas un UUID aléatoire (reproductibilité, évaluation gold/pred joignable).
- Supervision : `train.py:503-508` collecte `hyperedge_map` mais ne le supervise pas. Ajouter une loss auxiliaire sur les paires `joint_group_id` (contrastive ou BCE simple) — ou assumer explicitement que Phase C = stockage seul, supervision en follow-up (Phase E).
- Requêtes Pearl : les requêtes `CHAIN/SPOF/counterfactual` dans `pearl.rs` doivent grouper les arêtes `JointCause` par `joint_group_id` pour reconstruire la nécessité conjointe. Sans ce groupement, `joint_group_id` est stocké mais jamais exploité.

**Contrainte `all_pairs` :** `cgnp.py` construit les paires d'arêtes en mode adjacent par défaut (i<j, L387-391). Le détecteur joint (deux arêtes, même cible, sources liées par `cc`) a besoin de voir les deux co-sources simultanément — elles peuvent être non-adjacentes. Le détecteur doit soit exiger `all_pairs=True`, soit effectuer un passage dédié sur les arêtes groupées par cible, indépendamment du mode de pairage. Documenter cette limitation dans C.6.

**Protocole d'entraînement v3.0 à classes vides (obligatoire, rapatrié de Phase E) :** B+C ajoute 8 logits à support ~0. Sans protocole, la masse de probabilité fuit vers ces 8 logits non supervisés → confiances des 11 classes utiles baissent mécaniquement → ECE (T5) se dégrade → K2 peut se déclencher sur un artefact d'entraînement, pas sur l'utilité des features.

Deux options, choisir avant le premier entraînement v3.0 :
- **Option A (recommandée) : masquer/geler les 8 logits** — entraîner sur les 11 classes connues, activer les 8 à N_min (condition D4). Implémentation : `train.py` masque les colonnes 11-18 du softmax jusqu'à `epoch_unfreeze`.
- **Option B : coarse-first** — première phase 11 classes, deuxième phase 19 classes avec poids initiaux de la phase 1. Plus coûteux, plus propre.

Documenter le choix dans B.6 (script `init_v3_stub.py`) et dans le CHANGELOG. Sans ce point, l'option implicite est softmax 19 voies avec 8 zéros — résultat garanti bruité.

**Choix explicite à documenter avant implémentation :** Phase C = stockage + détecteur heuristique, supervision Pearl en Phase E. Sinon chiffrer +3-5j supplémentaires.

### C.4-ter — Consommateurs CausalEdge + label_builder (silencieux, dangereux)

**8 sites Rust à réparer (erreurs de compilation — détectées par `cargo build`) :**

| Fichier | Ligne | Action |
|---------|-------|--------|
| `gcn-frontend-fr/src/emitter.rs` | 60 | Ajouter `third: None, joint_group_id: None,` |
| `gcn-frontend-en/src/emitter.rs` | 59 | Idem |
| `gcn-frontend-code/src/common.rs` | 81 | Idem |
| `gcn-knowledge/src/inference.rs` | 407 | Idem |
| `gcn-ir/src/lib.rs` | 104 | Idem |
| `gcn-middleend/tests/integration_middleend.rs` | 31 | Idem |
| `gcn-backend/tests/integration_backend.rs` | 46 | Idem |

Ces sites n'ont pas de `..Default::default()`. Après ajout de `third` et `joint_group_id`, ils ne compilent plus. `cargo build` les signale — inclure dans le gate C. Ajouter `+1j` à la durée C (3-4j → 4-5j).

**`#[serde(default)]` sur `third` et `joint_group_id` :** nécessaire pour lire les vieux CIR (champs absents). Déjà prévu par `skip_serializing_if`, mais `#[serde(default)]` est symétrique à la déserialisation. Vérifier que edge.rs le porte (ajouter si absent).

**`label_builder.py` — bug silencieux (pas de compilation, sémantique fausse) :**
```python
# ACTUEL — indices positionnels fragiles :
_NT_ACTION     = NODE_TYPES[1]   # "action" → devient "etat_local" après réordre
_NT_ENTITE     = NODE_TYPES[5]   # "entite" → devient "concept" après réordre
_NT_ETAT_SYS   = NODE_TYPES[6]   # "etat_systemique" → devient "evenement" après réordre

# APRÈS — accès par nom :
_NT_CONDITION   = "condition"
_NT_ENTITE      = "entite"
_NT_ETAT_SYS    = "etat_systemique"
_NT_ACTION      = "action"
_NT_TRANSITION  = "transition"
```
Aucune erreur levée, sémantique fausse partout sauf `[4]=condition` (survit par chance). Corriger avant tout test B.

**`gcn-transformers/base.py:66` + `quick_test.py:14-15` :**
- `base.py:66` contient une garde `if d_clause == 79` (avec message citant 207) — deviendra fausse pour 106 silencieusement.
- `quick_test.py:14-15` : `207` et `877` en dur. À inclure dans l'inventaire B.6 (`grep -rn "== 79\b\|207\b\|877\b\|NODE_TYPES\[\|RELATION_TYPES\["`).

**Gold datasets et lexique :** vérifier avant merge B que les labels gold sont en strings (pas en indices numériques) — si indices, un remap est nécessaire après réordre de NODE_TYPES. Une ligne de vérification dans B.6 suffit.

### C.4-bis — Périmètre Rust/Python complet (19 relations)

Les fichiers suivants nécessitent une mise à jour pour couvrir les 19 variants — non cités en C.1-C.3 :

| Fichier | Raison |
|---------|--------|
| `gcn-core/crates/gcn-ir/src/query.rs` | ~20 match exhaustifs sur RelationType |
| `gcn-core/crates/gcn-ir/src/pearl.rs` | Requêtes Pearl typées sur RelationType |
| `gcn-python/src/gcn_python/constants.py` | RELATION_TYPES_INV (bidirectionnel 19→38 types) |
| `gcn-python/src/gcn_python/data/loader.py` | hyperedge_map, aplatissement normalize_edge |
| `gcn-python/src/gcn_python/evaluation/eval_runner.py` | filtre `final/` + nouveaux types |

Vérifier que les `match` Rust sont exhaustifs (`cargo test` détecte les variants manquants).

### C.5 — Orphelins D6 et Éq.7

Ces deux décisions gelées n'ont aucune tâche dans le plan. Elles doivent être implémentées en Phase C :

**D6 — Formule de confiance unique (`ir_emitter.py`) :**
Transcrire à l'identique depuis `ETUDE_LINGUISTIQUE_NLU.md §D6` (décision gelée) — **ne pas inventer de valeurs hors-ETUDE**. L'ETUDE spécifie les bases par rung, les facteurs `f(Mood)` et `g(connecteur)` avec leurs valeurs numériques exactes. Si une valeur est absente de l'ETUDE, ouvrir un ticket d'amendement ETUDE avant d'implémenter.

**D6-shadow (défaut, remplacement irréversible évité) :** pendant 1 version, émettre les deux champs dans le CIR : `confidence_ml` (actuel) et `confidence_d6` (formule ETUDE). Comparer sur val avant de supprimer `confidence_ml`. Si D6 > ML sur les 11 classes → basculer. Sinon → amender D6 dans l'ETUDE.

**Éq.7 — Normalisation voix (avant construction CIR) :**
Avant émission CIR, si `Voice=Pass` → inverser Agent/Patient dans le tuple `(src, dst)` pour que `src` soit toujours l'Agent logique. Implémenter dans `ir_emitter.py` avant la construction `attrs`.

**Ordre d'application avec D7 :** appliquer **D7 d'abord** (orientation source=subordonnée SCONJ), puis **Éq.7** (inversion voix passive). Raisonnement : D7 fixe le rôle sémantique des clauses (quelle clause est la cause), Éq.7 corrige ensuite l'agent logique au sein de la clause source. Inverser l'ordre produirait un `src` erroné que D7 réorienterait ensuite de façon incohérente.

**Ordre D6 → isotonie (T5) :** la pipeline de confiance finale est : `brut ML → D6 (rung×f×g) → isotonie (T5) → confidence_cal`. D6 s'applique en sortie d'`ir_emitter.py` avant sauvegarde dans le CIR ; la calibration isotonique s'applique lors de la lecture du CIR par `gcn-eval`. Cet ordre garantit que T5 calibre les sorties D6, pas les logits bruts — sinon la garantie ECE<0.15 est annulée sur les sorties finales.

**Test obligatoire — combinaison passif+conditionnel** (`test_ir_emitter.py`) :
```python
def test_voice_d7_order_passive_conditional():
    # Phrase : « Le médicament est prescrit si les traitements échouent »
    # D7 : source = clause SCONJ « si les traitements échouent »
    # Éq.7 : dans la clause cible passive, inverser Agent/Patient
    # Résultat attendu : src=traitements_clause, dst=médicament_agent_logique
    # (PAS src=médicament_sujet_grammatical qui serait l'inverse)
    ...
```

---

### C.7 — Analyse post-merge B+C (attribution du gain)

**Pourquoi :** B+C invalide 28 fichiers de tests, les checkpoints, le cache SHA, et ajoute 27 dims. Si edge F1 monte, on doit savoir pourquoi — sinon on ne sait pas quoi conserver si un composant casse plus tard.

**Métrique de référence :** macro-F1 sur les **11 classes partagées** v2↔v3 (pas la macro-F1 19 voies, diluée par 8 classes à support ~0). Rapport séparé pour les 8 nouvelles classes.

**Méthode — analyse de sensibilité (zero-out à l'inférence) :** désactiver successivement chaque groupe (dims à 0 sur le modèle v3.0 entraîné) et mesurer ΔF1 sur les 11 classes partagées :

| Groupe désactivé | Dims retirées | ΔF1(11 classes) attendu |
|------------------|---------------|------------------------|
| Voice (3 vals) | −3 | ? |
| PronType (7 vals) | −7 | ? |
| 12 positionnels | −12 | ? |
| 5 ternaires | −5 | ? |
| third / joint_group_id | struct Rust, pas dims | ? |

**Interprétation honnête :** zero-out mesure la sensibilité du réseau entraîné à une rupture train/test, pas la valeur intrinsèque de la feature (un réseau peut compenser — ΔF1≈0 n'implique pas que la feature est inutile). Seuil ΔF1 < 0.5% = **indicatif seulement** → candidat à re-ablation complète (réentraînement sans le groupe) si le budget le permet. Ne pas décider de supprimer un groupe sur la seule analyse de sensibilité.

**Outillage existant :** `metrics.py:42-87` contient déjà `edge_f1_per_class` et `node_f1_per_class` avec filtre `support > 0`. La macro filtrée des 11 classes partagées est obtenue en passant `class_subset=RELATION_TYPES_V2` — pas de nouveau code de métriques à écrire. Brancher `--per-class-report` de `gcn-eval` sur ce filtre plutôt qu'ajouter de nouvelles métriques.

**Kill criteria §KILL_CRITERIA K2** : si edge F1 (11 classes) post-B+C < F1_v2 + 1%, ouvrir Phase E-architecture avant publication.

---

## PHASE D — Preuves (gate T1-T5)

**Périmètre :** annotation manuelle + nouveaux fichiers de tests
**Objectif :** valider H1-H7, gate avant publication (D9)
**Durée estimée : 3-4 jours** (dont ~1-2j d'annotation)

### D.0 — Pré-annotation (bloquant pour T2 et T4)

**D.0-min (défaut, 0.5j, zéro annotation manuelle) :**
`generated_1000.json` a un lexique fermé (48 noms/50 verbes). Extraire par script les exemples contenant ≥ 1 verbe/nom hors de ce lexique → `gcn-datasets/test/oov_split_test.json`. Utiliser comme T2 provisoire. Pour T4 provisoire : filtrer les conditionnelles du gold-219 existant.

Si D.0-min donne node F1 > 70% → pari "OOV-domaine > OOV-lexique" est non nécessaire, ne pas dépenser D.0-médical.
Si D.0-min donne node F1 < 70% → D.0-médical justifié (annotation manuelle, owner Michel).

**D.0-médical (conditionnel, 1-2j) :**
25 phrases médicales OOV réelles (domaine distinct du corpus). Format : `gcn-datasets/test/oov_medical_25.json`. **Owner : Michel. Ne démarrer qu'après D.0-min.**

### D.1 — T1 logique : Généralisation structurelle (fixtures)

`tests/test_cross_lingual.py` — tokens UD allemands construits manuellement en fixture (`Mood=Sub`, `SCONJ lemma="wenn"`, `dep_rel="mark"`) → `SubordinationType.CONDITION` sans avoir vu le lemme en entraînement.

**Périmètre de cette preuve :** valide que `classify()` détecte CONDITION via la structure UD, indépendamment de la langue. Ce test ne prouve pas la robustesse end-to-end sur un vrai corpus DE.

**T1 DE (futur) :** preuve de généralisation réelle sur un parser UD allemand. Pas planifié dans ce document — dépend d'un frontend DE (stanza/de ou equivalent) non disponible dans le repo.

### D.2 — T2 : Hors-lexique OOV

25 phrases médicales OOV annotées (D.0 prérequis). Critère : node F1 > 70% sur ces phrases.

**Protocole deux conditions (obligatoire) :**
- **Condition A — sans fastText** : embeddings `_unk`/zéro pour les lemmes OOV. Mesure la robustesse des features UD structurelles seules.
- **Condition B — avec fastText** (`--fasttext subword`) : subword coverage, quasi-zéro OOV réel. Mesure fastText, pas les 27 dims UD.

Rapporter les deux F1 séparément. Interpréter :
- Si A > 70% : les features UD suffisent, H2 (lexique vs structure) tranchée en faveur de la structure.
- Si A < 70% mais B > 70% : le gain vient de fastText, pas de B+C — signal pour Phase E (features paire ou enrichissement lexical).
- Si les deux < 70% : revoir le seuil ou les annotations D.0.

### D.3 — T3 : Ablation Mood

Sur corpus annoté gold : classifier CONDITION vs CONCESSION avec et sans feature Mood. Mesurer ΔF1. Résultat documenté dans CHANGELOG, pas de cible fixe (test falsificateur H3).

### D.4 — T4 : Impact bridge fix

`tests/test_bridge_fix.py` :
- `node_type=condition` → `has_advcl=True` (test unitaire direct)
- Évaluer edge F1 sur 20 phrases conditionnelles annotées (D.0 prérequis) avec/sans la correction.
- **Critère :** edge F1 après fix ≥ edge F1 avant fix − 3%. Une dégradation supérieure à 3% signale une régression à investiguer.

### D.5 — T5 : Calibration

**T5-min (défaut, 0.5j) :** `temperature` existe déjà dans `cgnp.py:52,100,520`. Optimiser `temperature` sur le val set (grid search ou binary search sur ECE), mesurer ECE. Si ECE < 0.15 avec température → pari "isotonie > température" non nécessaire.

**T5-isotonie (conditionnel) :** seulement si ECE-température > 0.15.

`tests/test_calibration.py` : dès N_val ≥ 100, calibrer et vérifier ECE < 0.15.

**Hook calibration :** `checkpoint.py` sérialise en `.npz` (arrays numpy). Stocker un objet `sklearn.IsotonicRegression` complet n'est pas compatible avec `.npz` sans pickle, et le repo a un gate anti-RCE pickle explicite (`guarded_np_load`).

**Solution : stocker les paramètres nus** (pas l'objet sklearn) :
```python
# Après calibration isotonique (sklearn en mémoire seulement) :
np.savez(..., calibrator_x=iso.X_thresholds_, calibrator_y=iso.y_thresholds_)

# Au chargement : reconstruire en 2 lignes sans sklearn :
from numpy import interp
# Fallback obligatoire — checkpoint v3 pré-calibration ou checkpoint v2 : clés absentes
if "calibrator_x" in data and "calibrator_y" in data:
    confidence_cal = interp(confidence_raw, data["calibrator_x"], data["calibrator_y"])
else:
    confidence_cal = confidence_raw  # identité — pas de calibration
```
Cela reste dans les arrays numpy, compatible avec le gate existant. sklearn n'est requis que pendant l'entraînement (pas au chargement/inférence). Le fallback identité évite le `KeyError` sur `gcn-eval` avec un checkpoint v3 non encore calibré.

---

---

## PARIS_HYPOTHESES — Ce qu'il faut croire pour payer le plan complet

Chaque phase coûteuse du plan complet repose sur un pari. Ces paris ne sont pas documentés ailleurs. Si un pari est faux, la phase correspondante devient du shelfware — code testé, jamais activé en prod.

| Pari | Phase concernée | Coût si faux | Signal de vérification |
|------|----------------|--------------|----------------------|
| OOV-domaine (médical) apporte plus que OOV-lexique (split existant) | D.0-médical (1-2j) | 1-2j + SPOF Michel | D.0-min d'abord : si split discrimine, annuler |
| Un frontend UD réel alimentera `discuss.py` prochainement | A NLU complet (5-7j) | 5-7j de routage inactif en prod | Vérifier si pipeline UD existe avant de coder A.5-A.6 |
| Les 12 positionnels / 5 ternaires portent du signal au-delà de Voice/PronType | B dims complet | Découvert via C.7 (sensitiv.) | Flags `--no-positional` / `--no-ternary-flags` + C.7 |
| Des annotations CONDITIONAL/MEDIATED arriveront vite après C | C-complet (third, détecteur, 6 classes) | Code mort sans données ni supervision | Données D4 estimées avant de coder C-complet |
| Isotonie bat la température de > 1 point ECE sur N_val~100 | T5-isotonie | Dépendance sklearn, code interp, clés checkpoint | T5-min (température) d'abord |
| D6 > conf ML dès J1 sur val | D6 in-place (irréversible) | Remplacement sans comparaison | D6-shadow (deux champs pendant 1 version) |

---

## KILL_CRITERIA — Branches d'arrêt explicites

Ces critères doivent être évalués aux points indiqués. Si un critère déclenche, la branche suivante du plan est suspendue jusqu'à décision explicite de Michel.

### K1 — T3 ΔF1(Mood) ≈ 0 (évaluer en Phase B0, avant B+C)

**Signal :** ablation Mood → ΔF1 < 1% sur CONDITION vs CONCESSION.

**Conséquence :** l'hypothèse "Mood distingue condition/cause" est fausse. Tout l'édifice Mood s'effondre :
- Features PronType/Mood de B.3 : les 7 dims Mood sont à supprimer ou à réaffecter.
- `f(Mood)` de D6 : devient un facteur constant, formula simplifiée.
- §7.4 ETUDE ("Sub vs Ind distingue condition/cause") : ouvrir un ticket d'amendement ETUDE.

**Action :** avant de merger B+C, décider explicitement si les dims Mood restent. Ne pas merger en silence.

### K2 — edge F1 post-B+C < F1_v2 + 1% (évaluer en C.7)

**Signal :** après réentraînement v3.0 complet, edge F1 sur les **11 classes partagées** (comparables v2↔v3) n'a pas progressé de plus de 1 point vs baseline v2.

**Métrique exacte :** macro-F1 calculée uniquement sur les 11 classes présentes en v2 et en v3 — pas la macro-F1 19 voies (diluée mécaniquement par 8 classes à support ~0, F1=0). Rapport séparé obligatoire : « 8 nouvelles classes : support N, F1 attendue ~0, non considéré comme échec ».

**Conséquence :** le plafond edge était l'architecture paire (information paire manquante : [h_u;h_v;h_u−h_v;h_u⊙h_v], edge-features connecteur, convolution orientée arête — cf. `note.txt`), pas les features nœuds.

**Action :** ouvrir **Phase E-Architecture** avant toute publication. Ne pas publier v3.0 comme une amélioration si edge F1 (11 classes) n'a pas bougé.

### K3 — T4 dégrade > 3% (évaluer en Phase D)

**Signal :** edge F1 sur 20 phrases conditionnelles après bridge fix < F1_avant − 3%.

**Conséquence :** le fix has_advcl cause un distribution shift dans les cas où la condition était précédemment codée autrement (ou les phrases conditionnelles du corpus d'éval ne sont pas représentatives des conditions bridge).

**Action :** autopsie bornée 0.5j — distinguer (a) régression réelle (retirer B.4 ou le conditionner à un seuil de qualité bridge), (b) bruit d'éval (augmenter D.0 à 40 phrases). Si non résolu en 0.5j : reporter B.4 à Phase E avec une analyse plus large.

---

## DECISIONS_NON_COUVERTES — D3, D4, D7

Ces trois décisions gelées de l'ETUDE n'ont aucune tâche dans ce plan. Elles ne bloquent pas les Phases A-D telles que décrites, mais elles doivent être planifiées avant Phase E (ou avant toute mise en production).

### D3 — FILTER : relation directe vs CONDITIONAL_CAUSE

**Ce qui manque :** aucun test ne valide la distinction `FILTER` (« n'est prescrit que si… ») vs `CONDITIONAL_CAUSE`. Le modèle peut apprendre à confondre les deux sans que les gates A-D le détectent.

**Tâche à ajouter (Phase E ou follow-up) :**
- Annoter 10 exemples pivot FILTER (phrases restrictives FR + EN).
- Ajouter `test_filter_vs_conditional_cause.py` : vérifier que l'émetteur émet `FILTER`, pas `CONDITIONAL_CAUSE`, sur ces phrases.
- Critère : F1 FILTER > 60% sur les 10 pivots.

### D4 — Entraînement coarse→fin, N_min, bascule par type

**Ce qui manque :** aucune stratégie de bascule coarse/fine dans `train.py`. Avec peu de données ternaires (0 pour JOINT, <5 pour CONDITIONAL), le modèle convergera sur les types fréquents et ignorera les ternaires.

**Tâche à ajouter (Phase E) :**
- Implémenter dans `train.py` une phase coarse (11 relations actuelles) puis fine (19 relations), avec `N_min` par type comme seuil de bascule.
- Documenter `N_min` dans `REGLE_EQUILIBRE_DATASET.md` (valeur à définir avec Michel — probablement N_min ≥ 20 par type ternaire).

### D7 — Direction causale invariante (source = subordonnée SCONJ)

**Ce qui manque :** l'orientation des arêtes dans `ir_emitter.py` n'est pas invariante. La règle D7 (source = clause SCONJ subordonnée, sauf SEQUENCE où source = clause avant) n'est implémentée nulle part.

**Tâche à ajouter (Phase C ou follow-up) :**
- Ajouter dans `ir_emitter.py` une fonction `_orient_edge(src_clause, dst_clause, relation, tokens)` qui applique D7 avant construction du tuple.
- Cas SEQUENCE : source = clause chronologiquement antérieure (by token position).
- Test : `test_edge_orientation_d7.py` — vérifier sur 5 exemples cause/condition/sequence que l'orientation est correcte indépendamment de l'ordre des phrases dans le texte source.

---

## ALIGNEMENT SATELLITES (parallèle)

**`REGLE_EQUILIBRE_DATASET.md` :** 8 nœuds, 19 relations, quotas ±10% pour les nouveaux types ternaires, N_min/fine documentés, val set séparé obligatoire. **Owner : Michel. Gate : avant merge B+C.**

**`PLAN_NLU_SENTENCE_TYPE.md` :** aligner Phases 1-3 sur la nouvelle implémentation, SubordinationType avec 10 valeurs, signature `classify(tokens, markers=None)`.

---

## GATE D9 — Conditions de merge par phase

```
Phase A mergeable (indépendant) :
  - A.0 : grep -r '"id"' gcn-datasets/ | wc -l > 0
  - test_sentence_type.py : 60 tests verts
  - test_instructions.py existants (42) : 0 régression
  - grep sentence_type.py pour "_LEMMAS" = 0 résultat
  - lang_markers.json valide et chargeable

Phases B+C mergeables ensemble (breaking v3.0) :
  - Prérequis : kill criteria K1 et K2 évalués et non déclenchés
  - Prérequis : Phase E chiffrée (détecteur supervisé + loss joint + Pearl groupé) — sinon C = actif sans usage
  - scripts/init_v3_stub.py livré et testé (He-init, pas zéro)
  - REGLE_EQUILIBRE_DATASET.md mis à jour (8 nœuds / 19 relations)
  - test_layer1.py : tests d_clause statique : `grep -rn "== 80\b\|== 79\b\|== 107\b" tests/` → 0 résultat (la plupart des asserts sont dynamiques — vocab.d_clause)
  - has_advcl=True pour node_type=condition (T4 unitaire passé)
  - lemma_emb shape = (3, word_embedding.d_emb) — pas hardcodé 128
  - test_ir_emitter.py : 6 nouveaux tests verts
  - len(RELATION_TYPES) == 19
  - len(NODE_TYPES) == 8
  - joint_group_id identique sur les 2 arêtes JointCause
  - bootstrap._normalize_edge patché (chemin bootstrap couvert)
  - cargo test gcn-ir verts (match RelationType exhaustifs — 8 nouveaux, pas DataDependency/ControlDependency déjà présents)

Phase D — gate final avant toute publication :
  - D.0 annotation livrée (25 OOV + 20 conditionnelles)
  - Kill criteria K1/K2/K3 tous évalués et documentés (même s'ils ne déclenchent pas)
  - T1 logique : fixtures manuelles DE → SubordinationType.CONDITION
  - T2 : 25 OOV, node F1 > 70%
  - T3 : ΔF1 Mood documenté dans CHANGELOG
  - T4 : edge F1 bridge fix ≥ F1_avant − 3%
  - T5 : ECE < 0.15 sur N_val ≥ 100
  - C.7 ablation livrée (table ΔF1 par groupe de features dans CHANGELOG)

Note : T2 ne bloque pas A. A peut merger dès que son propre gate est vert.
      T2 bloque uniquement la publication finale (Phase D).
```

---

## VÉRIFICATION END-TO-END

```bash
# Après Phase A :
cd gcn-python && python -m pytest tests/test_sentence_type.py tests/test_instructions.py -v --tb=short

# Après Phases B+C :
cd gcn-python && python -m pytest tests/ -v --tb=short
cd gcn-core && cargo test

# Tests spécifiques :
python -m pytest tests/test_sentence_type.py -v    # Phase A : 60 tests
python -m pytest tests/test_layer1.py -v            # Phase B
python -m pytest tests/test_ir_emitter.py -v        # Phase C
python -m pytest tests/test_cross_lingual.py -v     # Phase D T1
python -m pytest tests/test_calibration.py -v       # Phase D T5
```

---

## FICHIERS MODIFIÉS — RÉSUMÉ

| Fichier | Phase | Action | Risque |
|---------|-------|--------|--------|
| `layer1/sentence_type.py` | A | Réécriture (198→~300 lignes) | Faible — zéro import actuel |
| `gcn-datasets/configs/lang_markers.json` | A | Nouveau | Zéro |
| `layer1/__init__.py` | A | +exports | Zéro |
| `layer1/representation.py` | A | +sentence_profile property | Très faible |
| `verbalizer/instructions.py` | A | +tokens/markers params, +3 méthodes | Faible (params optionnels) |
| `discuss.py` | A | +ponctuation, +fallback NLU conditionnel | Faible |
| `tests/test_sentence_type.py` | A | Nouveau (~60 tests) | Zéro |
| `constants.py` | B+C | 7→8 nœuds, 11→19 relations, +Voice/PronType | **Fort (cascade breaking)** |
| `layer1/features.py` | B | +Voice/PronType/positionnels/lemma_emb (d_emb), d_clause 79→106 | Fort |
| `frontend/bridge.py:187` | B | has_advcl dérivé de node_type | Faible |
| `scripts/init_v3_stub.py` | B+C | Nouveau — init stub v3 (He-init) | Zéro |
| `gcn-core/crates/gcn-ir/src/edge.rs` | C | +8 RelationType, +TernaryThird, +joint_group_id | Fort (Rust) |
| `gcn-core/crates/gcn-ir/src/query.rs` | C | Match RelationType exhaustifs | Fort |
| `gcn-core/crates/gcn-ir/src/pearl.rs` | C | Requêtes Pearl nouveaux types | Fort |
| `data/edge_norm.py:131` | C | Multi-sources → 2 arêtes + joint_group_id | Moyen (cascade loader) |
| `data/loader.py` | C | Aplatissement normalize_edge, hyperedge_map | Moyen |
| `pipeline/ir_emitter.py` | C | 9-tuple rétrocompat, D6 confiance, Éq.7 voix | Moyen |
| `training/checkpoint.py` | C | Hook calibration isotonique | Faible |
| `evaluation/eval_runner.py` | C | Filtre nouveaux types | Faible |
| `tests/test_ir_emitter.py` | C | +6 tests ternaires | Zéro |
| `tests/test_cross_lingual.py` | D | Nouveau (T1 logique fixtures + T2) | Zéro |
| `tests/test_bridge_fix.py` | D | Nouveau (T4) | Zéro |
| `tests/test_calibration.py` | D | Nouveau (T5 — params nus np.interp) | Zéro |
| `REGLE_EQUILIBRE_DATASET.md` | B+C | 8 nœuds / 19 relations / quotas | Zéro |
| `pipeline/cgnp.py` | C | Détecteur ternaire `_detect_ternary` (C.6) | Moyen |
| `training/bootstrap.py` | C | Patch `_normalize_edge` joint_cause | Faible |
