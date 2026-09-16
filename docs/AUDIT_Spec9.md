---

RAPPORT D'AUDIT — Moteur GCN-Core (pipeline complet)

Périmètre : dataset JSON → chargement → vectorisation → training → inference/verbalisation
Base : 120 tests passent (0 échecs). Code vérifié ligne par ligne.

---

FINDINGS HIGH (5)

---

H1 — Conflit encodeur → décodeur : dimension mismatch à l'inférence

Fichiers :
- trainable.py:279-303 — méthode decode(ir_json)
- trainable.py:113-126 — méthode _node_type_embeddings_from_ir()
- cgnp.py:279-283 — alimentation du décodeur en training conjoint

Problème exact :

En training conjoint (quand --verbalize-dir est passé sans --decoder-only), le décodeur est alimenté par les vecteurs enrichis produits par l'encodeur. Cela se passe à cgnp.py:279-283 :

_vecs = (self._cached_enriched_vecs         # sortie R-GCN : shape (N, d_clause=75)
         if self._cached_enriched_vecs is not None
         else self._cached_clause_vecs)       # sortie Layer 1 : shape (N, d_clause=75)

Le décodeur reçoit _vecs de shape (N, 75). Ses layers RNN sont initialisées paresseusement à trainable.py:141-142 :

d_in = node_embeddings.shape[1]   # → 75
self._init_layers(d_in)           # layer0 = _LinearLayer(75 + 64, 64)

Mais à l'inférence, decode() (trainable.py:279-281) appelle :

node_embs = self._node_type_embeddings_from_ir(ir_json)  # one-hot → shape (N, 7)

Le décodeur a ses layers initialisées pour d_in=75 mais reçoit des vecteurs de dimension 7. Le _rnn_step à trainable.py:106 fait :

rnn_in = np.concatenate([context, h_prev])   # [7 + 64] = 71 ≠ 75 + 64 = 139
z1 = self._layers[0].forward(rnn_in)         # W est (139, 64), rnn_in est (71,) → CRASH

Conséquence : Un décodeur entraîné conjointement ne peut pas être utilisé via decode(ir_json). L'inférence standalone est cassée.

Correction : decode() doit recevoir les vecteurs enrichis de l'encodeur, pas fabriquer du one-hot. Le chemin d'inférence doit être : encodeur.forward() → vecteurs enrichis → décodeur.decode(vecteurs).

---

H2 — decode() tronque toute sortie à 5 tokens en dur

Fichier : trainable.py:289 et trainable.py:302

Problème exact :

Ligne 289 (mode sans layers, fallback) :
filtered = [int(i) for i in top_indices if int(i) not in _skip][:5]

Ligne 302 (mode greedy avec layers) :
filtered = [i for i in tokens if i not in _skip][:5]

Le [:5] coupe systématiquement la sortie à 5 tokens maximum, alors que self.max_decode_len = 20 (ligne 81) contrôle déjà le nombre de pas RNN (ligne 295). La boucle greedy peut produire jusqu'à 20 tokens, mais le filtre post-traitement en jette 15.

Conséquence : Toute phrase verbalisée de plus de 5 mots est tronquée. "La hausse des prix entraîne une baisse de la demande" → "hausse prix entraîne baisse demande" (5 tokens max).

Correction : Supprimer [:5] aux deux lignes. La longueur est déjà limitée par max_decode_len.

---

H3 — Le fixture datasets_dir pointe vers un chemin inexistant

Fichier : tests/conftest.py:26-30

Problème exact :

@pytest.fixture
def datasets_dir() -> Path:
    d = Path(__file__).parents[2] / "gcn-core" / "datasets" / "examples"
    if not d.is_dir():
        pytest.skip(f"Datasets dir not found: {d}")
    return d

Le chemin résolu est <projet>/gcn-core/datasets/examples. Ce répertoire n'existe pas — la structure réelle est :

<projet>/gcn-datasets/examples/    ← les vrais fichiers JSON
<projet>/gcn-core/crates/          ← le code Rust, pas de datasets/

Tous les tests qui utilisent datasets_dir sont donc silencieusement skippés via pytest.skip().

Correction : Remplacer le chemin par :
d = Path(__file__).parents[2] / "gcn-datasets" / "examples"

(parents[2] depuis tests/conftest.py remonte à gcn-python/../../ = racine du projet.)

---

H4 — Les fichiers PL (Python/Rust) sont ignorés sans warning

Fichier : data/json_reader.py:8-24

Problème exact :

def load_sentences(path: Path, lang: str = "fr") -> list[SentenceRecord]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return []
    if "examples" in doc:                     # format paper_examples
        return [...]
    if "document" in doc:                     # format gcn-nl
        return [...]
    return []                                 # ← tout le reste est ignoré silencieusement

Les fichiers python_basic.json et rust_basic.json dans gcn-datasets/examples/ utilisent le format "snippets" (gcn-pl). Ils n'ont ni "examples" ni "document" → load_sentences retourne [] sans aucun warning.

load_all_sentences (json_reader.py:27-31) itère sur *.json et appelle load_sentences pour chaque fichier. Les 2 fichiers PL sont chargés puis silencieusement ignorés.

Conséquence : Un data scientist qui pointe --data-dir gcn-datasets/examples/ ne sait pas que 2/5 fichiers sont ignorés.

Correction : Ajouter un warning au return [] final :
warnings.warn(f"Format JSON non reconnu dans {path.name} — fichier ignoré.", UserWarning)
return []

---

H5 — load_checkpoint ne restaure pas les poids de RGCNLayerPT

Fichiers :
- layer3/pytorch_rgcn.py:150-155 — parameters() retourne des copies détachées
- training/checkpoint.py:74-78 — load_checkpoint écrit dans ces copies

Problème exact :

RGCNLayerPT.parameters() (pytorch_rgcn.py:150-155) :
def parameters(self) -> list[np.ndarray]:
    return [
        self.W_r.detach().cpu().numpy(),    # copie NumPy, déconnectée du tenseur PyTorch
        self.W_0.detach().cpu().numpy(),    # idem
    ]

load_checkpoint (checkpoint.py:74-78) :
for i, p in enumerate(graph_params):     # graph_params = pipeline.graph.parameters()
    key = f"graph_{i}"
    if key in data:
        p[:] = data[key]                  # écrit dans la copie NumPy, PAS dans W_r/W_0

p[:] = data[key] modifie la copie NumPy retournée par parameters(). Les vrais tenseurs PyTorch self.W_r et self.W_0 restent inchangés.

Conséquence : Après load_checkpoint, le R-GCN PyTorch a toujours ses poids aléatoires initiaux. Le modèle chargé ne correspond pas au checkpoint.

Note : Ce bug ne touche PAS RGCNLayer (NumPy référence) dont parameters() retourne des vues directes sur self.W_r et self.W_0 — l'écriture p[:] = data[key] modifie bien les poids réels.

Correction : Ajouter une méthode load_state à RGCNLayerPT :
def load_state(self, arrays: list[np.ndarray]) -> None:
    with torch.no_grad():
        self.W_r.copy_(torch.as_tensor(arrays[0], device=self._device))
        self.W_0.copy_(torch.as_tensor(arrays[1], device=self._device))
Et dans load_checkpoint, détecter si graph a load_state et l'utiliser.

---

FINDINGS MEDIUM (12)

---

M1 — Arêtes longue distance éliminées sans compteur agrégé

data/loader.py:81-88 — Les arêtes avec gap > 1 sont ignorées avec un warning par arête, mais aucun compteur total n'est reporté en fin de chargement. Sur un gros dataset, des centaines d'arêtes gold peuvent être silencieusement perdues sans vision d'ensemble.

M2 — Format legacy paper_examples diverge des schémas actuels

data/json_reader.py:15-16 — Le chemin "examples" in doc + "expected_cir" ne correspond à aucun fichier dans gcn-datasets/. Il est utilisé uniquement par le fixture gcn-core/tests/fixtures/paper_examples.json. Code mort potentiel qui pourrait diverger silencieusement.

M3 — Glob verbalize_*.json trop restrictif

data/verbalize_loader.py:79 — Seuls les fichiers commençant par verbalize_ sont chargés. Pas de feedback sur les fichiers ignorés dans le répertoire.

M4 — Corpus phrases_fr.txt vide

gcn-datasets/corpus/phrases_fr.txt — 21 lignes de commentaires, zéro phrase réelle. Le dataset réel d'entraînement se limite à 3 fichiers d'exemples (8 sentences NL).

M5 — _LinearLayer.backward silencieusement faux si batch 2-D

layer2/reference.py:26-28 — np.outer(d_out, x) suppose des vecteurs 1-D. Un appel batch 2-D produirait un gradient de forme incorrecte sans erreur. Le pipeline actuel itère nœud par nœud donc ça fonctionne, mais rien ne documente ni ne vérifie cette contrainte.

M6 — Bootstrap passe le texte comme argument CLI

training/bootstrap.py:42-48 — subprocess.run([gcn_bin, "analyze", text]) passe le texte de la phrase en argument de ligne de commande. Risque de dépassement de la limite de longueur d'argument OS (128 KB sur Linux) pour de longs textes. Devrait utiliser input=text via stdin.

M7 — Pas de test d'intégration CLI pour gcn-train

tests/ — Aucun test n'invoque train_cmd via CliRunner. Le CLI assemble loader + pipeline + checkpoint et pourrait avoir des bugs de glue non détectés.

M8 — P1 : pont texte brut → UDRepresentation absent

Aucun fichier frontend_bridge dans gcn-python/src/. Le moteur Python ne peut pas faire d'inférence autonome — il dépend du binaire Rust gcn-cli.

M9 — Indices hors-bornes ignorés dans vocab.decode()

trainable.py:38-39 — if 0 <= int(i) < len(self._i2t) filtre silencieusement les indices invalides. Un modèle corrompu produirait une chaîne tronquée sans diagnostic.

M10 — Format IR incompatible entre Rust et Python

training/bootstrap.py:72-117 — _cir_to_doc lit n.get("token_span") comme liste, mais l'IR du pipeline Python (ir_emitter.py:42) emballe le span dans {"source_span": {"token_span": {...}}}. Les deux formats sont incompatibles.

M11 — Métriques alignées par position

evaluation/metrics.py:103-118 — node_type_accuracy et edge_relation_accuracy alignent par index, pas par identité. Un décalage d'insertion produit des métriques pessimistes sans avertissement.

M12 — eval_runner ne couvre pas la verbalisation

evaluation/eval_runner.py — run_eval() évalue nœuds et arêtes uniquement. Les métriques BLEU, causal_fidelity, cross_modal_consistency existent dans metrics.py mais ne sont jamais appelées par le runner.

---

FINDINGS LOW (17)

┌─────┬──────────────────────────────┬────────────────────────────────────────────────────┐
│  #  │           Fichier            │                    Description                     │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L1  │ loader.py:39                 │ Docstring dit "YAML" au lieu de "JSON"             │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L2  │ loader.py:107                │ Docstring dit "tokens YAML annotés" au lieu de     │
│     │                              │ "tokens JSON"                                      │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L3  │ loader.py:209                │ "nobj" dans le set de détection — pas une relation │
│     │                              │  UD valide                                         │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L4  │ pipeline/label_builder.py:58 │ Même "nobj" fantôme                                │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L5  │ conftest.py:18               │ Fixture nommée paper_examples_yaml pour un fichier │
│     │                              │  .json                                             │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L6  │ fr_causal_cycles.json        │ cycle_type: "feedback_negative" devrait être       │
│     │                              │ "feedback_positive" (boucle d'amplification)       │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L7  │ fr_causal_basic.json         │ "lang": "french" au lieu de "fr" (ISO 639-1)       │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L8  │ verbalize_loader.py:19       │ Fallback silencieux idx=0 pour node_type inconnu   │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L9  │ checkpoint.py                │ Pas d'état optimizer/epoch sauvegardé              │
│     │                              │ (documentable)                                     │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L10 │ layer3/reference.py:49       │ Boucle Python O(E) au lieu de np.add.at (cohérence │
│     │                              │  avec backward)                                    │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L11 │ pytorch_rgcn.py:124          │ Boucle Python par type de relation (acceptable     │
│     │                              │ pour référence)                                    │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L12 │ train.py:137+185             │ gold_node recalculé deux fois identiquement        │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L13 │ cgnp.py:130-134              │ Premier MLP forward gaspillé quand R-GCN actif     │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L14 │ cgnp.py:430                  │ np.log(probs + 1e-9) — epsilon cosmétique, pas de  │
│     │                              │ risque réel                                        │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L15 │ pipeline/label_builder.py:7  │ _nom_cache module-level sans invalidation          │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L16 │ train.py:202-213             │ Loss encodeur + décodeur mélangées dans avg_loss   │
├─────┼──────────────────────────────┼────────────────────────────────────────────────────┤
│ L17 │ train.py:92-95               │ CSV sans colonnes décodeur quand training conjoint │
│     │                              │  actif                                             │
└─────┴──────────────────────────────┴────────────────────────────────────────────────────┘

---

DIAGNOSTIC TRANSVERSAL : Encodeur → Décodeur

Le problème central forme une chaîne de 3 maillons cassés :

① M8 : Pas de pont P1
   → Le moteur Python ne peut pas encoder du texte brut seul

② H1 : Dimension mismatch
   → Le décodeur entraîné avec l'encodeur (d_in=75) crashe en inférence standalone (d_in=7)

③ H2 : Troncature à 5 tokens
   → Même si le décodeur fonctionne, la sortie est mutilée

Seul le mode --decoder-only (qui entraîne avec d_in=7 via one-hot) produit un décodeur utilisable en standalone — mais il ne bénéficie pas des représentations enrichies du R-GCN.

