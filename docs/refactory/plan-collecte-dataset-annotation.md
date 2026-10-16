# Plan collecte / dataset / annotation — GCN 3M + gold-3000 (2026-10-11)

> Owner solo : Michel Tendeng. Local-only (`docs/` git-ignoré), tout offline sauf
> collecte web/RSS/OTA autorisée. Moteur jamais YAML-configuré ;
> `PYTHONPATH=gcn-python/src` requis. Pas de GPU.
> Pilotage opérationnel unifié ; fonds de référence : `plan-dataset-3M.md`
> (collecte brute) + `plan-annotation-cir.md` (phrases → CIR) ; verdicts : `GATES.md` ;
> table 11→19 : `table-11-19.md` (SQUELETTE, interdit prod).

## 0. État des lieux vérifié (ne pas recollecter)

- Supervision : 321 343 phr / 35 521 CIR / 0 err (`comptes_classes.json`). 11 relations legacy ≥1 491 occ. ; **8 nouvelles relations à 0** → gel R8/R10 actif.
- Brut v5 : 67 504 lignes (~2 % cible 3M). Registre v5 : table **`scraped_urls`** (2963 urls, integrity ok).
- Annotees 31M (fr 27M, reste quasi-vide), entraînement 66M (fr 57M / en 8,8M).
- Sains à garder : `generated_1000` (990) + `B_batch` (219 fr/en/es) + `merge_balanced` (829) + splits (714 sans fuite) + v4 (2193).
- Textes à re-annoter (CIR jetés) : smart 3154 + cir 1171 + combined 755 + final 812 + verbal 700 (~3,5–3,8k uniq).
- Gold-mini 85 + held-out gelés seed 42 + sha256 (gate socle 0c GO).
- Disque : 87 %, 59G libres après purge. Suffisant C0 + spikes + tranche verticale, insuffisant pour 3M complets sans purge étape 2 ou volume externe.
- Ph5 NO-GO (ECE 0,33–0,55) : conf moteur = `provisoire`, silver weight 0.0 tant que non prouvé.
- Arborescence constatée 2026-10-11 (`gcn-datasets/DATA/data_brut/`) : legacy `code/ de/ en/ es/ fr/`
  + `v5/` (avec `url_registry.db`, sous-dossiers `presse/ science/ science2/ science_web/`).
  **Absents** : `c1_prog/ c2_maths/ c3_phys/ c4_science/ c5_presse/ c6_educ/ c7_wiki/` et
  `url_registry_global.db` — création = T0, pas anticipée.

## 1. Ordre d'exécution (5 tranches, stop/go à chaque gate)

```text
T0 socle C0 (0,5j) → T1 spikes P0-6 (1j) → T2 tranche verticale gold-219 (2–3j)
  → T3 outillage annotation + table v1 (2j) → T4 campagnes fond C1–C7 (50–95j/3mach. ≈ 150–280j solo)
  → T5 gold-3000 au fil de l'eau + J7/J8.
```

Règle d'arrêt : aucun verdict [VALIDÉ] table sans ~30 arbitrages/classe sur minage large (§19 garde anti-circularité, `table-11-19.md`).
J7 bloqué 2 cycles → arbitrage périmètre (variante B : prog 28 % / 840k, +105k wiki/divers).

## 2. T0 — Socle C0 (0,5j, EXÉCUTÉ 2026-10-11 — gate J0 : GO)

```bash
cd gcn-datasets/
mkdir -p DATA/data_brut/{c1_prog,c2_maths,c3_phys,c4_science,c5_presse,c6_educ,c7_wiki}
cp DATA/data_brut/v5/url_registry.db DATA/data_brut/url_registry_global.db
python3 -c "import sqlite3; c=sqlite3.connect('DATA/data_brut/url_registry_global.db'); print(sorted(r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\"))); print(c.execute('SELECT COUNT(*) FROM scraped_urls').fetchone()[0], c.execute('PRAGMA integrity_check').fetchone()[0])"
# T0-conformité : vérification NON DESTRUCTIVE depuis la racine (ne pas re-jouer
# select_gold_mini.py : il écraserait les manifests gelés) —
cd /home/virus-one/Bureau/projet_CNM
PYTHONPATH=gcn-python/src python3 -c "import json; ..."  # cf. session 2026-10-11 : 85/85, 11 rel
```

1. `url_registry_global.db` : copie v5, **schéma réel `scraped_urls`**
   (`url_hash/url/campaign_id/source_name/output_file/scraped_at/sentence_count`).
   Ne jamais écrire dans `v5/url_registry.db` (source gelée).
2. T0-conformité `normalize.py` : 100 % conformes schéma 8/19 sur gold-mini —
   vérifié 2026-10-11 : **85/85, 11 relations × 7-10** (contrôle non destructif,
   manifests gelés intouchés).
3. Output-dir par campagne : `DATA/data_brut/{c1_prog,c2_maths,c3_phys,c4_science,c5_presse,c6_educ,c7_wiki}` —
   les dossiers legacy (`code/ de/ en/ es/ fr/`) restent intouchés, aucune campagne n'y écrit.
4. Config gelée : `--seed 42 --min-quality 0.3 --resume` identique, workers ≤8.
- Gate J0 : 7 dossiers créés + registry copié, stats OK, budget figé.

## 3. T1 — Spikes P0-6 (EXÉCUTÉS 2026-10-11 — double GO + module C2 livré)

- Densité code : **GO** — docstrings gcn-python (strate cible : commentaires/docstrings) :
  **44,4 %** (170/383) ≥ 15 % (marge ×3). Réserve : fonctions brutes 55,5 % (337/607)
  mais mesure polluée (`condition` 255 hits via `si/if` syntaxiques, `motivation` via `pour`
  de boucle) → le compteur devra exclure le code brut ou filtrer ces marqueurs sur le
  bucket code. Le seuil est tenu sur la bonne strate, pas un artefact exploitable tel quel.
- PDF LaTeX : **GO** — Exo7 `livre-algebre-1.pdf` (CC-BY-NC-SA, 246 p., TOC 72 entrées) :
  sections propres via `get_toc` + texte FR extractible + opacification par fonts maths
  (`MathDesign-*`, `MSAM10` → `[FORMULE]`, 294 blocs / 3 pages test). Réserve : variables
  isolées en fonte texte fragmentent (`[FORMULE] x[FORMULE]`) → règle d'agglutination à
  prévoir au module C2, non bloquante. `latexpand` non requis sur PDF compilé (utile
  seulement si sources `.tex` brutes) ; `pymupdf` 1.28.2 installé par pip.
- Preuves : session shell 2026-10-11 (re-jouable) ; PDF test `/tmp/opencode/spike-pdf/` (volatil).
- **Module C2 livré (J1 GO)** : `gcn-scraper/sources/pdf_latex.py` (sections TOC,
  opacification fontes + agglutination, licence exigée par source, repli sans-TOC)
  + `tests/test_pdf_latex.py` **6/6 verts** (100 % offline). Première extraction réelle :
  Exo7 algèbre-1 → **72 sections** (`c2_maths/exo7_algebre1_sections.jsonl`,
  31 762 blocs `[FORMULE]`, longueurs 3k-74k).
- **Finding F4** (cosmétique) : en-têtes collés (`etraisonnementsChapitre1`) — retours
  par bloc à insérer au câblage pipeline C2 (`skip_split`), non bloquant.
- **Finding F5** (OpenStax) : WEB-PDF vectorisent les maths → 0–12 blocs `[FORMULE`]/section
  (contre 98/section Exo7) : prose intacte, équations invisibles (absentes, pas corrompues).
  Voie CNX-HTML **écartée 2026-10-12** (prose OK mais maths rendues client-side, aucun
  MathML statique ; archive.cnx.org morte). Verdict : prose OpenStax acceptée
  (353 sections) ; couverture formules = PDF LaTeX natifs (Exo7) + sources `.tex`
  arXiv ultérieures (`latexpand` garde sa raison d'être).
- **Sésamath : impasse technique** (site = app JS sans PDF, SVN mort) → contourné par OpenStax ;
  à retenter via sources LaTeX si besoin FR.

## 4. T2 — Tranche verticale gold-219 (EXÉCUTÉE 2026-10-11 — GO avec 3 findings)

But : prouver la chaîne complète sur les 219 sains (B_batch fr/en/es) avant de brûler des semaines de collecte.
Sorties : `gcn-datasets/DATA/supervision/tranche_t2/` (`t2_gold.json`, `t2_silver.jsonl`,
`t2_train/val/test.json` split seed 42 → 135/41/43, `t2_train_oversampled.json` 135→206,
`t2_manifest.json` sha256).

1. Collecte micro : 0 (B_batch 118+77+24 + gold-mini 85 réutilisés).
2. Silver-B sur 219 : **219/219 `silver-B`, 0 fallback** (modèle `es_core_news_sm` installé
   par `spacy download` — sans lui, es tombait à 24 fallbacks, taux 11 % > seuil 5 %).
   `source=silver-B`, weight 0.0, loss masquée en fallback (règle tenue).
3. Contrôle : **LLM NON JOUÉ** (décision humaine : qwen-coder non fiable + RAM insuffisante ;
   job ollama tué, 0 modèle chargé) → **arbitrage humain (assistant)** : census des 26
   multi-clauses + échantillon stratifié 33 phrases (1/motif/langue) + revue des 65
   divergences compteur-vs-gold. Segmentation **97,7 %** (5 défauts : s014, s016, s069,
   s077, s115 + bruit s060) ; connecteurs **~97 % pertinents** (1 FP : s067 `pilote`-nom ;
   bruit multi-hints attendu : s010, s013, s028).
4. `validate_cir` **219/219 (100 %)** + `normalize` 8/19 + manifests sha256 + `content_sha256`
   par record silver + `published_at:null` (B_batch non daté — assumé) + alias réversibles (ids `sid`).
5. Traversée entraînement : split seed 42 + `oversample_rare.py` rejoué (prevent 14→42,
   motivation →30, opposition →28, sequence →26) + **loader `GCNDataLoader` 135/135 et
   `reps_from_sentence` 135/135** (adaptateur format slice). Métrique harnais unique :
   **reportée** (0,468 vs 0,1867 toujours non réconciliés — exige gold-3000).
- Livrable : chaîne prouvée bout-en-bout jusqu'à l'entrée du train, pas de valeur SOTA attendue.
- **Finding F1** (compteur, pas silver-B) : rappel hints **154/219 (70 %)** ; 65 manques =
  gaps table documentés (flexions `basée`, contractions `du/des`, élisions `lorsqu'`,
  lexique `réservé/nécessite/coordonne/contre/finir par/une fois/pourtant/s'oppose`,
  sur-spécificités `basé sur les données/gère l'exécution`, compositionnels `si` concessif,
  `autorise` nié) + **es sans tables** → patch-list pour C8, à verser en table v1.
- **Finding F2** (gold) : s081/s083/s106 — arêtes `motivation` en sens inverse (src>tgt),
  **ignorées silencieusement** au load (relations asymétriques) → convention src<tgt à
  trancher en guide §4 (ré-annoter ou remap loader), sinon perte sèche au train.
- **Finding F3** (silver-B) : sur-segmentation spaCy (tires titres, `Celles-ci`, `Big Bus`,
  virgules s077, `utilise` s115) → pré-filtre coupures presse à prévoir en C8 (pas bloquant T2).

## 5. T3 — Outillage annotation (2j, BLOQUÉ par table v1)

1. **Table 11→19 v1 VALIDÉE 2026-10-11** (arbitrage humain sur minage 146k uniques :
   5 GARDÉS patchés, `joint_prevent` GELÉ, `analogy` deux-tiers, `counterfactual`
   REFORMULÉ — `table-11-19.md`, garde anti-circularité §19 satisfaite).
   Consommateurs débloqués : fusion C8/J7, silver-B, guide §4 (F2 src<tgt à trancher).
   Cas ambigus : remplir depuis LITIGE P0-5 (`like a/an` déjà versé en LITIGE).
2. Silver-B branché pipeline : **FAIT 2026-10-11** — module `gcn-scraper/silver_b.py`
   (brut JSONL/doc-JSON → silver JSONL + quarantaine `causal_none` + manifest ;
   `weight=0.0`, `format_refacto` s0, routage langue par phrase, jamais d'UD legacy)
   + `tests/test_silver_b.py` **8/8 verts**, suite scraper **57 passed**.
   Modèles fr+en+es installés (`es_core_news_sm` ajouté par `spacy download`).
3. Protocole contrôle (décision humaine 2026-10-11 : **pas de LLM** — qwen-coder non
   fiable + RAM insuffisante) : **arbitrage assistant** — census des multi-clauses +
   échantillon stratifié 1/motif/langue + revue exhaustive des divergences
   compteur-vs-gold. Étalonné en T2 (97,7 % segmentation, ~97 % connecteurs). À chaque
   jalon : même grille, taux publiés, montée poids 0→0,3 seulement si erreur assistée
   ≤ 20 % sur l'échantillon (règle GATES silver 0→0,3 inchangée, juge = humain).
4. Guide annotation §4 + kappa : campagne gold avec double-annotation, **kappa ≥0,70** exigé, guide versionné (`guide-annotation-v0.md` à faire monter en v1 avec la table).
5. Cas spéciaux : code source → tree-sitter (pas silver ML) ; commentaires/docstrings → prose ; maths/phys : latexpand avant annotation, formules opaques.

## 6. T4 — Campagnes fond C1–C7 (NOUVEAU GELÉ 2026-10-13 — lot socle NI, `plan-socle-ni.md`)

> Gel T4-nouveau 2026-10-13 (décision owner, session chemin tok) : aucune campagne
> nouvelle avant le verdict go/no-go du lot socle NI (80 phrases FR : zéros nœuds +
> intent → run tok fin). Motif : le run coarse `3ded799` prouve mémorisation sans
> généralisation à classes non servies — collecter avant le verdict accumule du
> brut inévaluable. Runs fond déjà engagés (C1-code run 9, C7 seed-2) laissés en
> cours (coût marginal nul). Reprise T4 à quotas corrigés (ciblage nœuds) si GO,
> dossier modèle rouvert si NO-GO (règle STOP GATES.md).

- Pilote mécanique : wiki FR seul, target 200, seed 42, `--registry-db` global, vers
  `c1_prog/` (fond, résultat en attente) — prouve output-dir + registre partagé + resume.
- C1-code **RUN 9 2026-10-12 (fond)** : run 8 +383 (cumul **~1 436**, python plafonne à 322,
  rust +129) ; run 9 relancé `--resume` (ne rejoue que les vagues flapées).
- C7 **124 156/270 000 (46 %, 6 wikis + web-search, TOUTES sources done)** ;
  seed-2 (123, nouvelle session, dédup registre) **LANCÉ** pour le solde + pt/it.
- **Première écriture C1 validée 2026-10-12** : vague python → 52 fonctions
  (`github_*.jsonl`, filtre qualité 0.3 passé — `--min-quality 0.0` inutile).
- **C5 vague 1 news TERMINÉE** : 5 920 phrases (fr 1855/en 4065), flux épuisés.
- **C5 vague 2 web-search TERMINÉE** : +4 201 (DDG seul) → presse **10 121/315k**, solde en rattrapage.
- **Câblage education FAIT** (C6) : flag `--education/--no-education` + tâche pipeline
  + entrée `sources.yaml` + `tests/test_education.py` (stub offline), suite **69 passed**.
  (Incident : suppression accidentelle de `feeds:` aussitôt réparée + vérifiée —
  `sources.yaml` intact, 7 flux FR.)

| Campagne | Quota | Point de vigilance |
|---|---|---|
| C1-code 756k | 80/20 code/doc, 7 langs ±20 %, `source=github_*` 100 %, `--no-doc`, garde anti-doc | Compteur repart de ~0 (ancien "code" = doc) |
| C1-doc 187k | 44 URLs + SO/manuels/tutoriels (scraper 1–2j, TOS vérifiées, sortie {text,lang,source,url,quality}) | **Source SO LIVRÉE 2026-10-12** : `sources/so_qa.py` (16 requêtes causales × 7 tags, Q+meilleure réponse, code opacifié, attribution url+auteur+licence CC BY-SA) + flag `--so` + tâche pipeline + tests 4/4 + câblage 1/1, suite **75 passed** ; vague 1 SO **TERMINÉE : 77 488** (27 Mo, quota tenu) + vague doc officielle **TERMINÉE : +1 067** (44 URLs) → **c1_doc 78 555/187 000 (42 %)** ; doc URLs officielles **44→72** (+28 tutoriels) ; solde = scraper tutoriels tiers |
| C2-maths 660k | Exo7→Sésamath→OpenStax→LibreTexts→MIT OCW, **licence auditée AVANT extraction**, chunking PAR SECTION | Module livré + Exo7 72 sections FR + **OpenStax Algebra 2e : 103 sections EN (CC-BY)** ; Sésamath impasse (F5) ; reste LibreTexts/MIT OCW + voie CNX-HTML/MathML |
| C3-phys 240k | Même module + HAL phys + arXiv physics + OpenStax Univ.Phys + MIT 8.01/8.02 | **OpenStax Phys Vol.1 : 177 sections EN** (prose OK, formules invisibles — F5) ; reste HAL/arXiv/MIT |
| C4 science 330k | hal+arxiv+web_search/openalex/pubmed, **vagues séquentielles par source**
  (parallèle ThreadPool affame les sources lentes — cause probable du 0-ligne arXiv
  20261002 ; source OK testée seule 2026-10-11), budget partagé via `--resume`
  (`pipeline.py:144-147`) | HAL ~26k/run prouvé ; vague 1 HAL **LANCÉE 2026-10-11** (fond) |
| C5 presse 315k | news_rss (15 flux FR/EN) + web_search, rotation seed | Vague 1 news
  **TERMINÉE 2026-10-12 : 5 920** (fr 1855/en 4065, flux épuisés) ; vague 2 web-search
  **TERMINÉE : +4 201** (DDG seul, OpenAlex 403 + PubMed KO) → total presse **10 121/315k**,
  solde en rattrapage ultérieur (rotation seed / flux neufs) |
| C6 educ 240k | education_web (Vikidia/Simple étendus) + wikipedia + ressources langue ~60k (licences 1 par 1) | Vague 1 education **TERMINÉE 2026-10-12 : 15 052** (fr 5078/en 9974, ~5 min, liste URLs épuisée) → **extension URLs requise** (Vikidia/Simple + ressources langue) pour viser 240k |
| C7 wiki/divers 270k | wikipedia 6 langues + web_search, 2 runs seed distincts (compenser pt/it) | — |

Filtres prose : 30–600c (optimum 150–300) + quality≥0,3 + ≥1 connecteur causal (sinon `causal_none.jsonl`, pas suppression) + multi-hints rares priorisés. Exception maths/phys : unités = SECTIONS, filtre connecteur sur texte porteur uniquement.
Dédup : SHA256 partagé + MD5 exact + Jaccard>0,7 persisté + hash global avant fusion. Quarantaine : trop_court_long / qualite_lt_0,3 / sans_connecteur / near_dedup + errata/tombstones (exclus train, réinjection sur revue).
Débit réaliste solo 1 mach. : 10–20k net/j → **150–280j pour 3M**. À 59G libres : phaser (verticale → 100k utile → fond). Ne pas lancer C1-code plein sans purge étape 2 ou disque externe.

## 7. T5 — Fusion C8 + splits J7/J8 (3–4j par cycle)

- C8 : MD5 global + Jaccard (2000→20000) + bacs quarantaine + `causal_hints` 100 % + répartition ±3pts + split seed 42 + oversample_rares rejoué AVANT gel, jamais après.
- Équilibrage 5 dimensions (gates bloquants, détail `plan-dataset-3M.md` §5) : D1 relations (cause 20–25, condition 15–20, enable 10–15, sequence 10±2, conc/opp/motiv 5–8 ch., prevent/filter/data_dep/control_dep 3–7 ch., plancher 3 %, N_min=30/classe/split) ; D2 langues (fr/en 35±3, es 10±2, de 8±2, pt/it/code-doc 12±3 ; code = bucket à part) ; D3 domaines (quotas ±3pts) ; D4 prog interne (±20 %, csharp inclus) ; D5 garde-fous longueur/actes (150–300c ≥50 %, 3 actes aucune <10 %, impératif ≥5 %).
- Gate J7 : matrice langue×relation×domaine publiée, chaque case ±tolérance sinon campagne rattrapage AVANT gel.
- J8 : split stratifié (langue×relation×domaine) train/val/test + errata exclus → main vers entraînement.
- Livrables/jalon : `sentences_<ts>.jsonl` intouchés + checkpoint + `registry.stats()` + histogrammes len/quality/lang/source/hints.

## 8. Gold-3000 (au fil de l'eau, lève les gels)

- Cible ~3000 (19 rel × 8 types × 3 actes, langue×relation×domaine couverts, aucun zéro). Socle 60–100 suffit pour démarrer.
- R8/R10 : <30/classe/split → gel (pas de seuils, pas de calibration). Levée après gold-3000 + re-mesure harnais unique.
- Calibration Ph5 rejouée sur gold-3000 (ECE + monotonie, N et seed notés) → conditionne SA (`provisoire` levé si GO).

## 9. Risques assumés

- 3M solo = ~1 an calendaire à 1 machine ; le plan ordonne : vertical → utile → fond.
- Disque : C1-code plein + C2 PDF = ~10–15G + marge ; sans purge étape 2, plafonner à ~100k puis arbitrer.
- 8 relations à 0 occ. supervision : tout seuil/calibration avant gold-3000 = invalide.
- Table circulaire (règles minées sur 0,5 % puis validées sur même corpus) : garde §19, minage large exigé.

## 10. Prochaines actions (ordre)

1. ~~Arbitrer 8 déclencheurs OU lancer minage large~~ FAIT 2026-10-11 (v1 validée).
2. Choisir purge étape 2 ou volume externe.
3. Exécuter T0 (registry global schéma `scraped_urls` + T0-conformité).
4. Jouer spikes P0-6.
5. Tranche verticale T2 avant tout C1 plein.
