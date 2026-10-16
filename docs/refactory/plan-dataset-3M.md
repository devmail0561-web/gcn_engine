# Plan dataset — 3M de phrases multi-domaines (standalone)

> Doc spécifique dataset, indépendante des plans refacto (`plan-remodelage-tout-appris.md`,
> `plan-remodelage-inference.md`). Cible : **3 000 000 phrases**. Stock actuel
> **~146k uniques (5%)**, gap **~2 854k (95%)**. Legacy annoté : textes réutilisés,
> CIR jetés (sauf `generated_1000`, `B_batch`, `v4` sains). Token GitHub dispo.
> L'annotation (phrases → CIR) est dans `plan-annotation-cir.md` — ce plan ne
> traite que la collecte brute.
>
> Note d'ordre : « Maths > Code » = ordre d'effort et de qualité, pas de volume.
> Les maths exigent l'extraction la plus soigneuse (LaTeX, sections) et passent
> en premier dans l'effort, même si la programmation reste le premier volume.
>
> Note 2026-10-11 : pilotage opérationnel unifié dans
> `plan-collecte-dataset-annotation.md` (T0→T5, tranche verticale gold-219 d'abord).
> Correctif C0 : le registre v5 utilise la table **`scraped_urls`** (pas `urls`) —
> `url_registry_global.db` doit reprendre ce schéma
> (`url_hash/url/campaign_id/source_name/output_file/scraped_at/sentence_count`).

## 1. Mix figé et quotas

```text
Programmation 31.5% = 945 000 :
  code 80% ≈ 756 000 — GitHub avec token, --no-doc, 6 langs + csharp,
    source=github_{lang} exigée, garde anti-doc (URLs docs bannies,
    contrôle code-like), fonctions (ast/regex, ≥50c, ≤5000) +
    unités causales (commentaires/docstrings/issues/commits).
    Compteur repart de ~0 (l'ancien "code" n'était que de la doc).
  doc 20% ≈ 189 000 — ~2k existantes comptent → ~187k à collecter,
    périmètre ÉLARGI : docs officielles 6-7 langages + MDN/realpython
    + tutoriels + StackOverflow + manuels (scraper à écrire : TOS à
    vérifier, sortie {text,lang,source,url,quality}, 1-2j dev).
    Débit ~5-10k/j (trafilatura + URLs bornées).
Maths 22% = 660 000 (effort prioritaire) :
  Sup Exo7 (CC-BY-NC-SA, LaTeX) + Sésamath (parties publiques GNU-FDL/CC-BY-SA)
  + OpenStax (CC-BY) + LibreTexts (licence à auditer par page) + MIT OCW
  (cours maths, conditions à vérifier). EXCLUS : Khan Academy (conditions
  anti-scraping/IA), Annabac/Hatier/manuels sous droits.
  Chunking PAR SECTION (énoncé + éléments de preuve gardés ensemble), jamais
  par phrase isolée. Nouveau module PDF à écrire : pymupdf + latexpand
  (extraction texte + formules en blocs opaques).
Physique 8% = 240 000 :
  HAL physique + arXiv physics + OpenStax University Physics (CC-BY) +
  MIT OCW physique 8.01/8.02 (conditions à vérifier). Même chunking par
  section et même module PDF que les maths (campagne commune d'extraction).
Presse 10.5% = 315 000 : news_rss (15 flux FR/EN) + web_search, rotation seed.
Science hors-physique 11% = 330 000 : hal + arxiv + web_search/openalex/pubmed
  (fixer arXiv à 0 ligne du run 20261002). HAL prouvé ~26k/run.
Éducatif 8% = 240 000 (dont ~60k contenus de langue) : education_web
  (Vikidia/Simple étendus) + wikipedia + ressources langue (grammaires et
  cours ouverts, licences à vérifier une par une).
Wiki/Divers 9% = 270 000 : wikipedia 6 langues + web_search
  (compenser pt/it 0.8 par 2 runs seed distincts).
Total quotas = 3 000 000 (marge dédup/quarantaine ~2% incluse dans les
débits, pas dans les quotas).
Variante B (si GitHub rate-limite) : prog 28% (840k), +105k sur wiki/divers.
```

## 2. Stock existant (ne pas recollecter)

```text
Brut : legacy FR 47k + EN 27k + DE/ES 9k + v5 75k (divers 18k, educ 18k,
  presse 7.5k, science 32k, doc-code 1k) − overlaps ≈ 146k uniques.
  (Gonflement ~300k lignes ÷2 : sentences=wikipedia+news, presse×3, DE/ES×2.)
Sains à garder : generated_1000 (990 synthétiques) + B_batch (219 fr/en/es
  manuels) + merge_balanced (829) + splits (714, seuls sans fuite) + v4 (2193).
À jeter (CIR) / garder (textes) : smart 3154 + cir 1171 + combined 755 +
  final 812 + verbal 700 (≈3.5-3.8k uniq réels à re-annoter).
À exclure : phrases_fr.txt (vide), quarantaine 152, cy/lv artefacts,
  science/ vides, .v2bak/miroirs, final/val fuyard (100% du val dans final).
```

## 3. Campagnes (pipeline `gcn-scrape` existant, 1 registry global partagé)

> Convention chemins (2026-10-08) : les commandes ci-dessous s'exécutent depuis `gcn-datasets/` (données sous `DATA/` — existant vérifié : `DATA/data_brut/v5/url_registry.db`).

```text
C0 Socle (0.5j) : registry global (copie v5/url_registry.db), output-dir par
  campagne (c1_prog, c2_maths, c3_phys, c4_science, c5_presse, c6_educ,
  c7_wiki), seeds fixés, config gelée (langs/target/code-ratio), checkpoint
  --resume prêt. Principe : --seed fixe, --min-quality 0.3, --resume à config
  identique, workers auto (≤8), campagnes parallèles sur output-dir distincts.

C1-code 756k (40-70j/1 mach.) :
  gcn-scrape --output-dir DATA/data_brut/c1_prog --langs none --prog-langs python,rust,javascript,typescript,java,go,csharp --target-total 756000 --github --github-token $GITHUB_TOKEN --no-doc --no-wiki --no-hal --no-arxiv --no-news --no-web-search --max-per-query 30 --seed 42 --registry-db DATA/data_brut/url_registry_global.db
  Débit ~10-20k/j (2s, 30 req/min). Voie code séparée (pas de split 30-600).
  Correctif 2026-10-11 : `--langs none` OBLIGATOIRE (sans lui, code_ratio 0.10
  plafonne le code à 10 % — run mort-né à ~75k ; `balance_tracker.py:42-46`).

C1-doc 187k (20-35j) : --doc (44 URLs + étendues) PUIS scraper
  StackOverflow/manuels/tutoriels (1-2j dev, cf §1).

C2-maths 660k (effort prioritaire, 30-60j) : module PDF LaTeX d'abord (1-2j
  dev : pymupdf + latexpand, sortie {text,lang,source,url,section,quality}),
  PUIS extraction Exo7 → Sésamath → OpenStax → LibreTexts → MIT OCW, avec
  audit de licence par source AVANT extraction. Chunking par section imposé
  par le module (pas de phrase isolée).

C3-phys 240k (15-25j) : même module PDF que C2 + HAL physique + arXiv physics
  + OpenStax University Physics + MIT OCW 8.01/8.02.

C4 science hors-phys 330k (15-25j) : --langs fr,en --hal --arxiv --web-search
  --max-per-query 100.
C5 presse 315k (25-40j) : --news --web-search, rotation seed (flux finis).
C6 educ 240k (15-25j) : --wiki + education_web (étendre URLs) + ressources
  langue (~60k, licences vérifiées une par une).
C7 wiki/divers 270k (15-25j) : --langs fr,en,es,de,pt,it --wiki --web-search.

C8 fusion (3-4j) : MD5 global + Jaccard (mémoire 2000→20000) + bacs quarantaine
  (longueur/qualité/sans-connecteur/near-dedup) + causal_hints 100%
  (compteur causal, 11 indices → table 11→19 du plan annotation) +
  répartition ±3pts + split seed 42 + oversample_rares rejoué.
Débit global : 15-30k brut/j/mach. → 10-20k net/j → ~150-280j à 1 mach.,
  ~50-95j à 3 mach. (registry partagé) +20% reprises.
```

## 4. Qualité, quarantaine, traçabilité

```text
Filtres prose : 30-600 chars (optimum 150-300) + quality≥0.3 + lang budgétée
  + ≥1 connecteur causal (sinon quarantaine causal_none.jsonl, pas suppression)
  + multi-hints priorisés (rares : filter/data_dep/control_dep/motivation/sequence/opposition).
  Exception maths/physique : unités = SECTIONS (pas de borne 600c), formules
  en blocs opaques, filtre connecteur appliqué au texte porteur uniquement.
Dédup : URLRegistry SHA256 partagé + Deduplicator exact MD5 + near Jaccard>0.7
  persisté entre campagnes + hash texte global avant fusion.
Quarantaine : trop_court_long / qualite_lt_0.3 / sans_connecteur / near_dedup
  + errata/tombstones (modèle errata_supervision_v2) — exclus du train, réinjection sur revue.
Traçabilité : {text,lang,source,url,quality,causal_hints,seed,campaign_id} ;
  sentences_<ts>.jsonl intouchés + checkpoint + registry.stats() +
  histogrammes len/quality/lang/source/hints par jalon.
Distribution causale visée (post-filtre, par langue) : cause 20-25%, condition 15-20%,
  enable 10-15%, sequence 10%, concession/opposition/motivation 5-8% ch.,
  prevent/filter/data_dep/control_dep 3-7% ch.
```

## 5. Équilibrage (exigé, 5 dimensions, tolérances + gates)

```text
[D1] Relations causales (11 indices → 19 cibles via la table du plan
  annotation) — table cible % global ET par langue :
  cause 20-25 | condition 15-20 | enable 10-15 | sequence 10±2 |
  concession/opposition/motivation 5-8 ch. |
  prevent/filter/data_dep/control_dep 3-7 ch. (plancher 3% absolu,
  jamais 0 : N_min=30/classe/split pour démasquer, cf R8).
  Vérif : histogramme hints/relations par langue à chaque jalon ;
  gate J7 : chaque case dans sa fourchette ±3pts, sinon rééquilibrage
  avant fusion (pas après).
[D2] Langues (LE support des langues du moteur, T6) : fr 35±3 | en 35±3 | es 10±2 | de 8±2 | pt/it/code-doc 12±3
  (code source = bucket à part, pas une langue : 756k comptés dans prog,
  pas dans la balance des langues naturelles). Toute langue hors de cette
  liste = non supportée jusqu'à sa campagne de collecte (ajouter une langue
  = ajouter des données, pas du code). Arabe/CJK : pas de collecte dans ce
  plan → embeddings aléatoires assumés (T6), campagne ultérieure si besoin.
  Silver intent par famille de ponctuation (voir T6).
[D3] Domaines : quotas §1 ±3pts (prog 31.5 / maths 22 / presse 10.5 /
  science 11 / phys 8 / educ 8 / wiki-divers 9). Gate J7 : écart >3pts →
  campagne de rattrapage sur le domaine déficitaire avant gel.
[D4] Prog interne : 6-7 langages code ±20% entre eux (pas de python à 60%) ;
  doc/code 20/80 tenu ; csharp inclus dès C1 (pas 0 final).
[D5] Longueur/actes (garde-fous, pas quotas durs) : 150-300c ≥50% sur la
  prose (sections maths/physique hors borne), <100c et >400c ≤20% ch. ;
  3 actes (assert/question/request) : aucune classe <10% sur le sous-ensemble
  annoté intent ; sentence_type : impératif ≥5% (sinon tête act sans exemples).
Méthodes : split stratifié seed 42 (langue×relation×domaine) ;
  oversample_rares rejoué (déjà +0,07 val_edge_f1) AVANT le gel J7,
  jamais après ; undersampling du surplus cause/wiki uniquement vers
  quarantaine (pas suppression) ; multi-hints priorisés en collecte
  pour les 6 rares. Contrôle : matrice langue×relation×domaine publiée
  à J7, gate bloquant si une case hors fourchette.
```

## 6. Jalons J0-J8

```text
J0 registry global OK (copié v5, stats OK, output-dir par campagne).
J1 module PDF LaTeX OK (pymupdf + latexpand, sortie sections testée sur Exo7).
J2 C1-doc 187k (100% 30-600, registry sans doublon URL) + scraper SO/manuels.
J3 C1-code 756k (6-7 langs ±20%, source github_* 100%, 0 doc).
J4 C2-maths 660k (licences auditées, chunking section vérifié par sondage).
J5 C3-phys 240k + C4 science 330k (arxiv>5k, fix 0-ligne).
J6 C5 presse 315k (FR/EN équilibrés) + C6 educ 240k + C7 wiki/divers 270k.
J7 fusion 3M net (toutes fourchettes §5 tenues, quarantaine, hints 100%).
J8 split stratifié seed 42 (langue×relation×domaine) train/val/test + errata exclus + oversample rejoué → main vers plan-annotation-cir.md.
Livrables/jalon : sentences_<ts>.jsonl + checkpoint + registry.stats() + histogrammes.
```
