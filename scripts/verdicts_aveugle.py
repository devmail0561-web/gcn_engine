#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Verdicts aveugle_200 (gate silver) — assistant délégué, veto humain possible.

Lit `v5/gates/aveugle_200.json` ({fichier, id, text, verdict}), applique :
  - Q (ci-dessous) -> verdict "quarantaine" + motif (tombstone : exclu des
    merges, jamais supprimé du disque) ;
  - sinon -> verdict "garder".
Réécrit le gate en place + `AVEUGLE_200_verdicts.md` (trace).
Ne modifie aucun autre fichier.

Usage :
    python3 scripts/verdicts_aveugle.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v5" / "gates" / "aveugle_200.json"

# --- verdicts quarantaine (67) : relation non fondée / marqueur invalide /
#     débris structurels / fragment. Doctrine lots 01-11 + 20/21/22/23. ---
Q = {
 "divers-en-news-008847": "verbe d'état 'become', pas de séquence",
 "educ2-fr-001538": "titre + [modifier] accolés, spans à cheval",
 "presse-en-001020": "assemblage deux articles (suite 'What else we're reading')",
 "educ-fr-000575": "cause non fondée ('ainsi' sans antécédent causal)",
 "divers-en-news-003679": "conditionnel ('If...becomes') typé séquence",
 "educ-legacy-fr-000354": "marqueur partiel 'l'' (de 'l'expansion') + '« le' dédoublé",
 "educ2-en-003096": "'first' ordinal + débris citation [149] + phrase tronquée",
 "divers-en-news-000940": "'controls' nominal ('export controls'), mention",
 "educ2-en-000499": "assemblage (phrase + équation + phrase), spans à cheval",
 "educ2-fr-002615": "titre + [modifier] accolés",
 "educ-legacy-fr-001421": "'ainsi que' énumératif, pas causal",
 "divers-en-news-008723": "conditionnel ('when...becomes') typé séquence",
 "educ2-en-002070": "débris [change] médian, spans à cheval",
 "educ-en-000213": "'causes' mentionné (métalangage entre guillemets), pas de relation",
 "presse-en-003953": "'become' statif ('If the self has become'), pas de séquence",
 "presse-en-001733": "'first' ordinal (first-ever), pas de séquence",
 "presse-en-003767": "'first' ordinal (the first Black woman), pas de séquence",
 "educ2-fr-002145": "'nécessaire' prédicatif ('il est nécessaire'), pas de dépendance",
 "educ-en-002454": "'only if' conditionnel, pas restrictif (cf. r2-00124)",
 "educ-en-000212": "'such as' énumératif + 'causes' mentionné, pas de relation",
 "educ-en-004817": "'blocks' dans nom composé ('building blocks') + débris wiki",
 "divers-en-news-003813": "construction lexicale 'end up', pas de séquence",
 "educ-en-000021": "'if and only if' conditionnel, pas restrictif",
 "presse-en-003785": "'first' ordinal (were first moving), pas de séquence",
 "educ-legacy-fr-001151": "fragment tronqué (finit sur 'par') + résidu math",
 "educ-en-004765": "marqueur 'first' ordinal (pas le connecteur 'Earlier')",
 "divers-en-news-003039": "'first' ordinal (the first time), pas de séquence",
 "presse-en-004592": "construction lexicale 'end up', pas de séquence",
 "educ2-en-004128": "titre + section + citations accolés, marqueur partiel 'In'",
 "educ2-en-001712": "'first' ordinal + débris [change] (sections accolées)",
 "divers-en-news-007571": "'first' ordinal (for the first time), pas de séquence",
 "divers-en-news-001703": "'become' statif (has become popular) + doublon presse-en-002171",
 "educ2-fr-000942": "'ainsi que' énumératif, pas causal",
 "educ-en-000481": "'causes' mentionné (métalangage), pas de relation",
 "divers-en-news-003978": "assemblage deux articles + crédit image ('unless' hors portée)",
 "educ2-fr-000415": "'nécessaire' adjectival, pas de dépendance",
 "educ-en-003921": "'first' ordinal (was first introduced), pas de séquence",
 "divers-en-news-002801": "'first' ordinal (first batch), pas de séquence",
 "divers-en-news-006793": "'become' statif (has become policy), pas de séquence",
 "divers-en-news-006427": "'first' ordinal (one of the first), pas de séquence",
 "educ-fr-002048": "'nécessaire' adjectival, pas de dépendance",
 "educ-en-004792": "titre + [change] + liste accolés",
 "divers-en-news-008699": "'first' ordinal (speak first), pas de séquence réalisée",
 "educ2-fr-001919": "titre + [modifier] accolés",
 "educ-en-002067": "'first' nominal (the first = le premier), pas séquentiel",
 "presse-en-002521": "'first-person' adjectival (type de vue), pas de séquence",
 "divers-en-news-002180": "'first' ordinal (first reported), pas de séquence",
 "educ-en-001445": "marqueur 'is' erroné (pas le connecteur 'When')",
 "educ-en-000051": "exemple logique à placeholder + hypothétique, pas de séquence",
 "presse-fr-001915": "'devient' changement d'état, pas séquentiel",
 "educ-legacy-fr-000117": "'nécessaire' nominal ('ce nécessaire'), pas de relation",
 "divers-en-news-005632": "'first' ordinal (world's first), pas de séquence",
 "presse-en-001179": "'controls' nominal + marqueur bruité ('controls.\"')",
 "educ-fr-000787": "'ainsi que' énumératif, pas causal",
 "educ-en-005691": "'first' ordinal (one of the first), pas de séquence",
 "divers-en-news-008694": "'First' dans nom propre (First Amendment)",
 "divers-en-news-005470": "'become' statif, pas de séquence",
 "educ-legacy-en-000090": "'first' ordinal (first MP), pas de séquence",
 "divers-en-news-004360": "'first' calendaire (first half), pas de séquence",
 "divers-en-news-006309": "'first' ordinal (first flagged), pas de séquence",
 "divers-en-news-000431": "'become' statif (has become a cipher), pas de séquence",
 "presse-en-000133": "'first' ordinal (one of his first acts), pas de séquence",
 "educ2-fr-001648": "'ainsi que' énumératif, pas causal",
 "educ-legacy-fr-001186": "marqueur = ponctuation (apposition, pas de causalité)",
 "presse-en-002171": "'become' statif + doublon divers-en-news-001703",
 "educ-en-000636": "'become' statif (would become part), pas de séquence",
 "educ-fr-001090": "'n'empêche en rien' (empêchement nié) + titre accolé",
}


def main() -> None:
    items = json.load(open(GATE))
    ids = [x["id"] for x in items]
    assert len(ids) == len(set(ids)), "doublons dans aveugle_200"
    unknown = [k for k in Q if k not in set(ids)]
    assert not unknown, f"ids Q hors gate : {unknown}"
    ng = nq = 0
    qrel: Counter = Counter()
    grel: Counter = Counter()
    for x in items:
        rel = None
        d = json.load(open(ROOT / x["fichier"]))
        s = next((s for s in d["document"]["sentences"] if s["id"] == x["id"]), None)
        if s is not None:
            es = s.get("cir", {}).get("edges", [])
            rel = es[0].get("relation") if es else "_none"
        if x["id"] in Q:
            x["verdict"] = "quarantaine"
            x["motif"] = Q[x["id"]] + " (arbitrage 2026-10-04)"
            nq += 1
            qrel[rel] += 1
        else:
            x["verdict"] = "garder"
            x["motif"] = ""
            ng += 1
            grel[rel] += 1
    json.dump(items, open(GATE, "w"), ensure_ascii=False, indent=1)
    md = ["# Verdicts aveugle_200 — 200 verdicts (seed du tirage inconnu, arbitrage 2026-10-04)",
          "",
          f"Garder : {ng} | Quarantaine : {nq} (tombstones, jamais suppression).",
          f"Quar par relation : {dict(sorted(qrel.items()))}",
          "",
          "## Faiblesses systématiques proposeur",
          "- 'first' ordinal/nominal/propre → sequence (~20 cas) : règle de typage à durcir.",
          "- 'become' statif → sequence (~10 cas) : exiger deux événements.",
          "- 'nécessaire' adjectival/prédicatif/nominal → data_dependency (3 cas).",
          "- Titres/sections/citations accolés (éduc/wiki) : nettoyage pool amont.",
          "- Doublon inter-registres : divers-en-news-001703 = presse-en-002171.",
          "",
          "## Quarantaines"]
    for x in items:
        if x["verdict"] == "quarantaine":
            md.append(f"- {x['id']} : {x['motif']}")
    (GATE.parent / "AVEUGLE_200_verdicts.md").write_text("\n".join(md) + "\n",
                                                          encoding="utf-8")
    print(f"GARDER: {ng} | QUARANTAINE: {nq}")
    print("quar/rel:", dict(sorted(qrel.items())))


if __name__ == "__main__":
    main()
