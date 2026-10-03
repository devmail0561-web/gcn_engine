#!/usr/bin/env python3
# Arbitrage lot20 (48) + top-up lot11 (10) + gold auto lot20_05 (5) — doctrine lots 01-11.
# Produit : /tmp/opencode/arbitrage63_patch.json (sentences avec cir corrigés)
#         + /tmp/opencode/ARBITRAGE_63_propositions.md (relecture humaine)
# Ne touche à aucun fichier gold/quarantaine existant.
import json, re, sys

V4 = "gcn-datasets/DATA/supervision/v4/annotated"
fresh = {s["id"]: s for s in json.load(open("/tmp/opencode/lot20_rebuild.json"))["document"]["sentences"]}
top = json.load(open(f"{V4}/topup_lot11_pending.json"))
for s in top["document"]["sentences"]:
    fresh[s["id"]] = s

QNORM = str.maketrans({chr(c): "'" for c in (0x2019, 0x2018, 0x0060, 0x00B4)})
def norm(w): return w.translate(QNORM).lower()

def tok_idx(toks, a, b, label):
    for i in range(a - 1, b):
        t = toks[i]
        if norm(t["form"]) == norm(label) or norm(t.get("lemma", "")) == norm(label):
            return i + 1
    return None

def head_verb(toks, a, b, anchor):
    best, bd = None, 10**9
    for i in range(a - 1, b):
        if toks[i].get("pos") == "verb" and abs((i + 1) - anchor) < bd:
            bd, best = abs((i + 1) - anchor), i + 1
    return best if best else anchor

def window(a, b, head, marker):
    na, nb = max(a, head - 7), min(b, head + 7)
    if marker and na <= marker <= nb:
        if marker <= head: na = marker + 1
        else: nb = marker - 1
    return [na, nb]

PUNCT = set(",;:()\"«»")
def strip_punct(toks, a, b):
    while a < b and toks[a - 1]["form"].strip() in PUNCT: a += 1
    while b > a and toks[b - 1]["form"].strip("()«»\"") in PUNCT and toks[b - 1]["form"] in PUNCT: b -= 1
    return [a, b]

# ancrage manuel (bruit POS lattice) : sid -> (noeud, token-ancre ou label-ancre)
ANCHOR = {"s0138": ("n2", 16), "s2466": ("n1", "agents"),
          "s2251": ("n1", "indiquent")}
# fin faible (coupure) : on étend d'abord (<15, bornes orig), sinon on taille
TRIM = {"et", "ou", "ne", "se", "me", "te", "nous", "vous", "le", "la", "les",
        "l'", "de", "des", "du", "au", "aux", "à", "en", "y", "que", "qui",
        "dont", "où", "pour", "dans", "sur", "sous", "avec", "par", "vers",
        "contre", "entre", "comme", "est", "sont", "plus", "moins", "très",
        "davantage"}
SPAN15 = {"s2251": ("n1", [18, 32])}  # recentré sur 'indiquent', finit 'produit'
# étiquettes hors-span après redécoupe -> ré-ancrer (vérifié présent, sinon warn)
LABELFIX = {"s0138": ("n2", "réchauffent"), "s0400": None, "s2550": ("n1", "repéré"),
            "s3044": ("n1", "expriment"), "s0649": ("n1", "mettant")}
SPANFIX = {"s0400": ("n2", [2, 16])}  # ré-inclure le sujet 'transfert' (15 toks)

# action par phrase : (verdict, params)
# G = garder (+redecoupage auto), LBL = nouveau label n1, EXP0 = explicit:false,
# REL = nouvelle relation, Q = quarantaine(motif), OPTB = option B humaine
PLAN = {
 # 5 gold auto
 "s2684": ("G",), "s1327": ("G",), "s1695": ("G",), "s0038": ("G",),
 "s1679": ("Q", "'si' interrogatif indirect (savoir si) — relation condition non fondée ; aucun retype adapté"),
 # lot20 litigieux — garder + redecoupe
 "s0314": ("G", {"lbl": "inventorier"}), "s1514": ("G",), "s1599": ("G",),
 "s0793": ("G",), "s1326": ("G",), "s2736": ("G",), "s2695": ("G",),
 "s0400": ("G",), "s2479": ("G",), "s1857": ("G",),
 "s2987": ("G",), "s3044": ("G",), "s0611": ("G",), "s2394": ("G",),
 "s2386": ("G",), "s2550": ("G",),
 "s2209": ("G",), "s2132": ("G",), "s1684": ("G",), "s1693": ("G",),
 "s2251": ("G",), "s1546": ("G",), "s2192": ("G",), "s2236": ("G",),
 "s0654": ("G", {"lbl": "suggéré"}), "s0843": ("G", {"lbl": "posséder"}),
 "s1862": ("G", {"lbl": "envoyée"}), "s2466": ("G", {"lbl": "agents"}),
 "s2491": ("G", {"lbl": "utilisées"}), "s1058": ("G", {"lbl": "servir"}),
 "s0464": ("G", {"lbl": "retour"}), "s0973": ("G", {"lbl": "zoos"}),
 "s2002": ("G", {"lbl": "biens"}), "s1214": ("G",),
 "s0212": ("G",),
 # degenere [1,1] mal ancre sur marqueur -> re-ancre
 "s0138": ("G", {"n1span": [2, 3]}), "s1285": ("G", {"n1span": [2, 3]}),
 "s2920": ("G", {"n1span": [2, 3]}), "s3150": ("G", {"n1span": [2, 3]}),
 # 'ainsi' consequence/enumeratif -> explicit:false (doctrine lot05 s1763)
 "f0118": ("G",),
 "f0127": ("G", {"exp0": True}), "s1093": ("G", {"exp0": True}), "s0468": ("G", {"exp0": True}), "s1011": ("G", {"exp0": True}),
 # 'en raison de' / 'en cas de' : marqueur note
 "s0178": ("G",), "s2394b": ("G",),
 # retypes
 "s0870": ("G", {"rel": "concession"}), "s1225": ("G", {"rel": "filter"}),
 "s0649": ("G", {"rel": "cause"}), "s1713": ("G", {"rel": "opposition"}),
 # quarantaines
 "f0073": ("Q", "énumération légale multi-conditions (4 conditions, span [7,65] irréductible à ±7 sans perte ; cf. f0119 catalogue). Option B : garder + fenêtre [7,14]"),
 "s2467": ("Q", "nœud n1 = marqueur seul [1,1] ('Mais'), portée anaphorique hors-phrase ('mais aussi' discursif)"),
 "s1343": ("Q", "marqueur 'ainsi' anaphorique (réfère à la phrase précédente), n1 copule dégénérée [1,1]"),
 "s1111": ("Q", "faux marqueur : 'active' adjectival (population active), relation control non fondée — descriptif statistique"),
 "s1690": ("Q", "marqueur 'contre' dans nom propre (Ligue contre l'impérialisme), relation prevent non fondée"),
 "s2564": ("Q", "litote 'n\\u2019est pas sans rappeler' — 'sans' non préventif ; 0-arête possible (descriptif), à trancher humain"),
 "s2758": ("Q", "'sans' circonstanciel (manière) + bruit lexical ('tritre', 'utilisage')"),
 "s0475": ("Q", "'si' interrogatif indirect (déterminer si) — relation condition non fondée"),
 "s0107": ("Q", "'si' interrogatif indirect (savoir si) — relation condition non fondée"),
}
assert "s2394b" not in fresh  # garde-fou frappe
PLAN["s2394"] = ("G",)

JUSTIF = {
 "s0870": "'sans que' + subjonctif = concessif, pas préventif",
 "s1225": "parenthèse d'exception '(sans majuscule)' = sémantique filter (cf. lot08)",
 "s0649": "'si bien que' = consécutive, pas conditionnelle",
 "s1713": "'s\\u2019incline contre' = conteste électorale (opposition) ; option B : quarantaine si désaccord",
 "f0127": "'ainsi que' énumératif, pas causal (doctrine lot05 s1763)",
 "f0118": "marqueur partiel 'à' (de 'à cause de', césure lattice ; cf. lot08 s0415)",
 "s1093": "idem 'ainsi que' énumératif",
 "s0468": "'ainsi' manière/conséquence factuelle, pas connecteur causal",
 "s1011": "idem 'ainsi'",
 "s0314": "tête séquencée = 'inventorier' (pas 'facteurs', topique de phrase)",
 "s0138": "n1 [1,1] pointe le marqueur 'Après' → re-ancré sur 'l\u2019Arctique' [2,3] ; label n2 → 'réchauffent' (tête événementielle)",
 "s1285": "n1 [1,1] pointe 'Si' → re-ancré [2,3]",
 "s2920": "n1 [1,1] pointe 'Après' → re-ancré [2,3]",
 "s3150": "n1 [1,1] pointe 'Après' → re-ancré [2,3]",
}

patch, props, stats = [], [], {"G": 0, "Q": 0}
md20 = open(f"{V4}/../ARBITRAGE_lot20.md").read()
secs20 = dict(re.findall(r"^## (\S+)\n(.*?)(?=^## |\Z)", md20, flags=re.M | re.S))
md11 = open(f"{V4}/../ARBITRAGE_lot11.md").read()
secs11 = dict(re.findall(r"^## (\S+)\n(.*?)(?=^## |\Z)", md11, flags=re.M | re.S))

def md_text(sid):
    for d in (secs20, secs11):
        if sid in d:
            m = re.search(r"(.*?)\n- noeuds:", d[sid], flags=re.S)
            return (m.group(1).strip() if m else "").replace("\n", " ")
    return fresh[sid]["text"]

for sid in sorted(fresh):
    if sid not in PLAN:
        print("SANS PLAN:", sid); continue
    act = PLAN[sid]
    s = fresh[sid]; e = s["cir"]["edges"][0]; ns = s["cir"]["nodes"]
    toks = s["tokens"]; mk = e.get("marker_token")
    if act[0] == "Q":
        stats["Q"] += 1
        props.append((sid, "QUARANTAINE", act[1], None))
        continue
    stats["G"] += 1
    p = act[1] if len(act) > 1 else {}
    nodes = [dict(n, token_span=list(n["token_span"])) for n in ns]
    # re-ancrage n1 degenere
    if "n1span" in p:
        nodes[0]["token_span"] = p["n1span"]
    # re-ancrage label n1
    newlbl = p.get("lbl")
    if newlbl:
        nodes[0]["label"] = newlbl
    edge = dict(e)
    if "rel" in p: edge["relation"] = p["rel"]
    if p.get("exp0"):
        edge["explicit"] = False; edge.pop("marker_token", None); mk = None
    # redecoupe ±7 partout sauf spans imposees et spans <=15 deja OK
    newspans = []
    for i, n in enumerate(nodes):
        a, b = n["token_span"]
        if sid in SPAN15 and SPAN15[sid][0] == n["id"]:
            n["token_span"] = list(SPAN15[sid][1])
            n["label"] = "indiquent"
            newspans.append(list(SPAN15[sid][1])); continue
        if (i == 0 and "n1span" in p) or (b - a + 1) <= 15:
            # trim marqueur fonctionnel en bordure (marqueur exclu, doctrine) ;
            # jamais si le marqueur porte le label (déclencheur lexical composé)
            mkform = toks[mk - 1]["form"] if mk and 1 <= mk <= len(toks) else ""
            if (mk and a <= mk <= b and tok_idx(toks, a, b, n["label"]) != mk
                    and norm(n["label"]) not in norm(mkform)):
                if mk == b: b -= 1
                elif mk == a: a += 1
                n["token_span"] = [a, b]
            newspans.append([a, b]); continue
        anchor = tok_idx(toks, a, b, n["label"]) or (a + b) // 2
        force_head = None
        if sid in ANCHOR and ANCHOR[sid][0] == n["id"]:
            ov = ANCHOR[sid][1]
            if isinstance(ov, int):
                anchor = ov
            else:
                anchor = tok_idx(toks, a, b, ov) or anchor
                force_head = anchor
        head = force_head if force_head else head_verb(toks, a, b, anchor)
        nb = window(a, b, head, mk if p.get("exp0") is None else None)
        nb = strip_punct(toks, nb[0], nb[1])
        if sid == "s0314" and n["id"] == "n1":
            nb = [23, 37]  # +8 pour garder l'objet 'biodiversité' (span 15 ≤ gate)
        n["token_span"] = nb; newspans.append(nb)
        # note: marqueur exclu aussi en exp0 (marqueur retire)
        if p.get("exp0") and e.get("marker_token") and nb[0] <= e["marker_token"] <= nb[1]:
            pass  # marqueur retire de toute facon
        n["token_span"] = nb; newspans.append(nb)
    edge_flags = []
    # finition : ponctuation traînante + fin faible (étendre <15, sinon tailler)
    for n in nodes:
        a, b = n["token_span"]
        oa, ob = ns[[x["id"] for x in ns].index(n["id"])]["token_span"]
        while b > a and toks[b - 1]["form"] in (",", ";", ":"):
            b -= 1
        while b > a and norm(toks[b - 1]["form"]) in TRIM:
            if (b < ob and (b + 1 - a + 1) <= 15
                    and norm(toks[b]["form"]) not in TRIM):
                b += 1
                break
            b -= 1
        while b > a and toks[b - 1]["form"] in (",", ";", ":"):
            b -= 1
        n["token_span"] = [a, b]
    # resync newspans (finition a pu ajuster)
    newspans = [list(n["token_span"]) for n in nodes]
    if sid in SPANFIX:
        nid, sp = SPANFIX[sid]
        for n in nodes:
            if n["id"] == nid:
                n["token_span"] = list(sp)
        newspans = [list(n["token_span"]) for n in nodes]
    if sid in LABELFIX and LABELFIX[sid]:
        nid, lb = LABELFIX[sid]
        for n in nodes:
            if n["id"] == nid:
                a, b = n["token_span"]
                if tok_idx(toks, a, b, lb):
                    n["label"] = lb
                else:
                    print(f"WARN {sid} {nid}: label '{lb}' absent du span")
        newspans = [list(n["token_span"]) for n in nodes]
    # nœuds disjoints : clamp dans l'ordre textuel (re-ancrage [1,1]→[2,3] peut chevaucher n2)
    ordered = sorted(range(len(nodes)), key=lambda i: nodes[i]["token_span"][0])
    for k in range(1, len(ordered)):
        prev, cur = nodes[ordered[k - 1]], nodes[ordered[k]]
        if cur["token_span"][0] <= prev["token_span"][1]:
            cur["token_span"][0] = prev["token_span"][1] + 1
    for n in nodes:
        a, b = n["token_span"]
        if a > b:
            edge_flags.append(f"span {n['id']} VIDE après clamp")
        if n["token_span"][1] - n["token_span"][0] + 1 > 15:
            edge_flags.append(f"span {n['id']} >15 RESIDUEL")
    if mk and any(n["token_span"][0] <= mk <= n["token_span"][1] for n in nodes):
        edge_flags.append("marqueur DANS span")
    just = JUSTIF.get(sid, "")
    if newlbl: just = (just + " ; " if just else "") + f"label n1 re-ancre '{newlbl}' (marqueur-dans-nœud)"
    if p.get("exp0"): just = (just + " ; " if just else "") + "explicit:false"
    if "rel" in p: just = (just + " ; " if just else "") + f"retype {e['relation']}→{p['rel']}"
    props.append((sid, "GARDER", just or "spans ±7, conf revalidée (doctrine lots 08-10)", newspans))
    rec = dict(s); rec["cir"] = {"nodes": nodes, "edges": [edge]}
    patch.append(rec)

json.dump({"schema_version": "4.0", "document": {"id": "arbitrage63-propose", "sentences": patch}},
          open("/tmp/opencode/arbitrage63_patch.json", "w"), ensure_ascii=False, indent=1)

resid = [(sid, ns) for sid, v, j, ns in props if v == "GARDER" and ns and any(b - a + 1 > 15 for a, b in ns)]
print("GARDER:", stats["G"], "| QUARANTAINE:", stats["Q"], "| patch:", len(patch))
print("spans >15 residuels:", resid if resid else "aucun")
with open("/tmp/opencode/arbitrage63_resume.txt", "w") as f:
    for sid, v, j, ns in props:
        f.write(f"{sid} | {v} | {j} | spans={ns}\n")
print("resume ecrit")
