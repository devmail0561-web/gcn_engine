"""
Annotateur UD : texte brut → liste de tokens UD avec IDs alignés.

Utilise spaCy pour produire les tokens Universal Dependencies
(id, form, lemma, pos, dep_rel, dep_head, morph).
"""
import spacy
from typing import Literal

_NLP_CACHE: dict = {}


def _load_nlp(lang: str):
    if lang not in _NLP_CACHE:
        model = {"fr": "fr_core_news_sm", "en": "en_core_web_sm"}.get(lang)
        if model is None:
            raise ValueError(f"Langue non supportée : {lang!r}. Valeurs: 'fr', 'en'")
        _NLP_CACHE[lang] = spacy.load(model)
    return _NLP_CACHE[lang]


def annotate_ud(
    text: str,
    lang: str = "fr",
    id_convention: Literal["0based", "1based"] = "1based",
) -> list[dict]:
    """
    Texte brut → liste de dicts token compatibles gcn-nl.

    Chaque dict contient : id, form, lemma, pos, dep_rel, dep_head, morph.

    id_convention :
      "1based" → ids commencent à 1 (défaut, compatible avec le CIR annoté)
      "0based" → ids commencent à 0

    dep_head = 0 si le token est la racine syntaxique (root), id du parent sinon.
    morph = dict avec clés "Tense", "Aspect", "Mood", "Polarity" ; valeur "_absent"
            si la catégorie morphologique est absente.
    """
    nlp = _load_nlp(lang)
    doc = nlp(text)
    offset = 1 if id_convention == "1based" else 0
    tokens = []
    for tok in doc:
        tok_id = tok.i + offset
        if tok.dep_.lower() == "root":
            dep_head = 0
        else:
            dep_head = tok.head.i + offset
        morph = {
            "Tense": tok.morph.get("Tense", ["_absent"])[0],
            "Aspect": tok.morph.get("Aspect", ["_absent"])[0],
            "Mood": tok.morph.get("Mood", ["_absent"])[0],
            "Polarity": tok.morph.get("Polarity", ["_absent"])[0],
        }
        tokens.append({
            "id": tok_id,
            "form": tok.text,
            "lemma": tok.lemma_,
            "pos": tok.pos_,
            "dep_rel": tok.dep_,
            "dep_head": dep_head,
            "morph": morph,
        })
    return tokens
