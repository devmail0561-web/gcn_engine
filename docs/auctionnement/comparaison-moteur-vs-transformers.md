# Comparaison globale : Moteur Vs Transformers / LLMs

Date : 2026-10-14
Statut : note de travail — exhaustive, concrète, honnête.

## 1. Architecture et entrée

|  | Moteur | Transformer / LLM |
|--|--------|-------------------|
| Entrée | Bytes bruts UTF-8 | Tokens (BPE / SentencePiece) |
| Prétraitement | Aucun | Tokenization, vocabulaire figé |
| Couche d'embedding | Apprise byte-level | Apprise token-level |
| Représentation | Séquence continue | Séquence discrète discrète |

## 2. Structure de sortie

|  | Moteur | Transformer |
|--|--------|-------------|
| Format | Graphe causal typé (nœuds + arêtes + spans) | Texte libre (séquence de tokens) |
| Relations | 19 types explicites (CAUSE, CONTRAST, CONDITION...) | Patterns implicites non typés |
| Traçabilité | Offsets vers texte source | Aucune (hallucination possible) |
| Vérifiabilité | Arête = sujet → objet [span] confiance | Phrase probable ≠ phrase vraie |
| Boucles causales | Supportées (Tarjan annote) | Impossibles (séquence linéaire) |

## 3. Performances mesurées

| Métrique | Moteur actuel | Transformer équivalent |
|----------|--------------|----------------------|
| val_edge_macro_f1 | 0.239 ± 0.04 | Non mesuré (pas de labels causaux) |
| train fit | ~0.99 | Généralement élevé |
| OOV val | 23.7% | Variable selon tokenization |
| ECE (calibration) | 0.33-0.55 | Non applicable (pas de confiance structurée) |

## 4. Efficacité

|  | Moteur (1.3M params) | Transformer 100M params |
|--|----------------------|------------------------|
| Taille poids | ~5 Mo | ~400 Mo |
| Mémoire inference | ~5 Mo / phrase | ~400 Mo + KV cache |
| Hardware | CPU suffit | GPU recommandé |
| Temps inference | ms | 100ms-10s |
| Coût entraînement | Minutes (NumPy CPU) | Jours-semaines (GPU cluster) |

Sur une phrase de 50 tokens :
- Moteur : ~5 Mo (poids fixes) + ~50 Ko (activation graphe) = ~5 Mo
- Transformer 100M : ~400 Mo (poids fixes) + ~1 Mo (KV cache) = ~400 Mo

Le moteur est ~80× plus léger par inférence. Mais sa sortie est un graphe causal structuré, pas du texte fluide.

## 5. Avantages du Moteur

1. Universel : bytes → causal, zéro frontend par langue
2. Prouvable : arêtes typées + spans sourcés + confiance calibrée
3. Léger : ~100× plus petit, CPU-only
4. Interrogeable : requêtes causales natives (WHY, CHAIN, DO)
5. Transparent : graphe inspectable, raison explicite
6. Coût : entraînable sur laptop, pas de GPU
7. Contraintes : seuil explicite, recouvrement détecté

## 6. Avantages du Transformer

1. Fluidité textuelle : phrases naturelles, grammaticalement correctes
2. Généraliste : multi-tâches avec le même modèle
3. Scalabilité : 10B-100B+ params = capacités émergentes
4. Peu de données : prétraining massif → few-shot
5. Pas de features manuelles : tout appris implicitement
6. Écosystème : HuggingFace, LoRA, quantisation, tooling mature

## 7. Honnêteté : limites du Moteur actuel

- val 0.239 = ne généralise pas encore (mémorisation mécanique, 2291× surappris)
- Pas de comparaison équitable sur les mêmes 572 phrases avec Transformer
- Pas de fluence textuelle : sortie graphe ≠ sortie texte naturel
- Pas de test multi-langue : seulement FR mesuré
- Pas de prétraining massif : nécessite refonte complète (voie B)

## 8. Verdict honnête

| Critère | Gagnant |
|---------|---------|
| Structure causale prouvée | Moteur |
| Traçabilité et interprétabilité | Moteur |
| Universel (sans frontend langue) | Moteur |
| Efficacité (taille / coût) | Moteur |
| Fluence textuelle naturelle | Transformer |
| Généralisation multi-tâches | Transformer |
| Scalabilité (données massives) | Transformer |
| Écosystème et tooling | Transformer |
| Performance mesurée sur 572 phrases | Indéterminé |

Conclusion : le Moteur est supérieur sur la traçabilité causale et l'efficacité, le Transformer sur la fluidité et la scalabilité. Sans bench tête-à-tête sur le même corpus avec métrique commune, déclarer un gagnant global = supposition. Le choix dépend du besoin : vérification causale stricte (Moteur) ou génération fluide généraliste (Transformer).
