"""
TaxonomyIndex: charge les fichiers YAML de taxonomies GCN et expose
l'appartenance par gcn_class_key = "taxonomy_name.class_name".

Pas de règles ici — uniquement le chargement des données.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class TaxonomyIndex:
    """
    Mappe chaque gcn_class_key vers un frozenset de lemmes.
    Ex: "verbes.etat" -> {"être", "avoir", "savoir", ...}

    Pour une langue sans fichier → ensemble vide → feature = 0.
    Le modèle apprend le poids de chaque dimension.
    """
    data: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, taxonomies_dir: Path, lang_code: str = "fr") -> "TaxonomyIndex":
        """
        Charge les taxonomies depuis:
          1. taxonomies_dir/{lang_code}/  (spécifiques à la langue)
          2. taxonomies_dir/              (partagées, fallback)
        """
        index: dict[str, frozenset[str]] = {}

        dirs_to_scan = []
        lang_dir = taxonomies_dir / lang_code
        if lang_dir.is_dir():
            dirs_to_scan.append(lang_dir)
        if taxonomies_dir.is_dir():
            dirs_to_scan.append(taxonomies_dir)

        for scan_dir in dirs_to_scan:
            for yaml_path in sorted(scan_dir.glob("*.yaml")):
                try:
                    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(doc, dict) or "taxonomy" not in doc:
                    continue
                tax_name = doc["taxonomy"]
                classes = doc.get("classes") or {}
                for class_name, class_data in classes.items():
                    key = f"{tax_name}.{class_name}"
                    if key in index:
                        continue  # lang-specific already loaded
                    lemmas: set[str] = set()
                    examples = (class_data or {}).get("examples_fr") or []
                    for entry in examples:
                        if isinstance(entry, dict) and "lemma" in entry:
                            lemmas.add(entry["lemma"].lower().strip())
                        elif isinstance(entry, str):
                            lemmas.add(entry.lower().strip())
                    index[key] = frozenset(lemmas)

        return cls(data=index)

    def membership(self, lemma: str) -> dict[str, bool]:
        """Retourne {gcn_class_key: True/False} pour un lemme."""
        lower = lemma.lower().strip()
        return {key: lower in lemmas for key, lemmas in self.data.items()}

    def keys(self) -> list[str]:
        return sorted(self.data.keys())

    def __len__(self) -> int:
        return len(self.data)
