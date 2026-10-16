# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Scraper de sites educatifs francais."""

import time
import trafilatura


class EducationScraper:
    """Extrait du texte depuis des pages web educatives."""

    URLS = [
        "https://fr.wikipedia.org/wiki/Connecteur_logique",
        "https://fr.wikipedia.org/wiki/Proposition_causale",
        "https://fr.wikipedia.org/wiki/Proposition_conditionnelle",
        "https://fr.wikipedia.org/wiki/Proposition_concessive",
        "https://fr.wikipedia.org/wiki/Relation_causale",
        "https://fr.wikipedia.org/wiki/Logique",
        "https://fr.wikipedia.org/wiki/Physique",
        "https://fr.wikipedia.org/wiki/Chimie",
        "https://fr.wikipedia.org/wiki/Biologie",
        "https://fr.wikipedia.org/wiki/Mathematiques",
        "https://fr.wikipedia.org/wiki/Informatique",
        "https://fr.wikipedia.org/wiki/Astronomie",
        "https://fr.wikipedia.org/wiki/Ecologie",
        "https://fr.wikipedia.org/wiki/Medecine",
        # Vikidia (encyclopédie junior) — causalité simple
        "https://fr.vikidia.org/wiki/Cause",
        "https://fr.vikidia.org/wiki/Expérience_scientifique",
        "https://fr.vikidia.org/wiki/Gravité",
        "https://fr.vikidia.org/wiki/Photosynthèse",
        "https://fr.vikidia.org/wiki/Électricité",
        "https://fr.vikidia.org/wiki/Maladie",
        "https://fr.vikidia.org/wiki/Climat",
        "https://fr.vikidia.org/wiki/Volcan",
        # Wikiversité — cours
        "https://fr.wikiversity.org/wiki/Introduction_à_la_logique",
        "https://fr.wikiversity.org/wiki/Cause_et_conséquence",
        "https://fr.wikiversity.org/wiki/Biologie_cellulaire",
        "https://fr.wikiversity.org/wiki/Thermodynamique",
        # Sésamath (manuels maths)
        "https://www.sesamath.net/",
        # EN — Khan Academy / éducatif
        "https://en.wikipedia.org/wiki/Causality",
        "https://en.wikipedia.org/wiki/Causal_reasoning",
        "https://en.wikipedia.org/wiki/Scientific_method",
        "https://en.wikipedia.org/wiki/Photosynthesis",
        "https://en.wikipedia.org/wiki/Gravity",
        "https://en.wikipedia.org/wiki/Electricity",
        "https://en.wikipedia.org/wiki/Disease",
        "https://en.wikipedia.org/wiki/Climate_change",
        "https://simple.wikipedia.org/wiki/Cause_and_effect",
        "https://simple.wikipedia.org/wiki/Science",
        "https://simple.wikipedia.org/wiki/Biology",
        "https://simple.wikipedia.org/wiki/Physics",
        "https://simple.wikipedia.org/wiki/Chemistry",
        # Vague 2 (2026-10-02) : simple + avancé
        # Simple FR (Vikidia)
        "https://fr.vikidia.org/wiki/Eau",
        "https://fr.vikidia.org/wiki/Air",
        "https://fr.vikidia.org/wiki/Soleil",
        "https://fr.vikidia.org/wiki/Terre",
        "https://fr.vikidia.org/wiki/Corps_humain",
        "https://fr.vikidia.org/wiki/Animaux",
        "https://fr.vikidia.org/wiki/Plantes",
        "https://fr.vikidia.org/wiki/Énergie",
        # Simple EN
        "https://simple.wikipedia.org/wiki/Water",
        "https://simple.wikipedia.org/wiki/Air",
        "https://simple.wikipedia.org/wiki/Fire",
        "https://simple.wikipedia.org/wiki/Earth",
        "https://simple.wikipedia.org/wiki/Animals",
        "https://simple.wikipedia.org/wiki/Plants",
        "https://simple.wikipedia.org/wiki/Human_body",
        "https://simple.wikipedia.org/wiki/Energy",
        # Avancé FR
        "https://fr.wikipedia.org/wiki/Antibiotique",
        "https://fr.wikipedia.org/wiki/Vaccin",
        "https://fr.wikipedia.org/wiki/Ordinateur_quantique",
        "https://fr.wikipedia.org/wiki/Relativité_restreinte",
        "https://fr.wikipedia.org/wiki/Évolution_(biologie)",
        # Avancé EN
        "https://en.wikipedia.org/wiki/Antibiotic",
        "https://en.wikipedia.org/wiki/Vaccine",
        "https://en.wikipedia.org/wiki/Quantum_computing",
        "https://en.wikipedia.org/wiki/Theory_of_relativity",
        "https://en.wikipedia.org/wiki/Evolution",
    ]

    def __init__(self, user_agent: str = "GCN-Dataset/1.0 (research)"):
        self.user_agent = user_agent

    def scrape_url(self, url: str) -> str | None:
        """Telecharge et nettoie une page web."""
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
        return None

    def scrape_all(self) -> list[dict]:
        """Scrape toutes les URLs."""
        results = []
        for url in self.URLS:
            text = self.scrape_url(url)
            if text and len(text) > 200:
                title = url.split("/")[-1].replace("_", " ")
                results.append({
                    "title": title,
                    "text": text,
                    "url": url,
                    "source": "education",
                })
            time.sleep(0.5)
        return results
