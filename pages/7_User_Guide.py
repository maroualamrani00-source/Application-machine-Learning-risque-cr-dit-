from __future__ import annotations

import streamlit as st

from utils.ui import page_header, section_header

st.set_page_config(page_title="Guide d'utilisation • Credit Risk", layout="wide")

page_header("Guide d'utilisation", "Courte aide à l'usage des pages de l'application")

section_header("Data", "Aperçu et exploration")
st.write(
    "La page **Data** permet d'explorer le jeu de données brut : aperçu des colonnes, comptage des valeurs manquantes et visualisation de la distribution de la variable cible."
)

section_header("Cleaning", "Nettoyage et préparation")
st.write(
    "La page **Cleaning** offre des options pour gérer les valeurs manquantes et les valeurs extrêmes (clipping / percentiles). Après application, le jeu de données nettoyé est enregistré en session pour les étapes suivantes."
)

section_header("Modeling", "Entraînement et comparaison")
st.write(
    "La page **Modeling** permet d'entraîner plusieurs modèles standard, d'afficher un tableau comparatif de métriques et d'inspecter les performances au seuil décisionnel choisi. Elle permet aussi de sauvegarder un modèle comme artéfact réutilisable par la page de prédiction."
)

section_header("Predict", "Prédictions individuelles et en lot")
st.write(
    "La page **Predict** accepte soit une saisie manuelle pour une prédiction individuelle, soit un téléversement CSV pour des prédictions en lot. Une vérification simple des plages numériques (issues de l'entraînement) est affichée à titre informatif."
)

section_header("Conseils d'usage", None)
st.write(
    "- Vérifiez la configuration de nettoyage avant d'entraîner un modèle.\n"
    "- Comparez les modèles et inspectez les métriques avant de sauvegarder un artéfact.\n"
    "- Utilisez le mode en lot pour obtenir rapidement des prédictions sur un jeu de nouveaux exemples."
)
