from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import SessionKeys, get_raw_df
from src.constants import TARGET_COL
from src.modeling import load_artifact
from src.app_state import ARTIFACT_PATH
from utils.ui import page_header, section_header

st.set_page_config(page_title="Évaluation du modèle • Credit Risk", layout="wide")

page_header("Évaluation du modèle", "Présentation des performances calculées et interprétation")

# Prefer in-session results; fall back to saved artifact
if SessionKeys.MODEL_RESULTS in st.session_state:
    res_df = pd.DataFrame(st.session_state[SessionKeys.MODEL_RESULTS]).sort_values("f1", ascending=False)
    section_header("Résultats disponibles", "Tableau récapitulatif des métriques")
    st.dataframe(res_df, use_container_width=True)
else:
    if ARTIFACT_PATH.exists():
        try:
            loaded = load_artifact(ARTIFACT_PATH)
            meta = loaded.get("metadata", {}) or {}
            metrics = meta.get("metrics") or {}
            if isinstance(metrics, dict):
                section_header("Résultats (modèle sauvegardé)", "Métriques extraites du modèle sauvegardé")
                st.write(pd.DataFrame([metrics]))
            else:
                st.info("Aucun jeu de métriques disponible dans l'artéfact sauvegardé.")
        except Exception as e:
            st.warning(f"Impossible de lire l'artéfact: {e}")
    else:
        st.info("Aucun résultat d'entraînement disponible en session ni d'artéfact sauvegardé. Voir la page 'Modeling'.")

section_header("Interprétation des métriques", None)
st.write(
    "- **Accuracy** : proportion d'exemples correctement classés.\n"
    "- **Precision** : part des prédictions positives correctes (utile quand les faux positifs sont coûteux).\n"
    "- **Recall** : part des exemples positifs correctement identifiés (utile quand les faux négatifs sont coûteux).\n"
    "- **F1** : moyenne harmonique de la précision et du rappel; synthèse robuste quand la classe est déséquilibrée.\n"
    "- **ROC AUC** : area under the ROC curve; synthétise la capacité du modèle à séparer les classes sur l'ensemble des seuils."
)

section_header("Remarque", None)
st.write(
    "Les métriques présentées correspondent aux évaluations calculées pendant la phase d'entraînement/validation. "
    "Pour obtenir des métriques calculées au seuil choisi, utilisez l'option correspondante dans la page 'Modeling'."
)
