from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import get_raw_df, require_target_column
from src.constants import TARGET_COL
from utils.ui import page_header, section_header

st.set_page_config(page_title="Présentation • Credit Risk", layout="wide")

page_header("Présentation", "Résumé synthétique du projet")

# Contexte
section_header("Objectif du projet", "But et périmètre")
st.write(
    "Ce projet met en œuvre un modèle de classification destiné à estimer le risque de crédit. "
    "Le modèle vise à prédire la colonne cible : ``{}``.".format(TARGET_COL)
)

# Jeu de données
section_header("Jeu de données", "Taille et caractéristiques principales")
raw = get_raw_df()
require_target_column(raw)
st.write(f"Taille du jeu de données : **{raw.shape[0]}** observations × **{raw.shape[1]}** colonnes")

dtypes = raw.dtypes.astype(str).rename("dtype").reset_index().rename(columns={"index": "colonne"})
st.dataframe(dtypes, use_container_width=True)

with st.expander("Aperçu des premières lignes", expanded=False):
    st.dataframe(raw.head(6), use_container_width=True)

section_header("Remarques", None)
st.write(
    "Les informations ci-dessus sont extraites automatiquement du jeu de données chargé par l'application. "
    "Elles servent d'introduction générale et n'altèrent en rien les étapes d'entraînement ou d'évaluation du modèle."
)
