from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import get_raw_df, require_target_column
from src.constants import TARGET_COL
from utils.validators import get_feature_columns


st.set_page_config(page_title="Data • Credit Risk", layout="wide")

from utils.ui import page_header, section_header

page_header("Data", "Aperçu du jeu de données et résumés statistiques")

df = get_raw_df()
require_target_column(df)

show_details = st.session_state.get("show_details", False)

num_cols, cat_cols = get_feature_columns(df)

section_header("Valeurs manquantes", "Colonnes classées par nombre de valeurs manquantes (top 10).")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False).head(10)
if not missing.empty:
    st.dataframe(missing.rename("missing_count").reset_index().rename(columns={"index": "colonne"}), use_container_width=True)
else:
    st.info("Aucune valeur manquante détectée.")

st.divider()
# Quick filters when categorical columns are available
with st.expander("Filtres", expanded=False):
    st.caption("Appliquer un filtre simple sur les colonnes catégorielles.")
    filter_col = None
    filter_val = None
    if cat_cols:
        filter_col = st.selectbox("Filtrer par colonne catégorielle", options=["(none)"] + cat_cols, index=0)
        if filter_col and filter_col != "(none)":
            vals = sorted(df[filter_col].dropna().astype(str).unique().tolist()) if filter_col in df.columns else []
            filter_val = st.selectbox("Valeur", options=["(all)"] + vals, index=0)

# Ensure a default rows_preview is set and use the centralized sidebar value
if "rows_preview" not in st.session_state:
    st.session_state["rows_preview"] = 50

st.subheader("Aperçu")
st.caption("Affichage des premières lignes du jeu de données ou du sous-ensemble filtré.")
view = df
if filter_col and filter_col != "(none)" and filter_val and filter_val != "(all)":
    view = df[df[filter_col].astype(str) == str(filter_val)]

top = int(st.session_state.get("rows_preview", 50))
st.dataframe(view.head(top), use_container_width=True)

# Télécharger la vue actuelle
st.download_button(
    "Télécharger la vue (CSV)",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="dataset_view.csv",
    mime="text/csv",
)

# Télécharger le jeu de données complet
st.download_button(
    "Télécharger le jeu de données (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="dataset_full.csv",
    mime="text/csv",
)

st.divider()
st.header("Résumé")
st.caption("Mesures clés du jeu de données.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", int(df.shape[0]))
c2.metric("Colonnes", int(df.shape[1]))
c3.metric("Valeurs manquantes", int(df.isna().sum().sum()))
c4.metric("Doublons", int(df.duplicated().sum()))

with st.expander("Dtypes", expanded=show_details):
    st.json(df.dtypes.astype(str).to_dict())

with st.expander(f"Distribution de la cible : `{TARGET_COL}`", expanded=False if not show_details else True):
    vc = df[TARGET_COL].value_counts(dropna=False).rename_axis(TARGET_COL).reset_index(name="count")
    st.dataframe(vc, use_container_width=True)

st.divider()
if show_details:
    st.subheader("Pandas describe")
    st.dataframe(df.describe(include="all"), use_container_width=True)



