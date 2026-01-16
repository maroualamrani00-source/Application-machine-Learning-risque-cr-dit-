from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import (
    compute_cleaned,
    get_feature_columns,
    get_raw_df,
    require_target_column,
    set_cleaning_cfg,
)
from src.cleaning import CleaningConfig
from src.constants import TARGET_COL


st.set_page_config(page_title="Cleaning • Credit Risk", layout="wide")

from utils.ui import page_header, section_header

page_header("Cleaning", "Configurer et appliquer les stratégies de nettoyage du jeu de données")

raw_df = get_raw_df()
require_target_column(raw_df)
numeric_cols, _cat_cols = get_feature_columns(raw_df)

section_header("Configuration de nettoyage", "Sélectionnez les stratégies et appliquez-les via le formulaire ci-dessous.")

with st.form("cleaning_form", border=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        numeric_missing = st.selectbox(
            "Numeric missing values",
            options=["drop", "mean", "median", "most_frequent"],
            index=1,
        )
        categorical_missing = st.selectbox(
            "Categorical missing values",
            options=["drop", "mode", "most_frequent"],
            index=1,
        )
        drop_dups = st.checkbox("Drop duplicate rows", value=True)

    with c2:
        outlier_method = st.selectbox(
            "Outlier method",
            options=["none", "zscore", "mean_std", "clip", "clip_percentile"],
            index=0,
        )
        outlier_cols = st.multiselect(
            "Columns for outlier handling",
            options=numeric_cols,
            default=numeric_cols[:1] if numeric_cols else [],
        )

    with c3:
        z_thr = st.slider("Z-score threshold (zscore)", 1.0, 5.0, 3.0, 0.1, key="cleaning_z_thr")
        k_std = st.slider("k * std (mean±k·std)", 1.0, 5.0, 3.0, 0.1, key="cleaning_k_std")
        pct_lower = st.number_input("Lower percentile (clip)", value=1.0, min_value=0.0, max_value=49.0, step=0.1, key="cleaning_pct_lower")
        pct_upper = st.number_input("Upper percentile (clip)", value=99.0, min_value=51.0, max_value=100.0, step=0.1, key="cleaning_pct_upper")
        st.caption("Use 'clip' for mean±k·std or 'clip_percentile' to clip to [lower%, upper%].")

    submitted = st.form_submit_button("Apply cleaning", type="primary")

if submitted:
    cfg = CleaningConfig(
        target_col=TARGET_COL,
        numeric_missing=numeric_missing,
        categorical_missing=categorical_missing,
        drop_duplicates=drop_dups,
        outlier_method=outlier_method,
        outlier_cols=outlier_cols,
        zscore_threshold=float(z_thr),
        mean_std_k=float(k_std),
        clip_lower_pct=float(pct_lower),
        clip_upper_pct=float(pct_upper),
    )
    set_cleaning_cfg(cfg)
    st.success("Configuration enregistrée — consultez la section Résultats pour voir l'impact du nettoyage.")

st.divider()
st.divider()
section_header("Résultats du nettoyage", "Résumé des principales métriques après application de la configuration choisie.")

clean_df, report = compute_cleaned(raw_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Lignes (avant)", report["rows_before"])
m2.metric("Lignes (après)", report["rows_after"])
m3.metric("Valeurs manquantes (avant)", report.get("nulls_before", 0))
m4.metric("Valeurs manquantes (après)", report.get("nulls_after", 0))

if report.get("outliers", {}).get("method") in {"clip", "clip_percentile"}:
    with st.expander("Outlier clipping details", expanded=False):
        st.json(report.get("outliers", {}))

with st.expander("Cleaning report (full)", expanded=False if not st.session_state.get("show_details", False) else True):
    st.json(report)

st.subheader("Aperçu")
st.dataframe(clean_df.head(25), use_container_width=True)

# Extra feature (only one): allow download of cleaned dataset
st.download_button(
    "Télécharger le jeu de données nettoyé (CSV)",
    data=clean_df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)


