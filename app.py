from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Credit Risk", layout="wide")

# Shared settings in sidebar
if "decision_threshold" not in st.session_state:
    st.session_state["decision_threshold"] = 0.5
if "show_details" not in st.session_state:
    st.session_state["show_details"] = False
if "rows_preview" not in st.session_state:
    st.session_state["rows_preview"] = 50

with st.sidebar.expander("Paramètres", expanded=True):
    st.markdown("**Paramètres globaux**")
    st.caption("Réglages applicables à la prédiction et à l'affichage des rapports")

    st.session_state["decision_threshold"] = st.slider(
        "Seuil de décision",
        min_value=0.1,
        max_value=0.9,
        value=float(st.session_state.get("decision_threshold", 0.5)),
        step=0.01,
        key="sidebar_decision_threshold",
    )

    st.session_state["show_details"] = st.checkbox(
        "Afficher détails",
        value=bool(st.session_state.get("show_details", False)),
        key="sidebar_show_details",
    )

    # Centralized preview rows slider (used by Data page)
    st.session_state["rows_preview"] = st.sidebar.slider(
        "Nombre de lignes à afficher",
        min_value=10,
        max_value=200,
        value=int(st.session_state.get("rows_preview", 50)),
        step=10,
        key="sidebar_rows_preview",
    )

    st.caption("Les paramètres ci-dessus contrôlent l'affichage et l'évaluation des modèles.")

try:
    st.switch_page("pages/1_Data.py")
except Exception:
    # Fallback for environments where switching isn't available for some reason.
    st.title("Credit Risk")
    st.caption("Open the pages menu (top-left) and select **Data** to start.")

