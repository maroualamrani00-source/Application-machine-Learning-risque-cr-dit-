from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import (
    ARTIFACT_PATH,
    SessionKeys,
    compute_cleaned,
    get_feature_columns,
    get_raw_df,
    require_target_column,
)
from src.modeling import load_artifact


st.set_page_config(page_title="Prédiction • Credit Risk", layout="wide")

from utils.ui import page_header, section_header

page_header("Prédiction", "Prédictions individuelles et en lot à partir de nouvelles entrées")

raw_df = get_raw_df()
require_target_column(raw_df)

clean_df, _report = compute_cleaned(raw_df)
num_cols, cat_cols = get_feature_columns(clean_df)

# Resolve model source
pipe = None
meta: dict[str, object] = {}
numeric_ranges: dict[str, dict[str, float]] = {}

if ARTIFACT_PATH.exists():
    try:
        loaded = load_artifact(ARTIFACT_PATH)
        pipe = loaded["pipeline"]
        meta = loaded.get("metadata", {}) or {}
        numeric_ranges = (meta.get("metrics") or {}).get("numeric_ranges", {})  # type: ignore[assignment]
        st.success(f"Utilisation du modèle sauvegardé : `{ARTIFACT_PATH}`")
    except Exception as e:
        st.warning(f"Le modèle sauvegardé existe mais n'a pas pu être chargé: {e}")

if pipe is None:
    if SessionKeys.MODEL_RESULTS not in st.session_state or SessionKeys.MODEL_PIPES not in st.session_state:
        st.info("Entraînez et sauvegardez d'abord un modèle (voir la page **Modeling**).")
        st.stop()
    res_df = pd.DataFrame(st.session_state[SessionKeys.MODEL_RESULTS]).sort_values("f1", ascending=False)
    best = str(res_df.iloc[0]["model"])
    pipe = st.session_state[SessionKeys.MODEL_PIPES][best]["pipeline"]
    numeric_ranges = st.session_state[SessionKeys.MODEL_PIPES][best]["metrics"].get("numeric_ranges", {})
    st.info(f"Utilisation du modèle en session : **{best}** (non sauvegardé)")

# Prefer metadata columns when available
numeric_for_form = meta.get("numeric_cols", num_cols) if isinstance(meta, dict) else num_cols
categorical_for_form = meta.get("categorical_cols", cat_cols) if isinstance(meta, dict) else cat_cols

from utils import validators

thr = float(st.session_state.get("decision_threshold", 0.5))
show_details = st.session_state.get("show_details", False)


def _default_numeric(col: str) -> float:
    if col in clean_df.columns:
        return float(pd.to_numeric(clean_df[col], errors="coerce").median())
    return 0.0


section_header("Entrée", "Saisissez les caractéristiques numériques et catégorielles pour une prédiction individuelle.")

with st.form("predict_form", border=True):
    left, right = st.columns(2)
    payload: dict[str, object] = {}

    with left:
        st.caption("Numeric")
        for col in list(numeric_for_form):
            val = st.number_input(col, value=float(_default_numeric(col)))
            payload[col] = float(val)
            if col in numeric_ranges:
                r = numeric_ranges[col]
                if float(val) < r["min"] or float(val) > r["max"]:
                    st.caption(f"Remarque : `{col}` hors plage d'entraînement [{r['min']:.3g}, {r['max']:.3g}]")
        for col in list(categorical_for_form):
            if col in clean_df.columns:
                options = sorted(clean_df[col].dropna().astype(str).unique().tolist()) or ["UNKNOWN"]
            else:
                options = ["UNKNOWN"]
            payload[col] = st.selectbox(col, options=options, index=0)

    submitted = st.form_submit_button("Prédire", type="primary")


if submitted:
    x = pd.DataFrame([payload])
    # validate numerics
    issues = validators.validate_payload_numeric_ranges(payload, numeric_ranges)
    if issues:
        for it in issues:
            st.warning(it)

    try:
        proba = pipe.predict_proba(x)[0] if hasattr(pipe, "predict_proba") else None
        if proba is not None and len(proba) >= 2:
            p_pos = float(proba[1])
            label = "Risque Faible" if p_pos >= thr else "Risque Elevé"
            st.metric("Prédiction", label)

            if show_details:
                out = (
                    pd.DataFrame({"Class": ["Risque Elevé", "Risque Faible"], "Probability": [float(proba[0]), float(proba[1])]})
                    .sort_values("Probability", ascending=False)
                    .reset_index(drop=True)
                )
                out["Probability"] = out["Probability"].round(3)
                st.subheader("Probabilités")
                st.dataframe(out, use_container_width=True)
            else:
                st.caption(f"Probabilité (classe prédite): {p_pos:.3f}")
        else:
            pred = pipe.predict(x)[0]
            if isinstance(pred, (int, float)) and int(pred) in (0, 1):
                label = "Risque Faible" if int(pred) == 1 else "Risque Elevé"
            else:
                label = str(pred)
            st.metric("Prédiction", label)
    except Exception as e:
        st.error(f"Prédiction échouée : {e}")

# Batch CSV mode
st.divider()
st.header("Prédictions en lot (CSV)")
st.caption("Téléversez un fichier CSV contenant les colonnes d'entrée attendues pour obtenir des prédictions en masse.")
uploaded = st.file_uploader("Téléverser un CSV (lignes d'entrée)", type=["csv"], accept_multiple_files=False)
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        missing = [c for c in list(numeric_for_form) + list(categorical_for_form) if c not in df_in.columns]
        if missing:
            st.error(f"Colonnes requises manquantes : {missing}")
        else:
            # try to coerce numeric cols
            df_in_coerced, failed = validators.coerce_numeric_columns(df_in, numeric_for_form)
            if failed:
                st.warning(f"Impossible de convertir ces colonnes numériques : {failed}")
            try:
                if hasattr(pipe, "predict_proba"):
                    probs = pipe.predict_proba(df_in_coerced)
                    p_pos = [float(x[1]) if len(x) > 1 else None for x in probs]
                    preds = [1 if (p is not None and p >= thr) else 0 for p in p_pos]
                else:
                    preds = pipe.predict(df_in_coerced).tolist()
                    p_pos = [None] * len(preds)

                out = df_in.copy()
                out["pred_proba"] = p_pos
                out["pred_class"] = ["Risque Faible" if int(x) == 1 else "Risque Elevé" for x in preds]

                if show_details:
                    st.dataframe(out.head(25), use_container_width=True)
                else:
                    # show small summary when details are off
                    st.write("**Résumé des prédictions**")
                    st.write(out["pred_class"].value_counts())
                    st.caption("Activez 'Afficher détails' dans la barre latérale pour voir toutes les prédictions.")

                st.download_button(
                    "Télécharger les prédictions (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Prédiction en lot échouée : {e}")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")


