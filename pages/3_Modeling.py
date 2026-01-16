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
from src.constants import TARGET_COL
from src.modeling import TrainConfig, save_artifact, train_and_evaluate


st.set_page_config(page_title="Modeling • Credit Risk", layout="wide")

from utils.ui import page_header, section_header

page_header("Modeling", "Entraînez et comparez plusieurs modèles sur le jeu de données nettoyé")

raw_df = get_raw_df()
require_target_column(raw_df)

section_header("Configuration d'entraînement", "Réglages pour l'entraînement et la validation")

t1, t2 = st.columns(2)
with t1:
    test_size = st.slider("Taille du test", 0.10, 0.50, 0.30, 0.05, key="modeling_test_size")
with t2:
    seed = st.number_input("Graine aléatoire", value=42, step=1)

st.divider()

clean_df, _report = compute_cleaned(raw_df)
num_cols, cat_cols = get_feature_columns(clean_df)

models = ["LogReg", "RandomForest", "GradientBoosting"]

if st.button("Entraîner et évaluer (3 modèles)", type="primary"):
    results: list[dict[str, object]] = []
    pipes: dict[str, dict[str, object]] = {}

    with st.spinner("Entraînement des modèles..."):
        for name in models:
            pipe, metrics = train_and_evaluate(
                clean_df,
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                cfg=TrainConfig(
                    target_col=TARGET_COL,
                    test_size=float(test_size),
                    random_state=int(seed),
                    model_name=name,
                ),
            )
            results.append(
                {k: v for k, v in metrics.items() if k in {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}}
            )
            pipes[name] = {"pipeline": pipe, "metrics": metrics}

    st.session_state[SessionKeys.MODEL_RESULTS] = results
    st.session_state[SessionKeys.MODEL_PIPES] = pipes
    st.success("Done.")


if SessionKeys.MODEL_RESULTS not in st.session_state:
    st.info("Entraînez des modèles pour comparer les performances.")
    st.stop()

res_df = pd.DataFrame(st.session_state[SessionKeys.MODEL_RESULTS]).sort_values("f1", ascending=False)
st.subheader("Résultats")
st.caption("Tableau récapitulatif des métriques pour les modèles entraînés.")
st.dataframe(res_df, use_container_width=True)

best = str(res_df.iloc[0]["model"])
st.caption(f"Best by F1: **{best}**")

chosen = st.selectbox("Model to save / use", options=res_df["model"].tolist(), index=0)

show_details = st.session_state.get("show_details", False)
# Evaluation with adjustable threshold
thr = float(st.session_state.get("decision_threshold", 0.5))
with st.expander("Evaluate results with current threshold", expanded=show_details):
    st.caption(f"Threshold used for classification: **{thr:.2f}**")
    sel_model = st.selectbox("Select model to inspect", options=res_df["model"].tolist(), index=0)
    metrics_for = st.session_state[SessionKeys.MODEL_PIPES][sel_model]["metrics"]

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    y_true = metrics_for.get("y_test_values")
    y_proba = metrics_for.get("y_test_proba")

    if y_true is None:
        st.info("No test split details available for this model to compute thresholded metrics.")
    else:
        if y_proba is not None:
            import numpy as _np

            y_hat = [1 if p >= thr else 0 for p in y_proba]
        else:
            st.info("Model does not expose probabilities; using provided metrics instead.")
            y_hat = None

        if y_hat is not None:
            cm = confusion_matrix(y_true, y_hat)
            acc = accuracy_score(y_true, y_hat)
            prec = precision_score(y_true, y_hat, zero_division=0)
            rec = recall_score(y_true, y_hat, zero_division=0)
            f1 = f1_score(y_true, y_hat, zero_division=0)

            st.write("**Metrics at threshold**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("F1", f"{f1:.3f}")

            st.subheader("Confusion matrix")
            cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm_df, use_container_width=True)

col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("Sauvegarder le modèle sélectionné", type="secondary"):
        bundle = st.session_state[SessionKeys.MODEL_PIPES][chosen]
        meta = {
            "model_name": chosen,
            "metrics": bundle["metrics"],
            "cleaning_config": st.session_state.get(SessionKeys.CLEANING_CFG).__dict__
            if st.session_state.get(SessionKeys.CLEANING_CFG)
            else None,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
        }
        save_artifact(ARTIFACT_PATH, pipeline=bundle["pipeline"], metadata=meta)
        st.success(f"Sauvegardé : {ARTIFACT_PATH}")

with col_b:
    with st.expander("Rapport de classification du meilleur modèle", expanded=show_details):
        st.text(st.session_state[SessionKeys.MODEL_PIPES][best]["metrics"]["classification_report"]) 


