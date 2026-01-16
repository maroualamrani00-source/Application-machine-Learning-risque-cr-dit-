from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import get_raw_df, get_feature_columns, require_target_column
from utils.ui import page_header, section_header

st.set_page_config(page_title="Description des variables • Credit Risk", layout="wide")

page_header("Description des variables", "Détail structuré des colonnes présentes dans le jeu de données")

raw = get_raw_df()
require_target_column(raw)
num_cols, cat_cols = get_feature_columns(raw)

# Build descriptive table
rows: list[dict] = []
for c in raw.columns:
    s = raw[c]
    row = {
        "colonne": c,
        "type_suggérée": "numérique" if c in num_cols else ("catégorielle" if c in cat_cols else "autre"),
        "dtype": str(s.dtype),
        "n_missing": int(s.isna().sum()),
        "n_unique": int(s.nunique(dropna=True)),
    }
    # add compact sample
    if c in cat_cols:
        vals = s.dropna().astype(str).unique().tolist()[:5]
        row["exemples"] = ", ".join(vals)
    else:
        try:
            row["exemples"] = f"min={s.min():.3g}, méd={s.median():.3g}, max={s.max():.3g}"
        except Exception:
            row["exemples"] = "-"
    rows.append(row)

meta_df = pd.DataFrame(rows).sort_values("type_suggérée")
st.dataframe(meta_df, use_container_width=True)

section_header("Note méthodologique", None)
st.write(
    "Les types indiqués ci-dessus sont inférés à partir du contenu des colonnes. "
    "Pour toute documentation formelle des variables, se référer aux sources de données d'origine.")
