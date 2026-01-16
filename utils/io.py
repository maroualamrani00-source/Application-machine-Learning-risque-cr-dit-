from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.constants import DEFAULT_DATASET_PATH, ARTIFACTS_DIR
from src.data_io import read_dataset_from_path
from src.modeling import load_artifact


@st.cache_data(show_spinner=False)
def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path) if path is not None else DEFAULT_DATASET_PATH
    return read_dataset_from_path(p)


@st.cache_resource(show_spinner=False)
def load_model(path: Optional[Path] = None) -> dict | None:
    p = Path(path) if path is not None else ARTIFACTS_DIR / "credit_risk_model.joblib"
    if not p.exists():
        return None
    return load_artifact(p)
