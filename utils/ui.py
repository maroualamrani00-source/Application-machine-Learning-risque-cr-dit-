from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str | None = None) -> None:
    """Render a consistent page header: title, optional subtitle, and a divider."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()


def section_header(title: str, caption: str | None = None) -> None:
    """Render a consistent section header with optional caption and spacing."""
    st.header(title)
    if caption:
        st.caption(caption)
    st.markdown("\n")
