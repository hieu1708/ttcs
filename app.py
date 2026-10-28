
import streamlit as st

st.set_page_config(page_title="Code RAG Assistant", layout="wide")

from core.i18n import t  # noqa: E402
from views import admin, user  # noqa: E402

st.session_state.setdefault("lang", "vi")

with st.sidebar:
    st.markdown("##  Code RAG Assistant")
    st.selectbox("Ngôn ngữ / Language", ["vi", "en"], key="lang",
                 format_func=lambda x: {"vi": "Tiếng Việt", "en": "English"}[x])
    page = st.radio(t("nav"), ["user", "admin"], key="page", format_func=lambda p: t("nav_" + p))
    st.divider()
    st.caption(t("app_footer"))

(admin if page == "admin" else user).render()
