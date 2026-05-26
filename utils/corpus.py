import streamlit as st
from config import GEN_CORPUS_N, GEN_CORPUS_DAYS


@st.cache_resource
def load_builtin_corpus() -> "tuple[object, dict]":
    """Genere le corpus de reference au premier appel, mis en cache memoire."""
    from models.generator import CurveGenerator
    gen = CurveGenerator()
    return gen.build_corpus(GEN_CORPUS_N, GEN_CORPUS_DAYS)
