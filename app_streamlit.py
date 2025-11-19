import streamlit as st
import joblib
import os

# Caminho do modelo (pipeline TF-IDF + MLP) salvo no notebook
CAMINHO_MODELO = os.path.join(
    os.path.dirname(__file__),  # pasta onde está o app_streamlit.py
    "models",
    "modelo_hatespeech_mlp.pkl"
)

@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(CAMINHO_MODELO)
    return modelo

modelo = carregar_modelo()

# ---------------- INTERFACE ----------------

st.set_page_config(page_title="Detecção de Discurso Ofensivo", page_icon="⚠")

st.title("🧠 Detecção de Discurso Ofensivo em Português")
st.write(
    """
Esta aplicação usa um modelo de *Machine Learning* (TF-IDF + MLPClassifier) 
treinado no dataset **OLID-BR** para identificar automaticamente 
comentários potencialmente ofensivos.
"""
)

texto_usuario = st.text_area(
    "Digite ou cole um comentário para análise:",
    height=150,
    placeholder="Ex: Eu discordo de você, mas respeito sua opinião."
)

if st.button("Analisar texto"):
    if not texto_usuario.strip():
        st.warning("Por favor, insira algum texto para análise.")
    else:
        # Probabilidade da classe 1 (ofensivo)
        prob_ofensivo = modelo.predict_proba([texto_usuario])[0][1]

        # Limiar de decisão - você pode ajustar (0.7 é um bom começo)
        limiar = 0.7

        pred = 1 if prob_ofensivo >= limiar else 0

        if pred == 1:
            st.error("⚠ O modelo classificou o texto como **OFENSIVO**.")
        else:
            st.success("✅ O modelo classificou o texto como **NÃO OFENSIVO**.")
        st.write(f"Probabilidade estimada de ser ofensivo: **{prob_ofensivo:.2%}**")

        with st.expander("Como interpretar este resultado?"):
            st.markdown(
                """
- Este modelo foi treinado em um conjunto específico de dados (OLID-BR).
- Ele **pode errar**, especialmente em frases irônicas, ambíguas ou sem contexto.
- O objetivo é servir como **ferramenta de apoio** à moderação de conteúdo,
  e não substituir a análise humana.
"""
            )

st.markdown("---")
st.caption("Projeto acadêmico de detecção de discurso ofensivo em textos em língua portuguesa.")
