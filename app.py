import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Preditor de Qualidade de Vinhos",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Inteligência Artificial para Classificação de Vinhos")
st.markdown("---")
st.markdown("**Desenvolvido por:**")
st.markdown("Maria Cláudia Geloneze Cangussú | Mariana da Silva | Murilo Tozoni")
st.markdown("*3° Termo - Curso de Inteligência Artificial (UNIMAR)*")
st.markdown("---")

st.write(
    "Insira os parâmetros físico-químicos do vinho tinto abaixo para que a aplicação "
    "classifique se ele pertence à classe de vinho comum ou vinho excelente."
)

@st.cache_resource
def carregar_modelo():
    caminho_modelo = Path(__file__).parent / "modelo_classificacao_vinhos.pkl"
    return joblib.load(caminho_modelo)

try:
    modelo = carregar_modelo()
except FileNotFoundError:
    st.error(
        "Erro: o arquivo 'modelo_classificacao_vinhos.pkl' não foi encontrado. "
        "Confira se ele está na mesma pasta do app.py no GitHub."
    )
    st.stop()
except Exception as erro:
    st.error(f"Erro ao carregar o modelo: {erro}")
    st.stop()

st.subheader("📊 Características Físico-Químicas")

fixed_acidity = st.slider("Acidez Fixa (fixed acidity)", 4.0, 16.0, 8.3, step=0.1)
volatile_acidity = st.slider("Acidez Volátil (volatile acidity)", 0.1, 1.6, 0.52, step=0.01)
citric_acid = st.slider("Ácido Cítrico (citric acid)", 0.0, 1.0, 0.27, step=0.01)
residual_sugar = st.slider("Açúcar Residual (residual sugar)", 0.9, 15.5, 2.5, step=0.1)
chlorides = st.slider("Cloretos (chlorides)", 0.01, 0.6, 0.08, step=0.001)
free_sulfur_dioxide = st.slider("Dióxido de Enxofre Livre (free sulfur dioxide)", 1.0, 72.0, 14.0, step=1.0)
total_sulfur_dioxide = st.slider("Dióxido de Enxofre Total (total sulfur dioxide)", 6.0, 289.0, 46.0, step=1.0)
density = st.slider("Densidade (density)", 0.990, 1.004, 0.996, step=0.0001)
pH = st.slider("Potencial Hidrogeniônico (pH)", 2.7, 4.0, 3.31, step=0.01)
sulphates = st.slider("Sulfatos (sulphates)", 0.3, 2.0, 0.65, step=0.01)
alcohol = st.slider("Teor Alcoólico (% alcohol)", 8.0, 15.0, 10.4, step=0.1)

st.markdown("---")

if st.button("🔮 Classificar Qualidade do Vinho", type="primary"):
    dados_usuario = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    predicao = modelo.predict(dados_usuario)[0]
    probabilidade = modelo.predict_proba(dados_usuario)[0]

    st.subheader("🎯 Resultado da Avaliação da IA")

    if predicao == 1:
        st.success("🍇 **VINHO EXCELENTE (Nota >= 7)**")
        st.write(f"Confiança do modelo: {probabilidade[1] * 100:.2f}% de chance de ser um vinho excelente.")
    else:
        st.warning("🧪 **VINHO COMUM (Nota < 7)**")
        st.write(f"Confiança do modelo: {probabilidade[0] * 100:.2f}% de chance de ser um vinho comum.")

    st.caption(
        "Observação: este resultado é uma predição baseada em um modelo de Regressão Logística "
        "treinado com dados físico-químicos de vinhos tintos."
    )
