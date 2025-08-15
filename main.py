import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar el estilo
sns.set(style="whitegrid")
st.set_page_config(page_title="Análisis Exploratorio de Datos", layout="wide")

# ---------- 1. Generar datos aleatorios ----------
@st.cache_data
def generar_datos(n=100):
    np.random.seed(42)
    return pd.DataFrame({
        'Categoría': np.random.choice(['A', 'B', 'C', 'D'], size=n),
        'Mes': np.random.choice(['Ene', 'Feb', 'Mar', 'Abr'], size=n),
        'Ventas': np.random.randint(100, 1000, size=n),
        'Costo': np.random.randint(50, 900, size=n)
    })

df = generar_datos()

# ---------- 2. Mostrar datos ----------
st.title("📊 Análisis Exploratorio de Datos")
st.markdown("Este dashboard muestra un análisis exploratorio básico de un conjunto de datos simulados.")

st.subheader("🔍 Vista previa de los datos")
st.dataframe(df.head())

st.subheader("📈 Estadísticas descriptivas")
st.dataframe(df.describe())

# ---------- 3. Visualizaciones ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Promedio de Ventas por Categoría")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=df, x='Categoría', y='Ventas', estimator='mean', ci=None, ax=ax1)
    ax1.set_ylabel("Ventas promedio")
    st.pyplot(fig1)

with col2:
    st.subheader("📊 Promedio de Costo por Mes")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x='Mes', y='Costo', estimator='mean', ci=None, palette='muted', ax=ax2)
    ax2.set_ylabel("Costo promedio")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("📈 Tendencia de Ventas Promedio por Mes")
    ventas_mes = df.groupby('Mes')['Ventas'].mean().sort_index()
    fig3, ax3 = plt.subplots()
    ventas_mes.plot(kind='line', marker='o', ax=ax3)
    ax3.set_ylabel("Ventas promedio")
    ax3.set_xlabel("Mes")
    st.pyplot(fig3)

with col4:
    st.subheader("📉 Relación entre Ventas y Costo por Categoría")
    relacion = df.groupby('Categoría')[['Ventas', 'Costo']].mean()
    fig4, ax4 = plt.subplots()
    relacion.plot(kind='line', marker='o', ax=ax4)
    ax4.set_ylabel("Valor promedio")
    ax4.set_title("Ventas vs Costo por Categoría")
    st.pyplot(fig4)

st.success("Se han generado 4 visualizaciones interactivas.")
