"""
principal.py
Aplicacion Streamlit para analizar datos de precios, cantidades y costos
de 4 mercados (archivo tipo PBA_4_mercados.csv).

Ejecutar con:
    streamlit run principal.py
"""

import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import anthropic

st.set_page_config(page_title="Analisis de Mercados", layout="wide")

METRICAS = ["Precio de Venta", "Cantidad producida & vendida", "Costo unitario"]


# ------------------------------------------------------------------
# Carga y limpieza de datos
# ------------------------------------------------------------------
@st.cache_data
def cargar_datos(archivo) -> pd.DataFrame:
    """
    Lee el CSV con encabezado doble (nombre de mercado + metrica),
    separador ';', decimales con coma y encoding latin-1.
    Devuelve un DataFrame en formato largo (long) con columnas:
    Dia, Mercado, Precio de Venta, Cantidad producida & vendida, Costo unitario
    """
    raw = pd.read_csv(
        archivo,
        sep=";",
        decimal=",",
        encoding="latin-1",
        header=[0, 1],
    )

    # Elimina columnas vacias (por el ';' final de cada fila)
    raw = raw.loc[:, ~raw.columns.get_level_values(0).str.startswith("Unnamed")]

    dias = raw[("Dias", raw["Dias"].columns[0])]
    dias.name = "Dia"

    mercados = [c for c in raw.columns.get_level_values(0).unique() if c != "Dias"]

    bloques = []
    for mercado in mercados:
        bloque = raw[mercado].copy()
        bloque.columns = [str(c).strip() for c in bloque.columns]
        bloque = bloque[[m for m in METRICAS if m in bloque.columns]]
        bloque.insert(0, "Mercado", mercado.strip())
        bloque.insert(0, "Dia", dias.values)
        bloques.append(bloque)

    df = pd.concat(bloques, ignore_index=True)
    for col in METRICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=METRICAS, how="all")

    # Metricas derivadas
    df["Utilidad Unitaria"] = df["Precio de Venta"] - df["Costo unitario"]
    df["Utilidad Total"] = df["Utilidad Unitaria"] * df["Cantidad producida & vendida"]
    df["Ingreso Total"] = df["Precio de Venta"] * df["Cantidad producida & vendida"]
    df["Margen %"] = (df["Utilidad Unitaria"] / df["Precio de Venta"]) * 100

    return df


def datos_ejemplo() -> pd.DataFrame:
    """Genera datos sinteticos si el usuario no carga ningun archivo."""
    rng = np.random.default_rng(42)
    dias = np.arange(1, 501)
    filas = []
    for i in range(1, 5):
        precio = rng.normal(100, 10, size=500)
        cantidad = rng.normal(500, 100, size=500)
        costo = rng.normal(80, 8, size=500)
        for d, p, c, co in zip(dias, precio, cantidad, costo):
            filas.append([d, f"Mercado {i}", p, c, co])
    df = pd.DataFrame(
        filas, columns=["Dia", "Mercado", "Precio de Venta",
                         "Cantidad producida & vendida", "Costo unitario"]
    )
    df["Utilidad Unitaria"] = df["Precio de Venta"] - df["Costo unitario"]
    df["Utilidad Total"] = df["Utilidad Unitaria"] * df["Cantidad producida & vendida"]
    df["Ingreso Total"] = df["Precio de Venta"] * df["Cantidad producida & vendida"]
    df["Margen %"] = (df["Utilidad Unitaria"] / df["Precio de Venta"]) * 100
    return df


def construir_contexto(df_filtrado: pd.DataFrame, incluir_todo: bool) -> str:
    """
    Arma un texto en Markdown con estadisticas por mercado para dar contexto
    al modelo. Si incluir_todo=True, adjunta ademas los datos filtrados
    completos en CSV (mas preciso, mas tokens).
    """
    columnas = METRICAS + ["Utilidad Unitaria", "Utilidad Total", "Ingreso Total", "Margen %"]
    resumen = df_filtrado.groupby("Mercado")[columnas].describe().round(2)

    partes = [
        f"Rango de dias analizado: {int(df_filtrado['Dia'].min())} a "
        f"{int(df_filtrado['Dia'].max())}.",
        f"Mercados incluidos: {', '.join(sorted(df_filtrado['Mercado'].unique()))}.",
        "Estadisticas descriptivas por mercado (count, mean, std, min, 25%, 50%, "
        "75%, max):",
        resumen.to_markdown(),
    ]

    if incluir_todo:
        partes.append("Datos completos filtrados (CSV):")
        partes.append(df_filtrado.to_csv(index=False))

    return "\n\n".join(partes)


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
st.title("📊 Analisis de Mercados")
st.caption("Precio de venta, cantidad producida/vendida y costo unitario por mercado")

with st.sidebar:
    st.header("Datos")
    archivo = st.file_uploader("Cargar CSV (formato PBA_4_mercados)", type=["csv"])
    st.caption("Si no cargas un archivo se usan datos de ejemplo.")

    st.header("Asistente de IA")
    api_key_input = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Se usa solo en esta sesion, no se guarda en ningun lado. "
             "Tambien puedes definirla como variable de entorno ANTHROPIC_API_KEY "
             "o en .streamlit/secrets.toml.",
    )
    incluir_datos_completos = st.checkbox(
        "Incluir datos completos filtrados en el contexto",
        value=False,
        help="Mas preciso para preguntas puntuales, pero consume mas tokens (y "
             "cuesta mas) por pregunta.",
    )

if archivo is not None:
    try:
        df = cargar_datos(archivo)
        st.sidebar.success(f"Archivo cargado: {archivo.name}")
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el archivo: {e}")
        df = datos_ejemplo()
else:
    df = datos_ejemplo()

mercados_disponibles = sorted(df["Mercado"].unique())

with st.sidebar:
    st.header("Filtros")
    mercados_sel = st.multiselect(
        "Mercados", mercados_disponibles, default=mercados_disponibles
    )
    dia_min, dia_max = int(df["Dia"].min()), int(df["Dia"].max())
    rango_dias = st.slider("Rango de dias", dia_min, dia_max, (dia_min, dia_max))
    metrica_sel = st.selectbox(
        "Metrica principal",
        METRICAS + ["Utilidad Unitaria", "Utilidad Total", "Ingreso Total", "Margen %"],
    )

df_f = df[
    (df["Mercado"].isin(mercados_sel))
    & (df["Dia"].between(rango_dias[0], rango_dias[1]))
]

if df_f.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------
st.subheader("Resumen general")
cols = st.columns(len(mercados_sel) if mercados_sel else 1)
for col, mercado in zip(cols, mercados_sel):
    sub = df_f[df_f["Mercado"] == mercado]
    col.metric(
        mercado,
        f"${sub['Utilidad Total'].mean():,.0f}",
        f"Margen {sub['Margen %'].mean():.1f}%",
        help="Utilidad total promedio diaria / margen promedio",
    )

# ------------------------------------------------------------------
# Serie de tiempo
# ------------------------------------------------------------------
st.subheader(f"Evolucion diaria — {metrica_sel}")
fig_serie = px.line(
    df_f, x="Dia", y=metrica_sel, color="Mercado", template="plotly_white"
)
st.plotly_chart(fig_serie, use_container_width=True)

# ------------------------------------------------------------------
# Distribucion y comparacion entre mercados
# ------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribucion")
    fig_hist = px.histogram(
        df_f, x=metrica_sel, color="Mercado", barmode="overlay",
        opacity=0.6, template="plotly_white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    st.subheader("Comparacion (boxplot)")
    fig_box = px.box(
        df_f, x="Mercado", y=metrica_sel, color="Mercado", template="plotly_white"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------------------------------
# Tabla de estadisticas descriptivas
# ------------------------------------------------------------------
st.subheader("Estadisticas descriptivas por mercado")
tabla_stats = (
    df_f.groupby("Mercado")[METRICAS + ["Utilidad Unitaria", "Utilidad Total", "Margen %"]]
    .agg(["mean", "std", "min", "max"])
    .round(2)
)
st.dataframe(tabla_stats, use_container_width=True)

# ------------------------------------------------------------------
# Correlacion entre variables (por mercado seleccionado)
# ------------------------------------------------------------------
st.subheader("Correlacion entre variables")
mercado_corr = st.selectbox("Mercado para correlacion", mercados_sel)
sub_corr = df_f[df_f["Mercado"] == mercado_corr][
    METRICAS + ["Utilidad Total", "Margen %"]
]
fig_corr = go.Figure(
    data=go.Heatmap(
        z=sub_corr.corr().values,
        x=sub_corr.columns,
        y=sub_corr.columns,
        colorscale="RdBu",
        zmid=0,
        text=sub_corr.corr().round(2).values,
        texttemplate="%{text}",
    )
)
fig_corr.update_layout(template="plotly_white")
st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------------------------------------------
# Descarga de datos procesados
# ------------------------------------------------------------------
st.subheader("Descargar datos procesados")
buffer = io.StringIO()
df_f.to_csv(buffer, index=False)
st.download_button(
    "Descargar CSV filtrado",
    data=buffer.getvalue(),
    file_name="mercados_procesado.csv",
    mime="text/csv",
)

# ------------------------------------------------------------------
# Asistente de IA (chat sobre los datos filtrados)
# ------------------------------------------------------------------
st.divider()
st.subheader("💬 Preguntale a la IA sobre estos datos")
st.caption(
    "El asistente responde con base en las estadisticas del recorte actual "
    "(mercados y rango de dias elegidos en la barra lateral)."
)

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

pregunta = st.chat_input("Ej: ¿Que mercado tiene el mejor margen promedio?")

if pregunta:
    if not api_key_input:
        st.error(
            "Falta la API key de Anthropic. Ingresala en la barra lateral, "
            "o defina la variable de entorno ANTHROPIC_API_KEY."
        )
    else:
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        contexto = construir_contexto(df_f, incluir_datos_completos)
        system_prompt = (
            "Eres un analista de datos que ayuda a interpretar un dataset de "
            "precios, cantidades y costos de varios mercados. Responde en "
            "español, de forma clara y concisa, basandote unicamente en el "
            "contexto entregado. Si la pregunta requiere un dato exacto que no "
            "esta en el resumen (y no se incluyeron los datos completos), dilo "
            "explicitamente en vez de inventar numeros.\n\n"
            f"CONTEXTO DE DATOS:\n{contexto}"
        )

        try:
            cliente = anthropic.Anthropic(api_key=api_key_input)
            with st.chat_message("assistant"):
                marcador = st.empty()
                texto_completo = ""
                with cliente.messages.stream(
                    model="claude-sonnet-5",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.mensajes
                    ],
                ) as stream:
                    for texto in stream.text_stream:
                        texto_completo += texto
                        marcador.markdown(texto_completo + "▌")
                marcador.markdown(texto_completo)
            st.session_state.mensajes.append(
                {"role": "assistant", "content": texto_completo}
            )
        except anthropic.AuthenticationError:
            st.error("La API key no es valida.")
        except Exception as e:
            st.error(f"Error al llamar a la API de Anthropic: {e}")
