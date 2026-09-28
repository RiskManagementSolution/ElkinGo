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
import openai

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODELOS_GROQ = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
]

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


def estimar_tokens(texto: str) -> int:
    """Aproximacion rapida: ~4 caracteres por token."""
    return len(texto) // 4


def construir_contexto(
    df_filtrado: pd.DataFrame, incluir_todo: bool, max_tokens_contexto: int = 2500
) -> str:
    """
    Arma un texto compacto (no describe() completo, que es muy verboso) con
    mean/std/min/max por mercado. Si incluir_todo=True intenta adjuntar los
    datos filtrados completos en CSV, mostrando una muestra si no caben en
    el presupuesto de tokens.
    """
    columnas = ["Precio de Venta", "Cantidad producida & vendida",
                "Costo unitario", "Utilidad Total", "Margen %"]

    lineas = [
        f"Rango de dias: {int(df_filtrado['Dia'].min())} a {int(df_filtrado['Dia'].max())}.",
        f"Mercados: {', '.join(sorted(df_filtrado['Mercado'].unique()))}.",
        "Estadisticas por mercado (media | desv.est | min | max):",
    ]
    for mercado, grupo in df_filtrado.groupby("Mercado"):
        partes_mercado = [mercado]
        for col in columnas:
            s = grupo[col]
            partes_mercado.append(
                f"{col}: {s.mean():.2f}|{s.std():.2f}|{s.min():.2f}|{s.max():.2f}"
            )
        lineas.append(" — ".join(partes_mercado))

    contexto = "\n".join(lineas)

    if incluir_todo:
        csv_completo = df_filtrado.to_csv(index=False)
        presupuesto_csv = max_tokens_contexto - estimar_tokens(contexto) - 100
        if estimar_tokens(csv_completo) > presupuesto_csv:
            # No cabe completo: se recorta a una muestra y se avisa al modelo
            max_chars = max(presupuesto_csv * 4, 0)
            csv_recortado = csv_completo[:max_chars]
            contexto += (
                "\n\nMuestra parcial de los datos filtrados (se recorto por "
                "limite de tamaño, no son todos los registros):\n" + csv_recortado
            )
        else:
            contexto += "\n\nDatos completos filtrados (CSV):\n" + csv_completo

    return contexto


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
st.title("📊 Analisis de Mercados")
st.caption("Precio de venta, cantidad producida/vendida y costo unitario por mercado")

with st.sidebar:
    st.header("Datos")
    archivo = st.file_uploader("Cargar CSV (formato PBA_4_mercados)", type=["csv"])
    st.caption("Si no cargas un archivo se usan datos de ejemplo.")

    st.header("Asistente de IA (Groq)")
    api_key_input = st.text_input(
        "Groq API key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Se usa solo en esta sesion, no se guarda en ningun lado. "
             "Tambien puedes definirla como variable de entorno GROQ_API_KEY "
             "o en .streamlit/secrets.toml.",
    )
    modelo_sel = st.selectbox("Modelo", MODELOS_GROQ, index=0)
    incluir_datos_completos = st.checkbox(
        "Incluir datos completos filtrados en el contexto",
        value=False,
        help="Mas preciso para preguntas puntuales, pero consume mas tokens. "
             "Si no caben en el limite de tokens por minuto de tu cuenta de "
             "Groq, se envia automaticamente una muestra parcial.",
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
            "Falta la API key de Groq. Ingresala en la barra lateral, "
            "o defina la variable de entorno GROQ_API_KEY."
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
            cliente = openai.OpenAI(api_key=api_key_input, base_url=GROQ_BASE_URL)
            with st.chat_message("assistant"):
                marcador = st.empty()
                texto_completo = ""
                stream = cliente.chat.completions.create(
                    model=modelo_sel,
                    max_tokens=1024,
                    messages=[{"role": "system", "content": system_prompt}]
                    + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.mensajes
                    ],
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        texto_completo += delta
                        marcador.markdown(texto_completo + "▌")
                marcador.markdown(texto_completo)
            st.session_state.mensajes.append(
                {"role": "assistant", "content": texto_completo}
            )
        except openai.AuthenticationError:
            st.error("La API key de Groq no es valida.")
        except openai.RateLimitError as e:
            st.error(
                "Se supero el limite de tokens por minuto de tu cuenta de Groq "
                f"para el modelo `{modelo_sel}`. Prueba con un modelo con mayor "
                "limite (ej. llama-3.1-8b-instant o llama-3.3-70b-versatile), "
                "desactiva 'Incluir datos completos', o reduce el rango de "
                f"dias/mercados en los filtros.\n\nDetalle: {e}"
            )
        except Exception as e:
            st.error(f"Error al llamar a la API de Groq: {e}")
