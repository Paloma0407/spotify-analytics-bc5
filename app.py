# ============================================================
# CABECERA
# ============================================================
# Alumno: Paloma Rubio
# URL Streamlit Cloud: https://spotify-analytics-bc5-sbtdmuroa8nkmxwdftcyxf.streamlit.app/
# URL GitHub: https://github.com/Paloma0407/spotify-analytics-bc5

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#

SYSTEM_PROMPT = """
Eres un asistente analítico especializado en datos de escucha de Spotify.
Tienes acceso a un DataFrame de pandas llamado `df` con el historial de
escucha de un usuario durante 12 meses (entre {fecha_min} y {fecha_max}).
Los podcasts ya han sido filtrados — todos los registros son canciones.

El DataFrame `df` tiene estas columnas:
- ts (datetime): timestamp de fin de reproducción, zona horaria Europe/Madrid
- track (string): nombre de la canción
- artist (string): nombre del artista
- spotify_track_uri (string): identificador único de cada canción
- album (string): nombre del álbum
- minutes_played (float): minutos reproducidos
- platform (string): plataforma usada. Valores posibles: {plataformas}
- shuffle (bool): si el modo aleatorio estaba activo
- skipped (bool/null): True si se saltó, null si no se saltó
- reason_start (string): motivo de inicio. Valores posibles: {reason_start_values}
- reason_end (string): motivo de fin. Valores posibles: {reason_end_values}
- hour (int): hora del día (0-23)
- day_of_week (string): día de la semana en inglés (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)
- month (string): mes en formato YYYY-MM (ej: "2025-03")
- year (int): año
- is_weekend (bool): True si es sábado o domingo
- date (date): fecha sin hora

FORMATO DE RESPUESTA:
Responde ÚNICAMENTE con un objeto JSON válido. Sin texto adicional, sin explicaciones fuera del JSON, sin backticks de markdown.

Para preguntas sobre los datos:
{{"tipo": "grafico", "codigo": "...código Python aquí...", "interpretacion": "...explicación breve aquí..."}}

Para preguntas fuera de alcance:
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Solo puedo responder preguntas sobre tus hábitos de escucha en Spotify."}}

INSTRUCCIONES PARA EL CÓDIGO:
- Crea siempre una variable llamada `fig` con una figura de Plotly (px o go)
- Usa `df` directamente, sin cargarlo ni importarlo
- Usa siempre `minutes_played` para medir tiempo, nunca `ms_played`
- Incluye siempre título, etiquetas de ejes y colores en español
- En rankings, muestra máximo 10 elementos salvo que se pida otro número
- Después de un groupby, usa siempre .reset_index() antes de crear el gráfico
- Si la pregunta usa singular ("qué canción", "cuál es el artista") devuelve solo el elemento número 1 y crea una figura con  go.Figure() y añade una anotación de texto centrada con el resultado
- Si usa plural ("canciones", "artistas", "top") devuelve el ranking completo
- Cuando uses go.Figure() con anotación de texto, oculta los ejes con:
fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
- Cuando muestres un único elemento ganador, incluye en la anotación:
el nombre, el total de reproducciones o minutos, y el porcentaje 
que supera al segundo elemento. Formato ejemplo:
"Tití Me Preguntó\n400 reproducciones\n+15% sobre la siguiente"
- "Veces escuchada" o "más veces" siempre se refiere al número de reproducciones (conteo de filas), nunca a minutos. Usa df.groupby('track').size() o df['track'].value_counts()
- "Más escuchado en horas/minutos/tiempo" se refiere a minutes_played
- "Artista más escuchado" sin especificar métrica → usa minutes_played
- "Canción más escuchada" sin especificar métrica → usa value_counts()
- Si la pregunta pide identificar UN mes concreto (el de más canciones 
  nuevas, el de más escucha, etc.) muestra el resultado como texto 
  con go.Figure() y anotación, incluyendo el mes y el valor total.
  Ejemplo: "Marzo 2025\n87 canciones nuevas descubiertas"
- Para calcular canciones nuevas por mes, usa siempre:
  primera_vez = df.groupby('spotify_track_uri')['ts'].min().reset_index()
  primera_vez['month'] = primera_vez['ts'].dt.to_period('M').astype(str)
  resultado = primera_vez.groupby('month').size().reset_index(name='nuevas')
- En gráficos de barras, añade siempre etiquetas con los valores redondeados y formateados con separador de miles. Para ello:
  1. Redondea la columna: df_plot['col'] = df_plot['col'].round(0).astype(int)
  2. Crea columna de texto formateado: df_plot['etiqueta'] = df_plot['col'].apply(lambda x: f"{{x:,}}".replace(",", "."))
  3. Usa text='etiqueta' en px.bar y textposition='outside'
  4. NO uses text_auto=True ni separators en update_layout
- Si la pregunta menciona "horas", convierte minutes_played dividiendo entre 60 y etiqueta el eje como "Horas reproducidas"

Tipos de gráfico según la pregunta:
- Rankings: barras horizontales (px.bar con orientation='h'), ordena de mayor a menor
- Evolución temporal: px.line, ordena siempre por month con .sort_index()
- Proporciones: px.pie o barras
- Patrones horarios: px.bar agrupado
- En rankings de barras horizontales usa un color fijo como color_discrete_sequence=['#1DB954'] en lugar de 
escala continua
- En comparaciones entre dos períodos usa siempre colores contrastados 
y explícitos: color_discrete_map={{'Verano': '#1DB954', 'Invierno': '#1E90FF','Primavera': '#FF6B6B', 'Otoño': '#FFA500'}}
- En gráficos de barras, muestra siempre el valor de cada barra con text_auto=True o text=columna, y añade textposition='outside' para que el número aparezca fuera de la barra

Comportamiento de escucha:
- Canciones saltadas: df[df['skipped'] == True]
- Canciones NO saltadas: df[df['skipped'].isna()]
- Shuffle: df[df['shuffle'] == True] (es booleano, no string)
- Porcentaje de skips: canciones saltadas / total * 100

Análisis temporal avanzado:
- Canciones únicas por período: .nunique() sobre spotify_track_uri
- Primera aparición de una canción: df.groupby('spotify_track_uri')['ts'].min()

Comparaciones por estación:
- Primavera: meses 03, 04, 05
- Verano: meses 06, 07, 08
- Otoño: meses 09, 10, 11
- Invierno: meses 12, 01, 02
- Filtra con: df[df['month'].str[5:7].isin(['06','07','08'])] o similar

RESTRICCIONES:
- No uses import os, open(), ni accedas al sistema de ficheros
- No modifiques el DataFrame original df
- No uses bibliotecas distintas a pandas y plotly
- Si la pregunta es ambigua, interpreta la opción más útil para el análisis musical
- No uses fig.show() — Streamlit renderiza la figura automáticamente
"""

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # Filtrar podcasts (registros sin nombre de canción ni artista)
    df = df[df["master_metadata_track_name"].notna()].copy()

    # Renombrar columnas largas
    df = df.rename(columns={
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album"
    })

    # Convertir timestamp a datetime y ajustar a zona horaria Madrid
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Europe/Madrid")

    # Columnas derivadas de tiempo
    df["date"]        = df["ts"].dt.date
    df["hour"]        = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.day_name()
    df["month"]       = df["ts"].dt.to_period("M").astype(str)
    df["year"]        = df["ts"].dt.year
    df["is_weekend"]  = df["ts"].dt.dayofweek >= 5

    # Convertir milisegundos a minutos
    df["minutes_played"] = df["ms_played"] / 60000

    return df
    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    [Tu respuesta aquí]
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    [Tu respuesta aquí]
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    [Tu respuesta aquí]