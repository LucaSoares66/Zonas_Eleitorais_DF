"""
App Streamlit — Zonas Eleitorais do DF
recortado pelos Setores Censitários + Análise de Votação
---------------------------------------------------------
Dependências:
    pip install streamlit geopandas folium streamlit-folium branca scipy shapely

Rodar:
    streamlit run app.py
"""

import re
import warnings
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
from pathlib import Path

from streamlit_folium import st_folium
from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import Voronoi


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

LAT_COL  = "latitude"
LON_COL  = "longitude"
ZONA_COL = "zona"

CSV_PATH  = "Zonas_pontos.csv"
GPKG_PATH = "DF_setores_CD2022.gpkg"

CRS_GEO  = "EPSG:4326"
CRS_PROJ = "EPSG:31983"

COL_ZONA_VOTACAO   = "Zona"
COL_CARGO          = "Cargo"
COL_NOME_CANDIDATO = "Nome candidato"
COL_VOTOS_NOMINAIS = "Votos nominais"
COL_PARTIDO        = "Partido"
COL_SITUACAO       = "Situação totalização"
COL_TURNO          = "Turno"

PASTA_VOTACAO      = Path("votacao_df")
ANOS_ELEICAO       = [2002, 2006, 2010, 2014, 2018, 2022]
PADRAO_CSV_VOTACAO = "votacao_candidato_{ano}.csv"

TOP_N_CANDIDATOS = 10
TILE_MAPA        = "CartoDB positron"
ZOOM_INICIAL     = 10
ALTURA_MAPA      = 640

FILL_OPACITY_ZONA        = 0.45
FILL_OPACITY_HOVER       = 0.75
FILL_OPACITY_SETOR       = 0.08
FILL_OPACITY_SETOR_HOVER = 0.35
COR_BORDA_ZONA    = "#ffffff"
PESO_BORDA_ZONA   = 1.2
COR_BORDA_SETOR   = "#1a3a5c"
PESO_BORDA_SETOR  = 1.0
DASH_BORDA_SETOR  = "3 3"
COR_CONTORNO_DF   = "#1a3a5c"
PESO_CONTORNO_DF  = 2
DASH_CONTORNO_DF  = "5 4"

CORES_ZONAS = [
    "#1a5c1d", "#3ca42e", "#4ac491", "#6baed6",
    "#e19ed6", "#2c7bb6", "#7c1d77", "#bb5b5b",
    "#6b55cc", "#3a6fa8", "#772552", "#b07f40",
]


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Zonas Eleitorais — DF",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Layout geral */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    section[data-testid="stSidebar"] { background-color: #f7f9fc; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* Tipografia */
    h1 { font-size: 1.4rem !important; color: #1a3a5c !important; margin-bottom: 0 !important; }
    h2 { font-size: 1.1rem !important; color: #1a3a5c !important; }
    h3 { font-size: 0.95rem !important; color: #2e6da4 !important; }
    .caption-sub { color: #6b7a8d; font-size: 0.82rem; margin-top: 2px; }

    /* Cards de métrica */
    div[data-testid="metric-container"] {
        background: #f0f4f8;
        border: 1px solid #dce6f0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label { font-size: 0.72rem !important; color: #6b7a8d; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.3rem !important; color: #1a3a5c !important; font-weight: 700;
    }

    /* Badge de zona */
    .zona-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: #1a3a5c; color: white;
        padding: 5px 16px; border-radius: 20px;
        font-size: 0.88rem; font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Header de cargo */
    .cargo-header {
        background: #eef4fb;
        border-left: 4px solid #2e6da4;
        padding: 7px 14px;
        border-radius: 0 6px 6px 0;
        margin: 1.2rem 0 0.4rem 0;
        font-weight: 600; font-size: 0.9rem;
        color: #1a3a5c;
    }

    /* Tabelas */
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Botão primário */
    .stButton > button[kind="primary"] {
        background-color: #1a3a5c; color: white;
        border: none; border-radius: 6px;
        padding: 8px 20px; font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover { background-color: #2e6da4; }

    /* Separador lateral */
    .sidebar-section {
        background: white; border-radius: 8px;
        padding: 12px 14px; margin-bottom: 12px;
        border: 1px solid #dce6f0;
    }
    .sidebar-label {
        font-size: 0.72rem; font-weight: 700;
        color: #6b7a8d; text-transform: uppercase;
        letter-spacing: 0.05em; margin-bottom: 6px;
    }

    /* Info box */
    .info-box {
        background: #eef4fb; border: 1px solid #c8d8e8;
        border-radius: 8px; padding: 12px 16px;
        color: #1a3a5c; font-size: 0.88rem;
        margin: 0.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for key, val in [("pagina", "mapa"), ("zona_selecionada", None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESSAMENTO GEOESPACIAL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def carregar_dados():
    df      = pd.read_csv(CSV_PATH)
    setores = gpd.read_file(GPKG_PATH)
    return df, setores


@st.cache_data
def preparar_pontos(_df):
    gdf = gpd.GeoDataFrame(
        _df.copy(),
        geometry=gpd.points_from_xy(_df[LON_COL], _df[LAT_COL]),
        crs=CRS_GEO,
    )
    return gdf.to_crs(CRS_PROJ)


@st.cache_data
def construir_contorno(_setores):
    s = _setores.copy()
    if s.crs is None:
        s = s.set_crs(CRS_GEO)
    s = s.to_crs(CRS_PROJ)
    return gpd.GeoDataFrame(
        {"nome": ["Distrito Federal"]},
        geometry=[unary_union(s.geometry)],
        crs=CRS_PROJ,
    )


@st.cache_data
def calcular_voronoi(_gdf_pontos, _gdf_contorno):
    contorno_geom = _gdf_contorno.geometry.unary_union
    coords = np.array([(g.x, g.y) for g in _gdf_pontos.geometry])

    minx, miny, maxx, maxy = contorno_geom.bounds
    dx, dy = (maxx - minx) * 3, (maxy - miny) * 3
    mirror = np.array([
        [minx - dx, miny - dy], [maxx + dx, miny - dy],
        [minx - dx, maxy + dy], [maxx + dx, maxy + dy],
    ])
    vor = Voronoi(np.vstack([coords, mirror]))

    polys, indices = [], []
    for pt_idx, reg_idx in enumerate(vor.point_region):
        if pt_idx >= len(coords):
            continue
        region = vor.regions[reg_idx]
        if -1 in region or not region:
            continue
        poly = ShapelyPolygon([vor.vertices[v] for v in region])
        clipped = poly.intersection(contorno_geom)
        if not clipped.is_empty:
            polys.append(clipped)
            indices.append(pt_idx)

    gdf_raw = gpd.GeoDataFrame(
        _gdf_pontos.iloc[indices].reset_index(drop=True),
        geometry=polys, crs=_gdf_pontos.crs,
    )
    return gdf_raw.dissolve(by=ZONA_COL, as_index=False).reset_index(drop=True)


@st.cache_data
def recortar_por_setores(_gdf_vor, _setores):
    warnings.filterwarnings("ignore")
    setores_proj = _setores.copy()
    if setores_proj.crs is None:
        setores_proj = setores_proj.set_crs(CRS_GEO)
    setores_proj = setores_proj.to_crs(CRS_PROJ).reset_index(drop=True)
    vor_proj = _gdf_vor.to_crs(CRS_PROJ)

    zonas_atribuidas = []
    for _, setor in setores_proj.iterrows():
        melhor_zona, melhor_area = None, 0.0
        for _, zona_row in vor_proj.iterrows():
            try:
                area = setor.geometry.intersection(zona_row.geometry).area
            except Exception:
                area = 0.0
            if area > melhor_area:
                melhor_area = area
                melhor_zona = zona_row[ZONA_COL]
        zonas_atribuidas.append(melhor_zona)

    setores_proj[ZONA_COL] = zonas_atribuidas

    gdf_zonas = (
        setores_proj.dissolve(by=ZONA_COL, as_index=False)
        .reset_index(drop=True)[[ZONA_COL, "geometry"]]
    )

    colunas = [ZONA_COL, "geometry"]
    if "CD_SETOR" in setores_proj.columns:
        colunas = [ZONA_COL, "CD_SETOR", "geometry"]

    return gdf_zonas, setores_proj[colunas].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  VOTAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def ler_csv_votacao(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, sep=";", encoding="windows-1252")
    df.columns = df.columns.str.strip()
    if COL_ZONA_VOTACAO in df.columns:
        df[COL_ZONA_VOTACAO] = pd.to_numeric(df[COL_ZONA_VOTACAO], errors="coerce")
    if COL_VOTOS_NOMINAIS in df.columns:
        df[COL_VOTOS_NOMINAIS] = pd.to_numeric(df[COL_VOTOS_NOMINAIS], errors="coerce")
    return df


@st.cache_data
def carregar_todos_votos():
    resultado, avisos = {}, []
    for ano in ANOS_ELEICAO:
        caminho = PASTA_VOTACAO / PADRAO_CSV_VOTACAO.format(ano=ano)
        if caminho.exists():
            resultado[ano] = ler_csv_votacao(str(caminho))
        else:
            avisos.append(str(caminho))
    return resultado, avisos


def top_por_cargo(df_votos, zona, turno):
    mask = (
        (df_votos[COL_ZONA_VOTACAO].astype(str) == str(zona)) &
        (df_votos[COL_TURNO] == turno)
    )
    df_z = df_votos[mask].copy()
    if df_z.empty:
        return {}

    resultado = {}
    for cargo in sorted(df_z[COL_CARGO].dropna().unique()):
        df_c = df_z[df_z[COL_CARGO] == cargo]
        cols = [COL_NOME_CANDIDATO, COL_PARTIDO]
        if COL_SITUACAO in df_c.columns:
            cols.append(COL_SITUACAO)
        agg = (
            df_c.groupby(cols, as_index=False)[COL_VOTOS_NOMINAIS].sum()
            .sort_values(COL_VOTOS_NOMINAIS, ascending=False)
            .head(TOP_N_CANDIDATOS)
            .reset_index(drop=True)
        )
        agg.index += 1
        resultado[cargo] = agg
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
#  MAPA
# ══════════════════════════════════════════════════════════════════════════════

def _cor_por_zona(zonas_unicas):
    return {z: CORES_ZONAS[i % len(CORES_ZONAS)] for i, z in enumerate(sorted(zonas_unicas))}


def _camada_zonas(gdf, cor_map):
    features = [
        {
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {"zona": str(row[ZONA_COL]), "cor": cor_map.get(str(row[ZONA_COL]), "#4a90c4")},
        }
        for _, row in gdf.iterrows()
    ]
    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Zonas Eleitorais",
        style_function=lambda f: {
            "fillColor": f["properties"]["cor"],
            "color": COR_BORDA_ZONA, "weight": PESO_BORDA_ZONA,
            "fillOpacity": FILL_OPACITY_ZONA,
        },
        highlight_function=lambda f: {
            "fillColor": f["properties"]["cor"],
            "fillOpacity": FILL_OPACITY_HOVER, "weight": 2, "color": "#1a3a5c",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["zona"], aliases=["Zona Eleitoral:"],
            sticky=True,
            style=(
                "background-color:#fff;color:#1a3a5c;"
                "font-family:Arial,sans-serif;font-size:13px;font-weight:600;"
                "border:1px solid #c8d8e8;border-radius:4px;padding:6px 10px;"
                "box-shadow:0 1px 4px rgba(0,0,0,0.10);"
            ),
        ),
        popup=folium.GeoJsonPopup(
            fields=["zona"], aliases=["Zona Eleitoral:"],
            style=(
                "background-color:#fff;color:#1a3a5c;"
                "font-family:Arial,sans-serif;font-size:13px;"
                "border:1px solid #c8d8e8;border-radius:6px;padding:10px 14px;"
            ),
        ),
    )


def _camada_setores(gdf, cor_map):
    features = [
        {
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "zona": str(row[ZONA_COL]),
                "cd_setor": str(row["CD_SETOR"]) if "CD_SETOR" in row.index else "–",
                "cor": cor_map.get(str(row[ZONA_COL]), "#4a90c4"),
            },
        }
        for _, row in gdf.iterrows()
    ]
    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Setores Censitários",
        style_function=lambda f: {
            "fillColor": f["properties"]["cor"],
            "fillOpacity": FILL_OPACITY_SETOR,
            "color": COR_BORDA_SETOR, "weight": PESO_BORDA_SETOR,
            "dashArray": DASH_BORDA_SETOR,
        },
        highlight_function=lambda f: {
            "fillColor": f["properties"]["cor"],
            "fillOpacity": FILL_OPACITY_SETOR_HOVER,
            "weight": 2, "color": COR_BORDA_SETOR, "dashArray": "",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["cd_setor", "zona"],
            aliases=["Setor Censitário:", "Zona Eleitoral:"],
            sticky=True,
            style=(
                "background-color:#fff;color:#1a3a5c;"
                "font-family:Arial,sans-serif;font-size:12px;font-weight:500;"
                "border:1px solid #c8d8e8;border-radius:4px;padding:6px 10px;"
            ),
        ),
    )


def construir_mapa(gdf_zonas, gdf_setores, gdf_contorno, zonas_sel, modo):
    contorno_geo = gdf_contorno.to_crs(CRS_GEO)
    centro = contorno_geo.geometry.unary_union.centroid
    mapa = folium.Map(
        location=[centro.y, centro.x],
        zoom_start=ZOOM_INICIAL,
        tiles=TILE_MAPA,
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=False,
        doubleClickZoom=False,
        touchZoom=False,
        keyboard=False,
    )

    folium.GeoJson(
        contorno_geo.__geo_interface__, name="Contorno do DF",
        style_function=lambda _: {
            "fillColor": "transparent", "color": COR_CONTORNO_DF,
            "weight": PESO_CONTORNO_DF, "dashArray": DASH_CONTORNO_DF,
        },
    ).add_to(mapa)

    cor_map    = _cor_por_zona(gdf_zonas[ZONA_COL].astype(str).unique().tolist())
    zonas_str  = [str(z) for z in zonas_sel]
    zonas_fil  = gdf_zonas[gdf_zonas[ZONA_COL].astype(str).isin(zonas_str)].to_crs(CRS_GEO)
    setores_fil = gdf_setores[gdf_setores[ZONA_COL].astype(str).isin(zonas_str)].to_crs(CRS_GEO)

    _camada_zonas(zonas_fil, cor_map).add_to(mapa)
    if modo == "zona_setores":
        _camada_setores(setores_fil, cor_map).add_to(mapa)

    folium.LayerControl(collapsed=True).add_to(mapa)
    try:
        from folium.plugins import Fullscreen
        Fullscreen(position="topright").add_to(mapa)
    except Exception:
        pass
    return mapa


# ══════════════════════════════════════════════════════════════════════════════
#  CARREGAMENTO INICIAL
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Carregando dados geográficos…"):
    df_base, setores = carregar_dados()
    gdf_pontos   = preparar_pontos(df_base)
    gdf_contorno = construir_contorno(setores)
    gdf_vor      = calcular_voronoi(gdf_pontos, gdf_contorno)

with st.spinner("Recortando zonas pelos setores censitários…"):
    gdf_zonas_setores, gdf_setores_zona = recortar_por_setores(gdf_vor, setores)

with st.spinner("Carregando dados de votação…"):
    df_votos_todos, avisos_arquivos = carregar_todos_votos()

todas_zonas    = sorted(gdf_zonas_setores[ZONA_COL].astype(str).unique().tolist())
anos_carregados = sorted(df_votos_todos.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Logo / título lateral
    st.markdown("## 🗳️ TRE-DF")
    st.markdown('<p class="caption-sub">Zonas Eleitorais — Distrito Federal</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Navegação
    st.markdown('<div class="sidebar-label">Navegação</div>', unsafe_allow_html=True)
    pagina_escolhida = st.radio(
        "", options=["mapa", "votos"],
        format_func=lambda x: {"mapa": "🗺️  Mapa de Zonas", "votos": "📊  Análise de Votos"}[x],
        index=0 if st.session_state.pagina == "mapa" else 1,
        key="nav_radio", label_visibility="collapsed",
    )
    if pagina_escolhida != st.session_state.pagina:
        st.session_state.pagina = pagina_escolhida
        st.rerun()

    st.markdown("---")

    # ── Seção contextual por página
    if st.session_state.pagina == "mapa":

        st.markdown('<div class="sidebar-label">Modo de Exibição</div>', unsafe_allow_html=True)
        modo = st.radio(
            "", options=["zona", "zona_setores"],
            format_func=lambda x: {
                "zona":         "🔲  Apenas zonas",
                "zona_setores": "🔳  Zonas + Setores censitários",
            }[x],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-label">Filtro de Zonas</div>', unsafe_allow_html=True)
        selecionar_todas = st.toggle("Exibir todas as zonas", value=True)

        if selecionar_todas:
            zonas_selecionadas = todas_zonas
        else:
            zonas_selecionadas = st.multiselect(
                "Zonas visíveis:",
                options=todas_zonas,
                default=todas_zonas[:3],
                placeholder="Selecione as zonas…",
            )

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Total", len(todas_zonas))
        col2.metric("Visíveis", len(zonas_selecionadas))

    else:
        modo = "zona"
        zonas_selecionadas = todas_zonas

        st.markdown('<div class="sidebar-label">Dados Disponíveis</div>', unsafe_allow_html=True)
        if anos_carregados:
            st.success(f"✅ {len(anos_carregados)} eleições carregadas")
            for ano in anos_carregados:
                st.caption(f"• {ano}")
        if avisos_arquivos:
            with st.expander("⚠️ Arquivos não encontrados"):
                for av in avisos_arquivos:
                    st.caption(av)


# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA 1 — MAPA
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.pagina == "mapa":

    # Cabeçalho
    st.markdown("# 🗺️ Zonas Eleitorais do Distrito Federal")
    st.markdown('<p class="caption-sub">Colégios eleitorais agrupados por Voronoi e recortados pelos setores censitários — TRE-DF</p>', unsafe_allow_html=True)
    st.divider()

    if not zonas_selecionadas:
        st.warning("Selecione ao menos uma zona no painel lateral.")
    else:
        mapa = construir_mapa(
            gdf_zonas_setores, gdf_setores_zona,
            gdf_contorno, zonas_selecionadas, modo,
        )

        dados_mapa = st_folium(
            mapa, use_container_width=True,
            height=ALTURA_MAPA,
            returned_objects=["last_object_clicked_popup"],
        )

        # Detecta clique e oferece navegação para votos
        clicked = dados_mapa.get("last_object_clicked_popup")
        if clicked:
            zona_clicada = None
            if isinstance(clicked, dict):
                zona_clicada = str(clicked.get("zona", "")).strip()
            elif isinstance(clicked, str):
                m = re.search(r"\d+", clicked)
                if m:
                    zona_clicada = m.group()

            if zona_clicada and zona_clicada in todas_zonas:
                st.session_state.zona_selecionada = zona_clicada
                st.markdown(
                    f'<div class="info-box">📍 Zona <strong>{zona_clicada}</strong> selecionada. '
                    f'Acesse <strong>📊 Análise de Votos</strong> no menu lateral para explorar os candidatos.</div>',
                    unsafe_allow_html=True,
                )
                if anos_carregados:
                    if st.button("📊 Ver candidatos desta zona →", type="primary"):
                        st.session_state.pagina = "votos"
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA 2 — ANÁLISE DE VOTOS
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.pagina == "votos":

    # Cabeçalho com botão voltar
    col_h, col_v = st.columns([5, 1])
    with col_h:
        st.markdown("# 📊 Análise de Votação")
        st.markdown('<p class="caption-sub">Top candidatos por zona eleitoral — TRE-DF</p>', unsafe_allow_html=True)
    with col_v:
        st.write("")
        st.write("")
        if st.button("← Voltar ao mapa"):
            st.session_state.pagina = "mapa"
            st.rerun()

    st.divider()

    if not anos_carregados:
        st.info(f"Nenhum CSV de votação encontrado em `{PASTA_VOTACAO}`.")
        st.stop()

    # ── Filtros em linha
    col_zona, col_ano, col_turno, col_cargo = st.columns([2, 1, 1, 2])

    with col_zona:
        zona_analise = st.selectbox(
            "🗳️ Zona eleitoral",
            options=todas_zonas,
            index=todas_zonas.index(st.session_state.zona_selecionada)
                  if st.session_state.zona_selecionada in todas_zonas else 0,
        )
        st.session_state.zona_selecionada = zona_analise

    with col_ano:
        ano_analise = st.selectbox(
            "📅 Eleição",
            options=anos_carregados,
            index=len(anos_carregados) - 1,
        )

    df_ano = df_votos_todos[ano_analise]

    # Turnos disponíveis para zona + ano selecionados (sem "Todos")
    mask_zona = df_ano[COL_ZONA_VOTACAO].astype(str) == str(zona_analise)
    turnos_disp = sorted(df_ano[mask_zona][COL_TURNO].dropna().unique().tolist())

    with col_turno:
        if turnos_disp:
            turno_analise = st.selectbox(
                "🔄 Turno",
                options=turnos_disp,
                format_func=lambda t: f"{int(t)}º Turno",
            )
        else:
            st.selectbox("🔄 Turno", options=["–"], disabled=True)
            turno_analise = None

    # Cargos disponíveis para zona + turno
    if turno_analise is not None:
        mask_turno = mask_zona & (df_ano[COL_TURNO] == turno_analise)
        cargos_disp = sorted(df_ano[mask_turno][COL_CARGO].dropna().unique().tolist())
    else:
        cargos_disp = []

    with col_cargo:
        if cargos_disp:
            cargo_analise = st.selectbox("🏛️ Cargo", options=["Todos os cargos"] + cargos_disp)
        else:
            st.selectbox("🏛️ Cargo", options=["–"], disabled=True)
            cargo_analise = None

    st.divider()

    # Badge
    st.markdown(
        f'<div class="zona-badge">🗳️ Zona {zona_analise} &nbsp;·&nbsp; {ano_analise} '
        f'&nbsp;·&nbsp; {int(turno_analise)}º Turno</div>' if turno_analise else
        f'<div class="zona-badge">🗳️ Zona {zona_analise} &nbsp;·&nbsp; {ano_analise}</div>',
        unsafe_allow_html=True,
    )

    # ── Sem dados
    if turno_analise is None or cargo_analise is None:
        st.warning("Nenhum dado encontrado para a seleção.")
        st.stop()

    # ── Dados filtrados
    mask_base = (
        (df_ano[COL_ZONA_VOTACAO].astype(str) == str(zona_analise)) &
        (df_ano[COL_TURNO] == turno_analise)
    )
    df_filtrado = df_ano[mask_base].copy()

    if cargo_analise != "Todos os cargos":
        df_filtrado = df_filtrado[df_filtrado[COL_CARGO] == cargo_analise]

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
        st.stop()

    # ── Métricas de resumo
    total_votos  = int(df_filtrado[COL_VOTOS_NOMINAIS].sum())
    n_candidatos = df_filtrado[COL_NOME_CANDIDATO].nunique()
    n_cargos     = df_filtrado[COL_CARGO].nunique()

    m1, m2, m3, _ = st.columns([1, 1, 1, 2])
    m1.metric("Votos nominais", f"{total_votos:,}")
    m2.metric("Candidatos", n_candidatos)
    m3.metric("Cargos", n_cargos)

    st.write("")

    # ── Tabelas por cargo
    if cargo_analise == "Todos os cargos":
        for cargo in sorted(df_filtrado[COL_CARGO].dropna().unique()):
            df_c = df_filtrado[df_filtrado[COL_CARGO] == cargo]
            cols_grupo = [COL_NOME_CANDIDATO, COL_PARTIDO]
            if COL_SITUACAO in df_c.columns:
                cols_grupo.append(COL_SITUACAO)

            agg = (
                df_c.groupby(cols_grupo, as_index=False)[COL_VOTOS_NOMINAIS].sum()
                .sort_values(COL_VOTOS_NOMINAIS, ascending=False)
                .head(TOP_N_CANDIDATOS)
                .reset_index(drop=True)
            )
            agg.index += 1

            st.markdown(f'<div class="cargo-header">🏛️ {cargo}</div>', unsafe_allow_html=True)

            df_exib = agg.rename(columns={
                COL_NOME_CANDIDATO: "Candidato",
                COL_PARTIDO: "Partido",
                COL_VOTOS_NOMINAIS: "Votos",
                COL_SITUACAO: "Situação",
            })
            df_exib.index.name = "#"

            st.dataframe(
                df_exib, use_container_width=True,
                height=min(58 + len(df_exib) * 36, 420),
                column_config={"Votos": st.column_config.NumberColumn(format="%d")},
            )

    else:
        cols_grupo = [COL_NOME_CANDIDATO, COL_PARTIDO]
        if COL_SITUACAO in df_filtrado.columns:
            cols_grupo.append(COL_SITUACAO)

        agg = (
            df_filtrado.groupby(cols_grupo, as_index=False)[COL_VOTOS_NOMINAIS].sum()
            .sort_values(COL_VOTOS_NOMINAIS, ascending=False)
            .head(TOP_N_CANDIDATOS)
            .reset_index(drop=True)
        )
        agg.index += 1

        st.markdown(f'<div class="cargo-header">🏛️ {cargo_analise} — Top {TOP_N_CANDIDATOS}</div>', unsafe_allow_html=True)

        df_exib = agg.rename(columns={
            COL_NOME_CANDIDATO: "Candidato",
            COL_PARTIDO: "Partido",
            COL_VOTOS_NOMINAIS: "Votos",
            COL_SITUACAO: "Situação",
        })
        df_exib.index.name = "#"

        st.dataframe(
            df_exib, use_container_width=True,
            height=min(58 + len(df_exib) * 36, 460),
            column_config={"Votos": st.column_config.NumberColumn(format="%d")},
        )

    st.markdown("---")

    # ── Download
    csv_bytes = df_filtrado.to_csv(index=False).encode("utf-8")
    nome_cargo = cargo_analise.lower().replace(" ", "_") if cargo_analise != "Todos os cargos" else "todos"
    st.download_button(
        label=f"⬇️  Baixar dados — Zona {zona_analise} · {ano_analise} · {int(turno_analise)}º Turno",
        data=csv_bytes,
        file_name=f"votos_{nome_cargo}_zona{zona_analise}_{ano_analise}_t{int(turno_analise)}.csv",
        mime="text/csv",
    )

    # ── Comparativo entre anos (expander)
    if len(anos_carregados) > 1 and cargo_analise != "Todos os cargos":
        with st.expander("📈 Evolução de votos de um candidato entre eleições"):
            candidatos_historico = set()
            for df_c in df_votos_todos.values():
                mask = (
                    (df_c[COL_ZONA_VOTACAO].astype(str) == str(zona_analise)) &
                    (df_c[COL_CARGO] == cargo_analise)
                )
                candidatos_historico.update(df_c[mask][COL_NOME_CANDIDATO].dropna().unique())

            if candidatos_historico:
                candidato_comp = st.selectbox("Candidato:", sorted(candidatos_historico))
                rows = []
                for ano_c, df_c in df_votos_todos.items():
                    mask = (
                        (df_c[COL_ZONA_VOTACAO].astype(str) == str(zona_analise)) &
                        (df_c[COL_CARGO] == cargo_analise) &
                        (df_c[COL_NOME_CANDIDATO] == candidato_comp) &
                        (df_c[COL_TURNO] == turno_analise)
                    )
                    votos = df_c[mask][COL_VOTOS_NOMINAIS].sum()
                    if votos > 0:
                        rows.append({"Ano": ano_c, "Votos": int(votos)})

                if rows:
                    st.bar_chart(pd.DataFrame(rows).set_index("Ano"))
                else:
                    st.info("Candidato sem votos registrados no turno selecionado.")
            else:
                st.info("Nenhum candidato encontrado para esse cargo.")