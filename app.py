"""
App Streamlit — Voronoi dos Colégios Eleitorais do DF
recortado pelos Setores Censitários
------------------------------------------------------
Dependências:
    pip install streamlit geopandas folium streamlit-folium branca scipy shapely

Rodar:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import numpy as np

from streamlit_folium import st_folium
from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import Voronoi

# ── CONFIG ────────────────────────────────────────────────────────────────────
LAT_COL   = "latitude"
LON_COL   = "longitude"
ZONA_COL  = "zona"
CRS_GEO   = "EPSG:4326"
CRS_PROJ  = "EPSG:31983"
CSV_PATH  = "Zonas_pontos.csv"
GPKG_PATH = "DF_setores_CD2022.gpkg"

CORES_FORMAIS = [
    "#1a3a5c", "#2e6da4", "#4a90c4", "#6baed6",
    "#9ecae1", "#2c7bb6", "#1d4e7c", "#357abd",
    "#5599cc", "#3a6fa8", "#254e77", "#4078b0",
]
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Zonas Eleitorais — DF",
    page_icon="🗳️",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.5rem !important; color: #1a3a5c !important; }
    .stMetric label { font-size: 0.75rem; color: #555; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE PROCESSAMENTO
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
    contorno = unary_union(s.geometry)
    return gpd.GeoDataFrame(
        {"nome": ["Distrito Federal"]},
        geometry=[contorno],
        crs=CRS_PROJ,
    )


@st.cache_data
def calcular_voronoi(_gdf_pontos, _gdf_contorno):
    """Calcula Voronoi bruto (sem dissolve ainda — dissolve ocorre após recorte)."""
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
        geometry=polys,
        crs=_gdf_pontos.crs,
    )

    # Dissolve por zona (Voronoi puro, sem setores)
    return gdf_raw.dissolve(by=ZONA_COL, as_index=False).reset_index(drop=True)


@st.cache_data
def recortar_por_setores(_gdf_vor, _setores):
    """
    Para cada setor censitário, atribui à zona de Voronoi com maior
    área de interseção. Depois dissolve os setores por zona → fronteiras
    finais seguem os limites dos setores censitários.

    Retorna
    -------
    gdf_zonas_setores : zonas recortadas pelos setores (dissolve final)
    gdf_setores_zona  : setores individuais com coluna 'zona' atribuída
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Garantir mesmo CRS
    setores_proj = _setores.copy()
    if setores_proj.crs is None:
        setores_proj = setores_proj.set_crs(CRS_GEO)
    setores_proj = setores_proj.to_crs(CRS_PROJ).reset_index(drop=True)

    vor_proj = _gdf_vor.to_crs(CRS_PROJ)

    # ── Para cada setor, calcular área de interseção com cada zona Voronoi
    zonas_atribuidas = []

    for idx_s, setor in setores_proj.iterrows():
        melhor_zona = None
        melhor_area = 0.0

        for _, zona_row in vor_proj.iterrows():
            try:
                inter = setor.geometry.intersection(zona_row.geometry)
                area  = inter.area
            except Exception:
                area = 0.0

            if area > melhor_area:
                melhor_area = area
                melhor_zona = zona_row[ZONA_COL]

        zonas_atribuidas.append(melhor_zona)

    setores_proj[ZONA_COL] = zonas_atribuidas

    # ── Dissolve: une setores da mesma zona → fronteiras seguem setores
    gdf_zonas_setores = (
        setores_proj
        .dissolve(by=ZONA_COL, as_index=False)
        .reset_index(drop=True)[[ZONA_COL, "geometry"]]
    )

    # Setores individuais com zona atribuída (para camada de setores no mapa)
    gdf_setores_zona = setores_proj[[ZONA_COL, "CD_SETOR", "geometry"]].copy()

    return gdf_zonas_setores, gdf_setores_zona


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUÇÃO DO MAPA
# ══════════════════════════════════════════════════════════════════════════════

def _cor_por_zona(zonas_unicas):
    return {
        z: CORES_FORMAIS[i % len(CORES_FORMAIS)]
        for i, z in enumerate(sorted(zonas_unicas))
    }


def _geojson_features(gdf, zona_col, cor_map):
    features = []
    for _, row in gdf.iterrows():
        zona_val = str(row[zona_col])
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "zona": zona_val,
                "cor":  cor_map.get(zona_val, "#4a90c4"),
                # --- campos futuros ---
                # "eleitores": "–",
                # "secoes":    "–",
            },
        })
    return features


def _camada_voronoi(features, name):
    """Retorna camada GeoJson de polígonos de zona."""
    tooltip_fields  = ["zona"]
    tooltip_aliases = ["Zona Eleitoral:"]
    # Futuro: tooltip_fields += ["eleitores"]; tooltip_aliases += ["Eleitores:"]

    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name=name,
        style_function=lambda feat: {
            "fillColor":   feat["properties"]["cor"],
            "color":       "#ffffff",
            "weight":      1.2,
            "fillOpacity": 0.45,
        },
        highlight_function=lambda feat: {
            "fillColor":   feat["properties"]["cor"],
            "fillOpacity": 0.75,
            "weight":      2,
            "color":       "#1a3a5c",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=True,
            style=(
                "background-color:#ffffff;color:#1a3a5c;"
                "font-family:Arial,sans-serif;font-size:13px;font-weight:500;"
                "border:1px solid #c8d8e8;border-radius:4px;"
                "padding:6px 10px;box-shadow:0 1px 4px rgba(0,0,0,0.12);"
            ),
        ),
    )


def _camada_setores(gdf_setores, cor_map):
    """Retorna camada GeoJson dos setores censitários individuais."""
    features = []
    for _, row in gdf_setores.iterrows():
        zona_val  = str(row[ZONA_COL])
        setor_val = str(row["CD_SETOR"]) if "CD_SETOR" in row.index else "–"
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "zona":     zona_val,
                "cd_setor": setor_val,
                "cor":      cor_map.get(zona_val, "#4a90c4"),
            },
        })

    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Setores Censitários",
        style_function=lambda feat: {
            "fillColor":   "transparent",
            "color":       "#7a9cbf",   # linhas azul-acinzentadas, discretas
            "weight":      0.5,
            "fillOpacity": 0,
            "dashArray":   "2 3",
        },
        highlight_function=lambda feat: {
            "fillColor":   feat["properties"]["cor"],
            "fillOpacity": 0.30,
            "weight":      1.5,
            "color":       "#1a3a5c",
            "dashArray":   "",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["cd_setor", "zona"],
            aliases=["Setor Censitário:", "Zona Eleitoral:"],
            sticky=True,
            style=(
                "background-color:#ffffff;color:#1a3a5c;"
                "font-family:Arial,sans-serif;font-size:12px;font-weight:500;"
                "border:1px solid #c8d8e8;border-radius:4px;"
                "padding:6px 10px;box-shadow:0 1px 4px rgba(0,0,0,0.10);"
            ),
        ),
    )


def construir_mapa(gdf_zonas, gdf_setores_zona, gdf_contorno,
                   zonas_selecionadas, modo_visualizacao):
    """
    modo_visualizacao:
        "zona"          → apenas zonas recortadas pelos setores
        "zona_setores"  → zonas + divisão interna dos setores
    """
    contorno_geo = gdf_contorno.to_crs(CRS_GEO)
    centro = contorno_geo.geometry.unary_union.centroid

    mapa = folium.Map(
        location=[centro.y, centro.x],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    # Contorno do DF
    folium.GeoJson(
        contorno_geo.__geo_interface__,
        name="Contorno do DF",
        style_function=lambda _: {
            "fillColor": "transparent",
            "color":     "#1a3a5c",
            "weight":    2,
            "dashArray": "5 4",
        },
    ).add_to(mapa)

    # Filtrar por zonas selecionadas
    mask_z = gdf_zonas[ZONA_COL].astype(str).isin([str(z) for z in zonas_selecionadas])
    mask_s = gdf_setores_zona[ZONA_COL].astype(str).isin([str(z) for z in zonas_selecionadas])
    zonas_fil   = gdf_zonas[mask_z].to_crs(CRS_GEO)
    setores_fil = gdf_setores_zona[mask_s].to_crs(CRS_GEO)

    zonas_unicas = gdf_zonas[ZONA_COL].astype(str).unique().tolist()
    cor_map = _cor_por_zona(zonas_unicas)

    if modo_visualizacao == "zona_setores":
        # Camada de setores por baixo (divisão interna visível)
        _camada_setores(setores_fil, cor_map).add_to(mapa)

    # Camada de zonas (sempre presente — fronteira externa da zona)
    features_zonas = _geojson_features(zonas_fil, ZONA_COL, cor_map)
    _camada_voronoi(features_zonas, "Zonas Eleitorais").add_to(mapa)

    folium.LayerControl(collapsed=True).add_to(mapa)

    try:
        from folium.plugins import Fullscreen
        Fullscreen(position="topright").add_to(mapa)
    except Exception:
        pass

    return mapa


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

st.title("🗳️ Zonas Eleitorais do Distrito Federal")
st.caption("Colégios eleitorais agrupados por Voronoi e recortados pelos setores censitários — TRE-DF")
st.divider()

with st.spinner("Carregando dados…"):
    df, setores = carregar_dados()

with st.spinner("Calculando Voronoi…"):
    gdf_pontos   = preparar_pontos(df)
    gdf_contorno = construir_contorno(setores)
    gdf_vor      = calcular_voronoi(gdf_pontos, gdf_contorno)

with st.spinner("Recortando zonas pelos setores censitários (pode levar alguns segundos)…"):
    gdf_zonas_setores, gdf_setores_zona = recortar_por_setores(gdf_vor, setores)

todas_zonas = sorted(gdf_zonas_setores[ZONA_COL].astype(str).unique().tolist())

# ── Sidebar
with st.sidebar:
    st.markdown("### Visualização")
    st.markdown("---")

    modo = st.radio(
        "Modo de exibição:",
        options=["zona", "zona_setores"],
        format_func=lambda x: {
            "zona":         "🗺️ Apenas zonas",
            "zona_setores": "🗺️ Zonas + Setores censitários",
        }[x],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Filtros")

    selecionar_todas = st.checkbox("Todas as zonas", value=True)

    if selecionar_todas:
        zonas_selecionadas = todas_zonas
    else:
        zonas_selecionadas = st.multiselect(
            "Zonas:",
            options=todas_zonas,
            default=todas_zonas[:5],
            placeholder="Selecione as zonas…",
        )

    st.markdown("---")
    st.metric("Total de zonas", len(todas_zonas))
    st.metric("Zonas exibidas", len(zonas_selecionadas))

# ── Mapa
if not zonas_selecionadas:
    st.warning("Selecione ao menos uma zona no painel lateral.")
else:
    mapa = construir_mapa(
        gdf_zonas_setores,
        gdf_setores_zona,
        gdf_contorno,
        zonas_selecionadas,
        modo_visualizacao=modo,
    )
    st_folium(
        mapa,
        use_container_width=True,
        height=680,
        returned_objects=[],
    )
