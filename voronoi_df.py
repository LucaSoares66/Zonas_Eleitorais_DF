"""
Voronoi dos Colégios Eleitorais do DF recortado pela malha do DF
----------------------------------------------------------------
Dependências:
    pip install geopandas shapely scipy matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import Voronoi

# ── CONFIG (ajuste se necessário) ─────────────────────────────────────────────
LAT_COL  = "latitude"
LON_COL  = "longitude"
ZONA_COL = "zona"
CRS_GEO  = "EPSG:4326"
CRS_PROJ = "EPSG:31983"   # SIRGAS 2000 / UTM 23S — ideal para o DF
OUTPUT   = "voronoi_df.gpkg"
# ─────────────────────────────────────────────────────────────────────────────


def preparar_pontos(df, lat_col=LAT_COL, lon_col=LON_COL,
                    crs_geo=CRS_GEO, crs_proj=CRS_PROJ):
    """Converte df para GeoDataFrame projetado."""
    gdf = df.copy()

    if "geometry" not in gdf.columns or gdf.geometry.isna().all():
        gdf = gpd.GeoDataFrame(
            gdf,
            geometry=gpd.points_from_xy(gdf[lon_col], gdf[lat_col]),
            crs=crs_geo,
        )
    else:
        if gdf.crs is None:
            gdf = gdf.set_crs(crs_geo)

    return gdf.to_crs(crs_proj)


def poligono_df(setores, crs_proj=CRS_PROJ):
    """Une todos os setores censitários num único polígono (contorno do DF)."""
    if setores.crs is None:
        setores = setores.set_crs(CRS_GEO)

    setores_proj = setores.to_crs(crs_proj)
    contorno = unary_union(setores_proj.geometry)

    return gpd.GeoDataFrame(
        {"nome": ["Distrito Federal"]},
        geometry=[contorno],
        crs=crs_proj,
    )


def voronoi_recortado(gdf_pontos, gdf_contorno):
    """Gera Voronoi a partir dos pontos e recorta pelo contorno do DF."""
    contorno_geom = gdf_contorno.geometry.unary_union

    coords = np.array([(geom.x, geom.y) for geom in gdf_pontos.geometry])

    # Pontos-espelho para fechar regiões infinitas nas bordas
    minx, miny, maxx, maxy = contorno_geom.bounds
    dx = (maxx - minx) * 3
    dy = (maxy - miny) * 3
    mirror = np.array([
        [minx - dx, miny - dy],
        [maxx + dx, miny - dy],
        [minx - dx, maxy + dy],
        [maxx + dx, maxy + dy],
    ])
    all_coords = np.vstack([coords, mirror])

    vor = Voronoi(all_coords)

    polys   = []
    indices = []

    for point_idx, region_idx in enumerate(vor.point_region):
        if point_idx >= len(coords):
            continue  # pula pontos-espelho

        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue

        vertices = [vor.vertices[v] for v in region]
        poly = ShapelyPolygon(vertices)
        clipped = poly.intersection(contorno_geom)

        if clipped.is_empty:
            continue

        polys.append(clipped)
        indices.append(point_idx)

    gdf_vor = gpd.GeoDataFrame(
        gdf_pontos.iloc[indices].reset_index(drop=True),
        geometry=polys,
        crs=gdf_pontos.crs,
    )

    return gdf_vor


def plotar(gdf_vor, gdf_pontos, gdf_contorno, zona_col=ZONA_COL):
    """Gera visualização estática matplotlib."""
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    gdf_contorno.boundary.plot(ax=ax, color="#58a6ff", linewidth=2, zorder=3)

    gdf_vor.plot(
        ax=ax,
        column=zona_col if zona_col in gdf_vor.columns else gdf_vor.index,
        cmap="tab20",
        alpha=0.55,
        edgecolor="#30363d",
        linewidth=0.6,
        zorder=2,
    )

    gdf_pontos.plot(ax=ax, color="#f0883e", markersize=18, zorder=4)

    if zona_col in gdf_vor.columns:
        for _, row in gdf_vor.iterrows():
            ax.annotate(
                str(row[zona_col]),
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                ha="center", va="center",
                fontsize=5.5, color="white", alpha=0.85,
            )

    ax.set_title("Zonas Eleitorais do DF — Diagrama de Voronoi",
                 color="white", fontsize=16, fontweight="bold", pad=18)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    legend_patch = mpatches.Patch(color="#f0883e", label="Colégios Eleitorais")
    ax.legend(handles=[legend_patch], facecolor="#161b22",
              labelcolor="white", framealpha=0.8, loc="lower right")

    plt.tight_layout()
    plt.savefig("voronoi_df.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("✔  Mapa salvo em voronoi_df.png")


def processar(df, setores,
              lat_col=LAT_COL, lon_col=LON_COL,
              zona_col=ZONA_COL,
              output=OUTPUT,
              plotar_mapa=True):
    """
    Pipeline completo: pontos -> polígono DF -> Voronoi recortado -> arquivo.

    Parâmetros
    ----------
    df          : DataFrame com os colégios eleitorais (lat/lon ou geometry)
    setores     : GeoDataFrame com a malha de setores censitários do DF
    lat_col     : nome da coluna de latitude
    lon_col     : nome da coluna de longitude
    zona_col    : coluna identificadora da zona
    output      : caminho do .gpkg de saída
    plotar_mapa : se True, exibe mapa matplotlib

    Retorna
    -------
    gdf_vor      : GeoDataFrame com polígonos de Voronoi recortados
    gdf_contorno : GeoDataFrame com o polígono do DF
    """
    print("── 1/4  Preparando pontos …")
    gdf_pontos = preparar_pontos(df, lat_col, lon_col)
    print(f"        {len(gdf_pontos)} pontos  |  CRS: {gdf_pontos.crs}")

    print("── 2/4  Construindo polígono do DF …")
    gdf_contorno = poligono_df(setores)
    area_km2 = gdf_contorno.geometry.area.sum() / 1e6
    print(f"        Área total: {area_km2:,.1f} km²")

    print("── 3/4  Calculando Voronoi e recortando …")
    gdf_vor = voronoi_recortado(gdf_pontos, gdf_contorno)
    print(f"        {len(gdf_vor)} regiões geradas")

    print(f"── 4/4  Salvando em {output} …")
    gdf_vor.to_file(output, layer="voronoi_zonas", driver="GPKG")
    gdf_contorno.to_file(output, layer="contorno_df", driver="GPKG")
    print("        Camadas: 'voronoi_zonas' e 'contorno_df'")

    if plotar_mapa:
        plotar(gdf_vor, gdf_pontos, gdf_contorno, zona_col)

    return gdf_vor, gdf_contorno
