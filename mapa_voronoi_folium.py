"""
Visualização Folium — Voronoi dos Colégios Eleitorais do DF
-----------------------------------------------------------
Dependências:
    pip install folium geopandas branca

Como usar no notebook:
    from voronoi_df import processar, preparar_pontos
    from mapa_voronoi_folium import gerar_mapa

    gdf_vor, gdf_contorno = processar(df, setores, zona_col="zona")
    gdf_pontos = preparar_pontos(df)
    mapa = gerar_mapa(gdf_vor, gdf_contorno, gdf_pontos=gdf_pontos, zona_col="zona")
    mapa
"""

import folium
import geopandas as gpd
import branca.colormap as cm

CRS_GEO  = "EPSG:4326"
ZONA_COL = "zona"


def gerar_mapa(gdf_vor, gdf_contorno, gdf_pontos=None,
               zona_col=ZONA_COL, output_html="mapa_voronoi_df.html"):
    """
    Gera mapa Folium interativo com os polígonos de Voronoi recortados no DF.

    Parâmetros
    ----------
    gdf_vor      : GeoDataFrame com os polígonos de Voronoi
    gdf_contorno : GeoDataFrame com o polígono do DF
    gdf_pontos   : GeoDataFrame com os pontos dos colégios (opcional)
    zona_col     : coluna com o nome/número da zona
    output_html  : caminho do HTML de saída

    Retorna
    -------
    mapa : folium.Map (exibe direto no Jupyter com mapa na última linha da célula)
    """

    # ── Reprojetar para lat/lon (Folium exige EPSG:4326)
    vor_geo      = gdf_vor.to_crs(CRS_GEO)
    contorno_geo = gdf_contorno.to_crs(CRS_GEO)

    # ── Centro do mapa
    centro = contorno_geo.geometry.unary_union.centroid
    mapa = folium.Map(
        location=[centro.y, centro.x],
        zoom_start=10,
        tiles="CartoDB dark_matter",
    )

    # ══════════════════════════════════════════════════════════════════════
    # 1. CONTORNO DO DF
    # ══════════════════════════════════════════════════════════════════════
    folium.GeoJson(
        contorno_geo.__geo_interface__,
        name="Contorno do DF",
        style_function=lambda _: {
            "fillColor": "transparent",
            "color":     "#58a6ff",
            "weight":    2.5,
            "dashArray": "6 4",
        },
    ).add_to(mapa)

    # ══════════════════════════════════════════════════════════════════════
    # 2. POLÍGONOS DE VORONOI
    # ══════════════════════════════════════════════════════════════════════
    colormap = cm.LinearColormap(
        colors=["#1f4e79", "#2e86c1", "#48c9b0", "#f9e79f", "#e67e22", "#c0392b"],
        vmin=0,
        vmax=max(len(vor_geo) - 1, 1),
    )

    features = []
    for i, (_, row) in enumerate(vor_geo.iterrows()):
        zona_val = str(row[zona_col]) if zona_col in row.index else str(i)
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "zona": zona_val,
                "cor":  colormap(i),
            },
        })

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Zonas Eleitorais (Voronoi)",
        style_function=lambda feat: {
            "fillColor":   feat["properties"]["cor"],
            "color":       "#1a1a2e",
            "weight":      1,
            "fillOpacity": 0.55,
        },
        highlight_function=lambda _: {
            "fillOpacity": 0.85,
            "weight":      2.5,
            "color":       "#ffffff",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["zona"],
            aliases=["🗳️ Zona:"],
            localize=True,
            sticky=True,
            style=(
                "background-color: #1e1e2e;"
                "color: #cdd6f4;"
                "font-family: monospace;"
                "font-size: 13px;"
                "border: 1px solid #45475a;"
                "border-radius: 6px;"
                "padding: 6px 10px;"
            ),
        ),
    ).add_to(mapa)

    # ══════════════════════════════════════════════════════════════════════
    # 3. PONTOS DOS COLÉGIOS (opcional)
    # ══════════════════════════════════════════════════════════════════════
    if gdf_pontos is not None:
        pts_geo = gdf_pontos.to_crs(CRS_GEO)
        camada_pontos = folium.FeatureGroup(name="Colégios Eleitorais")

        for _, row in pts_geo.iterrows():
            zona_val = str(row[zona_col]) if zona_col in row.index else "–"
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color="#f0883e",
                fill=True,
                fill_color="#f0883e",
                fill_opacity=0.9,
                weight=1.5,
                tooltip=folium.Tooltip(
                    f"<b style='color:#f0883e'>🗳️ Zona {zona_val}</b>",
                    sticky=True,
                ),
            ).add_to(camada_pontos)

        camada_pontos.add_to(mapa)

    # ══════════════════════════════════════════════════════════════════════
    # 4. CONTROLES
    # ══════════════════════════════════════════════════════════════════════
    folium.LayerControl(collapsed=False).add_to(mapa)

    try:
        from folium.plugins import MiniMap
        MiniMap(toggle_display=True).add_to(mapa)
    except Exception:
        pass

    try:
        from folium.plugins import Fullscreen
        Fullscreen().add_to(mapa)
    except Exception:
        pass

    mapa.save(output_html)
    print(f"✔  Mapa salvo em {output_html}")

    return mapa
