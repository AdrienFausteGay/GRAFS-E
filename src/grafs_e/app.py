# %%
import json
import os
from importlib.metadata import version

import branca
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

from grafs_e.donnees import *
from grafs_e.N_class import DataLoader, NitrogenFlowModel
from grafs_e.sankey import (
    streamlit_sankey_app,
    streamlit_sankey_fertilization,
    streamlit_sankey_food_flows,
    streamlit_sankey_systemic_flows,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

geojson_path = os.path.join(DATA_DIR, "contour-GRAFS.geojson")
image_path = os.path.join(DATA_DIR, "metabolism.png")
icon_path = os.path.join(DATA_DIR, "logo.jpg")

st.set_page_config(page_title="GRAFS-E App", page_icon=icon_path)  # , layout="wide")


# Charger les données
@st.cache_data
def load_data():
    # votre DataLoader (ou équivalent)
    return DataLoader()


if "data" not in st.session_state:
    st.session_state["data"] = load_data()

data = st.session_state["data"]

# %%
# Initialisation de l'interface Streamlit
st.title("GRAFS-E")
__version__ = version("grafs_e")
st.write(f"📦 GRAFS-E version: {__version__}")
st.title("Nitrogen Flow Simulation Model: A Territorial Ecology Approach")

# 🔹 Initialiser les valeurs dans session_state si elles ne sont pas encore définies
if "selected_region" not in st.session_state:
    st.session_state.selected_region = None
if "year" not in st.session_state:
    st.session_state.year = None

if st.session_state.selected_region:
    st.write(f"✅ Selected territory: {st.session_state.selected_region}")
else:
    st.warning("⚠️ Please select a region")

if st.session_state.year:
    st.write(f"✅ Selected year : {st.session_state.year}")
else:
    st.warning("⚠️ Please select a year")

# -- Sélection des onglets --
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Documentation", "Run", "Sankey", "Detailed data", "Map"]
)

with tab1:
    st.title("Documentation")

    st.header(
        "GRAFS-Extended: Comprehensive Analysis of Nitrogen Flux in Agricultural Systems"
    )

    st.subheader("Overview")

    st.text(
        "The GRAFS-extended model serves as an advanced tool designed to analyze and map the evolution of nitrogen utilization within agricultural systems, with a particular focus on 33 regions of France from 1852 to 2014. This model builds upon the GRAFS framework developped at IEES and integrates graph theory to provide a detailed analysis of nitrogen flows in agriculture, identifying key patterns, transformations, and structural invariants. The model enables researchers to construct robust prospective scenarios and examine the global structure of nitrogen flows in agricultural ecosystems."
    )

    # 🔹 Mise en cache du chargement de l'image
    @st.cache_data
    def load_image():
        img = Image.open(image_path)
        width, height = img.size
        aspect_ratio = width / height  # Calcul du ratio
        return img, width, height, aspect_ratio

    # 🔹 Création d'une carte avec l'image en cache
    def create_image_map():
        img, width, height, aspect_ratio = load_image()

        # Définir les bounds pour garder les proportions
        bounds = [[-0.5, -0.5 * aspect_ratio], [0.5, 0.5 * aspect_ratio]]

        # Créer une carte sans fond de carte
        m = folium.Map(location=[0, 0.3], zoom_start=9.5, zoom_control=True, tiles=None)

        # Ajouter l'image avec les bonnes proportions
        image_overlay = folium.raster_layers.ImageOverlay(
            image=image_path,
            bounds=bounds,
            opacity=1,
            interactive=True,
            cross_origin=False,
        )
        image_overlay.add_to(m)

        return m

    # 🔹 Affichage de l'image dans Streamlit
    st.subheader("Metabolism Overview")

    m = create_image_map()  # Création de la carte avec l'image mise en cache

    # 🟢 Affichage de la carte avec Streamlit-Folium
    st_folium(m, width=900, height=600)

    st.subheader("Features")

    st.text(
        "Historical Data: Covers nitrogen flow analysis for the period from 1852 to 2014 across 33 French regions.      \nComprehensive Nitrogen Flux Model: Includes 36 varieties of crops, 6 livestock categories, 2 population categories, 2 industrial sectors and 20 mores objects for environmental interactions and trade."
    )
    st.text(
        "Go to 'Run' tab to select a year and region to run GRAFS-E. This will display the nitrogen transition matrix for this territory"
    )
    st.text(
        "Then use 'Sankey' tab to analyse direct input and output flows for each object."
    )

    st.text("Use map tab to display agricultural maps.")

    st.subheader("Methods")

    st.text(
        "The GRAFS-E model is designed to encapsulate the nitrogen utilization process in agricultural systems by considering historical transformations in French agricultural practices. It captures the transition from traditional crop-livestock agriculture to more intensive, specialized systems."
    )
    st.text(
        "GRAFS-E uses optimization model to allocate plant productions to livestock, population and trade."
    )

    st.text(
        "By integrating optimization techniques and new mechanisms, GRAFS-E allows for the detailed study of nitrogen flows at a finer resolution than the original GRAFS model, covering 64 distinct objects, including various types of crops, livestock, population groups, industrial sectors, import/export category, and 6 environment category. The extension of GRAFS makes it possible to examine the topology and properties of the graph build with this flow model. This approach, provides a deeper understanding of the structure of the system, notably identifying invariants and hubs."
    )

    st.subheader("Results")
    st.text(
        "The model generates extensive transition matrices representing nitrogen flows between different agricultural components."
    )
    st.text(
        "These matrices can be used for several analysis as network analysis, input-output analysis, environmental footprint analysis and so on."
    )

    st.subheader("Future Work")

    st.text(
        "- Prospective tool: a prospective mode will be developped to allow creation of various agricultural futurs and analyse their realism."
    )
    st.text(
        "- Implementation in a general inductrial ecology model : GRAFS-E will be integrated to MATER as agricultural sector sub-model."
    )
    st.text(
        "- Network Resilience: Further analysis using Ecological Network Analysis (ENA) can help improve the model's understanding of resilience in nitrogen flows."
    )
    st.text(
        "- Multi-Layer Models: Future versions may include additional structural flows such as energy, water, and financial transfers."
    )

    st.subheader("Data")
    st.text(
        "The GRAFS-E model relies on agronomic data and technical coefficients from previous research, which were initially used in the seminal GRAFS framework. It consists in production and area data from all cultures, livestock size and production, mean use of synthetic fertilization and total net feed import."
    )

    st.subheader("License")
    st.text(
        "This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the LICENSE file for details."
    )
    st.subheader("Contact")
    st.text(
        "For any questions or contributions, feel free to reach out to Adrien Fauste-Gay at adrien.fauste-gay@univ-grenoble-alpes.fr."
    )

with tab2:
    st.title("Territory selection")
    st.write("Please select a year and a territory then click on Run.")

    # 🟢 Sélection de l'année
    st.subheader("Select a year")
    st.session_state.year = st.selectbox("", annees_disponibles, index=0)

    # 🔹 Vérifier la connexion Internet
    @st.cache_data
    def is_online():
        try:
            requests.get("https://www.google.com", timeout=3)
            return True
        except requests.ConnectionError:
            return False

    # 🔹 Charger les données GeoJSON avec cache
    @st.cache_data
    def load_geojson():
        with open(geojson_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 🔹 Création de la carte avec ou sans fond de carte
    def create_map():
        geojson_data = load_geojson()  # Charger les données JSON
        map_center = [48.8566, 2.3522]  # Centre de la carte (ex: Paris)

        # Vérifier la connexion Internet
        online = is_online()

        # Si Internet est disponible, utiliser un fond de carte normal
        if online:
            m = folium.Map(location=map_center, zoom_start=6)
        else:
            st.warning(
                "⚠️ No Internet connection detected. The map will be displayed without background tiles."
            )
            m = folium.Map(
                location=map_center, zoom_start=6, tiles=None
            )  # Pas de fond de carte

        # Style des régions survolées
        def on_click(feature):
            return {
                "fillColor": "#ffaf00",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.6,
                "highlight": True,
            }

        # Ajouter les polygones GeoJSON
        geo_layer = folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                "fillColor": "#0078ff",
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.5,
            },
            tooltip=folium.GeoJsonTooltip(fields=["nom"], aliases=["Région :"]),
            highlight_function=on_click,
        )

        geo_layer.add_to(m)
        return m

    # 🔹 Affichage de la carte dans Streamlit
    st.subheader("Select a territory")

    m = create_map()  # Crée la carte en fonction de l'état de la connexion Internet

    # 🟢 Affichage de la carte avec Streamlit-Folium
    map_data = st_folium(m, width=700, height=500)

    # 🔹 Mettre à jour `st.session_state.selected_region` avec la sélection utilisateur
    if map_data and "last_active_drawing" in map_data:
        last_drawing = map_data["last_active_drawing"]
        if (
            last_drawing
            and "properties" in last_drawing
            and "nom" in last_drawing["properties"]
        ):
            st.session_state.selected_region = last_drawing["properties"]["nom"]

    # ✅ Affichage des sélections (se met à jour dynamiquement)
    if st.session_state.selected_region:
        st.write(f"✅ Région sélectionnée : {st.session_state.selected_region}")
    else:
        st.warning("⚠️ Veuillez sélectionner une région")

    if st.session_state.year:
        st.write(f"✅ Année sélectionnée : {st.session_state.year}")
    else:
        st.warning("⚠️ Veuillez sélectionner une année")

    # 🟢 Fonction pour générer la heatmap et éviter les recalculs inutiles
    @st.cache_data
    def generate_heatmap(_model, year, region):
        return _model.plot_heatmap_interactive()

    # 🔹 Bouton "Run" avec les valeurs mises à jour
    if st.button("Run"):
        if st.session_state.selected_region and st.session_state.year:
            # Initialiser le modèle avec les paramètres
            st.session_state.model = NitrogenFlowModel(
                data=data,
                year=st.session_state.year,
                region=st.session_state.selected_region,
                categories_mapping=categories_mapping,
                labels=labels,
                cultures=cultures,
                legumineuses=legumineuses,
                prairies=prairies,
                betail=betail,
                Pop=Pop,
                ext=ext,
            )

            # st.session_state["model"] = model

            # ✅ Générer la heatmap et la stocker
            st.session_state.heatmap_fig = generate_heatmap(
                st.session_state.model,
                st.session_state.year,
                st.session_state.selected_region,
            )
        else:
            st.warning(
                "❌ Please select a year and a region before running the analysis."
            )

    # 🔹 Indépendance de l'affichage de la heatmap 🔹
    if "heatmap_fig" in st.session_state:
        st.subheader(
            f"Heatmap of the nitrogen flows for {st.session_state.selected_region} in {st.session_state.year}"
        )
        st.plotly_chart(st.session_state.heatmap_fig, use_container_width=True)

with tab3:
    st.title("Sankey")

    # Vérifier si le modèle a été exécuté
    if "model" not in st.session_state:
        st.warning("⚠️ Please run the model first in the 'Run' tab.")
    else:
        # Récupérer l'objet model
        model = st.session_state["model"]

        # 🔹 Ajouter un bouton de mode
        mode_complet = st.toggle("Detailed view", value=False, key="first")

        streamlit_sankey_app(model, mode_complet)

        st.subheader("Fertilization in the territory")

        mode_complet_ferti = st.toggle("Detailed view", value=False, key="ferti")

        if mode_complet_ferti:
            merge = {
                "Population": ["urban", "rural"],
                "Livestock": [
                    "bovines",
                    "ovines",
                    "equine",
                    "poultry",
                    "porcines",
                    "caprines",
                ],
                "Industry": ["Haber-Bosch", "other sectors"],
            }
            tre = 1e-1
        else:
            merge = {
                "Livestock and human": [
                    "bovines",
                    "ovines",
                    "equine",
                    "poultry",
                    "porcines",
                    "caprines",
                    "urban",
                    "rural",
                ],
                "Industry": ["Haber-Bosch", "other sectors"],
                "Cereals": [
                    "Wheat",
                    "Oat",
                    "Barley",
                    "Grain maize",
                    "Rye",
                    "Other cereals",
                    "Rice",
                ],
                "Grassland and forages": [
                    "Natural meadow ",
                    "Straw",
                    "Forage maize",
                    "Non-legume temporary meadow",
                    "Forage cabbages",
                ],
                "Oleaginous": ["Rapeseed", "Sunflower", "Hemp", "Flax"],
                "Leguminous": [
                    "Soybean",
                    "Other oil crops",
                    "Horse beans and faba beans",
                    "Peas",
                    "Other protein crops",
                    "Green peas",
                    "Dry beans",
                    "Green beans",
                    "Alfalfa and clover",
                ],
                "Fruits and vegetables": [
                    "Dry vegetables",
                    "Dry fruits",
                    "Squash and melons",
                    "Cabbage",
                    "Leaves vegetables",
                    "Fruits",
                    "Olives",
                    "Citrus",
                ],
                "Roots": ["Sugar beet", "Potatoes", "Other roots"],
            }
            tre = 1

        st.write(
            f"Nodes for which throughflow is below {tre} ktN/yr are not shown here."
        )

        streamlit_sankey_fertilization(
            model, cultures, legumineuses, prairies, merges=merge, THRESHOLD=tre
        )

        st.subheader("Feed for livestock and Food for local population")

        mode_complet_food = st.toggle("Detailed view", value=False, key="food")

        if mode_complet_food:
            merge = {
                "Cereals (excluding rice) trade": [
                    "cereals (excluding rice) food trade",
                    "cereals (excluding rice) feed trade",
                ],
                "Fruits and vegetables trade": [
                    "fruits and vegetables food trade",
                    "fruits and vegetables feed trade",
                ],
                "Leguminous trade": ["leguminous food trade", "leguminous feed trade"],
                "Oleaginous trade": ["oleaginous food trade", "oleaginous feed trade"],
            }
            tre = 1e-1
        else:
            merge = {
                "Cereals (excluding rice) trade": [
                    "cereals (excluding rice) food trade",
                    "cereals (excluding rice) feed trade",
                ],
                "Fruits and vegetables trade": [
                    "fruits and vegetables food trade",
                    "fruits and vegetables feed trade",
                ],
                "Leguminous trade": ["leguminous food trade", "leguminous feed trade"],
                "Oleaginous trade": ["oleaginous food trade", "oleaginous feed trade"],
                "Population": ["urban", "rural"],
                "Livestock": [
                    "bovines",
                    "ovines",
                    "equine",
                    "poultry",
                    "porcines",
                    "caprines",
                ],
                "Cereals": [
                    "Wheat",
                    "Oat",
                    "Barley",
                    "Grain maize",
                    "Rye",
                    "Other cereals",
                    "Rice",
                ],
                "Grassland and forages": [
                    "Natural meadow ",
                    "Straw",
                    "Forage maize",
                    "Non-legume temporary meadow",
                    "Forage cabbages",
                ],
                "Oleaginous": ["Rapeseed", "Sunflower", "Hemp", "Flax"],
                "Leguminous": [
                    "Soybean",
                    "Other oil crops",
                    "Horse beans and faba beans",
                    "Peas",
                    "Other protein crops",
                    "Green peas",
                    "Dry beans",
                    "Green beans",
                    "Alfalfa and clover",
                ],
                "Fruits and vegetables": [
                    "Dry vegetables",
                    "Dry fruits",
                    "Squash and melons",
                    "Cabbage",
                    "Leaves vegetables",
                    "Fruits",
                    "Olives",
                    "Citrus",
                ],
                "Roots": ["Sugar beet", "Potatoes", "Other roots"],
            }
            tre = 1

        st.write(
            f"Nodes for which throughflow is below {tre} ktN/yr are not shown here."
        )

        trades = [
            "animal trade",
            "cereals (excluding rice) food trade",
            "fruits and vegetables food trade",
            "leguminous food trade",
            "oleaginous food trade",
            "roots food trade",
            "rice food trade",
            "cereals (excluding rice) feed trade",
            "forages feed trade",
            "leguminous feed trade",
            "oleaginous feed trade",
            "grasslands feed trade",
        ]

        streamlit_sankey_food_flows(
            model, cultures, legumineuses, prairies, trades, merges=merge
        )

        st.subheader("Territorial Systemic Overview")
        st.write(
            "This Sankey diagram presents the primary flows (>1ktN/yr) within the model, organized by key categories."
        )
        st.write("For optimal visualization, please switch to full screen mode.")

        streamlit_sankey_systemic_flows(model, THRESHOLD=1)

with tab4:
    st.title("Detailed data")

    if "model" not in st.session_state:
        st.warning("⚠️ Please run the model first in the 'Run' tab.")
    else:
        st.text(
            "This tab is to access to detailed data used in input but also processed by the model"
        )

        st.subheader("Cultures data")

        st.dataframe(model.df_cultures)

        st.subheader("Livestock data")

        st.dataframe(model.df_elevage)

        st.subheader("Culture allocation to livestock and population")

        st.dataframe(model.allocation_vege)

        st.subheader("Diet deviations from defined diet")

        st.dataframe(model.deviations_df)


# 📌 Stocker et récupérer les modèles pour chaque région en cache
@st.cache_resource
def run_models_for_all_regions(year, regions, _data_loader):
    models = {}
    for region in regions:
        models[region] = NitrogenFlowModel(
            data=_data_loader,
            year=year,
            region=region,
            categories_mapping=categories_mapping,
            labels=labels,
            cultures=cultures,
            legumineuses=legumineuses,
            prairies=prairies,
            betail=betail,
            Pop=Pop,
            ext=ext,
        )
    return models


# 📌 Calculer et stocker les métriques pour chaque région en cache
@st.cache_data
def get_metrics_for_all_regions(_models, metric_name):
    metric_dict = {"Imported nitrogen": "imported_nitrogen"}
    metric_function_name = metric_dict[metric_name]
    metrics = {}
    for region, model in _models.items():
        metric_function = getattr(model, metric_function_name, None)
        if callable(metric_function):
            metrics[region] = metric_function()
        else:
            metrics[region] = None  # Si la méthode n'existe pas, on met None
    return metrics


@st.cache_data
def get_metric_range(metrics):
    """Récupère les valeurs min et max de l'indicateur sélectionné."""
    values = np.array(list(metrics.values()), dtype=np.float64)
    return np.nanmin(values), np.nanmax(values)  # Ignore les NaN


def add_color_legend(m, vmin, vmax, cmap, metric_name):
    """Ajoute une légende de la colormap à la carte."""
    colormap = branca.colormap.LinearColormap(
        vmin=vmin,
        vmax=vmax,
        colors=[mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, 256)],
    ).to_step(index=np.linspace(vmin, vmax, 25))  # 5 niveaux de couleur

    colormap.caption = str(metric_name)
    colormap.add_to(m)


# 📌 Fonction pour créer la carte et stocker dans `st.session_state`
@st.cache_resource
def create_map_with_metrics(geojson_data, metrics, metric_name):
    map_center = [48.8566, 2.3522]  # Centre (Paris)
    m = folium.Map(location=map_center, zoom_start=6, tiles="OpenStreetMap")

    for feature in geojson_data["features"]:
        region_name = feature["properties"]["nom"]
        metric_value = metrics.get(region_name, None)

        if metric_value is not None:
            # 📌 Obtenir min et max du metric sélectionné
            min_val, max_val = get_metric_range(metrics)

            # 📌 Normaliser et mapper les couleurs
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
            cmap = cm.get_cmap("plasma")  # Utilisation de la colormap "plasma"

            for feature in geojson_data["features"]:
                region_name = feature["properties"]["nom"]
                metric_value = metrics.get(
                    region_name, np.nan
                )  # Valeur de l'indicateur

                if np.isnan(metric_value):  # Si pas de donnée, couleur grise
                    color = "#CCCCCC"
                else:
                    rgba = cmap(norm(metric_value))  # Convertir en RGBA
                    color = mcolors.to_hex(rgba)  # Convertir en HEX

                # Ajouter le polygone à la carte
                folium.GeoJson(
                    feature,
                    style_function={
                        "fillColor": color,
                        "color": "black",
                        "weight": 1,
                        "fillOpacity": 0.6,
                    },
                    tooltip=folium.Tooltip(f"{region_name}: {metric_value:.2f} ktN/yr"),
                ).add_to(m)
    add_color_legend(m, min_val, max_val, cmap, metric_name)
    return m


# 🔹 Assurer la persistance de la carte dans `st.session_state`
if "map_html" not in st.session_state:
    st.session_state.map_html = None

with tab5:
    st.title("Map Configuration")

    # 🟢 Sélection de l'année
    st.session_state.map_year = st.selectbox(
        "Select a year", annees_disponibles, index=0, key="year_map_selection"
    )

    # 🟢 Sélection de la métrique
    metric = ["Imported nitrogen"]
    st.session_state.metric = st.selectbox(
        "Select a metric", metric, index=0, key="metric_selection"
    )

    # 🔹 Bouton "Run"
    if st.button("Run", key="map_button"):
        # 📌 Exécuter les modèles et récupérer les métriques
        models = run_models_for_all_regions(st.session_state.map_year, regions, data)
        metrics = get_metrics_for_all_regions(models, st.session_state.metric)

        # 📌 Charger le GeoJSON et créer la carte
        geojson_data = load_geojson()

        @st.cache_resource
        def get_cached_map(geojson_data, metrics, metric_name):
            return create_map_with_metrics(geojson_data, metrics, metric_name)

        map_obj = get_cached_map(geojson_data, metrics, st.session_state.metric)

        # 📌 Convertir la carte en HTML pour éviter la disparition
        st.session_state.map_html = map_obj._repr_html_()

        # 🔹 Vérifier si la carte est déjà générée et l'afficher
        st.title("Nitrogen Flow Map")

        if st.session_state.map_html:
            st.components.v1.html(st.session_state.map_html, height=600, scrolling=True)
        else:
            st.warning("Please run the model to generate the map.")


# %%
