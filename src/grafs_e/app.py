# %%
import copy
import io
import json
import os
import pickle
import tempfile
from importlib.metadata import version
from pathlib import Path

import branca
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from grafs_e.donnees import *
from grafs_e.N_class import DataLoader, NitrogenFlowModel
from grafs_e.prospective import NitrogenFlowModel_prospect, scenario
from grafs_e.sankey import (
    streamlit_sankey_app,
    streamlit_sankey_fertilization,
    streamlit_sankey_food_flows,
)
from grafs_e.system_flows_svg import (
    mapping_svg_fluxes_hist,
    mapping_svg_fluxes_pros,
    streamlit_sankey_systemic_flows_svg,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

geojson_path = os.path.join(DATA_DIR, "contour-GRAFS.geojson")
image_path = os.path.join(DATA_DIR, "metabolism.png")
icon_path = os.path.join(DATA_DIR, "logo.jpg")

# Déterminer le chemin du fichier de config Streamlit
config_dir = os.path.expanduser("~/.streamlit")
config_path = os.path.join(config_dir, "config.toml")

# Vérifier si le dossier ~/.streamlit existe, sinon le créer
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

# Écrire ou modifier le fichier config.toml pour imposer le dark mode
with open(config_path, "w") as config_file:
    config_file.write("[theme]\nbase='dark'\n")

st.set_page_config(page_title="GRAFS-E App", page_icon=icon_path, layout="wide")  # , layout="wide")


# Charger les données
@st.cache_data
def load_data():
    # votre DataLoader (ou équivalent)
    return DataLoader()


if "data" not in st.session_state:
    st.session_state["data"] = load_data()

data = st.session_state["data"]

# Initialisation de l'état des variables
for k, v in {
    "model": None,
    "name": None,
    "year": None,
    "year_run": None,
    "year_pros": None,
    "selected_region": None,
    "selected_region_run": None,
    "selected_region_pros": "",
    "prod_func": "",
    "heatmap_fig_pros": None,
    "pkl_blob": None,  # ← binaire pour download
    "load_error": "",
    # état provisoire (Excel upload)
    "prep_name": None,
    "prep_region": None,
    "prep_year": None,
    "prep_func": None,
    "prep_xlsx_path": None,
    "excel_uploaded_done": False,
    "orig": None,
}.items():
    st.session_state.setdefault(k, v)
# %%
# Initialisation de l'interface Streamlit
st.title("GRAFS-E")
__version__ = version("grafs_e")
st.write(f"📦 GRAFS-E version: {__version__}")
st.text("Contact: Adrien Fauste-Gay, adrien.fauste-gay@univ-grenoble-alpes.fr")
st.title("Nitrogen Flow Simulation Model: A Territorial Ecology Approach")

# # 🔹 Initialiser les valeurs dans session_state si elles ne sont pas encore définies
# if "selected_region" not in st.session_state:
#     st.session_state.selected_region = None
# if "year" not in st.session_state:
#     st.session_state.year = None

if st.session_state.selected_region:
    st.write(f"✅ Selected territory: {st.session_state.selected_region}")
else:
    st.warning("⚠️ Please select a region")

if st.session_state.year:
    st.write(f"✅ Selected year : {st.session_state.year}")
else:
    st.warning("⚠️ Please select a year")

# -- Sélection des onglets --
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Documentation",
        "Run",
        "Sankey",
        "Detailed data",
        "Map",
        "Historic Evolution",
        "Scenario Generator",
        "Prospective Mode",
    ]
)

with tab1:
    st.title("Documentation")

    st.header("GRAFS-Extended: Comprehensive Analysis of Nitrogen Flux in Agricultural Systems")

    st.subheader("Overview")

    st.write(
        """
    <p style='text-align: justify'>
        The GRAFS-extended model serves as an advanced tool designed to analyze and map the evolution of nitrogen utilization within agricultural systems, with a particular focus on 33 regions of France from 1852 to 2014. This model builds upon the GRAFS framework developed at IEES and integrates graph theory to provide a detailed analysis of nitrogen flows in agriculture, identifying key patterns, transformations, and structural invariants. The model enables researchers to construct robust prospective scenarios and examine the global structure of nitrogen flows in agricultural ecosystems. Technical documentation can be accessed here :
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Create a link button to the documentation
    st.link_button(
        "📖  Python Documentation",
        "https://adrienfaustegay.github.io/grafs-e-docs/",
        help="Opens the project's technical documentation in a new tab.",
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
        m = folium.Map(location=[0, 0], zoom_start=9.5, zoom_control=True, tiles=None)

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
    st_folium(m, height=600, use_container_width=True)

    st.subheader("Features")

    st.text(
        "Historical Data: Covers nitrogen flow analysis for the period from 1852 to 2014 across 33 French regions.      \nComprehensive Nitrogen Flux Model: Includes 36 varieties of crops, 6 livestock categories, 2 population categories, 2 industrial sectors and 20 mores objects for environmental interactions and trade."
    )
    st.text(
        "Go to 'Run' tab to select a year and region to run GRAFS-E. This will display the nitrogen transition matrix for this territory"
    )
    st.text("Then use 'Sankey' tab to analyse direct input and output flows for each object.")

    st.text("Use map tab to display agricultural maps.")

    st.subheader("Methods")

    st.write(
        """
    <p style='text-align: justify'>
        The GRAFS-E model is designed to encapsulate the nitrogen utilization process in agricultural systems by considering historical transformations in French agricultural practices. It captures the transition from traditional crop-livestock agriculture to more intensive, specialized systems.
    </p>
    <p style='text-align: justify'>
        GRAFS-E uses optimization model to allocate plant productions to livestock, population and trade.
    </p>
    <p style='text-align: justify'>
        By integrating optimization techniques and new mechanisms, GRAFS-E allows for the detailed study of nitrogen flows at a finer resolution than the original GRAFS model, covering 64 distinct objects, including various types of crops, livestock, population groups, industrial sectors, import/export category, and 6 environment category. The extension of GRAFS makes it possible to examine the topology and properties of the graph built with this flow model. This approach provides a deeper understanding of the structure of the system, notably identifying invariants and hubs.
    </p>
    """,
        unsafe_allow_html=True,
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

    st.write(
        """
    <p style='text-align: justify'>
        The GRAFS-E model relies on agronomic data and technical coefficients from previous research, which were initially used in the seminal GRAFS framework. It consists in production and area data from all cultures, livestock size and production, mean use of synthetic fertilization and total net feed import.
    </p>
    """,
        unsafe_allow_html=True,
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
    st.session_state.year_run = st.selectbox("", annees_disponibles, index=0)

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

    # 🔹 Fonction pour ajouter le rectangle englobant toute la France (AU DÉBUT)
    def add_france_rectangle(geojson_data):
        all_coords = []
        for feature in geojson_data["features"]:
            if feature["geometry"]["type"] == "Polygon":
                all_coords.extend(feature["geometry"]["coordinates"][0])
            elif feature["geometry"]["type"] == "MultiPolygon":
                for polygon in feature["geometry"]["coordinates"]:
                    all_coords.extend(polygon[0])

        # Extraire les bornes extrêmes
        coords = np.array(all_coords)
        min_lon, min_lat = coords.min(axis=0)
        max_lon, max_lat = coords.max(axis=0)

        # Créer le rectangle pour la France entière
        france_rectangle = {
            "type": "Feature",
            "properties": {"nom": "France"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat], [min_lon, min_lat]]
                ],
            },
        }

        # ✅ Insérer le rectangle AVANT les autres régions => En dessous des autres polygones
        geojson_data["features"].insert(0, france_rectangle)
        return geojson_data

    # 🔹 Création de la carte avec ou sans fond de carte
    def create_map():
        geojson_data = load_geojson()  # Charger les données JSON
        geojson_data = add_france_rectangle(geojson_data)
        map_center = [46.6034, 1.8883]  # Centre approximatif de la France

        # Vérifier la connexion Internet
        online = is_online()

        # Si Internet est disponible, utiliser un fond de carte normal
        if online:
            m = folium.Map(location=map_center, zoom_start=6)
        else:
            st.warning("⚠️ No Internet connection detected. The map will be displayed without background tiles.")
            m = folium.Map(location=map_center, zoom_start=6, tiles=None)  # Pas de fond de carte

        # Style des régions survolées
        def on_click(feature):
            return {
                "fillColor": "#ffaf00" if feature["properties"]["nom"] == "France" else "#0078ff",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.6,
                "highlight": True,
            }

        # Ajouter les polygones GeoJSON
        geo_layer = folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                "fillColor": "#ffaf00" if feature["properties"]["nom"] == "France" else "#0078ff",
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.5,
            },
            tooltip=folium.GeoJsonTooltip(fields=["nom"], aliases=["Territory :"]),
            highlight_function=on_click,
        )

        geo_layer.add_to(m)
        return m

    # 🔹 Affichage de la carte dans Streamlit
    st.subheader("Select a territory")

    m = create_map()  # Crée la carte en fonction de l'état de la connexion Internet

    # 🟢 Affichage de la carte avec Streamlit-Folium
    map_data = st_folium(m, height=600, use_container_width=True)

    # 🔹 Mettre à jour `st.session_state.selected_region` avec la sélection utilisateur
    if map_data and "last_active_drawing" in map_data:
        last_drawing = map_data["last_active_drawing"]
        if last_drawing and "properties" in last_drawing and "nom" in last_drawing["properties"]:
            st.session_state.selected_region_run = last_drawing["properties"]["nom"]

    # ✅ Affichage des sélections (se met à jour dynamiquement)
    if st.session_state.selected_region_run:
        st.write(f"✅ Région sélectionnée : {st.session_state.selected_region_run}")
    else:
        st.warning("⚠️ Veuillez sélectionner une région")

    if st.session_state.year_run:
        st.write(f"✅ Année sélectionnée : {st.session_state.year_run}")
    else:
        st.warning("⚠️ Veuillez sélectionner une année")

    # 🟢 Fonction pour générer la heatmap et éviter les recalculs inutiles
    @st.cache_data
    def generate_heatmap(_model, year, region):
        _model = copy.deepcopy(_model)
        return _model.plot_heatmap_interactive()

    # 🔹 Bouton "Run" avec les valeurs mises à jour
    if st.button("Run"):
        st.session_state.selected_region = st.session_state.selected_region_run
        st.session_state.year = st.session_state.year_run
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
            st.warning("❌ Please select a year and a region before running the analysis.")

    # 🔹 Indépendance de l'affichage de la heatmap 🔹
    if "heatmap_fig" in st.session_state:
        if st.session_state.model:
            st.text(f"Total Throughflow : {np.round(st.session_state.model.get_transition_matrix().sum(), 1)} ktN/yr.")
        st.subheader(f"Heatmap of the nitrogen flows for {st.session_state.selected_region} in {st.session_state.year}")
        st.plotly_chart(st.session_state.heatmap_fig, use_container_width=True)

with tab3:
    st.title("Sankey")

    # Vérifier si le modèle a été exécuté
    if not st.session_state.model:
        st.warning("⚠️ Please run the model first in the 'Run' tab or 'Prospective mode' tab.")
    else:
        # Récupérer l'objet model
        model = st.session_state["model"]

        st.subheader("Territorial Systemic Overview")
        st.write(
            f"This Sankey diagram presents nitrogen flows in the agricultural system organized by key categories (10% = {np.round(model.adjacency_matrix.sum() / 10, 0)} ktN/yr)."
        )
        # st.write("For optimal visualization, please switch to full screen mode.")

        # streamlit_sankey_systemic_flows(model)
        # os.path.join(os.getcwd(), "data/system_flows.svg")
        # 1) Point de départ : le dossier racine du projet
        base = Path(__file__).parent.parent  # par exemple, un dossier au-dessus de app.py
        # 2) Construire le chemin vers le SVG
        # svg_template_path = base / "grafs_e" / "data" / "system_flows.svg"

        if int(st.session_state.year) > 2014:
            mapping_svg_fluxes = mapping_svg_fluxes_pros
            svg_template_path = base / "grafs_e" / "data" / "system_flows_pros.svg"
        else:
            mapping_svg_fluxes = mapping_svg_fluxes_hist
            svg_template_path = base / "grafs_e" / "data" / "system_flows_hist.svg"

        streamlit_sankey_systemic_flows_svg(model, mapping_svg_fluxes, svg_template_path)

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
                "Forages": [
                    "Straw",
                    "Forage maize",
                    "Forage cabbages",
                ],
                "Temporary meadows": ["Non-legume temporary meadow", "Alfalfa and clover"],
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

        st.write(f"Nodes for which throughflow is below {tre} ktN/yr are not shown here.")

        streamlit_sankey_fertilization(model, cultures, legumineuses, prairies, merges=merge, THRESHOLD=tre)

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
                "Forages": [
                    "Straw",
                    "Forage maize",
                    "Forage cabbages",
                ],
                "Temporary meadows": ["Non-legume temporary meadow", "Alfalfa and clover"],
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

        st.write(f"Nodes for which throughflow is below {tre} ktN/yr are not shown here.")

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
            "temporary meadows feed trade",
            "natural meadows feed trade",
        ]

        streamlit_sankey_food_flows(model, cultures, legumineuses, prairies, trades, merges=merge)

with tab4:
    st.title("Detailed data")

    if not st.session_state.model:
        st.warning("⚠️ Please run the model first in the 'Run' tab or in the 'Prospective mode' tab.")
    else:
        st.text("This tab is to access to detailed data used in input but also processed by the model")

        st.subheader("Cultures data")

        st.dataframe(model.df_cultures_display)

        st.subheader("Livestock data")

        st.dataframe(model.df_elevage_display)

        st.subheader("Culture allocation to livestock and population")

        st.dataframe(model.allocation_vege)

        st.subheader("Diet deviations from defined diet")

        st.dataframe(model.deviations_df)

        if int(st.session_state.year) > 2024:
            st.subheader("Energy from biomass")
            st.text("Methaniser allocation")
            st.dataframe(model.allocation_energy_df)

            st.text("Input deviation for methanizer")
            st.dataframe(model.meth_deviation_df)


# 📌 Stocker et récupérer les modèles pour chaque région en cache
@st.cache_resource
def run_models_for_all_regions(year, regions, _data_loader):
    models = {}
    for region in regions:
        if region == "Savoie" and year == "1852":
            pass
        else:
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
def get_metrics_for_all_regions(_models, metric_name, year):
    metric_dict = {
        "Total imported nitrogen": "imported_nitrogen",
        "Total net plant import": "net_imported_plant",
        "Total net animal import": "net_imported_animal",
        "Total plant production": "total_plant_production",
        "Environmental Footprint": "net_footprint",
        "NUE": "NUE",
        "System NUE": "NUE_system",
        "Livestock density": "LU_density",
        "Cereals production": "cereals_production",
        "Leguminous production": "leguminous_production",
        "Oleaginous production": "oleaginous_production",
        "Grassland and forage production": "grassland_and_forages_production",
        "Roots production": "roots_production",
        "Fruits and vegetables production": "fruits_and_vegetable_production",
        "Relative Cereals production": "cereals_production_r",
        "Relative Leguminous production": "leguminous_production_r",
        "Relative Oleaginous production": "oleaginous_production_r",
        "Relative Grassland and forage production": "grassland_and_forages_production_r",
        "Relative Roots production": "roots_production_r",
        "Relative Fruits and vegetables production": "fruits_and_vegetable_production_r",
        "Total animal production": "animal_production",
        "NH3 volatilization": "NH3_vol",
        "N2O emission": "N2O_em",
        "Effective number of nodes": "N_eff",
        "Effective connectivity": "C_eff",
        "Effective number of links": "F_eff",
        "Effective number of role": "R_eff",
    }
    metric_function_name = metric_dict[metric_name]
    metrics = {}
    area = {}
    for region, model in _models.items():
        metric_function = getattr(model, metric_function_name, None)
        if callable(metric_function):
            metrics[region] = metric_function()
        else:
            metrics[region] = None  # Si la méthode n'existe pas, on met None
        area[region] = model.surfaces_tot()
    return metrics, area


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
def create_map_with_metrics(geojson_data, metrics, metric_name, year):
    map_center = [46.6034, 1.8883]  # Centre approximatif de la France
    m = folium.Map(location=map_center, zoom_start=6, tiles="OpenStreetMap")

    min_val, max_val = get_metric_range(metrics)

    if "net" in metric_name or "Footprint" in metric_name:
        cmap = cm.get_cmap("bwr")
        min_val = min(min_val, -abs(max_val))
        max_val = max(abs(min_val), max_val)
    else:
        cmap = cm.get_cmap("plasma")  # Utilisation de la colormap "plasma"

    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    if "Relative" in metric_name or "NUE" in metric_name:
        unit = "%"
    elif "Eff" in metric_name:
        unit = ""
    elif "Footprint" in metric_name:
        unit = "Mha"
    elif "Livestock density" == metric_name:
        unit = "LU/ha"
    else:
        unit = "ktN/yr"

    # 📊 Nouveau : créer le tableau des valeurs régionales
    table_data = []

    for feature in geojson_data["features"]:
        region_name = feature["properties"]["nom"]
        metric_value = metrics.get(region_name, np.nan)

        if np.isnan(metric_value):
            color = "#CCCCCC"
        else:
            rgba = cmap(norm(metric_value))
            color = mcolors.to_hex(rgba)

        def style_function(x, fill_color=color):
            return {
                "fillColor": fill_color,
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.6,
            }

        folium.GeoJson(
            feature,
            style_function=style_function,
            tooltip=folium.Tooltip(f"{region_name}: {metric_value:.2f} {unit}"),
        ).add_to(m)

        table_data.append(
            {
                "Région": region_name,
                "Valeur": round(metric_value, 2) if not np.isnan(metric_value) else None,
                # "Unité": unit,
            }
        )

    add_color_legend(m, min_val, max_val, cmap, metric_name)

    df = pd.DataFrame(table_data)

    return m, df


def weighted_mean(metrics, area):
    # Calcul de la somme des produits des valeurs et des poids
    weighted_sum = sum(metrics[key] * area[key] for key in metrics)

    # Calcul de la somme des poids
    total_area = sum(area[key] for key in area)

    # Retourner la moyenne pondérée
    return weighted_sum / total_area if total_area != 0 else 0


# 🔹 Assurer la persistance de la carte dans `st.session_state`
if "map_html" not in st.session_state:
    st.session_state.map_html = None

with tab5:
    st.title("Map Configuration")

    # 🟢 Sélection de l'année
    st.session_state.map_year = st.selectbox("Select a year", annees_disponibles, index=0, key="year_map_selection")

    # 🟢 Sélection de la métrique
    metric_map = [
        "Total imported nitrogen",
        "Total net plant import",
        "Total net animal import",
        "Total plant production",
        "Total animal production",
        "Environmental Footprint",
        "NUE",
        "System NUE",
        "Livestock density",
        "Cereals production",
        "Leguminous production",
        "Grassland and forage production",
        "Roots production",
        "Oleaginous production",
        "Fruits and vegetables production",
        "Relative Cereals production",
        "Relative Leguminous production",
        "Relative Grassland and forage production",
        "Relative Roots production",
        "Relative Oleaginous production",
        "Relative Fruits and vegetables production",
        "NH3 volatilization",
        "N2O emission",
        # "Effective number of nodes",
        # "Effective connectivity",
        # "Effective number of links",
        # "Effective number of role",
    ]
    st.session_state.metric = st.selectbox("Select a metric", metric_map, index=0, key="metric_selection")

    # 🔹 Bouton "Run"
    if st.button("Run", key="map_button"):

        @st.cache_resource
        def get_cached_map(geojson_data, metrics, metric_name, year):
            return create_map_with_metrics(geojson_data, metrics, metric_name, st.session_state.map_year)

        with st.spinner("🚀 Running models and calculating metrics..."):
            # 📌 Exécuter les modèles et récupérer les métriques
            models = run_models_for_all_regions(st.session_state.map_year, regions, data)
            metrics, area = get_metrics_for_all_regions(models, st.session_state.metric, st.session_state.map_year)

            # 📌 Charger le GeoJSON et créer la carte
            geojson_data = load_geojson()

            map_obj, table = get_cached_map(geojson_data, metrics, st.session_state.metric, st.session_state.map_year)

            # 📌 Convertir la carte en HTML pour éviter la disparition
            st.session_state.map_html = map_obj._repr_html_()

            # 🔹 Vérifier si la carte est déjà générée et l'afficher
            st.title("Nitrogen Map")
            if st.session_state.metric in [
                "NUE",
                "System NUE",
                "Livestock density",
                "Relative Cereals production",
                "Relative Leguminous production",
                "Relative Grassland and forage production",
                "Relative Roots production",
                "Relative Oleaginous production",
                "Relative Fruits and vegetables production",
                "Environmental Footprint",
            ]:
                if st.session_state.metric == "Livestock density":
                    st.text(f"Mean for France: {np.round(weighted_mean(metrics, area), 2)} LU/ha")
                elif st.session_state.metric == "Environmental Footprint":
                    # st.text(f"Mean for France: {np.round(weighted_mean(metrics, area), 2)} Mha")
                    pass
                else:
                    st.text(f"Mean for France: {np.round(weighted_mean(metrics, area), 2)} %")
            else:
                st.text(f"Total for France: {np.round(np.sum(list(metrics.values())), 2)} ktN/yr")

        if st.session_state.map_html:
            st.components.v1.html(st.session_state.map_html, height=800, scrolling=True)
            st.dataframe(table)
            st.download_button("Download Data", table.to_csv(index=False), file_name="data.csv")
        else:
            st.warning("Please run the model to generate the map.")


@st.cache_resource
def run_models_for_all_years(region, _data_loader):
    models = {}
    for year in annees_disponibles:
        if year == "1852" and region == "Savoie":
            pass
        else:
            models[year] = NitrogenFlowModel(
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
def get_metrics_for_all_years(_models, metric_name, region):
    metric_dict = {
        "Total imported nitrogen": "imported_nitrogen",
        "Total net plant import": "net_imported_plant",
        "Total net animal import": "net_imported_animal",
        "Total plant production": "stacked_plant_production",
        "Area": "surfaces",
        "Environmental Footprint": "env_footprint",
        "Area tot": "surfaces_tot",
        "Total Fertilization": "tot_fert",
        "Relative Fertilization": "rel_fert",
        "Primary Nitrogen fertilization use": "primXsec",
        "Emissions": "emissions",
        "NUE": "NUE",
        "System NUE": "NUE_system_2",
        "Livestock density": "LU_density",
        "Self-Sufficiency": "N_self_sufficient",
        "Cereals production": "cereals_production",
        "Leguminous production": "leguminous_production",
        "Oleaginous production": "oleaginous_production",
        "Grassland and forage production": "grassland_and_forages_production",
        "Roots production": "roots_production",
        "Fruits and vegetables production": "fruits_and_vegetable_production",
        "Relative Cereals production": "cereals_production_r",
        "Relative Leguminous production": "leguminous_production_r",
        "Relative Oleaginous production": "oleaginous_production_r",
        "Relative Grassland and forage production": "grassland_and_forages_production_r",
        "Relative Roots production": "roots_production_r",
        "Relative Fruits and vegetables production": "fruits_and_vegetable_production_r",
        "Total animal production": "animal_production",
        "Effective number of nodes": "N_eff",
        "Effective connectivity": "C_eff",
        "Effective number of links": "F_eff",
        "Effective number of role": "R_eff",
    }
    metric_function_name = metric_dict[metric_name]
    metrics = {}
    for year, model in _models.items():
        metric_function = getattr(model, metric_function_name, None)
        if callable(metric_function):
            metrics[year] = metric_function()
        else:
            metrics[year] = None  # Si la méthode n'existe pas, on met None
    return metrics


def plot_standard_graph(_models, metric, region):
    metrics = get_metrics_for_all_years(_models, metric, region)

    # 📊 Préparation des données pour le graphique
    years = sorted([int(year) for year in metrics.keys()])
    values = list(metrics.values())

    # 🔄 Affichage conditionnel des labels
    min_gap = 4  # ⚙️ Seuil minimum entre deux labels
    visible_years = []
    last_visible_year = None

    for year in years:
        if last_visible_year is None or (year - last_visible_year) >= min_gap:
            visible_years.append(str(year))
            last_visible_year = year
        else:
            visible_years.append("")  # Pas de label si trop proche

    # 🔵 Détection du thème actuel
    current_theme = st.get_option("theme.base")

    if current_theme == "light":
        line_color = "royalblue"
    else:
        line_color = "white"

    # 📊 Création du graphique avec Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=values,
            mode="lines+markers",
            name=st.session_state.metric_hist,
            line=dict(color=line_color, width=2),
            marker=dict(size=8, color="royalblue", symbol="circle"),
        )
    )

    if "Eff" in metric:
        y_label = "#"
    elif "NUE" in metric or "Primary" in metric or "Sufficiency" in metric or "Relative" in metric:
        y_label = "%"
    elif "Livestock density" in metric:
        y_label = "LU/ha"
    else:
        y_label = "ktN/yr"

    # 🎨 Personnalisation du style du graphique
    fig.update_layout(
        title=f"Historical Evolution of {st.session_state.metric_hist} in {st.session_state.selected_region_hist}",
        xaxis_title="Year",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(x=0.05, y=0.95, bgcolor="rgba(255, 255, 255, 0.5)"),
    )

    # 🔍 Amélioration des axes
    fig.update_xaxes(
        showgrid=True,
        tickmode="array",
        tickvals=years,
        ticktext=visible_years,
        tickangle=45,
        # tickfont=dict(size=10),
    )

    fig.update_yaxes(showgrid=True, zeroline=True)

    # 🚀 Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)


def stacked_area_chart(_models, metric, region):
    """Affiche un graphique en courbes empilées pour l'évolution des surfaces cultivées
    avec une légende par catégorie et un hover individuel par culture."""

    # -----------------------------------------------------------------
    # 1) Récupération & Transformation des données
    # -----------------------------------------------------------------
    metrics = get_metrics_for_all_years(_models, metric, region)  # Dict { année(str) : df_cultures }

    all_years = sorted(metrics.keys(), key=int)

    # Suppose qu'on prend la première année comme référence pour l'index (les cultures)
    df = pd.DataFrame(index=metrics[all_years[0]].index, columns=all_years, dtype=float)

    # Remplir le DataFrame (Cultures x Années)
    for year, df_year in metrics.items():
        df.loc[df_year.index, year] = df_year

    df.fillna(0, inplace=True)

    # Calcul cumulatif pour l'affichage empilé par colonnes
    df_cumsum = df.cumsum(axis=0)
    # Ajouter une ligne "Base" (0) pour le fill='tonexty'
    df_cumsum.loc["Base"] = 0
    # df_cumsum = df_cumsum.sort_index()

    # -----------------------------------------------------------------
    # 2) Création du Sankey Plotly
    # -----------------------------------------------------------------
    fig = go.Figure()

    # On veut regrouper les cultures par catégorie pour la légende
    # => On utilise 'legendgroup' + 'showlegend' seulement au 1er trace de chaque catégorie
    categories_seen = set()

    # On veut hover = seulement la courbe survolée => hovermode='closest'
    # => On n'utilise plus 'x unified'
    if metric == "Area":
        fig.update_layout(
            title=f"Agricultural Area - {region}",
            xaxis_title="Year",
            yaxis_title="Cumulated Area (ha)",
            hovermode="closest",  # ❗️ Montre seulement la courbe survolée
            showlegend=True,
        )

        # Parcourir chaque culture dans l'ordre de l'index (df.index)
        # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
        cultures_list = [c for c in df_cumsum.index if c != "Base"]

        for culture in cultures_list:
            # Courbe du haut = df_cumsum.loc[culture]
            # fill='tonexty' => se remplit entre cette courbe et la précédente
            # => l'ordre du df_cumsum doit être correct (Base, ..., culture)
            # Couleur en fonction de la catégorie
            cat = categories_mapping.get(culture, "Unknown")
            culture_color = node_color[label_to_index[culture]]

            # Groupe de légende = cat
            # On affiche la légende qu'une seule fois par catégorie
            show_in_legend = cat not in categories_seen
            if show_in_legend:
                categories_seen.add(cat)

            fig.add_trace(
                go.Scatter(
                    x=all_years,
                    y=df_cumsum.loc[culture],
                    fill="tonexty",
                    mode="lines",
                    line=dict(color=culture_color, width=0.5),
                    name=cat,  # ❗️ Nom affiché = Catégorie
                    legendgroup=cat,  # ❗️ On groupe par Catégorie
                    customdata=df.loc[culture].tolist(),
                    showlegend=show_in_legend,  # ❗️ Un seul trace par groupe dans la légende
                    hovertemplate=("Culture: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} ha<extra></extra>"),
                    text=[culture] * len(all_years),  # Pour afficher le nom de la culture au survol
                )
            )

    if metric == "Total plant production":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(
            title=f"Agricultural Production - {region}",
            xaxis_title="Year",
            hovermode="closest",  # ❗️ Montre seulement la courbe survolée
            showlegend=True,
        )

        # Parcourir chaque culture dans l'ordre de l'index (df.index)
        # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
        cultures_list = [c for c in df_cumsum.index if c != "Base"]

        for culture in cultures_list:
            # Courbe du haut = df_cumsum.loc[culture]
            # fill='tonexty' => se remplit entre cette courbe et la précédente
            # => l'ordre du df_cumsum doit être correct (Base, ..., culture)
            # Couleur en fonction de la catégorie
            cat = categories_mapping.get(culture, "Unknown")
            culture_color = node_color[label_to_index[culture]]

            # Groupe de légende = cat
            # On affiche la légende qu'une seule fois par catégorie
            show_in_legend = cat not in categories_seen
            if show_in_legend:
                categories_seen.add(cat)

            fig.add_trace(
                go.Scatter(
                    x=all_years,
                    y=df_cumsum.loc[culture],
                    fill="tonexty",
                    mode="lines",
                    line=dict(color=culture_color, width=0.5),
                    name=cat,
                    legendgroup=cat,
                    customdata=df.loc[culture].tolist(),
                    showlegend=show_in_legend,
                    hovertemplate="Culture: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} ha<extra></extra>",
                    text=[culture] * len(all_years),
                ),
                secondary_y=False,  # Axe Y principal
            )

        prod_tot = get_metrics_for_all_years(_models, "Area tot", region)

        # Ajouter la ligne de production végétale totale
        fig.add_trace(
            go.Scatter(
                x=list(prod_tot.keys()),
                y=list(prod_tot.values()),
                mode="lines+markers",
                line=dict(color="white", width=3, dash="dash"),
                name="Total agricultural Area",
                hovertemplate="Year: %{x}<br>Value: %{y:.2f} ha<extra></extra>",
            ),
            secondary_y=True,  # Axe Y secondaire (à droite)
        )

        fig.update_yaxes(title_text="Cumulated Production (ktN/yr)", secondary_y=False)
        fig.update_yaxes(title_text="Total Area (ha)", secondary_y=True)
    if metric == "Emissions":
        fig.update_layout(
            title=f"Nitrogen emissions - {region}",
            xaxis_title="Year",
            yaxis_title="kTon/yr",
            hovermode="closest",  # ❗️ Montre seulement la courbe survolée
            showlegend=True,
        )

        # Parcourir chaque culture dans l'ordre de l'index (df.index)
        # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
        emissions_list = [c for c in df_cumsum.index if c != "Base"]

        color = {"N2O emission": "red", "atmospheric N2": "white", "NH3 volatilization": "blue"}

        for emission in emissions_list:
            # Courbe du haut = df_cumsum.loc[culture]
            # fill='tonexty' => se remplit entre cette courbe et la précédente
            # => l'ordre du df_cumsum doit être correct (Base, ..., culture)

            fig.add_trace(
                go.Scatter(
                    x=all_years,
                    y=df_cumsum.loc[emission],
                    fill="tonexty",
                    mode="lines",
                    line=dict(color=color[emission], width=0.5),
                    customdata=df.loc[emission].tolist(),
                    name=emission,  # ❗️ Nom affiché = Catégorie
                    hovertemplate=(
                        "Emission: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} kton/yr<extra></extra>"
                    ),
                    text=[emission] * len(all_years),  # Pour afficher le nom de la culture au survol
                )
            )

    if metric == "Total Fertilization":
        fig.update_layout(
            title=f"Total Fertilization Use - {region}",
            xaxis_title="Year",
            yaxis_title="ktN",
            hovermode="closest",  # ❗️ Montre seulement la courbe survolée
            showlegend=True,
        )

        # Parcourir chaque culture dans l'ordre de l'index (df.index)
        # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
        emissions_list = [c for c in df_cumsum.index if c != "Base"]

        color = {
            "Haber-Bosch": "purple",
            "Atmospheric deposition": "red",
            "atmospheric N2": "white",
            "Mining": "gray",
            "Seeds": "pink",
            "Animal excretion": "lightblue",
            "Human excretion": "darkblue",
            "Leguminous soil enrichment": "darkgreen",
        }

        for emission in emissions_list:
            # Courbe du haut = df_cumsum.loc[culture]
            # fill='tonexty' => se remplit entre cette courbe et la précédente
            # => l'ordre du df_cumsum doit être correct (Base, ..., culture)

            fig.add_trace(
                go.Scatter(
                    x=all_years,
                    y=df_cumsum.loc[emission],
                    fill="tonexty",
                    mode="lines",
                    line=dict(color=color[emission], width=0.5),
                    customdata=df.loc[emission].tolist(),
                    name=emission,  # ❗️ Nom affiché = Catégorie
                    hovertemplate=(
                        "Fertilization vector: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} ktN/yr<extra></extra>"
                    ),
                    text=[emission] * len(all_years),  # Pour afficher le nom de la culture au survol
                )
            )

        prod_tot = get_metrics_for_all_years(_models, "Total plant production", region)

        # Ajouter la ligne de production végétale totale
        fig.add_trace(
            go.Scatter(
                x=list(prod_tot.keys()),  # Clés du dictionnaire comme années
                y=list(prod_tot.values()),  # Valeurs du dictionnaire comme données
                mode="lines+markers",  # Ligne avec des marqueurs
                line=dict(color="white", width=3, dash="dash"),  # Ligne noire en pointillés pour la distinguer
                name="Total Plant Production",  # Légende
                hovertemplate="Year: %{x}<br>Value: %{y:.2f} ktN/yr<extra></extra>",  # Tooltip personnalisé
            )
        )

    if metric == "Relative Fertilization":
        fig.update_layout(
            title=f"Relative Fertilization Use - {region}",
            xaxis_title="Year",
            yaxis_title="%",
            hovermode="closest",  # ❗️ Montre seulement la courbe survolée
            showlegend=True,
        )

        # Parcourir chaque culture dans l'ordre de l'index (df.index)
        # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
        emissions_list = [c for c in df_cumsum.index if c != "Base"]

        color = {
            "Haber-Bosch": "purple",
            "Atmospheric deposition": "red",
            "atmospheric N2": "white",
            "Mining": "gray",
            "Seeds": "pink",
            "Animal excretion": "lightblue",
            "Human excretion": "darkblue",
            "Leguminous soil enrichment": "darkgreen",
        }

        for emission in emissions_list:
            # Courbe du haut = df_cumsum.loc[culture]
            # fill='tonexty' => se remplit entre cette courbe et la précédente
            # => l'ordre du df_cumsum doit être correct (Base, ..., culture)

            fig.add_trace(
                go.Scatter(
                    x=all_years,
                    y=df_cumsum.loc[emission],
                    fill="tonexty",
                    mode="lines",
                    line=dict(color=color[emission], width=0.5),
                    customdata=df.loc[emission].tolist(),
                    name=emission,  # ❗️ Nom affiché = Catégorie
                    hovertemplate=(
                        "Fertilization vector: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} %<extra></extra>"
                    ),
                    text=[emission] * len(all_years),  # Pour afficher le nom de la culture au survol
                )
            )

    if metric == "Environmental Footprint":
        color = {
            "Local Food": "blue",
            "Local Feed": "lightgreen",
            "Import Food": "lightgray",
            "Import Feed": "darkgray",
            "Import Livestock": "cyan",
            "Export Livestock": "lightblue",
            "Export Feed": "green",
            "Export Food": "red",
        }

        color = {
            # Local – bleus
            "Local Food": "#1f77b4",
            "Local Feed": "#5fa2ce",
            # Import – violets
            "Import Food": "#9467bd",
            "Import Feed": "#b799d3",
            "Import Livestock": "#d4c2e5",
            # Export – rouges / corail
            "Export Food": "#d62728",
            "Export Feed": "#ff796c",
            "Export Livestock": "#ffb1a8",
        }

        net_curve_color = "#c48b00"  # très lisible sur fond noir

        # Séparer les catégories
        import_categories = ["Import Food", "Import Feed", "Import Livestock", "Local Food", "Local Feed"]
        export_categories = ["Export Food", "Export Feed", "Export Livestock"]

        for name in import_categories:
            fig.add_trace(
                go.Scatter(
                    x=all_years,
                    y=df.loc[name],  # pas de cumul
                    mode="none",  # juste l'aire
                    stackgroup="p",  # pile positive
                    name=name,
                    fillcolor=color[name],
                    customdata=(df.loc[name] / 1e6).values.reshape(-1, 1),
                    hovertemplate=f"<b>{name}</b><br>Year %{{x}}<br>%{{customdata[0]:.2f}} M ha<extra></extra>",
                )
            )

        # -------- EXPORTS (négatifs) -------------------------------
        for name in export_categories:
            fig.add_trace(
                go.Scatter(
                    x=all_years,
                    y=df.loc[name],  # on passe en négatif
                    mode="none",
                    stackgroup="n",  # pile négative
                    name=name,
                    fillcolor=color[name],
                    customdata=(-df.loc[name] / 1e6).values.reshape(-1, 1),
                    hovertemplate=f"<b>{name}</b><br>Year %{{x}}<br>%{{customdata[0]:.2f}} M ha<extra></extra>",
                )
            )

        # Calculer le total importé - exporté
        df_total_import = df.loc[["Import Food", "Import Feed", "Import Livestock"]].sum(axis=0)
        df_total_export = df.loc[export_categories].sum(axis=0)
        df_net_import_export = df_total_import + df_total_export

        # Ajouter la ligne total
        fig.add_trace(
            go.Scatter(
                x=all_years,
                y=df_net_import_export,
                mode="lines+markers",
                line=dict(
                    color=net_curve_color, width=4, dash="dash"
                ),  # line=dict(color="Black", width=4, dash="dash"),
                name="Net Land Import",
                hovertemplate="Year: %{x}<br>Value: %{customdata:.2f} Mha<extra></extra>",
                customdata=df_net_import_export.values.reshape(-1, 1)
                / 1e6,  # Utiliser les valeurs non cumulées pour le hover, divisées par 1e6
            )
        )

        # Mise à jour du layout
        fig.update_layout(
            title=f"Environmental Footprint - {region}",
            xaxis_title="Year",
            yaxis_title="ha",
            # hovermode="closest",
            showlegend=True,
            hovermode="x unified",
        )

    # -----------------------------------------------------------------
    # 3) Affichage
    # -----------------------------------------------------------------
    st.plotly_chart(fig, use_container_width=True)


with tab6:
    st.title("Historic evolution of agrarian landscape")

    st.text("Discover how agriculture changes during time. Choose a metric and a territory :")

    metric_hist = [
        "Total imported nitrogen",
        "Total net plant import",
        "Total net animal import",
        "Total plant production",
        "Total animal production",
        "Area",
        "Environmental Footprint",
        "Total Fertilization",
        "Relative Fertilization",
        "Primary Nitrogen fertilization use",
        "Emissions",
        "NUE",
        "System NUE",
        "Self-Sufficiency",
        "Livestock density",
        "Cereals production",
        "Leguminous production",
        "Grassland and forage production",
        "Roots production",
        "Oleaginous production",
        "Fruits and vegetables production",
        "Relative Cereals production",
        "Relative Leguminous production",
        "Relative Grassland and forage production",
        "Relative Roots production",
        "Relative Oleaginous production",
        "Relative Fruits and vegetables production",
        # "Effective number of nodes",
        # "Effective connectivity",
        # "Effective number of links",
        # "Effective number of role",
    ]

    st.session_state.metric_hist = st.selectbox("Select a metric", metric_hist, index=0, key="hist_metric_selection")

    m_hist = create_map()
    map_data_hist = st_folium(m_hist, height=600, use_container_width=True, key="hist_map")

    # 🔹 Mettre à jour `st.session_state.selected_region` avec la sélection utilisateur
    if map_data_hist and "last_active_drawing" in map_data_hist:
        last_drawing = map_data_hist["last_active_drawing"]
        if last_drawing and "properties" in last_drawing and "nom" in last_drawing["properties"]:
            st.session_state.selected_region_hist = last_drawing["properties"]["nom"]

    # ✅ Affichage des sélections (se met à jour dynamiquement)
    if "selected_region_hist" not in st.session_state:
        st.session_state.selected_region_hist = None
    if st.session_state.selected_region_hist:
        st.write(f"✅ Région sélectionnée : {st.session_state.selected_region_hist}")
    else:
        st.warning("⚠️ Veuillez sélectionner une région")

    if st.button("Run", key="map_button_hist"):
        with st.spinner("🚀 Running models and calculating metrics..."):
            # 📌 Exécuter les modèles et récupérer les métriques
            models = run_models_for_all_years(st.session_state.selected_region_hist, data)
            if st.session_state.metric_hist not in [
                "Area",
                "Emissions",
                "Relative Fertilization",
                "Total Fertilization",
                "Total plant production",
                "Environmental Footprint",
            ]:
                plot_standard_graph(models, st.session_state.metric_hist, st.session_state.selected_region_hist)
            else:
                stacked_area_chart(models, st.session_state.metric_hist, st.session_state.selected_region_hist)

    with tab7:
        st.title("Scenario generator")

        st.text(
            "Welcome to the prospective mode. Here you can imagine the future of agriculture according to your vision."
        )

        st.subheader("How to proceed ?")

        st.text(
            "You have to fill a scenario excel. Several tokens are needed to run the model in prospective mode. They are splitted in 3 tabs :"
        )
        st.markdown(
            "- Main: In this tab, you have to fill the main characteristics of the future of your territory : population, access to international trade, access to industrial input..."
        )
        st.markdown(
            "- Area: In this tab, you have to distribute the total agricultural area between crops and check the parameters of the production function. The production function gives the yield (Y) in function of the fertilisation amount (F). There is one set of parameters by crop type. The parameters of this function depend of the production function chosen :"
        )
        # st.markdown("The yield is computed as $Y = Y_{\\text{max}} \\cdot (1 - e^{-F/k})$")
        st.markdown(" Ratio: $Y(F) = \\frac{Y_{max}F}{Y_{max}+F}$")
        st.markdown(" Linear: $Y(F) = min(a*F, b)$")
        st.markdown(" Exponential: $Y(F) = Y_{max}(1-e^{F/F^*})$")

        st.markdown(
            "- Technical: This tab encompass all technical coefficient (excretion per LU, weight of the optimization model, time spend by livestock in crops). It reflects potential technical evolution in agriculture and physical constraints."
        )

        #     st.subheader("Scenario file")

        #     st.markdown(
        #         "Here you can find a blank scenario sheet. Please fill all items in main and technical tabs. Fill as many lines in area as you need crops. Make sure the sum of proportion column is 1 and for each crop $Y_{max}$<$k$."
        #     )

        #     # Absolute path to the folder where the script is located
        #     base_path = os.path.dirname(os.path.abspath(__file__))

        #     # Join with your relative file
        #     file_path = os.path.join(base_path, "data", "scenario.xlsx")
        #     # Read the binary content of the file
        #     with open(file_path, "rb") as file:
        #         file_bytes = file.read()

        #     # Create the download button
        #     st.download_button(
        #         label="📥 Download blank Scenario Excel",
        #         data=file_bytes,
        #         file_name="scenario.xlsx",
        #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        #     )

        #     st.markdown("Once you have your scenario ready, go to Prospective mode tab.")

        #     st.subheader("Hard to find these numbers ?")
        #     st.text(
        #         "It might be hard to create from scratch a physical functioning agro-system. To oversome this difficulty, you can use the scenario generator. To do so, choose a territory, a futur year. The scenario generator will automatically create a 'Business as usual' scenario. This scenario is created based on historical trajectory of the territory."
        #     )

        #     st.title("Scenario Generator")

        # 🟢 Sélection de l'année
        st.session_state.pros_year = st.selectbox(
            "Select a year", [str(y) for y in range(2025, 2061)], index=0, key="year_pros_selection"
        )
        # 🟢 Sélection de la fonction de production
        st.session_state.pros_func = st.selectbox(
            "Select Production function", ["Ratio", "Linear", "Exponential"], index=0, key="func_pros_selection"
        )
        st.text_input("Scenario name", key="scenario_name_input")

        m_hist = create_map()
        map_data_pros = st_folium(m_hist, height=600, use_container_width=True, key="pros_map")

        # 🔹 Mettre à jour `st.session_state.selected_region` avec la sélection utilisateur
        if map_data_pros and "last_active_drawing" in map_data_pros:
            last_drawing = map_data_pros["last_active_drawing"]
            if last_drawing and "properties" in last_drawing and "nom" in last_drawing["properties"]:
                st.session_state.selected_region_pros = last_drawing["properties"]["nom"]

        # ✅ Affichage des sélections (se met à jour dynamiquement)
        if "selected_region_pros" not in st.session_state:
            st.session_state.selected_region_pros = None
        if st.session_state.selected_region_pros:
            st.write(f"✅ Région sélectionnée : {st.session_state.selected_region_pros}")
        else:
            st.warning("⚠️ Veuillez sélectionner une région")

        def generate_scenario(year, region, name, func):
            scenar.generate_scenario_excel(year, region, name, func)

        if st.button("Create business as usual scenario", key="map_button_scenario"):
            with tempfile.TemporaryDirectory() as temp_dir:
                with st.spinner("Generating the business as usual scenario..."):
                    st.session_state.name = st.session_state.scenario_name_input
                    scenar = scenario(temp_dir)
                    # threading.Thread(
                    #     target=generate_scenario,
                    #     args=(
                    #         st.session_state.pros_year,
                    #         st.session_state.selected_region_pros,
                    #         st.session_state.name,
                    #         st.session_state.pros_func,
                    #     ),
                    #     daemon=True,
                    # ).start()
                    generate_scenario(
                        st.session_state.pros_year,
                        st.session_state.selected_region_pros,
                        st.session_state.name,
                        st.session_state.pros_func,
                    )

                    with open(os.path.join(temp_dir, st.session_state.name + ".xlsx"), "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label="📥 Download scenario sheet",
                        data=file_data,
                        file_name=st.session_state.name + ".xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                    st.markdown("Once you have your scenario ready, go to Prospective mode tab.")

    with tab8:
        st.title("Prospective mode")

        st.subheader("How to use GRAFS-E Prospective Mode")

        st.markdown(
            """To run the prospective mode, two options are available:\\
        1. **Upload an Excel sheet scenario** then click *Run scenario*  
        2. **Upload a model \*.pkl** saved from a previous session"""
        )

        # ─────────── A • Excel scenario
        st.subheader("① Excel scenario → Run")

        def excel_uploaded():
            # if st.session_state.excel_uploaded_done:
            #     return
            up = st.session_state["xlsx_up"]
            if not up:
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(up.read())
                st.session_state.prep_xlsx_path = tmp.name

            df = pd.read_excel(st.session_state.prep_xlsx_path)
            st.session_state.prep_name = df.iloc[14, 1]
            st.session_state.prep_region = df.iloc[15, 1]
            st.session_state.prep_year = int(df.iloc[16, 1])
            st.session_state.prep_func = df.iloc[17, 1]

        st.file_uploader("📂 Upload .xlsx", type=["xlsx"], key="xlsx_up", on_change=excel_uploaded)

        if st.session_state.prep_name:
            st.info(
                f"Scenario: **{st.session_state.prep_name}**  "
                f"({st.session_state.prep_region}, {st.session_state.prep_year}, {st.session_state.prep_func})"
            )
            if st.button("🚀 Run scenario"):
                with st.spinner("Running prospective model…"):
                    model = NitrogenFlowModel_prospect(st.session_state.prep_xlsx_path)
                    # remplir les variables finales
                    st.session_state.model = model
                    st.session_state.orig = model.adjacency_matrix.copy()
                    st.session_state.name = st.session_state.prep_name
                    st.session_state.year_pros = st.session_state.prep_year
                    st.session_state.year = st.session_state.year_pros
                    st.session_state.selected_region_pros = st.session_state.prep_region
                    st.session_state.selected_region = st.session_state.selected_region_pros
                    st.session_state.prod_func = st.session_state.prep_func
                    st.session_state.heatmap_fig_pros = generate_heatmap(model, model.year, model.region)
                    st.session_state.excel_uploaded_done = True
                    buf = io.BytesIO()
                    pickle.dump(model, buf)
                    buf.seek(0)
                    st.session_state.pkl_blob = buf.getvalue()
                st.success("Model generated!")
                st.rerun()
                # on peut laisser Streamlit relancer automatiquement (pas de st.rerun)

        # ─────────── B • Load .pkl
        st.subheader("② Load existing model (.pkl)")

        def load_pkl():
            up = st.session_state["pkl_up"]
            if not up:
                return
            try:
                obj = pickle.load(io.BytesIO(up.getvalue()))
                if not isinstance(obj, NitrogenFlowModel_prospect):
                    st.session_state.load_error = "⛔️ Wrong file."
                    return
                st.session_state.model = obj
                st.session_state.name = os.path.splitext(up.name)[0]
                st.session_state.year_pros = obj.year
                st.session_state.year = obj.year
                st.session_state.selected_region_pros = obj.region
                st.session_state.selected_region = obj.region
                st.session_state.prod_func = obj.prod_func
                st.session_state.heatmap_fig_pros = generate_heatmap(obj, obj.year, obj.region)
                st.session_state.pkl_blob = up.getvalue()
                st.session_state.load_error = ""
            except Exception as e:
                st.session_state.load_error = str(e)

        st.file_uploader("📥 Upload .pkl", type=["pkl", "pickle", "joblib"], key="pkl_up", on_change=load_pkl)

        if st.session_state.load_error:
            st.error(st.session_state.load_error)

        # ─────────── C • Résultats
        st.subheader("Active model")
        if st.session_state.model is None:
            st.warning("No model loaded or generated yet.")
        else:
            st.success("Model ready")
            st.markdown(
                f"📘 Name : **{st.session_state.name}**  \n"
                f"🗺️ Region : **{st.session_state.selected_region_pros}**  \n"
                f"📆 Year : **{st.session_state.year_pros}**  \n"
                f"⚙️ Prod. function : **{st.session_state.prod_func}**"
            )

            if st.session_state.pkl_blob:
                st.download_button(
                    "💾 Download this model",
                    data=st.session_state.pkl_blob,
                    file_name=f"{st.session_state.name}.pkl",
                    mime="application/octet-stream",
                )

            if st.button("🔄 Reset model"):
                for k in [
                    "model",
                    "name",
                    "year_pros",
                    "selected_region_pros",
                    "prod_func",
                    "heatmap_fig_pros",
                    "pkl_blob",
                ]:
                    st.session_state[k] = None
                st.session_state.excel_uploaded_done = False
                st.rerun()

            if st.session_state.heatmap_fig_pros:
                if st.session_state.model:
                    SA = st.toggle("See Solver Analytics", value=False, key="SA")
                    if SA:
                        st.subheader("Solver Analytics")
                        # ----------------------------
                        # 1. Préparer les données
                        # ----------------------------

                        final = st.session_state.model.log[-1]

                        weights = {
                            "diet_dev": st.session_state.model.w_diet,
                            "fert_dev": st.session_state.model.w_Nsyn,
                            "imp_dev": st.session_state.model.w_imp,
                            "exp_dev": st.session_state.model.w_exp,
                            "energy_dev": st.session_state.model.w_energy,
                            "energy_prod": st.session_state.model.w_share_energy,
                        }

                        # --- table contraintes "cible vs réalisé" ---
                        rows = [
                            ("Synthetic N (crops)", final["Nsyn crop target"], final["Nsyn crop model"]),
                            ("Synthetic N (grass)", final["Nsyn grasslands target"], final["Nsyn grasslands model"]),
                            ("Imports", final["import target"], final["import model"]),
                            ("Exports", final["export target"], final["export model"]),
                            ("Energy Production", final["methanisation target"], final["methanisation model"]),
                        ]
                        df = pd.DataFrame(rows, columns=["Constraint", "Target", "Model"])

                        EPS = 1  # ← tolérance (ktN/an)
                        # CAP = 400  # ← % maxi affiché

                        df["%"] = 100 * (df["Model"] + EPS) / (df["Target"] + EPS)
                        # df["%"] = df["%"].clip(upper=CAP)  # on limite à 400 %

                        max_pct = df["%"].max()
                        CAP = 400 if 1.1 * max_pct >= 400 else 1.1 * max_pct

                        # --- radar normalisé -------------------------------------------------
                        theta = df["Constraint"].tolist()
                        theta_closed = theta + [theta[0]]  # boucle fermée

                        r_target = [100] * len(theta) + [100]
                        r_model = df["%"].tolist() + [df["%"].iloc[0]]

                        # 1) ligne 100 %
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatterpolar(
                                r=r_target,
                                theta=theta_closed,
                                mode="lines",
                                line=dict(color="grey", dash="dot"),
                                name="Target (100 %)",
                            )
                        )

                        # 2) surface modèle
                        fig.add_trace(
                            go.Scatterpolar(
                                r=r_model,
                                theta=theta_closed,
                                fill="toself",
                                line=dict(color="dodgerblue"),
                                name="Model",
                            )
                        )

                        # 3) mise à l’échelle radiale
                        max_pct = df["%"].max()
                        CAP = 400 if 1.1 * max_pct >= 400 else 1.1 * max_pct  # 400 % ou 1.1×max
                        ticks = [0, 50, 100, 200, 400] if CAP >= 400 else [0, 50, 100, 200]

                        fig.update_layout(
                            title="Model vs Target (normalised)",
                            polar=dict(
                                radialaxis=dict(
                                    range=[0, CAP],
                                    tickvals=[v for v in ticks if v <= CAP],
                                    ticksuffix="%",
                                    color="black",
                                )
                            ),
                            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                            height=450,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        def color_diff(model, target, constraint):
                            if target == 0:
                                return "" if model == 0 else "background-color:red"  # rouge si mod>0 alors que cible 0
                            ratio = model / target
                            diff = abs(model - target)
                            if (
                                0.9 <= ratio <= 1.1
                                or diff < 5
                                or (
                                    ratio <= 1
                                    and constraint in ["Synthetic N (crops)", "Synthetic N (grass)", "Imports"]
                                )
                            ):
                                return "background-color:green"  # vert (±10 %)
                            elif ratio > 1 and constraint in ["Exports"]:
                                return "background-color:green"
                            elif 0.5 <= ratio <= 2 or diff < 10:
                                return "background-color:orange"  # orange
                            return "background-color:red"  # rouge

                        styled = df.style.apply(
                            lambda row: [
                                color_diff(row.Model, row.Target, row.Constraint) if col == "Model" else ""
                                for col in row.index
                            ],
                            axis=1,
                        ).format({"Target": "{:.2f}", "Model": "{:.2f}"})

                        st.dataframe(styled)

                        objective = final["objective"]
                        good = 1.5  # à adapter : zone « OK »
                        warn = 2.5  # au-delà = rouge
                        end = 4

                        fig_g = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=objective
                                / sum(
                                    list(weights.values())
                                ),  # On s'intéresse à une moyenne pondérée des termes de la fonctions objectif
                                title={"text": "Objective function"},
                                gauge={
                                    "axis": {"range": [0, end]},
                                    "steps": [
                                        {"range": [0, good], "color": "#c9f7d4"},
                                        {"range": [good, warn], "color": "#fff3b0"},
                                        {"range": [warn, end], "color": "#f9c0c0"},
                                    ],
                                },
                            )
                        )
                        st.plotly_chart(fig_g, use_container_width=True)

                    st.text(
                        f"Total Throughflow : {np.round(st.session_state.model.get_transition_matrix().sum(), 1)} ktN/yr."
                    )
                st.subheader(f"Heatmap – {st.session_state.selected_region_pros} in {st.session_state.year_pros}")
                st.plotly_chart(st.session_state.heatmap_fig_pros, use_container_width=True)
