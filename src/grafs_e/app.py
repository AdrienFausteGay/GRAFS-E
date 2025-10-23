import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]  # .../repo/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import copy
import io
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
from plotly.colors import qualitative as qcolors

from grafs_e.N_class import DataLoader, NitrogenFlowModel
from grafs_e.sankey import (
    streamlit_sankey,
    streamlit_sankey_fertilization,
    streamlit_sankey_food_flows,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

icon_path = os.path.join(ROOT_DIR, "docs", "source", "_static", "logo.png")

# Déterminer le chemin du fichier de config Streamlit
config_dir = os.path.expanduser("~/.streamlit")
config_path = os.path.join(config_dir, "config.toml")

# Vérifier si le dossier ~/.streamlit existe, sinon le créer
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

# Écrire ou modifier le fichier config.toml pour imposer le dark mode
with open(config_path, "w") as config_file:
    config_file.write("[theme]\nbase='dark'\n")

st.set_page_config(
    page_title="GRAFS-E App", page_icon=icon_path, layout="wide"
)  # , layout="wide")

# data = st.session_state["data"]
# st.session_state.available_years = data.years
# regions = data.regions
# crops = data.crops
# meta = data.metadata
# labels = data.labels
# label_to_index = data.label_to_index

# Initialisation de l'état des variables
for k, v in {
    "project": None,
    "data": None,
    "dataloader": None,
    "model": None,
    "available_years": None,
    "available_regions": None,
    "name": None,
    "year": None,
    "year_run": None,
    "selected_region": None,
    "region_run": None,
}.items():
    st.session_state.setdefault(k, v)
# %%
# Initialisation de l'interface Streamlit
st.title("GRAFS-E")
__version__ = "1.0.0"  # version("grafs_e")
st.write(f"📦 GRAFS-E version: {__version__}")
st.text("Contact: Adrien Fauste-Gay, adrien.fauste-gay@univ-grenoble-alpes.fr")
st.title("Nitrogen Flow Simulation Model: A Territorial Ecology Approach")

# # 🔹 Initialiser les valeurs dans session_state si elles ne sont pas encore définies
# if "selected_region" not in st.session_state:
#     st.session_state.region = None
# if "year" not in st.session_state:
#     st.session_state.year = None

# if st.session_state.data and st.session_state.project:
#     st.write(f"✅ Project file: {st.session_state.project}")
#     st.write(f"✅ Data file: {st.session_state.data}")
#     if st.session_state.region:
#         st.write(f"✅ Selected territory: {st.session_state.region}")
#     else:
#         st.warning("⚠️ Please select a region")

#     if st.session_state.year:
#         st.write(f"✅ Selected year : {st.session_state.year}")
#     else:
#         st.warning("⚠️ Please select a year")
# else:
#     st.warning("⚠️ Please upload a project file and a dataset")

# -- Sélection des onglets --
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Documentation",
        "Data uploading",
        "Run GRAFS-E",
        "Sankey",
        "Detailed data",
        "Historic Evolution",
    ]
)

with tab1:
    st.title("Documentation")

    st.header(
        "GRAFS-Extended: Comprehensive Analysis of Nitrogen Flux in Agricultural Systems"
    )

    st.subheader("Overview")

    st.write(
        """
    <p style='text-align: justify'>
        The GRAFS-E model (General Representation of Agro-Food Systems Extended) serves as an advanced tool designed to analyze and map the evolution of nitrogen utilization within agricultural systems. This model builds upon the GRAFS framework developed at IEES (Jussieu, Paris) provides a detailed analysis of nitrogen flows in agriculture. The model enables researchers to construct robust prospective scenarios and examine the global structure of nitrogen flows in agricultural ecosystems. Full documentation can be accessed here (access must be given by creator, see contact):
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Create a link button to the documentation
    st.link_button(
        "📖 Full Documentation",
        "https://grafs-e-f43b79.gricad-pages.univ-grenoble-alpes.fr/index.html",
        help="Opens GRAFS-E documentation in a new tab.",
    )

    st.subheader("Features")

    st.text(
        "First upload your project and dat files in the 'Upload data' tab. these files must be formated as described in the documentation."
    )
    st.text(
        "Go to 'Run' tab to select a year and region to run GRAFS-E. This will display the nitrogen transition matrix for this territory"
    )
    st.text(
        "The 'Sankey' tab offerts first visualization tools to analyse direct input and output flows for each object."
    )

    st.text("Use 'Time evolution' tab to display evolution of basic indexes.")

    st.subheader("Results")
    st.text(
        "The model generates extensive transition matrices representing nitrogen flows between different agricultural compartments."
    )
    st.text(
        "These matrices can be used for several analysis as network analysis, input-output analysis, environmental footprint analysis and so on."
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
    st.title("Data Uploading")
    st.write(
        "Please, provide project and data excel files. Ensure that the data are correctly formatted using the input page of the documentation."
    )

    if "files_loaded" not in st.session_state:
        st.session_state.files_loaded = False

    def load_files():
        """Charge les fichiers, initialise le DataLoader et définit le marqueur de succès."""

        project_path = st.session_state.get("project_path")
        data_path = st.session_state.get("data_path")

        # NOUVEAU : Récupération des noms originaux des fichiers
        project_name = st.session_state.get("project_name")
        data_name = st.session_state.get("data_name")

        if project_path and data_path:
            try:
                # Initialisation du DataLoader
                st.session_state.dataloader = DataLoader(project_path, data_path)
                st.session_state.available_years = (
                    st.session_state.dataloader.available_years
                )
                st.session_state.available_regions = (
                    st.session_state.dataloader.available_regions
                )

                # Définition du marqueur de succès
                st.session_state.files_loaded = True

                # Stockage du message de succès formaté pour l'affichage
                st.session_state.success_message = f"Fichiers chargés! Projet: **{project_name}**, Données: **{data_name}**"

            except Exception as e:
                st.session_state.files_loaded = False
                st.session_state.load_error = f"Erreur de chargement du DataLoader: {e}"

    def _handle_upload(key_uploader, key_path, key_name):
        """
        Gère le téléchargement, sauvegarde le fichier temporaire, et enregistre
        le chemin TEMPORAIRE et le NOM ORIGINAL dans st.session_state.
        """
        uploaded_file = st.session_state[key_uploader]

        if uploaded_file is None:
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_file.read())

            # 1. Stocke le chemin d'accès au fichier temporaire
            st.session_state[key_path] = tmp.name

            # 2. NOUVEAU : Stocke le nom ORIGINAL du fichier téléchargé
            st.session_state[key_name] = uploaded_file.name

        # Tente de charger les fichiers
        load_files()

    # --- Mise à Jour des Widgets Streamlit ---

    # 1. Uploader du fichier Projet
    st.file_uploader(
        "📂 Upload Project file",
        type=["xlsx"],
        key="project_uploader",
        # Ajout du paramètre pour stocker le nom original
        on_change=lambda: _handle_upload(
            "project_uploader", "project_path", "project_name"
        ),
    )

    # 2. Uploader du fichier Données
    st.file_uploader(
        "📂 Upload Data file",
        type=["xlsx"],
        key="data_uploader",
        # Ajout du paramètre pour stocker le nom original
        on_change=lambda: _handle_upload("data_uploader", "data_path", "data_name"),
    )

    # --- Affichage Conditionnel du Message ---
    if st.session_state.files_loaded:
        # Affiche le message formaté récupéré de st.session_state
        st.success(st.session_state.success_message)
    elif st.session_state.get("load_error"):
        st.error(st.session_state.load_error)

with tab3:
    st.title("Run GRAFS-E")
    if not st.session_state.dataloader:
        st.warning(
            "⚠️ Please upload project and data files first in the 'Data Uploading' tab."
        )
    else:
        st.subheader("Territory and year selection")
        st.write("Please select a year and a territory then click on Run.")

        # 🟢 Sélection de la région
        # st.subheader("Select an Area")
        st.session_state.region_run = st.selectbox(
            "Select an Area", st.session_state.available_regions, index=0
        )

        # 🟢 Sélection de l'année
        # st.subheader("Select a year")
        st.session_state.year_run = st.selectbox(
            "Select a year", st.session_state.available_years, index=0
        )

        # ✅ Affichage des sélections (se met à jour dynamiquement)
        if st.session_state.region_run:
            st.write(f"✅ Selected Area: {st.session_state.region_run}")
        else:
            st.warning("⚠️ Please select an Area")

        if st.session_state.year_run:
            st.write(f"✅ Selected Year: {st.session_state.year_run}")
        else:
            st.warning("⚠️ Please select a Year")

        # 🟢 Fonction pour générer la heatmap et éviter les recalculs inutiles
        @st.cache_data
        def generate_heatmap(_model):
            _model = copy.deepcopy(_model)
            return _model.plot_heatmap_interactive()

        # 🔹 Bouton "Run" avec les valeurs mises à jour
        if st.button("Run"):
            st.session_state.region = st.session_state.region_run
            st.session_state.year = st.session_state.year_run
            if st.session_state.region and st.session_state.year:
                # Initialiser le modèle avec les paramètres
                st.session_state.model = NitrogenFlowModel(
                    data=st.session_state.dataloader,
                    area=st.session_state.region,
                    year=st.session_state.year,
                )

                # ✅ Générer la heatmap et la stocker
                st.session_state.heatmap_fig = generate_heatmap(st.session_state.model)
            else:
                st.warning(
                    "❌ Please select a year and a region before visiting 'Sankey' and 'Detailed data' tabs."
                )

        # 🔹 Indépendance de l'affichage de la heatmap 🔹
        if "heatmap_fig" in st.session_state:
            if st.session_state.model:
                st.text(
                    f"Total Throughflow : {np.round(st.session_state.model.get_transition_matrix().sum(), 1)} ktN/yr."
                )
            st.subheader(
                f"Heatmap of the nitrogen flows for {st.session_state.region} in {st.session_state.year}"
            )
            st.plotly_chart(st.session_state.heatmap_fig, use_container_width=True)

            # Bouton pour télécharger la matrice
            # ───── Création du DataFrame à partir de la matrice ──────────
            matrix = st.session_state.model.get_transition_matrix()
            df_matrix = pd.DataFrame(
                matrix,
                index=st.session_state.model.labels,
                columns=st.session_state.model.labels,
            )

            @st.cache_data
            def convert_df_to_excel(df):
                """Convertit le DataFrame en bytes au format Excel (xlsx)."""
                # Crée un buffer de mémoire pour stocker les données Excel
                output = io.BytesIO()

                # Écrit le DataFrame dans le buffer en tant que fichier Excel
                # index=True inclut l'index du DataFrame dans le fichier
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=True, sheet_name="Matrice")

                # Récupère les bytes du buffer
                return output.getvalue()

            # Génère les bytes du fichier Excel à partir du DataFrame
            excel_bytes = convert_df_to_excel(df_matrix)

            # ───── Bouton de téléchargement ─────────────────────────────
            st.download_button(
                label="📥 Download matrix (xlsx)",
                data=excel_bytes,
                file_name=f"transition_matrix_{st.session_state.region}_{st.session_state.year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="hist_xlsx",  # Il est préférable de changer la clé
            )

with tab4:
    st.title("Sankey")

    # Vérifier si le modèle a été exécuté
    if not st.session_state.model:
        st.warning("⚠️ Please run the model first in the 'Run GRAFS-E' tab.")
    else:
        model = st.session_state["model"]

        # -- UI : mode d'affichage et profondeur --
        mode_complet = st.toggle(
            "Detailed view", value=False, key="first"
        )  # True = sans merge
        scope = 1

        # -- Préparer les merges si on est en mode simplifié --
        def build_simplified_merges_with_attachments(model) -> dict:
            """
            Construit un dict 'merges' :
            - crops groupés par Category
            - products attachés au groupe du crop d'origine (Origin compartment)
            - livestock groupés en 'Livestock'
            - excretions attachées au groupe Livestock (ou au groupe contenant l'élevage d'origine)
            - ajoute Trade/Industry/Environment/Population comme avant, si présents
            """
            merges: dict[str, list[str]] = {}

            # 1) Crops par Category
            if hasattr(model, "df_cultures") and not model.df_cultures.empty:
                for cat, idxs in model.df_cultures.groupby("Category").groups.items():
                    merges[cat] = list(idxs)  # labels des crops de cette catégorie
            else:
                merges = {}

            # 2) Groupe Livestock
            if hasattr(model, "df_elevage") and not model.df_elevage.empty:
                merges["Livestock"] = model.df_elevage.index.tolist()

            # 3) Population (optionnel mais courant)
            if hasattr(model, "df_pop") and not model.df_pop.empty:
                merges["Population"] = model.df_pop.index.tolist()

            # 4) Trade (tous les labels contenant 'trade')
            trade_labels = [lbl for lbl in model.labels if "trade" in lbl.lower()]
            if trade_labels:
                merges["Trade"] = trade_labels

            # 5) Industry (si présents)
            industry_candidates = [
                lbl for lbl in ["Haber-Bosch", "other sectors"] if lbl in model.labels
            ]
            if industry_candidates:
                merges["Industry"] = industry_candidates

            # 6) Environment (si présents)
            env_candidates = [
                "atmospheric NH3",
                "atmospheric N2O",
                "atmospheric N2",
                "soil stock",
                "hydro-system",
                "other losses",
            ]
            env_kept = [lbl for lbl in env_candidates if lbl in model.labels]
            if env_kept:
                merges["Environment"] = env_kept

            # ------- Attacher PRODUCTS au groupe du crop d'origine -------
            if hasattr(model, "df_prod") and not model.df_prod.empty:
                for prod_label, row in model.df_prod.iterrows():
                    origin = row.get("Origin compartment")
                    if not isinstance(origin, str):
                        continue

                    # chercher le groupe qui contient 'origin'
                    dest_group = None
                    for gname, members in merges.items():
                        if origin in members:
                            dest_group = gname
                            break
                    # sinon, fallback : fusionner directement avec le crop d'origine
                    if dest_group is None:
                        dest_group = origin
                        merges.setdefault(dest_group, []).append(
                            origin
                        )  # s’assurer que le crop y est

                    merges.setdefault(dest_group, []).append(prod_label)

            # ------- Attacher EXCRETION au groupe contenant l'élevage d'origine -------
            if hasattr(model, "df_excr") and not model.df_excr.empty:
                for ex_label, row in model.df_excr.iterrows():
                    origin = row.get("Origin compartment")
                    if not isinstance(origin, str):
                        continue

                    dest_group = None
                    for gname, members in merges.items():
                        if origin in members:
                            dest_group = gname
                            break
                    # si on n’a pas trouvé, mais on a un groupe Livestock, rattacher là
                    if dest_group is None and "Livestock" in merges:
                        dest_group = "Livestock"
                    if dest_group is None:
                        dest_group = origin
                        merges.setdefault(dest_group, []).append(origin)

                    merges.setdefault(dest_group, []).append(ex_label)

            return merges

        if mode_complet:
            merges = None  # pas de merge (ou tes micro-fusions spécifiques si tu veux)
        else:
            merges = build_simplified_merges_with_attachments(model)

        # appel inchangé : la sélection exclut déjà products/excretions et on affiche
        # les sorties produits/excrétions en aval quand on choisit crop/livestock
        streamlit_sankey(
            model=model,
            do_merge=not mode_complet,
            merges=merges,
            scope=scope,
        )

        st.subheader("Fertilization in the territory")

        mode_complet_ferti = st.toggle("Detailed view", value=False, key="ferti")

        if mode_complet_ferti:
            merges = {
                "Population": model.df_pop.index.tolist(),
                "Livestock": model.df_elevage.index.tolist(),
                "Atmospheric deposition": ["atmospheric NH3", "atmospheric N2O"],
            }
        else:
            merges = build_simplified_merges_with_attachments(model)

        # Seuil central = max/100
        base_threshold = float(np.max(model.get_transition_matrix()) / 100.0)

        # Étendue du slider : /100 → ×100 autour de la base
        min_thr = float(base_threshold / 50.0)
        max_thr = float(base_threshold * 50.0)

        # Slider sur la VALEUR du seuil (pas d'exponentiel visible)
        tre = st.slider(
            "Flow threshold (ktN/yr)",
            min_value=min_thr,
            max_value=max_thr,
            value=float(base_threshold),
            step=(max_thr - min_thr) / 200.0,  # granularité fine ; ajuste si besoin
        )

        # % du flux total représenté par ce seuil
        total_flow = float(np.sum(model.get_transition_matrix()))
        pct_total = (tre / total_flow * 100.0) if total_flow > 0 else 0.0

        st.write(
            f"Threshold = {tre:.2e} ktN/yr  ·  {pct_total:.3f}% of total flow ({total_flow:.2e} ktN/yr)"
        )

        # Utilisation du seuil
        streamlit_sankey_fertilization(model, merges, THRESHOLD=tre)

        st.subheader("Feed for livestock and Food for local population")

        mode_complet_food = st.toggle("Detailed view", value=False, key="food")

        # -- merges
        if mode_complet_food:
            merges = {}  # <- (coquille: avant c'était 'merge = {}')
        else:
            cat_groups = model.df_cultures.groupby("Category").groups
            sub_type_groups = model.df_prod.groupby("Sub Type").groups
            merges = {key: list(indices) for key, indices in cat_groups.items()}

            merges["Livestock"] = model.df_elevage.index.tolist()
            merges["Population"] = model.df_pop.index.tolist()
            merges["Excretion"] = model.df_excr.index.tolist()

            merges.update(
                {key: list(indices) for key, indices in sub_type_groups.items()}
            )

            trade_labels = [
                label
                for label in model.labels
                if isinstance(label, str) and "trade" in label.lower()
            ]
            if trade_labels:
                merges["Trade"] = trade_labels

            # (coquille précédente: la virgule finale faisait un tuple)
            env_candidates = [
                "atmospheric NH3",
                "atmospheric N2O",
                "atmospheric N2",
                "soil stock",
                "hydro-system",
                "other losses",
            ]
            merges["Environment"] = [
                lbl for lbl in env_candidates if lbl in model.labels
            ]

            industry_candidates = [
                lbl for lbl in ["Haber-Bosch", "other sectors"] if lbl in model.labels
            ]
            if industry_candidates:
                merges["Industry"] = industry_candidates

        # -- slider de seuil (valeur directe), centré sur max/100 et borné à /100 ↔ ×100
        base_threshold = float(np.max(model.get_transition_matrix()) / 1000.0)
        min_thr = float(base_threshold / 100.0)
        max_thr = float(base_threshold * 100.0)

        tre = st.slider(
            "Flow threshold (ktN/yr)",
            min_value=min_thr,
            max_value=max_thr,
            value=float(base_threshold),
            step=(max_thr - min_thr) / 200.0,  # ajuste la granularité si besoin
        )

        # % du flux total
        total_flow = float(np.sum(model.get_transition_matrix()))
        pct_total = (tre / total_flow * 100.0) if total_flow > 0 else 0.0

        st.write(
            f"Threshold = {tre:.2e} ktN/yr · {pct_total:.3f}% of total flow ({total_flow:.2e} ktN/yr)"
        )

        # -- appel
        streamlit_sankey_food_flows(model, merges=merges, THRESHOLD=tre)

with tab5:
    st.title("Detailed data")

    if not st.session_state.model:
        st.warning(
            "⚠️ Please run the model first in the 'Run' tab or in the 'Prospective mode' tab."
        )
    else:
        st.text(
            "This tab is to access to detailed data used in input but also processed by the model"
        )

        st.subheader("Cultures data")

        st.dataframe(model.df_cultures_display)

        st.subheader("Livestock data")

        st.dataframe(model.df_elevage_display)

        st.subheader("Excretion data")

        st.dataframe(model.df_excr_display)

        st.subheader("Products data")

        st.dataframe(model.df_prod_display)

        st.subheader("Products allocation to livestock and population")

        st.dataframe(model.allocations_df)

        st.subheader("Diet deviations from defined diet")

        st.dataframe(model.deviations_df)

        st.subheader("Population data")

        st.dataframe(model.df_pop)

        st.subheader("Energy data")

        st.dataframe(model.methanizer_overview_df)

        # --- Boutons de téléchargement ---
        st.markdown("### Download")

        # 1) Excel multi-feuilles
        def _clean_sheet_name(name: str, used: set) -> str:
            # Excel : max 31 chars, pas : \ / ? * [ ]
            forbidden = {":", "\\", "/", "?", "*", "[", "]"}
            cleaned = (
                "".join("_" if ch in forbidden else ch for ch in str(name))[:31]
                or "Sheet"
            )
            # assure l'unicité si doublon
            base, suffix, i = cleaned, "", 1
            while cleaned in used:
                suffix = f"_{i}"
                cleaned = base[: 31 - len(suffix)] + suffix
                i += 1
            used.add(cleaned)
            return cleaned

        dfs = {
            "Cultures": model.df_cultures_display,
            "Livestock": model.df_elevage_display,
            "Excretion": model.df_excr,
            "Products": model.df_prod_display,
            "Allocations": model.allocations_df,
            "Diet deviations": model.deviations_df,
            "Population": model.df_pop,
            "Energy": model.methanizer_overview_df,
        }

        buffer = BytesIO()
        used_names = set()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, df in dfs.items():
                sheet = _clean_sheet_name(name, used_names)
                # index=True pour garder les index sémantiques
                df.to_excel(writer, sheet_name=sheet, index=True)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download all (Excel, multi-sheet)",
            data=buffer,
            file_name="model_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # 2) (Optionnel) Boutons individuels en CSV
        with st.expander("Download individual CSV files"):
            for name, df in dfs.items():
                csv_bytes = df.to_csv(index=True).encode("utf-8")
                st.download_button(
                    label=f"⬇️ {name}.csv",
                    data=csv_bytes,
                    file_name=f"{name.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"dl_{name.lower().replace(' ', '_')}",
                )


# =========================================================
# 1) Modèles par année
# =========================================================
@st.cache_resource(show_spinner="Running GRAFS-E over all territories...")
def run_models_for_all_years(region, _data_loader):
    models = {}
    for year in st.session_state.available_years:
        if str(year) == "1852" and region == "Savoie":
            continue
        models[str(year)] = NitrogenFlowModel(data=_data_loader, area=region, year=year)
    return models


# =========================================================
# 2) Définition des métriques disponibles
#    (uniquement celles effectivement implémentées)
# =========================================================
SUPPORTED_METRICS = [
    # scalaires
    "Total imported nitrogen",
    "Total net plant import",
    "Total net animal import",
    "Total plant production",
    "Total animal production",
    "Livestock density",
    # séries (stacked)
    "Area",
    "Relative Area",
    "Emissions",
    "Environmental Footprint",
    # cultures (déjà ajouté précédemment)
    "Crop production by category",
    "Relative crop production by category",
    # >>> fertilisation par source
    "Fertilization by source",
    "Relative fertilization by source",
]


# pour les métriques par catégorie, on déduit la catégorie depuis le libellé
_CATEGORY_NAMES = {
    "Cereals",
    "Leguminous",
    "Oleaginous",
    "Grassland and forage",
    "Roots",
    "Fruits and vegetables",
}


def _crop_production_series(_model, relative: bool) -> pd.Series:
    """Retourne une Series indexée par catégorie de culture."""
    if _model.df_cultures.empty:
        return pd.Series(dtype=float)

    s = (
        _model.df_cultures.groupby("Category")["Total Nitrogen Production (ktN)"]
        .sum()
        .sort_values(ascending=False)
    )
    if relative:
        total = float(_model.total_plant_production()) or 1.0
        s = s.astype(float) * 100.0 / total
    return s


def _fert_series(_model, relative: bool) -> pd.Series:
    """
    Utilise model.tot_fert() pour retourner une Series indexée par source.
    relative=True -> % du total annuel.
    """
    s = _model.tot_fert()  # Series (ktN) index = sources (Haber-Bosch, Seeds, ...)
    if s is None or s.empty:
        return pd.Series(dtype=float)
    s = s.astype(float).sort_values(ascending=False)
    if relative:
        total = float(s.sum()) or 1.0
        s = s * 100.0 / total
    return s


def _area_by_category_series(_model, relative: bool = False) -> pd.Series:
    """
    Retourne la surface des cultures agrégée par Category.
    Utilise _model.surfaces() (index=cultures) + df_cultures['Category'] pour agréger.
    relative=True -> parts en % du total annuel.
    """
    s = _model.surfaces()  # Series: index=cultures, values=ha
    if s is None or s.empty:
        return pd.Series(dtype=float)

    # Associer chaque culture à sa catégorie puis sommer
    df = pd.DataFrame({"area": s.astype(float)})
    # On aligne sur l'index de s pour éviter les KeyError
    df["Category"] = _model.df_cultures.loc[df.index, "Category"].values
    out = df.groupby("Category", sort=False)["area"].sum().sort_values(ascending=False)

    if relative:
        total = float(out.sum()) or 1.0
        out = out * 100.0 / total
    return out


# =========================================================
# 3) Récupération de métrique (1 modèle -> 1 valeur ou 1 série)
# =========================================================
def compute_metric_for_model(_model, metric_name: str):
    # --- nouvelles métriques fertilisation ---
    if metric_name == "Fertilization by source":
        return _fert_series(_model, relative=False)
    if metric_name == "Relative fertilization by source":
        return _fert_series(_model, relative=True)

    # --- déjà présent : production par catégorie ---
    if metric_name == "Crop production by category":
        return _crop_production_series(_model, relative=False)
    if metric_name == "Relative crop production by category":
        return _crop_production_series(_model, relative=True)

    # --- scalaires simples ---
    if metric_name == "Total imported nitrogen":
        return float(_model.imported_nitrogen())
    if metric_name == "Total net plant import":
        return float(_model.net_imported_plant())
    if metric_name == "Total net animal import":
        return float(_model.net_imported_animal())
    if metric_name == "Total plant production":
        return float(_model.total_plant_production())
    if metric_name == "Total animal production":
        return float(_model.animal_production())
    if metric_name == "Livestock density":
        return float(_model.LU_density())

    # --- séries déjà existantes ---
    if metric_name == "Area":
        # désormais: surfaces agrégées par catégorie
        return _area_by_category_series(_model, relative=False)
    if metric_name == "Relative Area":
        return _area_by_category_series(_model, relative=True)
    if metric_name == "Emissions":
        return _model.emissions()  # Series (atmospheric N2O, NH3, N2, ...)
    if metric_name == "Environmental Footprint":
        return _model.env_footprint()  # Series (peut contenir des valeurs <0)

    return None


@st.cache_data
def compute_metrics_over_years(_models, metric_name: str, cache_key: tuple):
    """
    _models: dict {year: model}  (ignored by Streamlit's hasher)
    cache_key: ONLY thing that is hashed (make it stable & small)
    """
    out = {}
    for y, m in _models.items():
        val = compute_metric_for_model(m, metric_name)
        out[int(y)] = val
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


# =========================================================
# 4) Affichages graphiques
# =========================================================
def _theme_line_color():
    base = st.get_option("theme.base")
    return "royalblue" if base == "light" else "white"


def _ylabel_for(metric_name: str):
    if metric_name.startswith("Relative "):
        return "%"
    if "density" in metric_name:
        return "LU/ha"
    if metric_name in ("Emissions",):
        return "ktN/yr"
    if metric_name == "Area":
        return "ha"
    # défaut
    return "ktN/yr"


def plot_scalar_timeseries(series_by_year: dict, title: str, metric_name: str):
    years = list(series_by_year.keys())
    vals = [series_by_year[y] for y in years]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=vals,
            mode="lines+markers",
            name=metric_name,
            line=dict(color=_theme_line_color(), width=2),
            marker=dict(size=8, symbol="circle"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=_ylabel_for(metric_name),
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, tickangle=45)
    fig.update_yaxes(showgrid=True, zeroline=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_stacked_series(
    series_by_year: dict, title: str, metric_name: str, allow_negative=False
):
    """
    series_by_year: {year: pandas.Series} avec index = catégories
    allow_negative: True pour footprints (imports positifs / exports négatifs).
    """
    years = list(series_by_year.keys())
    # union de toutes les catégories
    cats = set()
    for s in series_by_year.values():
        cats.update(list(s.index))
    cats = sorted(cats)

    # palette lisible et répétable
    palette = (
        qcolors.Set3 + qcolors.Set2 + qcolors.Set1 + qcolors.Pastel1 + qcolors.Pastel2
    )

    def color_for(i):
        return palette[i % len(palette)]

    fig = go.Figure()

    if allow_negative:
        # séparer catégories "globalement" positives vs négatives
        sign_by_cat = {}
        for c in cats:
            vals = np.array([float(series_by_year[y].get(c, 0) or 0) for y in years])
            sign_by_cat[c] = vals.mean() >= 0.0

        for i, c in enumerate(cats):
            ys = [float(series_by_year[y].get(c, 0) or 0) for y in years]
            stack = "pos" if sign_by_cat[c] else "neg"
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=ys,
                    mode="none",
                    stackgroup=stack,
                    name=c,
                    fillcolor=color_for(i),
                    hovertemplate=f"<b>{c}</b><br>Year %{{x}}<br>%{{y:.2f}}<extra></extra>",
                )
            )
    else:
        for i, c in enumerate(cats):
            ys = [float(series_by_year[y].get(c, 0) or 0) for y in years]
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=ys,
                    mode="none",
                    stackgroup="one",
                    name=c,
                    fillcolor=color_for(i),
                    hovertemplate=f"<b>{c}</b><br>Year %{{x}}<br>%{{y:.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=_ylabel_for(metric_name),
        template="plotly_white",
        hovermode="x",
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, tickangle=45)
    fig.update_yaxes(showgrid=True, zeroline=True)
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 5) UI - onglet historique
# =========================================================
with tab6:
    st.title("Historic evolution of agrarian landscape")
    st.text(
        "Discover how agriculture changes over time. Choose a metric and a territory:"
    )

    if not st.session_state.dataloader:
        st.warning(
            "⚠️ Please upload project and data files first in the 'Data Uploading' tab."
        )
    else:
        # Sélecteurs
        available_regions = st.session_state.get("available_regions") or [
            st.session_state.get("region", "France")
        ]
        region = st.selectbox(
            "Select an area", available_regions, index=0, key="hist_area_selection"
        )

        metric = st.selectbox(
            "Select a metric", SUPPORTED_METRICS, index=0, key="hist_metric_selection"
        )

        if st.button("Run", key="map_button_hist"):
            with st.spinner("🚀 Running models and calculating metrics..."):
                models = run_models_for_all_years(region, st.session_state.dataloader)

                years_tuple = tuple(sorted(models.keys()))
                metrics_over_years = compute_metrics_over_years(
                    models,
                    metric,
                    cache_key=(region, metric, years_tuple),
                )

                # Scalaires vs séries
                sample_val = next(iter(metrics_over_years.values()))
                title = f"{metric} — {region}"

                if isinstance(sample_val, (int, float, np.floating)):
                    plot_scalar_timeseries(metrics_over_years, title, metric)
                elif isinstance(sample_val, pd.Series):
                    allow_negative = metric == "Environmental Footprint"
                    try:
                        plot_stacked_series(
                            metrics_over_years,
                            title,
                            metric,
                            allow_negative=allow_negative,
                        )
                    except Exception as e:
                        st.error(f"Plot failed for '{metric}': {e}")
                        st.info(
                            "Tip: ensure the model method returns a pandas.Series (index = categories)."
                        )
                else:
                    st.warning("Selected metric returned no data.")


# # 📌 Stocker et récupérer les modèles pour chaque région en cache
# @st.cache_resource(show_spinner="Running GRAFS-E over regions...")
# def run_models_for_all_regions(year, regions, _data_loader):
#     models = {}
#     for region in regions:
#         models[region] = NitrogenFlowModel(
#             data=_data_loader, year=str(year), region=region
#         )
#     return models


# # 📌 Calculer et stocker les métriques pour chaque région en cache
# @st.cache_data(show_spinner="Computing metrics...")
# def get_metrics_for_all_regions(_models, metric_name, year):
#     metric_dict = {
#         "Total imported nitrogen": "imported_nitrogen",
#         "Total net plant import": "net_imported_plant",
#         "Total net animal import": "net_imported_animal",
#         "Total plant production": "total_plant_production",
#         "Environmental Footprint": "net_footprint",
#         "NUE": "NUE",
#         "System NUE": "NUE_system",
#         "Livestock density": "LU_density",
#         "Cereals production": "cereals_production",
#         "Leguminous production": "leguminous_production",
#         "Oleaginous production": "oleaginous_production",
#         "Grassland and forage production": "grassland_and_forages_production",
#         "Roots production": "roots_production",
#         "Fruits and vegetables production": "fruits_and_vegetable_production",
#         "Relative Cereals production": "cereals_production_r",
#         "Relative Leguminous production": "leguminous_production_r",
#         "Relative Oleaginous production": "oleaginous_production_r",
#         "Relative Grassland and forage production": "grassland_and_forages_production_r",
#         "Relative Roots production": "roots_production_r",
#         "Relative Fruits and vegetables production": "fruits_and_vegetable_production_r",
#         "Total animal production": "animal_production",
#         "NH3 volatilization": "NH3_vol",
#         "N2O emission": "N2O_em",
#     }
#     metric_function_name = metric_dict[metric_name]
#     metrics = {}
#     area = {}
#     for region, model in _models.items():
#         metric_function = getattr(model, metric_function_name, None)
#         if callable(metric_function):
#             metrics[region] = metric_function()
#         else:
#             metrics[region] = None  # Si la méthode n'existe pas, on met None
#         area[region] = model.surfaces_tot()
#     return metrics, area


# @st.cache_data
# def get_metric_range(metrics):
#     """
#     Récupère les valeurs min et max de l'indicateur sélectionné, en ignorant
#     la valeur correspondant à la clé 'France'.
#     """
#     # Crée une liste de valeurs en ignorant la clé 'France'
#     values = []
#     for key, value in metrics.items():
#         if key != "France":
#             values.append(value)

#     # Convertit la liste en un tableau NumPy pour le calcul
#     values_array = np.array(values, dtype=np.float64)

#     return np.nanmin(values_array), np.nanmax(values_array)


# def add_color_legend(m, vmin, vmax, cmap, metric_name):
#     """Ajoute une légende de la colormap à la carte."""
#     colormap = branca.colormap.LinearColormap(
#         vmin=vmin,
#         vmax=vmax,
#         colors=[mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, 256)],
#     ).to_step(index=np.linspace(vmin, vmax, 25))  # 5 niveaux de couleur

#     colormap.caption = str(metric_name)
#     colormap.add_to(m)


# # 📌 Fonction pour créer la carte et stocker dans `st.session_state`
# @st.cache_resource
# def create_map_with_metrics(geojson_data, metrics, metric_name, year):
#     map_center = [46.6034, 1.8883]  # Centre approximatif de la France
#     m = folium.Map(location=map_center, zoom_start=6, tiles="OpenStreetMap")

#     min_val, max_val = get_metric_range(metrics)

#     if "net" in metric_name or "Footprint" in metric_name:
#         cmap = cm.get_cmap("bwr")
#         min_val = min(min_val, -abs(max_val))
#         max_val = max(abs(min_val), max_val)
#     else:
#         cmap = cm.get_cmap("plasma")  # Utilisation de la colormap "plasma"

#     norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

#     if "Relative" in metric_name or "NUE" in metric_name:
#         unit = "%"
#     elif "Eff" in metric_name:
#         unit = ""
#     elif "Footprint" in metric_name:
#         unit = "Mha"
#     elif "Livestock density" == metric_name:
#         unit = "LU/ha"
#     else:
#         unit = "ktN/yr"

#     # 📊 Nouveau : créer le tableau des valeurs régionales
#     table_data = []

#     for feature in geojson_data["features"]:
#         region_name = feature["properties"]["nom"]
#         metric_value = metrics.get(region_name, np.nan)

#         if np.isnan(metric_value):
#             color = "#CCCCCC"
#         else:
#             rgba = cmap(norm(metric_value))
#             color = mcolors.to_hex(rgba)

#         def style_function(x, fill_color=color):
#             return {
#                 "fillColor": fill_color,
#                 "color": "black",
#                 "weight": 1,
#                 "fillOpacity": 0.6,
#             }

#         folium.GeoJson(
#             feature,
#             style_function=style_function,
#             tooltip=folium.Tooltip(f"{region_name}: {metric_value:.2f} {unit}"),
#         ).add_to(m)

#         table_data.append(
#             {
#                 "Région": region_name,
#                 "Valeur": round(metric_value, 2)
#                 if not np.isnan(metric_value)
#                 else None,
#                 # "Unité": unit,
#             }
#         )

#     add_color_legend(m, min_val, max_val, cmap, metric_name)

#     df = pd.DataFrame(table_data)

#     return m, df


# def weighted_mean(metrics, area):
#     # Calcul de la somme des produits des valeurs et des poids
#     weighted_sum = sum(metrics[key] * area[key] for key in metrics)

#     # Calcul de la somme des poids
#     total_area = sum(area[key] for key in area)

#     # Retourner la moyenne pondérée
#     return weighted_sum / total_area if total_area != 0 else 0


# # 🔹 Assurer la persistance de la carte dans `st.session_state`
# if "map_html" not in st.session_state:
#     st.session_state.map_html = None

# with tab5:
#     st.title("Map Configuration")

#     # 🟢 Sélection de l'année
#     st.session_state.map_year = st.selectbox(
#         "Select a year",
#         st.session_state.available_years,
#         index=0,
#         key="year_map_selection",
#     )

#     # 🟢 Sélection de la métrique
#     metric_map = [
#         "Total imported nitrogen",
#         "Total net plant import",
#         "Total net animal import",
#         "Total plant production",
#         "Total animal production",
#         "Environmental Footprint",
#         "NUE",
#         "System NUE",
#         "Livestock density",
#         "Cereals production",
#         "Leguminous production",
#         "Grassland and forage production",
#         "Roots production",
#         "Oleaginous production",
#         "Fruits and vegetables production",
#         "Relative Cereals production",
#         "Relative Leguminous production",
#         "Relative Grassland and forage production",
#         "Relative Roots production",
#         "Relative Oleaginous production",
#         "Relative Fruits and vegetables production",
#         "NH3 volatilization",
#         "N2O emission",
#         # "Effective number of nodes",
#         # "Effective connectivity",
#         # "Effective number of links",
#         # "Effective number of role",
#     ]
#     st.session_state.metric = st.selectbox(
#         "Select a metric", metric_map, index=0, key="metric_selection"
#     )

#     # 🔹 Bouton "Run"
#     if st.button("Run", key="map_button"):

#         @st.cache_resource
#         def get_cached_map(geojson_data, metrics, metric_name, year):
#             return create_map_with_metrics(
#                 geojson_data, metrics, metric_name, st.session_state.map_year
#             )

#         with st.spinner("🚀 Running models and calculating metrics..."):
#             # 📌 Exécuter les modèles et récupérer les métriques
#             models = run_models_for_all_regions(
#                 st.session_state.map_year, regions, data
#             )
#             metrics, area = get_metrics_for_all_regions(
#                 models, st.session_state.metric, st.session_state.map_year
#             )

#             # 📌 Charger le GeoJSON et créer la carte
#             geojson_data = load_geojson()

#             map_obj, table = get_cached_map(
#                 geojson_data,
#                 metrics,
#                 st.session_state.metric,
#                 st.session_state.map_year,
#             )

#             # 📌 Convertir la carte en HTML pour éviter la disparition
#             st.session_state.map_html = map_obj._repr_html_()

#             # 🔹 Vérifier si la carte est déjà générée et l'afficher
#             st.title("Nitrogen Map")
#             if st.session_state.metric in [
#                 "NUE",
#                 "System NUE",
#                 "Livestock density",
#                 "Relative Cereals production",
#                 "Relative Leguminous production",
#                 "Relative Grassland and forage production",
#                 "Relative Roots production",
#                 "Relative Oleaginous production",
#                 "Relative Fruits and vegetables production",
#                 "Environmental Footprint",
#             ]:
#                 if st.session_state.metric == "Livestock density":
#                     st.text(
#                         f"Mean for France: {np.round(weighted_mean(metrics, area), 2)} LU/ha"
#                     )
#                 elif st.session_state.metric == "Environmental Footprint":
#                     # st.text(f"Mean for France: {np.round(weighted_mean(metrics, area), 2)} Mha")
#                     pass
#                 else:
#                     st.text(
#                         f"Mean for France: {np.round(weighted_mean(metrics, area), 2)} %"
#                     )
#             else:
#                 st.text(
#                     f"Total for France: {np.round(np.sum(list(metrics.values())), 2)} ktN/yr"
#                 )

#         if st.session_state.map_html:
#             st.components.v1.html(st.session_state.map_html, height=800, scrolling=True)
#             st.dataframe(table)
#             st.download_button(
#                 "Download Data",
#                 table.to_csv(index=False),
#                 file_name="data.csv",
#                 key="mapbutton",
#             )
#         else:
#             st.warning("Please run the model to generate the map.")


# @st.cache_resource(show_spinner="Running GRAFS-E over all territories...")
# def run_models_for_all_years(region, _data_loader):
#     models = {}
#     for year in st.session_state.available_years:
#         if str(year) == "1852" and region == "Savoie":
#             pass
#         else:
#             models[year] = NitrogenFlowModel(
#                 data=_data_loader, year=str(year), region=region
#             )
#     return models


# # 📌 Calculer et stocker les métriques pour chaque région en cache
# @st.cache_data
# def get_metrics_for_all_years(_models, metric_name, region):
#     metric_dict = {
#         "Total imported nitrogen": "imported_nitrogen",
#         "Total net plant import": "net_imported_plant",
#         "Total net animal import": "net_imported_animal",
#         "Total plant production": "stacked_plant_production",
#         "Area": "surfaces",
#         "Environmental Footprint": "env_footprint",
#         "Area tot": "surfaces_tot",
#         "Total Fertilization": "tot_fert",
#         "Relative Fertilization": "rel_fert",
#         "Primary Nitrogen fertilization use": "primXsec",
#         "Emissions": "emissions",
#         "NUE": "NUE",
#         "System NUE": "NUE_system_2",
#         "Livestock density": "LU_density",
#         "Self-Sufficiency": "N_self_sufficient",
#         "Cereals production": "cereals_production",
#         "Leguminous production": "leguminous_production",
#         "Oleaginous production": "oleaginous_production",
#         "Grassland and forage production": "grassland_and_forages_production",
#         "Roots production": "roots_production",
#         "Fruits and vegetables production": "fruits_and_vegetable_production",
#         "Relative Cereals production": "cereals_production_r",
#         "Relative Leguminous production": "leguminous_production_r",
#         "Relative Oleaginous production": "oleaginous_production_r",
#         "Relative Grassland and forage production": "grassland_and_forages_production_r",
#         "Relative Roots production": "roots_production_r",
#         "Relative Fruits and vegetables production": "fruits_and_vegetable_production_r",
#         "Total animal production": "animal_production",
#     }
#     metric_function_name = metric_dict[metric_name]
#     metrics = {}
#     for year, model in _models.items():
#         metric_function = getattr(model, metric_function_name, None)
#         if callable(metric_function):
#             metrics[year] = metric_function()
#         else:
#             metrics[year] = None  # Si la méthode n'existe pas, on met None
#     return metrics


# def plot_standard_graph(_models, metric, region):
#     metrics = get_metrics_for_all_years(_models, metric, region)

#     # 📊 Préparation des données pour le graphique
#     years = sorted([int(year) for year in metrics.keys()])
#     values = list(metrics.values())

#     # 🔄 Affichage conditionnel des labels
#     min_gap = 4  # ⚙️ Seuil minimum entre deux labels
#     visible_years = []
#     last_visible_year = None

#     for year in years:
#         if last_visible_year is None or (year - last_visible_year) >= min_gap:
#             visible_years.append(str(year))
#             last_visible_year = year
#         else:
#             visible_years.append("")  # Pas de label si trop proche

#     # 🔵 Détection du thème actuel
#     current_theme = st.get_option("theme.base")

#     if current_theme == "light":
#         line_color = "royalblue"
#     else:
#         line_color = "white"

#     # 📊 Création du graphique avec Plotly
#     fig = go.Figure()

#     fig.add_trace(
#         go.Scatter(
#             x=years,
#             y=values,
#             mode="lines+markers",
#             name=st.session_state.metric_hist,
#             line=dict(color=line_color, width=2),
#             marker=dict(size=8, color="royalblue", symbol="circle"),
#         )
#     )

#     if "Eff" in metric:
#         y_label = "#"
#     elif (
#         "NUE" in metric
#         or "Primary" in metric
#         or "Sufficiency" in metric
#         or "Relative" in metric
#     ):
#         y_label = "%"
#     elif "Livestock density" in metric:
#         y_label = "LU/ha"
#     else:
#         y_label = "ktN/yr"

#     # 🎨 Personnalisation du style du graphique
#     fig.update_layout(
#         title=f"Historical Evolution of {st.session_state.metric_hist} in {st.session_state.region_hist}",
#         xaxis_title="Year",
#         yaxis_title=y_label,
#         template="plotly_white",
#         hovermode="x unified",
#         showlegend=True,
#         legend=dict(x=0.05, y=0.95, bgcolor="rgba(255, 255, 255, 0.5)"),
#     )

#     # 🔍 Amélioration des axes
#     fig.update_xaxes(
#         showgrid=True,
#         tickmode="array",
#         tickvals=years,
#         ticktext=visible_years,
#         tickangle=45,
#         # tickfont=dict(size=10),
#     )

#     fig.update_yaxes(showgrid=True, zeroline=True)

#     # 🚀 Affichage dans Streamlit
#     st.plotly_chart(fig, use_container_width=True)


# # 2. Créer une fonction de génération de couleurs dynamiques
# def get_unique_colors(new_labels, fixed_colors_map):
#     labels_to_color = sorted(list(set(new_labels) - set(fixed_colors_map.keys())))

#     # 2.2 Choisir une Colormap (e.g., 'tab20', 'hsv', 'nipy_spectral'). 'tab20' est bonne pour max 20 catégories.
#     # Si le nombre de labels est élevé (> 20), 'hsv' ou 'nipy_spectral' est préférable.
#     N = len(labels_to_color)
#     colormap_name = "hsv" if N > 20 else "tab20"
#     cmap = cm.get_cmap(colormap_name)

#     # 2.3 Générer les couleurs RGB pour les labels manquants
#     dynamic_colors = {
#         label: cmap(i / N)  # Donne une couleur distincte de 0 à 1
#         for i, label in enumerate(labels_to_color)
#     }

#     all_colors = fixed_colors_map.copy()
#     all_colors.update(dynamic_colors)

#     new_index_to_color = {
#         i: all_colors.get(new_labels[i]) for i in range(len(new_labels))
#     }

#     from matplotlib.colors import to_hex

#     final_index_to_color = {
#         idx: to_hex(color) if isinstance(color, tuple) else color
#         for idx, color in new_index_to_color.items()
#     }

#     return final_index_to_color


# def stacked_area_chart(_models, metric, region):
#     """Affiche un graphique en courbes empilées pour l'évolution des surfaces cultivées
#     avec une légende par catégorie et un hover individuel par culture."""

#     # -----------------------------------------------------------------
#     # 1) Récupération & Transformation des données
#     # -----------------------------------------------------------------
#     metrics = get_metrics_for_all_years(
#         _models, metric, region
#     )  # Dict { année(str) : df_cultures }

#     all_years = sorted(metrics.keys(), key=int)

#     # Suppose qu'on prend la première année comme référence pour l'index (les cultures)
#     df = pd.DataFrame(index=metrics[all_years[0]].index, columns=all_years, dtype=float)

#     # Remplir le DataFrame (Cultures x Années)
#     for year, df_year in metrics.items():
#         df.loc[df_year.index, year] = df_year

#     df.fillna(0, inplace=True)

#     # Calcul cumulatif pour l'affichage empilé par colonnes
#     df_cumsum = df.cumsum(axis=0)
#     # Ajouter une ligne "Base" (0) pour le fill='tonexty'
#     df_cumsum.loc["Base"] = 0
#     # df_cumsum = df_cumsum.sort_index()

#     # -----------------------------------------------------------------
#     # 2) Création du Sankey Plotly
#     # -----------------------------------------------------------------
#     fig = go.Figure()

#     # On veut regrouper les cultures par catégorie pour la légende
#     # => On utilise 'legendgroup' + 'showlegend' seulement au 1er trace de chaque catégorie
#     categories_seen = set()

#     # On veut hover = seulement la courbe survolée => hovermode='closest'
#     # => On n'utilise plus 'x unified'
#     if metric == "Area":
#         fig.update_layout(
#             title=f"Agricultural Area - {region}",
#             xaxis_title="Year",
#             yaxis_title="Cumulated Area (ha)",
#             hovermode="closest",  # ❗️ Montre seulement la courbe survolée
#             showlegend=True,
#         )

#         # Parcourir chaque culture dans l'ordre de l'index (df.index)
#         # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
#         cultures_list = [c for c in df_cumsum.index if c != "Base"]

#         for culture in cultures_list:
#             # Courbe du haut = df_cumsum.loc[culture]
#             # fill='tonexty' => se remplit entre cette courbe et la précédente
#             # => l'ordre du df_cumsum doit être correct (Base, ..., culture)
#             # Couleur en fonction de la catégorie

#             node_color = get_unique_colors(model.labels, {})
#             cat = model.df_cultures.loc[
#                 model.df_cultures.index == culture, "Category"
#             ].item()
#             culture_color = node_color[culture]

#             # Groupe de légende = cat
#             # On affiche la légende qu'une seule fois par catégorie
#             show_in_legend = cat not in categories_seen
#             if show_in_legend:
#                 categories_seen.add(cat)

#             fig.add_trace(
#                 go.Scatter(
#                     x=all_years,
#                     y=df_cumsum.loc[culture],
#                     fill="tonexty",
#                     mode="lines",
#                     line=dict(color=culture_color, width=0.5),
#                     name=cat,  # ❗️ Nom affiché = Catégorie
#                     legendgroup=cat,  # ❗️ On groupe par Catégorie
#                     customdata=df.loc[culture].tolist(),
#                     showlegend=show_in_legend,  # ❗️ Un seul trace par groupe dans la légende
#                     hovertemplate=(
#                         "Culture: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} ha<extra></extra>"
#                     ),
#                     text=[culture]
#                     * len(all_years),  # Pour afficher le nom de la culture au survol
#                 )
#             )

#     if metric == "Total plant production":
#         fig = make_subplots(specs=[[{"secondary_y": True}]])
#         fig.update_layout(
#             title=f"Agricultural Production - {region}",
#             xaxis_title="Year",
#             hovermode="closest",  # ❗️ Montre seulement la courbe survolée
#             showlegend=True,
#         )

#         # Parcourir chaque culture dans l'ordre de l'index (df.index)
#         # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
#         cultures_list = [c for c in df_cumsum.index if c != "Base"]

#         for culture in cultures_list:
#             # Courbe du haut = df_cumsum.loc[culture]
#             # fill='tonexty' => se remplit entre cette courbe et la précédente
#             # => l'ordre du df_cumsum doit être correct (Base, ..., culture)
#             # Couleur en fonction de la catégorie
#             cat = model.df_cultures.loc[
#                 model.df_cultures.index == culture, "Category"
#             ].item()
#             culture_color = node_color[model.label_to_index[culture]]

#             # Groupe de légende = cat
#             # On affiche la légende qu'une seule fois par catégorie
#             show_in_legend = cat not in categories_seen
#             if show_in_legend:
#                 categories_seen.add(cat)

#             fig.add_trace(
#                 go.Scatter(
#                     x=all_years,
#                     y=df_cumsum.loc[culture],
#                     fill="tonexty",
#                     mode="lines",
#                     line=dict(color=culture_color, width=0.5),
#                     name=cat,
#                     legendgroup=cat,
#                     customdata=df.loc[culture].tolist(),
#                     showlegend=show_in_legend,
#                     hovertemplate="Culture: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} ha<extra></extra>",
#                     text=[culture] * len(all_years),
#                 ),
#                 secondary_y=False,  # Axe Y principal
#             )

#         prod_tot = get_metrics_for_all_years(_models, "Area tot", region)

#         # Ajouter la ligne de production végétale totale
#         fig.add_trace(
#             go.Scatter(
#                 x=list(prod_tot.keys()),
#                 y=list(prod_tot.values()),
#                 mode="lines+markers",
#                 line=dict(color="white", width=3, dash="dash"),
#                 name="Total agricultural Area",
#                 hovertemplate="Year: %{x}<br>Value: %{y:.2f} ha<extra></extra>",
#             ),
#             secondary_y=True,  # Axe Y secondaire (à droite)
#         )

#         fig.update_yaxes(title_text="Cumulated Production (ktN/yr)", secondary_y=False)
#         fig.update_yaxes(title_text="Total Area (ha)", secondary_y=True)
#     if metric == "Emissions":
#         fig.update_layout(
#             title=f"Nitrogen emissions - {region}",
#             xaxis_title="Year",
#             yaxis_title="kTon/yr",
#             hovermode="closest",  # ❗️ Montre seulement la courbe survolée
#             showlegend=True,
#         )

#         # Parcourir chaque culture dans l'ordre de l'index (df.index)
#         # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
#         emissions_list = [c for c in df_cumsum.index if c != "Base"]

#         color = {
#             "N2O emission": "red",
#             "atmospheric N2": "white",
#             "NH3 volatilization": "blue",
#         }

#         for emission in emissions_list:
#             # Courbe du haut = df_cumsum.loc[culture]
#             # fill='tonexty' => se remplit entre cette courbe et la précédente
#             # => l'ordre du df_cumsum doit être correct (Base, ..., culture)

#             fig.add_trace(
#                 go.Scatter(
#                     x=all_years,
#                     y=df_cumsum.loc[emission],
#                     fill="tonexty",
#                     mode="lines",
#                     line=dict(color=color[emission], width=0.5),
#                     customdata=df.loc[emission].tolist(),
#                     name=emission,  # ❗️ Nom affiché = Catégorie
#                     hovertemplate=(
#                         "Emission: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} kton/yr<extra></extra>"
#                     ),
#                     text=[emission]
#                     * len(all_years),  # Pour afficher le nom de la culture au survol
#                 )
#             )

#     if metric == "Total Fertilization":
#         fig.update_layout(
#             title=f"Total Fertilization Use - {region}",
#             xaxis_title="Year",
#             yaxis_title="ktN",
#             hovermode="closest",  # ❗️ Montre seulement la courbe survolée
#             showlegend=True,
#         )

#         # Parcourir chaque culture dans l'ordre de l'index (df.index)
#         # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
#         emissions_list = [c for c in df_cumsum.index if c != "Base"]

#         color = {
#             "Haber-Bosch": "purple",
#             "Atmospheric deposition": "red",
#             "atmospheric N2": "white",
#             "Mining": "gray",
#             "Seeds": "pink",
#             "Animal excretion": "lightblue",
#             "Human excretion": "darkblue",
#             "Leguminous soil enrichment": "darkgreen",
#         }

#         for emission in emissions_list:
#             # Courbe du haut = df_cumsum.loc[culture]
#             # fill='tonexty' => se remplit entre cette courbe et la précédente
#             # => l'ordre du df_cumsum doit être correct (Base, ..., culture)

#             fig.add_trace(
#                 go.Scatter(
#                     x=all_years,
#                     y=df_cumsum.loc[emission],
#                     fill="tonexty",
#                     mode="lines",
#                     line=dict(color=color[emission], width=0.5),
#                     customdata=df.loc[emission].tolist(),
#                     name=emission,  # ❗️ Nom affiché = Catégorie
#                     hovertemplate=(
#                         "Fertilization vector: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} ktN/yr<extra></extra>"
#                     ),
#                     text=[emission]
#                     * len(all_years),  # Pour afficher le nom de la culture au survol
#                 )
#             )

#         prod_tot = get_metrics_for_all_years(_models, "Total plant production", region)

#         # Ajouter la ligne de production végétale totale
#         fig.add_trace(
#             go.Scatter(
#                 x=list(prod_tot.keys()),  # Clés du dictionnaire comme années
#                 y=list(prod_tot.values()),  # Valeurs du dictionnaire comme données
#                 mode="lines+markers",  # Ligne avec des marqueurs
#                 line=dict(
#                     color="white", width=3, dash="dash"
#                 ),  # Ligne noire en pointillés pour la distinguer
#                 name="Total Plant Production",  # Légende
#                 hovertemplate="Year: %{x}<br>Value: %{y:.2f} ktN/yr<extra></extra>",  # Tooltip personnalisé
#             )
#         )

#     if metric == "Relative Fertilization":
#         fig.update_layout(
#             title=f"Relative Fertilization Use - {region}",
#             xaxis_title="Year",
#             yaxis_title="%",
#             hovermode="closest",  # ❗️ Montre seulement la courbe survolée
#             showlegend=True,
#         )

#         # Parcourir chaque culture dans l'ordre de l'index (df.index)
#         # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
#         emissions_list = [c for c in df_cumsum.index if c != "Base"]

#         color = {
#             "Haber-Bosch": "purple",
#             "Atmospheric deposition": "red",
#             "atmospheric N2": "white",
#             "Mining": "gray",
#             "Seeds": "pink",
#             "Animal excretion": "lightblue",
#             "Human excretion": "darkblue",
#             "Leguminous soil enrichment": "darkgreen",
#         }

#         for emission in emissions_list:
#             # Courbe du haut = df_cumsum.loc[culture]
#             # fill='tonexty' => se remplit entre cette courbe et la précédente
#             # => l'ordre du df_cumsum doit être correct (Base, ..., culture)

#             fig.add_trace(
#                 go.Scatter(
#                     x=all_years,
#                     y=df_cumsum.loc[emission],
#                     fill="tonexty",
#                     mode="lines",
#                     line=dict(color=color[emission], width=0.5),
#                     customdata=df.loc[emission].tolist(),
#                     name=emission,  # ❗️ Nom affiché = Catégorie
#                     hovertemplate=(
#                         "Fertilization vector: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} %<extra></extra>"
#                     ),
#                     text=[emission]
#                     * len(all_years),  # Pour afficher le nom de la culture au survol
#                 )
#             )

#     if metric == "Environmental Footprint":
#         color = {
#             "Local Food": "blue",
#             "Local Feed": "lightgreen",
#             "Import Food": "lightgray",
#             "Import Feed": "darkgray",
#             "Import Livestock": "cyan",
#             "Export Livestock": "lightblue",
#             "Export Feed": "green",
#             "Export Food": "red",
#         }

#         color = {
#             # Local – bleus
#             "Local Food": "#1f77b4",
#             "Local Feed": "#5fa2ce",
#             # Import – violets
#             "Import Food": "#9467bd",
#             "Import Feed": "#b799d3",
#             "Import Livestock": "#d4c2e5",
#             # Export – rouges / corail
#             "Export Food": "#d62728",
#             "Export Feed": "#ff796c",
#             "Export Livestock": "#ffb1a8",
#         }

#         net_curve_color = "#c48b00"  # très lisible sur fond noir

#         # Séparer les catégories
#         import_categories = [
#             "Import Food",
#             "Import Feed",
#             "Import Livestock",
#             "Local Food",
#             "Local Feed",
#         ]
#         export_categories = ["Export Food", "Export Feed", "Export Livestock"]

#         for name in import_categories:
#             fig.add_trace(
#                 go.Scatter(
#                     x=all_years,
#                     y=df.loc[name],  # pas de cumul
#                     mode="none",  # juste l'aire
#                     stackgroup="p",  # pile positive
#                     name=name,
#                     fillcolor=color[name],
#                     customdata=(df.loc[name] / 1e6).values.reshape(-1, 1),
#                     hovertemplate=f"<b>{name}</b><br>Year %{{x}}<br>%{{customdata[0]:.2f}} M ha<extra></extra>",
#                 )
#             )

#         # -------- EXPORTS (négatifs) -------------------------------
#         for name in export_categories:
#             fig.add_trace(
#                 go.Scatter(
#                     x=all_years,
#                     y=df.loc[name],  # on passe en négatif
#                     mode="none",
#                     stackgroup="n",  # pile négative
#                     name=name,
#                     fillcolor=color[name],
#                     customdata=(-df.loc[name] / 1e6).values.reshape(-1, 1),
#                     hovertemplate=f"<b>{name}</b><br>Year %{{x}}<br>%{{customdata[0]:.2f}} M ha<extra></extra>",
#                 )
#             )

#         # Calculer le total importé - exporté
#         df_total_import = df.loc[
#             ["Import Food", "Import Feed", "Import Livestock"]
#         ].sum(axis=0)
#         df_total_export = df.loc[export_categories].sum(axis=0)
#         df_net_import_export = df_total_import + df_total_export

#         # Ajouter la ligne total
#         fig.add_trace(
#             go.Scatter(
#                 x=all_years,
#                 y=df_net_import_export,
#                 mode="lines+markers",
#                 line=dict(
#                     color=net_curve_color, width=4, dash="dash"
#                 ),  # line=dict(color="Black", width=4, dash="dash"),
#                 name="Net Land Import",
#                 hovertemplate="Year: %{x}<br>Value: %{customdata:.2f} Mha<extra></extra>",
#                 customdata=df_net_import_export.values.reshape(-1, 1)
#                 / 1e6,  # Utiliser les valeurs non cumulées pour le hover, divisées par 1e6
#             )
#         )

#         # Mise à jour du layout
#         fig.update_layout(
#             title=f"Environmental Footprint - {region}",
#             xaxis_title="Year",
#             yaxis_title="ha",
#             # hovermode="closest",
#             showlegend=True,
#             hovermode="x unified",
#         )

#     if "Regime" in metric:
#         fig.update_layout(
#             title=f"{metric} - {region}",
#             xaxis_title="Year",
#             yaxis_title="ktN",
#             hovermode="closest",  # ❗️ Montre seulement la courbe survolée
#             showlegend=True,
#         )

#         # Parcourir chaque culture dans l'ordre de l'index (df.index)
#         # 'Base' doit être ignoré => On ne fait pas de trace pour 'Base'
#         cultures_list = [c for c in df_cumsum.index if c != "Base"]

#         for culture in cultures_list:
#             # Courbe du haut = df_cumsum.loc[culture]
#             # fill='tonexty' => se remplit entre cette courbe et la précédente
#             # => l'ordre du df_cumsum doit être correct (Base, ..., culture)
#             # Couleur en fonction de la catégorie
#             cat = model.df_cultures.loc[
#                 model.df_cultures.index == culture, "Category"
#             ].item()
#             culture_color = node_color[model.label_to_index[culture]]

#             # Groupe de légende = cat
#             # On affiche la légende qu'une seule fois par catégorie
#             show_in_legend = cat not in categories_seen
#             if show_in_legend:
#                 categories_seen.add(cat)

#             fig.add_trace(
#                 go.Scatter(
#                     x=all_years,
#                     y=df_cumsum.loc[culture],
#                     fill="tonexty",
#                     mode="lines",
#                     line=dict(color=culture_color, width=0.5),
#                     name=cat,  # ❗️ Nom affiché = Catégorie
#                     legendgroup=cat,  # ❗️ On groupe par Catégorie
#                     customdata=df.loc[culture].tolist(),
#                     showlegend=show_in_legend,  # ❗️ Un seul trace par groupe dans la légende
#                     hovertemplate=(
#                         "Product: %{text}<br>Year: %{x}<br>Value: %{customdata:.2f} ha<extra></extra>"
#                     ),
#                     text=[culture]
#                     * len(all_years),  # Pour afficher le nom de la culture au survol
#                 )
#             )

#     # -----------------------------------------------------------------
#     # 3) Affichage
#     # -----------------------------------------------------------------
#     st.plotly_chart(fig, use_container_width=True)


# with tab6:
#     st.title("Historic evolution of agrarian landscape")

#     st.text(
#         "Discover how agriculture changes during time. Choose a metric and a territory:"
#     )

#     if not st.session_state.dataloader:
#         st.warning(
#             "⚠️ Please upload project and data files first in the 'Data Uploading' tab."
#         )
#     else:
#         metric_hist = [
#             "Total imported nitrogen",
#             "Total net plant import",
#             "Total net animal import",
#             "Total plant production",
#             "Total animal production",
#             "Area",
#             "Environmental Footprint",
#             "Total Fertilization",
#             "Relative Fertilization",
#             "Primary Nitrogen fertilization use",
#             "Emissions",
#             "NUE",
#             "System NUE",
#             "Self-Sufficiency",
#             "Livestock density",
#             "Cereals production",
#             "Leguminous production",
#             "Grassland and forage production",
#             "Roots production",
#             "Oleaginous production",
#             "Fruits and vegetables production",
#             "Relative Cereals production",
#             "Relative Leguminous production",
#             "Relative Grassland and forage production",
#             "Relative Roots production",
#             "Relative Oleaginous production",
#             "Relative Fruits and vegetables production",
#         ]

#         st.session_state.region_hist = st.selectbox(
#             "Select an area", metric_hist, index=0, key="hist_area_selection"
#         )
#         st.session_state.metric_hist = st.selectbox(
#             "Select a metric", metric_hist, index=0, key="hist_metric_selection"
#         )

#         # ✅ Affichage des sélections (se met à jour dynamiquement)
#         if "selected_region_hist" not in st.session_state:
#             st.session_state.region_hist = None
#         if st.session_state.region_hist:
#             st.write(f"✅ Selected Area: {st.session_state.region_hist}")
#         else:
#             st.warning("⚠️ Please select an Area")

#         if st.button("Run", key="map_button_hist"):
#             with st.spinner("🚀 Running models and calculating metrics..."):
#                 # 📌 Exécuter les modèles et récupérer les métriques
#                 models = run_models_for_all_years(
#                     st.session_state.region_hist, st.session_state.dataloader
#                 )
#                 if st.session_state.metric_hist not in [
#                     "Area",
#                     "Emissions",
#                     "Relative Fertilization",
#                     "Total Fertilization",
#                     "Total plant production",
#                     "Environmental Footprint",
#                 ]:
#                     plot_standard_graph(
#                         models,
#                         st.session_state.metric_hist,
#                         st.session_state.region_hist,
#                     )
#                 else:
#                     stacked_area_chart(
#                         models,
#                         st.session_state.metric_hist,
#                         st.session_state.region_hist,
#                     )

# with tab7:
#     st.title("Scenario generator")

#     st.text(
#         "Welcome to the prospective mode. Here you can imagine the future of agriculture according to your vision."
#     )

#     st.subheader("How to proceed ?")

#     st.text(
#         "You have to fill a scenario excel. Several tokens are needed to run the model in prospective mode. They are splitted in 3 tabs :"
#     )
#     st.markdown(
#         "- Main: In this tab, you have to fill the main characteristics of the future of your territory : population, access to international trade, access to industrial input..."
#     )
#     st.markdown(
#         "- Area: In this tab, you have to distribute the total agricultural area between crops and check the parameters of the production function. The production function gives the yield (Y) in function of the fertilisation amount (F). There is one set of parameters by crop type. The parameters of this function depend of the production function chosen :"
#     )
#     # st.markdown("The yield is computed as $Y = Y_{\\text{max}} \\cdot (1 - e^{-F/k})$")
#     st.markdown(" Ratio: $Y(F) = \\frac{Y_{max}F}{Y_{max}+F}$")
#     st.markdown(" Linear: $Y(F) = min(a*F, b)$")
#     st.markdown(" Exponential: $Y(F) = Y_{max}(1-e^{F/F^*})$")

#     st.markdown(
#         "- Technical: This tab encompass all technical coefficient (excretion per LU, weight of the optimization model, time spend by livestock in crops). It reflects potential technical evolution in agriculture and physical constraints."
#     )

#     #     st.subheader("Scenario file")

#     #     st.markdown(
#     #         "Here you can find a blank scenario sheet. Please fill all items in main and technical tabs. Fill as many lines in area as you need crops. Make sure the sum of proportion column is 1 and for each crop $Y_{max}$<$k$."
#     #     )

#     #     # Absolute path to the folder where the script is located
#     #     base_path = os.path.dirname(os.path.abspath(__file__))

#     #     # Join with your relative file
#     #     file_path = os.path.join(base_path, "data", "scenario.xlsx")
#     #     # Read the binary content of the file
#     #     with open(file_path, "rb") as file:
#     #         file_bytes = file.read()

#     #     # Create the download button
#     #     st.download_button(
#     #         label="📥 Download blank Scenario Excel",
#     #         data=file_bytes,
#     #         file_name="scenario.xlsx",
#     #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#     #     )

#     #     st.markdown("Once you have your scenario ready, go to Prospective mode tab.")

#     #     st.subheader("Hard to find these numbers ?")
#     #     st.text(
#     #         "It might be hard to create from scratch a physical functioning agro-system. To oversome this difficulty, you can use the scenario generator. To do so, choose a territory, a futur year. The scenario generator will automatically create a 'Business as usual' scenario. This scenario is created based on historical trajectory of the territory."
#     #     )

#     #     st.title("Scenario Generator")

#     # 🟢 Sélection de l'année
#     st.session_state.pros_year = st.selectbox(
#         "Select a year",
#         [str(y) for y in range(2025, 2061)],
#         index=0,
#         key="year_pros_selection",
#     )
#     # 🟢 Sélection de la fonction de production
#     st.session_state.pros_func = st.selectbox(
#         "Select Production function",
#         ["Ratio", "Linear", "Exponential"],
#         index=0,
#         key="func_pros_selection",
#     )
#     st.text_input("Scenario name", key="scenario_name_input")

#     m_hist = create_map()
#     map_data_pros = st_folium(
#         m_hist, height=600, use_container_width=True, key="pros_map"
#     )

#     # 🔹 Mettre à jour `st.session_state.region` avec la sélection utilisateur
#     if map_data_pros and "last_active_drawing" in map_data_pros:
#         last_drawing = map_data_pros["last_active_drawing"]
#         if (
#             last_drawing
#             and "properties" in last_drawing
#             and "nom" in last_drawing["properties"]
#         ):
#             st.session_state.region_pros = last_drawing["properties"][
#                 "nom"
#             ]

#     # ✅ Affichage des sélections (se met à jour dynamiquement)
#     if "selected_region_pros" not in st.session_state:
#         st.session_state.region_pros = None
#     if st.session_state.region_pros:
#         st.write(
#             f"✅ Selected Area : {st.session_state.region_pros}"
#         )
#     else:
#         st.warning("⚠️ Veuillez sélectionner une région")

#     def generate_scenario(year, region, name, func):
#         scenar.generate_scenario_excel(year, region, name, func)

#     if st.button("Create business as usual scenario", key="map_button_scenario"):
#         with tempfile.TemporaryDirectory() as temp_dir:
#             with st.spinner("Generating the business as usual scenario..."):
#                 st.session_state.name = st.session_state.scenario_name_input
#                 scenar = scenario(temp_dir)
#                 # threading.Thread(
#                 #     target=generate_scenario,
#                 #     args=(
#                 #         st.session_state.pros_year,
#                 #         st.session_state.region_pros,
#                 #         st.session_state.name,
#                 #         st.session_state.pros_func,
#                 #     ),
#                 #     daemon=True,
#                 # ).start()
#                 generate_scenario(
#                     st.session_state.pros_year,
#                     st.session_state.region_pros,
#                     st.session_state.name,
#                     st.session_state.pros_func,
#                 )

#                 with open(
#                     os.path.join(temp_dir, st.session_state.name + ".xlsx"), "rb"
#                 ) as f:
#                     file_data = f.read()
#                 st.download_button(
#                     label="📥 Download scenario sheet",
#                     data=file_data,
#                     file_name=st.session_state.name + ".xlsx",
#                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                     key="scenar_button",
#                 )
#                 st.markdown(
#                     "Once you have your scenario ready, go to Prospective mode tab."
#                 )

# with tab8:
#     st.title("Prospective mode")

#     st.subheader("How to use GRAFS-E Prospective Mode")

#     st.markdown(
#         """To run the prospective mode, two options are available:\\
#     1. **Upload an Excel sheet scenario** then click *Run scenario*
#     2. **Upload a model \*.pkl** saved from a previous session"""
#     )

#     # ─────────── A • Excel scenario
#     st.subheader("① Excel scenario → Run")

#     def excel_uploaded():
#         # if st.session_state.excel_uploaded_done:
#         #     return
#         up = st.session_state["xlsx_up"]
#         if not up:
#             return
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
#             tmp.write(up.read())
#             st.session_state.prep_xlsx_path = tmp.name

#         df = pd.read_excel(st.session_state.prep_xlsx_path)
#         st.session_state.prep_name = df.iloc[14, 1]
#         st.session_state.prep_region = df.iloc[15, 1]
#         st.session_state.prep_year = int(df.iloc[16, 1])
#         st.session_state.prep_func = df.iloc[17, 1]

#     st.file_uploader(
#         "📂 Upload .xlsx", type=["xlsx"], key="xlsx_up", on_change=excel_uploaded
#     )

#     if st.session_state.prep_name:
#         st.info(
#             f"Scenario: **{st.session_state.prep_name}**  "
#             f"({st.session_state.prep_region}, {st.session_state.prep_year}, {st.session_state.prep_func})"
#         )
#         if st.button("🚀 Run scenario"):
#             with st.spinner("Running prospective model…"):
#                 model = NitrogenFlowModel_prospect(st.session_state.prep_xlsx_path)
#                 # remplir les variables finales
#                 st.session_state.model = model
#                 st.session_state.orig = model.adjacency_matrix.copy()
#                 st.session_state.name = st.session_state.prep_name
#                 st.session_state.year_pros = st.session_state.prep_year
#                 st.session_state.year = st.session_state.year_pros
#                 st.session_state.region_pros = st.session_state.prep_region
#                 st.session_state.region = (
#                     st.session_state.region_pros
#                 )
#                 st.session_state.prod_func = st.session_state.prep_func
#                 st.session_state.heatmap_fig_pros = generate_heatmap(
#                     model, model.year, model.region
#                 )
#                 st.session_state.excel_uploaded_done = True
#                 buf = io.BytesIO()
#                 pickle.dump(model, buf)
#                 buf.seek(0)
#                 st.session_state.pkl_blob = buf.getvalue()
#             st.success("Model generated!")
#             st.rerun()
#             # on peut laisser Streamlit relancer automatiquement (pas de st.rerun)

#     # ─────────── B • Load .pkl
#     st.subheader("② Load existing model (.pkl)")

#     def load_pkl():
#         up = st.session_state["pkl_up"]
#         if not up:
#             return
#         try:
#             obj = pickle.load(io.BytesIO(up.getvalue()))
#             if not isinstance(obj, NitrogenFlowModel_prospect):
#                 st.session_state.load_error = "⛔️ Wrong file."
#                 return
#             st.session_state.model = obj
#             st.session_state.name = os.path.splitext(up.name)[0]
#             st.session_state.year_pros = obj.year
#             st.session_state.year = obj.year
#             st.session_state.region_pros = obj.region
#             st.session_state.region = obj.region
#             st.session_state.prod_func = obj.prod_func
#             st.session_state.heatmap_fig_pros = generate_heatmap(
#                 obj, obj.year, obj.region
#             )
#             st.session_state.pkl_blob = up.getvalue()
#             st.session_state.load_error = ""
#         except Exception as e:
#             st.session_state.load_error = str(e)

#     st.file_uploader(
#         "📥 Upload .pkl",
#         type=["pkl", "pickle", "joblib"],
#         key="pkl_up",
#         on_change=load_pkl,
#     )

#     if st.session_state.load_error:
#         st.error(st.session_state.load_error)

#     # ─────────── C • Résultats
#     st.subheader("Active model")
#     if st.session_state.model is None:
#         st.warning("No model loaded or generated yet.")
#     else:
#         st.success("Model ready")
#         st.markdown(
#             f"📘 Name : **{st.session_state.name}**  \n"
#             f"🗺️ Region : **{st.session_state.region_pros}**  \n"
#             f"📆 Year : **{st.session_state.year_pros}**  \n"
#             f"⚙️ Prod. function : **{st.session_state.prod_func}**"
#         )

#         if st.session_state.pkl_blob:
#             st.download_button(
#                 "💾 Download this model",
#                 data=st.session_state.pkl_blob,
#                 file_name=f"{st.session_state.name}.pkl",
#                 mime="application/octet-stream",
#                 key="pros_pkl",
#             )

#         if st.button("🔄 Reset model"):
#             for k in [
#                 "model",
#                 "name",
#                 "year_pros",
#                 "selected_region_pros",
#                 "prod_func",
#                 "heatmap_fig_pros",
#                 "pkl_blob",
#             ]:
#                 st.session_state[k] = None
#             st.session_state.excel_uploaded_done = False
#             st.rerun()

#         if st.session_state.heatmap_fig_pros:
#             if st.session_state.model:
#                 SA = st.toggle("See Solver Analytics", value=False, key="SA")
#                 if SA:
#                     st.subheader("Solver Analytics")
#                     # ----------------------------
#                     # 1. Préparer les données
#                     # ----------------------------

#                     final = st.session_state.model.log[-1]

#                     weights = {
#                         "diet_dev": st.session_state.model.w_diet,
#                         "fert_dev": st.session_state.model.w_Nsyn,
#                         "imp_dev": st.session_state.model.w_imp,
#                         "exp_dev": st.session_state.model.w_exp,
#                         "energy_dev": st.session_state.model.w_energy,
#                         "energy_prod": st.session_state.model.w_share_energy,
#                     }

#                     # --- table contraintes "cible vs réalisé" ---
#                     rows = [
#                         (
#                             "Synthetic N (crops)",
#                             final["Nsyn crop target"],
#                             final["Nsyn crop model"],
#                         ),
#                         (
#                             "Synthetic N (grass)",
#                             final["Nsyn grasslands target"],
#                             final["Nsyn grasslands model"],
#                         ),
#                         ("Imports", final["import target"], final["import model"]),
#                         ("Exports", final["export target"], final["export model"]),
#                         (
#                             "Energy Production",
#                             final["methanisation target"],
#                             final["methanisation model"],
#                         ),
#                     ]
#                     df = pd.DataFrame(
#                         rows, columns=["Constraint", "Target", "Model"]
#                     )

#                     EPS = 1  # ← tolérance (ktN/an)
#                     # CAP = 400  # ← % maxi affiché

#                     df["%"] = 100 * (df["Model"] + EPS) / (df["Target"] + EPS)
#                     # df["%"] = df["%"].clip(upper=CAP)  # on limite à 400 %

#                     max_pct = df["%"].max()
#                     CAP = 400 if 1.1 * max_pct >= 400 else 1.1 * max_pct

#                     # --- radar normalisé -------------------------------------------------
#                     theta = df["Constraint"].tolist()
#                     theta_closed = theta + [theta[0]]  # boucle fermée

#                     r_target = [100] * len(theta) + [100]
#                     r_model = df["%"].tolist() + [df["%"].iloc[0]]

#                     # 1) ligne 100 %
#                     fig = go.Figure()
#                     fig.add_trace(
#                         go.Scatterpolar(
#                             r=r_target,
#                             theta=theta_closed,
#                             mode="lines",
#                             line=dict(color="grey", dash="dot"),
#                             name="Target (100 %)",
#                         )
#                     )

#                     # 2) surface modèle
#                     fig.add_trace(
#                         go.Scatterpolar(
#                             r=r_model,
#                             theta=theta_closed,
#                             fill="toself",
#                             line=dict(color="dodgerblue"),
#                             name="Model",
#                         )
#                     )

#                     # 3) mise à l’échelle radiale
#                     max_pct = df["%"].max()
#                     CAP = (
#                         400 if 1.1 * max_pct >= 400 else 1.1 * max_pct
#                     )  # 400 % ou 1.1×max
#                     ticks = (
#                         [0, 50, 100, 200, 400] if CAP >= 400 else [0, 50, 100, 200]
#                     )

#                     fig.update_layout(
#                         title="Model vs Target (normalised)",
#                         polar=dict(
#                             radialaxis=dict(
#                                 range=[0, CAP],
#                                 tickvals=[v for v in ticks if v <= CAP],
#                                 ticksuffix="%",
#                                 color="black",
#                             )
#                         ),
#                         legend=dict(
#                             orientation="h",
#                             yanchor="bottom",
#                             y=-0.25,
#                             xanchor="center",
#                             x=0.5,
#                         ),
#                         height=450,
#                     )

#                     st.plotly_chart(fig, use_container_width=True)

#                     def color_diff(model, target, constraint):
#                         if target == 0:
#                             return (
#                                 "" if model == 0 else "background-color:red"
#                             )  # rouge si mod>0 alors que cible 0
#                         ratio = model / target
#                         diff = abs(model - target)
#                         if (
#                             0.9 <= ratio <= 1.1
#                             or diff < 5
#                             or (
#                                 ratio <= 1
#                                 and constraint
#                                 in [
#                                     "Synthetic N (crops)",
#                                     "Synthetic N (grass)",
#                                     "Imports",
#                                 ]
#                             )
#                         ):
#                             return "background-color:green"  # vert (±10 %)
#                         elif ratio > 1 and constraint in ["Exports"]:
#                             return "background-color:green"
#                         elif 0.5 <= ratio <= 2 or diff < 10:
#                             return "background-color:orange"  # orange
#                         return "background-color:red"  # rouge

#                     styled = df.style.apply(
#                         lambda row: [
#                             color_diff(row.Model, row.Target, row.Constraint)
#                             if col == "Model"
#                             else ""
#                             for col in row.index
#                         ],
#                         axis=1,
#                     ).format({"Target": "{:.2f}", "Model": "{:.2f}"})

#                     st.dataframe(styled)

#                     objective = final["objective"]
#                     good = 1.5  # à adapter : zone « OK »
#                     warn = 2.5  # au-delà = rouge
#                     end = 4

#                     fig_g = go.Figure(
#                         go.Indicator(
#                             mode="gauge+number",
#                             value=objective
#                             / sum(
#                                 list(weights.values())
#                             ),  # On s'intéresse à une moyenne pondérée des termes de la fonctions objectif
#                             title={"text": "Objective function"},
#                             gauge={
#                                 "axis": {"range": [0, end]},
#                                 "steps": [
#                                     {"range": [0, good], "color": "#c9f7d4"},
#                                     {"range": [good, warn], "color": "#fff3b0"},
#                                     {"range": [warn, end], "color": "#f9c0c0"},
#                                 ],
#                             },
#                         )
#                     )
#                     st.plotly_chart(fig_g, use_container_width=True)

#                 st.text(
#                     f"Total Throughflow : {np.round(st.session_state.model.get_transition_matrix().sum(), 1)} ktN/yr."
#                 )
#             st.subheader(
#                 f"Heatmap – {st.session_state.region_pros} in {st.session_state.year_pros}"
#             )
#             st.plotly_chart(
#                 st.session_state.heatmap_fig_pros, use_container_width=True
#             )
#             # Bouton pour télécharger la matrice
#             # ───── Création du DataFrame à partir de la matrice ──────────
#             matrix = st.session_state.model.get_transition_matrix()
#             df_matrix = pd.DataFrame(
#                 matrix,
#                 index=st.session_state.model.labels,
#                 columns=st.session_state.model.labels,
#             )

#             # ───── Conversion en CSV (encodage UTF-8) ───────────────────
#             csv_bytes = df_matrix.to_csv(index=True).encode("utf-8")

#             # ───── Bouton de téléchargement ─────────────────────────────
#             st.download_button(
#                 label="📥 Download matrix (csv)",
#                 data=csv_bytes,
#                 file_name=f"transition_matrix_{st.session_state.region}_{st.session_state.year}.csv",
#                 mime="text/csv",
#                 key="pros_matrix",
#             )
