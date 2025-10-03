import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import warnings

from grafs_e.N_class import DataLoader, NitrogenFlowModel, FluxGenerator


# --- Classe Dataloader ---
class Dataloader_Carbon:
    def __init__(self, project_path, data_path, region, year):
        """Charger les données nécessaires pour le modèle"""
        # Instancier NitrogenFluxModel avec l'année et la région
        self.data = DataLoader(project_path, data_path)
        self.nitrogen_model = NitrogenFlowModel(
            self.data, region, year
        )  # Modèle de flux d'azote

        fixed_compartments = [
            "Haber-Bosch",
            "atmospheric CO2",
            "atmospheric CH4",
            "hydrocarbures",
            "fishery products",
            "other sectors",
            "other losses",
            "soil stock",
        ]

        trade = [i + " trade" for i in set(self.data.init_df_prod["Sub Type"])]

        mechanization = [
            i + " machines"
            for i in list(self.data.init_df_elevage.index)
            + list(self.data.init_df_cultures.index)
        ]

        suffixes = [" manure", " slurry", " grasslands excretion"]
        # Une seule list comprehension pour tout faire
        excretion_compartments = [
            i + suffix for i in self.data.init_df_elevage.index for suffix in suffixes
        ]

        self.labels = (
            list(self.data.init_df_cultures.index)
            + list(self.data.init_df_elevage.index)
            + excretion_compartments
            + mechanization
            + list(self.data.init_df_prod.index)
            + list(self.data.init_df_pop.index)
            + fixed_compartments
            + trade
        )

        self.n = len(self.labels)  # Nombre total de secteurs
        self.adjacency_matrix_N = (
            self.nitrogen_model.get_transition_matrix()
        )  # Récupérer la matrice des flux depuis N_class

        self.label_to_index = {
            self.label: index for index, self.label in enumerate(self.labels)
        }
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}


# --- Classe FluxModif ---
class CarbonFlowModel:
    def __init__(self, data, region, year):
        self.year = year
        self.region = region
        self.data_loader = data
        self.labels = data.labels

        self.df_elevage = self.data_loader.data.generate_df_elevage(region, year, True)
        self.df_excr = self.data_loader.data.generate_df_excr(region, year, True)
        self.df_cultures = self.data_loader.data.generate_df_cultures(
            region, year, True
        )
        self.df_pop = self.data_loader.data.generate_df_pop(region, year, True)
        self.df_prod = self.data_loader.data.generate_df_prod(region, year, True)
        self.df_global = self.data_loader.data.get_global_metrics(region, year, True)

        self.flux_generator = FluxGenerator(self.labels)
        self.carbon_matrix = self.flux_generator.adjacency_matrix

        self.compute_fluxes()

    def change_flow(self, source, target, factor):
        self.carbon_matrix[
            self.data_loader.label_to_index[source],
            self.data_loader.label_to_index[target],
        ] = (
            self.data_loader.nitrogen_model.flux_generator.get_coef(source, target)
            * factor
        )

    def plot_heatmap_interactive(self):
        """
        Generates an interactive heatmap using Plotly to visualize the nitrogen flux transition matrix.

        The heatmap has the following features:
        - Logarithmic scale (simulated via log10(z)) to handle wide-ranging values.
        - A horizontal colorbar placed at the bottom of the plot.
        - A legend that maps matrix indices to sector labels, positioned on the right, ensuring no overlap.
        - The X-axis is displayed at the top of the plot, and the title is centered above the plot.

        This visualization helps to understand the relative magnitudes of the nitrogen fluxes between sectors
        in a clear and interactive manner.

        Returns:
            plotly.graph_objects.Figure: An interactive Plotly figure containing the heatmap.
        """

        # 1) Préparation des labels numériques
        x_labels = list(range(1, len(self.labels) + 1))
        y_labels = list(range(1, len(self.labels) + 1))

        # Si vous ignorez la dernière ligne/colonne comme dans votre code :
        # adjacency_subset = self.adjacency_matrix[: len(self.labels), : len(self.labels)]

        adj = np.array(self.carbon_matrix)  # ou .copy()
        adjacency_subset = adj[: len(self.labels), : len(self.labels)].copy()

        # 2) Gestion min/max et transformation log10
        cmin = max(1e-2, np.min(adjacency_subset[adjacency_subset > 0]))
        cmax = 1e4  # np.max(adjacency_subset)
        log_matrix = np.where(adjacency_subset > 0, np.log10(adjacency_subset), np.nan)

        # 3) Construire un tableau 2D de chaînes pour le survol
        #    Même dimension que log_matrix
        strings_matrix = []
        for row_i, y_val in enumerate(y_labels):
            row_texts = []
            for col_i, x_val in enumerate(x_labels):
                # Valeur réelle (non log) => adjacency_subset[row_i, col_i]
                real_val = adjacency_subset[row_i, col_i]
                if np.isnan(real_val):
                    real_val_str = "0"
                else:
                    real_val_str = f"{real_val:.2e}"  # format décimal / exposant
                # Construire la chaîne pour la tooltip
                # y_val et x_val sont les indices 1..N
                # self.labels[y_val] = nom de la source, self.labels[x_val] = nom de la cible
                tooltip_str = f"Source : {self.labels[y_val - 1]}<br>Target : {self.labels[x_val - 1]}<br>Value  : {real_val_str} ktN/yr"
                row_texts.append(tooltip_str)
            strings_matrix.append(row_texts)

        # 3) Tracé Heatmap avec go.Figure + go.Heatmap
        #    On règle "zmin" et "zmax" en valeurs log10
        #    pour contrôler la gamme de couleurs
        trace = go.Heatmap(
            z=log_matrix,
            x=x_labels,
            y=y_labels,
            colorscale="Plasma_r",
            zmin=np.log10(cmin),
            zmax=np.log10(cmax),
            text=strings_matrix,  # tableau 2D de chaînes
            hoverinfo="text",  # on n'affiche plus x, y, z bruts
            # Colorbar horizontale
            colorbar=dict(
                title="ktN/year",
                orientation="h",
                x=0.5,  # centré horizontalement
                xanchor="center",
                y=-0.15,  # en dessous de la figure
                thickness=15,  # épaisseur
                len=1,  # longueur en fraction de la largeur
            ),
            # Valeurs de survol -> vous verrez log10(...) par défaut
            # Pour afficher la valeur réelle, on peut plus tard utiliser "customdata"
        )

        # Créer la figure et y ajouter le trace
        fig = go.Figure(data=[trace])

        # 4) Discrétisation manuelle des ticks sur la colorbar
        #    On veut afficher l'échelle réelle (et pas log10)
        #    => calcul de tickvals en log10, et ticktext en 10^(tickvals)
        tickvals = np.linspace(np.floor(np.log10(cmin)), np.ceil(np.log10(cmax)), num=7)
        ticktext = [
            10**x for x in range(int(np.log(cmin)), int(np.log(cmax)), 1)
        ]  # [f"{10**v:.2e}" for v in tickvals]
        # Mettre à jour le trace pour forcer l'affichage
        fig.data[0].update(
            colorbar=dict(
                title="ktN/year",
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.15,
                thickness=25,
                len=1,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
            )
        )

        # 5) Configuration de la mise en page
        fig.update_layout(
            width=1000,
            height=1000,
            margin=dict(t=0, b=0, l=0, r=220),  # espace à droite pour la légende
        )
        fig.update_layout(yaxis_scaleanchor="x")

        # Axe X en haut
        fig.update_xaxes(
            title="Target",
            side="top",  # place les ticks en haut
            tickangle=90,  # rotation
            tickmode="array",
            tickfont=dict(size=10),
            tickvals=x_labels,  # forcer l'affichage 1..N
            ticktext=[str(x) for x in x_labels],
        )

        # Axe Y : inverser l'ordre pour un style "matriciel" standard
        fig.update_yaxes(
            title="Source",
            autorange="reversed",
            tickmode="array",
            tickfont=dict(size=10),
            tickvals=y_labels,
            ticktext=[str(y) for y in y_labels],
        )

        # 6) Ajouter la légende à droite
        #    Format : "1: label[0]" ... vertical
        legend_text = "<br>".join(
            f"{i + 1} : {lbl}" for i, lbl in enumerate(self.labels)
        )
        fig.add_annotation(
            x=1.25,  # un peu à droite
            y=0.45,  # centré en hauteur
            xref="paper",
            yref="paper",
            showarrow=False,
            text=legend_text,
            align="left",
            valign="middle",
            font=dict(size=11),
            bordercolor="rgba(0,0,0,0)",
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(0,0,0,0)",
        )

        return fig

    def compute_fluxes(self):
        df_elevage = self.df_elevage
        df_excr = self.df_excr
        df_cultures = self.df_cultures
        df_pop = self.df_pop
        df_global = self.df_global
        df_prod = self.df_prod

        flux_generator = self.flux_generator

        ##flux des produits vers le reste
        df_prod["C/N"] = np.where(
            df_prod["Nitrogen Content (%)"] != 0,
            df_prod["Carbon Content (%)"] / df_prod["Nitrogen Content (%)"],
            0,
        )

        df_prod["Carbon Production (ktC)"] = (
            df_prod["Production (kton)"] * df_prod["Carbon Content (%)"] / 100
        )

        # Modifier tous les flux où la source ou la target est un produit
        # en multipliant par le C/N de la culture de base.

        for i in df_prod.index:
            for j in self.data_loader.nitrogen_model.labels:
                try:
                    self.change_flow(
                        i, j, df_prod.loc[df_prod.index == i, "C/N"].item()
                    )
                    self.change_flow(
                        j, i, df_prod.loc[df_prod.index == i, "C/N"].item()
                    )
                except:  # noqa: E722
                    pass

        ## Flux des excretions aux cultures (soil stock)

        # Humification

        df_excr["Humification (ktC)"] = (
            df_excr["Excretion to soil (ktN)"] * df_excr["C/N"]
        )
        df_excr["Excretion (ktC)"] = df_excr["Humification (ktC)"] / (
            df_excr["Humification coefficient (%)"] / 100
        )
        source = df_excr["Humification (ktC)"].to_dict()
        target = {"soil stock": 1}

        flux_generator.generate_flux(source, target)

        # CH4

        df_excr["Excretion to CH4 (ktC)"] = (
            df_excr["Excretion (ktC)"] * df_excr["CH4 EM (%)"] / 100
        )

        source = df_excr["Excretion to CH4 (ktC)"].to_dict()
        target = {"atmospheric CH4": 1}
        flux_generator.generate_flux(source, target)

        # CO2

        df_excr["Excretion to CO2 (ktC)"] = (
            df_excr["Excretion (ktC)"]
            * (100 - df_excr["CH4 EM (%)"] - df_excr["Humification coefficient (%)"])
            / 100
        )

        source = df_excr["Excretion to CO2 (ktC)"].to_dict()
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(source, target)

        ## Flux des animaux aux excretions

        for i in df_elevage.index:
            source = {i: 1}
            target = df_excr.loc[
                df_excr["Origin compartment"] == i, "Excretion (ktC)"
            ].to_dict()
            flux_generator.generate_flux(source, target)

        ##flux excretion humaine (même chose sauf que les excretions humaines sont gérées dans df_pop)
        # Humification
        df_pop["Humification (ktC)"] = df_pop["Excretion to soil (ktN)"] * df_pop["C/N"]
        df_pop["Excretion (ktC)"] = df_pop["Humification (ktC)"] / (
            df_pop["Humification coefficient (%)"] / 100
        )
        source = df_pop["Humification (ktC)"].to_dict()
        target = {"soil stock": 1}

        flux_generator.generate_flux(source, target)

        # CH4
        df_pop["Excretion to CH4 (ktC)"] = (
            df_pop["Excretion (ktC)"] * df_pop["CH4 EM (%)"] / 100
        )

        source = df_pop["Excretion to CH4 (ktC)"].to_dict()
        target = {"atmospheric CH4": 1}

        # CO2
        df_pop["Excretion to CO2 (ktC)"] = (
            df_pop["Excretion (ktC)"]
            * (100 - df_pop["CH4 EM (%)"] - df_pop["Humification coefficient (%)"])
            / 100
        )

        source = df_pop["Excretion to CO2 (ktC)"].to_dict()
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(source, target)

        # Flux de carbone liés à Haber Bosch
        # Prendre les données nécessaires dans df_global

        HB_CH4_intensity = df_global.loc[
            "Total Haber-Bosch methan input (kgC/kgN)"
        ].item()
        HB_prod = self.data_loader.nitrogen_model.flux_generator.get_coef(
            "atmospheric N2", "Haber-Bosch"
        )

        source = {"hydrocarbures": 1}
        target = {"Haber-Bosch": HB_prod * HB_CH4_intensity}
        flux_generator.generate_flux(source, target)

        source = {"Haber-Bosch": 1}
        target = {"atmospheric CO2": HB_prod * HB_CH4_intensity}
        flux_generator.generate_flux(source, target)

        # Flux de carbone liés à la respiration et à la fermentation entérique

        # CH4 Entérique
        df_elevage["CH4 enteric (ktC)"] = (
            df_elevage["LU"] * df_elevage["C-CH4 enteric/LU (kgC)"] / 1e6
        )
        source = df_elevage["CH4 enteric (ktC)"].to_dict()
        target = {"atmospheric CH4": 1}
        flux_generator.generate_flux(source, target)

        # Respiration CO2
        df_elevage["Ingestion (ktC)"] = df_elevage.index.map(
            lambda i: self.carbon_matrix[:, self.data_loader.label_to_index[i]].sum()
        )

        excretion_sum_aligned = (
            df_excr.groupby("Origin compartment")["Excretion (ktC)"]
            .sum()
            .reindex(df_elevage.index, fill_value=0)
        )

        prod_sum_aligned = (
            df_prod.groupby("Origin compartment")["Carbon Production (ktC)"]
            .sum()
            .reindex(df_elevage.index, fill_value=0)
        )

        df_elevage["Respiration (ktC)"] = (
            df_elevage["Ingestion (ktC)"]
            - df_elevage["CH4 enteric (ktC)"]
            - excretion_sum_aligned
            - prod_sum_aligned
        ).clip(lower=0)
        source = df_elevage["Respiration (ktC)"].to_dict()
        target = {"atmospheric CO2": 1}

        flux_generator.generate_flux(source, target)

        def _calculate_cn_critical(df, index_key):
            """Calcule C/N critique et gère les erreurs."""
            try:
                row = df.loc[df.index == index_key]
                if row.empty:
                    return np.nan

                # Extrait les valeurs scalaires
                N_emissions_percent = (
                    row[["N-NH3 EM (%)", "N-N2 EM (%)", "N-N2O EM (%)"]].iloc[0].sum()
                )
                Humif_coeff = row["Humification coefficient (%)"].iloc[0]
                CN_ratio = row["C/N"].iloc[0]

                if Humif_coeff == 0:
                    return np.nan

                return (
                    (100 - N_emissions_percent) / 100 / (Humif_coeff / 100) * CN_ratio
                )
            except Exception:
                return np.nan

        for index, row in df_elevage[df_elevage["Respiration (ktC)"] == 0].iterrows():
            excretion_value = excretion_sum_aligned.loc[index]

            products_df = df_prod.loc[df_prod["Origin compartment"] == index]
            production = products_df["Carbon Production (ktC)"].sum()

            somme_sorties = row["CH4 enteric (ktC)"] + excretion_value + production

            cn_mean_calc = (
                products_df["C/N"] * products_df["Carbon Production (ktC)"]
            ).sum() / production
            cn_mean_str = f"{cn_mean_calc:.4f}"

            cn_data = {
                "Manure (Fumier)": _calculate_cn_critical(df_excr, index + " manure"),
                "Slurry (Lisiers)": _calculate_cn_critical(df_excr, index + " slurry"),
                "Grasslands excretion (Pâture)": _calculate_cn_critical(
                    df_excr, index + " grasslands excretion"
                ),
            }

            cn_details = "\n".join(
                [
                    f"    {key}: {val:.4f}" if not np.isnan(val) else f"    {key}: N/A"
                    for key, val in cn_data.items()
                ]
            )

            cn_details.append(f"    C/N Production Moy. : {cn_mean_str}")

            message = (
                f"La respiration de '{index}' forcée à zéro (Calcul brut: {row['Ingestion (ktC)'] - somme_sorties:.4f} ktC). \n"
                f"  Détails des flux (ktC) : Ingestion: {row['Ingestion (ktC)']:.4f} | CH4: {row['CH4 enteric (ktC)']:.4f} | Excrétion: {excretion_value:.4f} | Produits animaux : {production:.4f} \n"
                f"  C/N feed moyen critique: \n{cn_details}"
            )
            warnings.warn(message, category=UserWarning)

        # Respiration humaine
        df_pop["Ingestion (ktC)"] = df_pop.index.map(
            lambda i: self.carbon_matrix[:, self.data_loader.label_to_index[i]].sum()
        )

        df_pop["Respiration (ktC)"] = (
            df_pop["Ingestion (ktC)"] - df_pop["Excretion (ktC)"]
        ).clip(lower=0)
        source = df_pop["Respiration (ktC)"].to_dict()
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(source, target)

        for index, row in df_pop[df_pop["Respiration (ktC)"] == 0].iterrows():
            message = (
                f"La respiration du groupe de population '{index}' a été forcée à zéro (Respiration (ktC) <= 0). "
                f"Détails des flux (ktC) : "
                f"Ingestion (Entrée): {row['Ingestion (ktC)']:.4f} | "
                f"Excrétion (Sortie): {row['Excretion (ktC)']:.4f} | "
            )
            warnings.warn(message, category=UserWarning)

        ## Flux de carbone liés aux machines / mécanisation

        # Flux liés aux machines des cultures

        df_cultures["Mecanisation Emission (ktC)"] = (
            df_cultures["Carbon Mechanisation Intensity (ktC/ha)"]
            * df_cultures["Area (ha)"]
        )
        dict_mecanisation = {
            index + " machines": value
            for index, value in df_cultures["Mecanisation Emission (ktC)"].items()
        }
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(dict_mecanisation, target)
        source = {"hydrocarbures": 1}
        flux_generator.generate_flux(source, dict_mecanisation)

        # Flux liés aux machines des élevages

        df_elevage["Mecanisation Emission (ktC)"] = (
            df_elevage["Infrastructure CO2 emissions/LU (kgC)"] * df_elevage["LU"]
        )
        dict_mecanisation = {
            index + " machines": value
            for index, value in df_elevage["Mecanisation Emission (ktC)"].items()
        }
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(dict_mecanisation, target)
        source = {"hydrocarbures": 1}
        flux_generator.generate_flux(source, dict_mecanisation)

        ## Import

        df_import = self.data_loader.nitrogen_model.allocations_df[
            self.data_loader.nitrogen_model.allocations_df["Type"].isin(
                ["Imported Food", "imported feed"]
            )
        ].copy()

        df_import = df_import.join(
            self.df_prod[["C/N", "Sub Type"]], on="Product", how="left"
        )

        df_import["Source Value"] = df_import["Allocated Nitrogen"] * df_import["C/N"]

        for consumer, sub_type, value in zip(
            df_import["Consumer"], df_import["Sub Type"], df_import["Source Value"]
        ):
            source_key = f"{sub_type} trade"
            self.flux_generator.generate_flux({source_key: value}, {consumer: 1})

        ## Cultures, photosynthèse

        # La production des produits est faite plus haut.
        # Résidus :

        carbon_prod_agg = df_prod.groupby("Origin compartment")[
            "Carbon Production (ktC)"
        ].sum()

        # Jointure de la production de carbone agrégée avec df_cultures
        df_cultures = df_cultures.join(carbon_prod_agg, how="left")

        # Calcul des Résidus
        df_cultures["Residue Production (ktC)"] = (
            df_cultures["Carbon Production (ktC)"]
            * (1 - df_cultures["Harvest Index"])
            / df_cultures["Harvest Index"]
        )

        df_cultures["Root Production (ktC)"] = (
            df_cultures["Surface Root Production (kgC/ha)"]
            * df_cultures["Area (ha)"]
            / 1e6
        )

        target = (
            df_cultures["Carbon Production (ktC)"]
            + df_cultures["Residue Production (ktC)"]
            + df_cultures["Root Production (ktC)"]
        ).to_dict()
        source = {"atmospheric CO2": 1}

        flux_generator.generate_flux(source, target)

        source = (
            df_cultures["Residue Production (ktC)"]
            * df_cultures["Residue Humification Coefficient (%)"]
            / 100
            + df_cultures["Root Production (ktC)"]
            * df_cultures["Root Humification Coefficient (%)"]
            / 100
        ).to_dict()
        target = {"soil stock": 1}

        flux_generator.generate_flux(source, target)

        source = (
            df_cultures["Residue Production (ktC)"]
            * (100 - df_cultures["Residue Humification Coefficient (%)"])
            / 100
            + df_cultures["Root Production (ktC)"]
            * (100 - df_cultures["Root Humification Coefficient (%)"])
            / 100
        ).to_dict()
        target = {"atmospheric CO2": 1}

        flux_generator.generate_flux(source, target)

        self.df_elevage = df_elevage
        self.df_excr = df_excr
        self.df_cultures = df_cultures
        self.df_pop = df_pop
        self.df_prod = df_prod

    def get_transition_matrix(self):
        """
        Returns the full nitrogen transition matrix.

        This matrix represents all nitrogen fluxes between sectors, including core and external processes.

        :return: A 2D NumPy array representing nitrogen fluxes between all sectors.
        :rtype: numpy.ndarray
        """
        return self.carbon_matrix


# Exemple d'utilisation
# year = '2025'  # Année sous forme de chaîne de caractères
# region = 'Europe'  # Exemple de région

# Création de l'objet FluxGenerator
# flux_modif = FluxModif(year, region)

# Visualiser la matrice des flux modifiés sous forme de heatmap (flux modifiés)
# flux_map = Fluxmap(flux_modif)
# flux_map.plot_heatmap()

# Flux de carbone liés à la volatilisation de CH4 et CO2 par les excréments

# def modify_flux_for_ch4_emissions(self):

#     ch4_df = pd.read_excel(
#         "./data/GRAFS_data.xlsx",
#         sheet_name="CH4 emission",         # feuille qui contient la colonne "CH4 rate"
#         usecols=["Elevage", "Valeur"]  # adapte le nom de la colonne si besoin
#     )

#     ch4_rates = dict(zip(ch4_df["Elevage"], ch4_df["Valeur"]))


#     # Définir les groupes d'animaux et leurs sous-groupes (en ligne 1150 à 1193, avec un espace entre les groupes)
#     animal_groups = {
#         "bovines": ["Milking cows", "Suckler cows", "Heifer for milking herd renewal over 2 yrs old",
#                     "Heifer for suckler cows renewal over 2 yrs old", "Heifer for slaughter over 2 yrs old",
#                     "Males of milking type over 2 yrs old", "Males of butcher type over 2 yrs old",
#                     "Heifer for milking herd renewal between 1 and 2 yrs old", "Heifer for suckler cows renewal between 1 and 2 yrs old",
#                     "Heifer for slaughter between 1 and 2 yrs old", "Males of milking type between 1 and 2 yrs old",
#                     "Males of butcher type between 1 and 2 yrs old", "Veal calves", "Other females under 1 yr", "Other males under 1 yr"],
#         "ovines": ["ewe lambs", "Sheep", "other ovines (incl. rams)"],
#         "caprines": ["kid goats", "female goats", "Other caprines (including male goats)"],
#         "porcines": ["piglets", "young pigs between 20 and 50 kg", "Sows over 50 kg", "Boars over 50 kg ", "fattening pigs over 50 kg "],
#         "poultry": ["Laying hens for hatching eggs", "Laying hens for consumption eggs", "young hens", "chickens",
#                     "Duck for 'foie gras'", "Ducks for roasting", "Turkeys", "Gooses", "Guinea fowls", "quails", "mother rabbits"],
#         "equine": ["horses ", "donkeys etc"]
#     }

#     ch4_idx = self.haber_bosch_labels.index("CH4 naturel")
# for group, subgroups in animal_groups.items():
#     total_emissions = 0.0
#     for sub in subgroups:

#         subgroup_rows = self.df[(self.df["index_excel"] >= 1150) & (self.df["index_excel"] <= 1193) &
#                                 (self.df["nom"] == sub)]

#         # Récupérer la valeur de 'head' pour la région choisie, en sélectionnant la colonne correspondant à la région
#         if not subgroup_rows.empty:
#             heads = subgroup_rows[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur
#         # récupérer rate et coef LU
#         rate = ch4_rates.get(sub, 0.0)

#         # contribution du sous-groupe
#         total_emissions += heads * rate / 10**6

#         # écrire dans la matrice : source=grand groupe, target=CH4 naturel
#         src_idx = self.haber_bosch_labels.index(group)
#         self.carbon_matrix[src_idx, ch4_idx] = total_emissions


# def modify_flux_for_C02_emissions_animal(self):
#     """Modéliser les émissions de CO2 en fonction des excréments et des flux associés"""

#     betail_normalized = [animal.lower() for animal in betail]  # Liste des animaux en minuscules

#     # Calculer les émissions de CO2 pour chaque animal
#     for animal in betail:  # Parcourir chaque animal de la liste 'betail'

#         normalized_animal = animal.lower()

#         animal_idx = self.data_loader.labels.index(animal)

#         # 1. Calculer la somme des flux allant des cultures vers l'animal
#         C_ingered = 0
#         for j in range(self.data_loader.n):
#             if self.haber_bosch_labels[j] in cultures + prairies + legumineuses + grains + straws and self.carbon_matrix[j, animal_idx] != 0:
#                     C_ingered += self.carbon_matrix[ j , animal_idx]  # Ajouter le flux à la somme

#         # 2. Calculer la somme des flux allant de cet animal vers les cultures
#         C_excreted = 0
#         for j in range(self.data_loader.n):  # Parcourir chaque secteur cible
#             # Si la cible est une culture et que le flux est non nul
#             if self.haber_bosch_labels[j] in cultures + prairies + legumineuses + grains + straws and self.carbon_matrix[animal_idx, j] != 0:
#                 C_excreted += self.carbon_matrix[animal_idx, j]  # Ajouter le flux à la somme

#         # 3. Soustraire les émissions de CH4 pour cet animal
#         ch4_emissions = self.carbon_matrix[animal_idx, self.haber_bosch_labels.index("CH4 naturel")]

#         # 4. Calculer le carbone fixé dans l'animal

#         C_edible_excel = pd.read_excel(
#             "./data/GRAFS_data.xlsx",
#             sheet_name="Volatilisation",         # feuille qui contient la colonne "CH4 rate"
#             usecols=["Elevage", "%C edible"]  # adapte le nom de la colonne si besoin
#         )
#         C_edible = C_edible_excel.loc[C_edible_excel["Elevage"] == animal, "%C edible"].values[0]

#         C_non_edible_excel = pd.read_excel(
#             "./data/GRAFS_data.xlsx",
#             sheet_name="Volatilisation",         # feuille qui contient la colonne "CH4 rate"
#             usecols=["Elevage", "%C non edible"]  # adapte le nom de la colonne si besoin
#         )

#         C_non_edible = C_non_edible_excel.loc[C_non_edible_excel["Elevage"] == animal, "%C non edible"].values[0]

#         carcasse_tot = self.df[(self.df["index_excel"] >= 1017) & (self.df["index_excel"] <= 1022) &
#                                     (self.df["nom"] == animal)]


#         if not carcasse_tot.empty:
#             carcasse = carcasse_tot[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur


#         C_fixé = C_edible / 100 * carcasse + C_non_edible / 100 * carcasse

#         if animal == 'bovins' :
#             extra = self.df[(self.df["index_excel"] >= 1023) & (self.df["index_excel"] <= 1025) &
#                                     (self.df["nom"] == 'cow milk')]

#             if not extra.empty:
#                 lait = extra[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur

#             C_fixé += lait * 5.48/100 #pourcentage de carbone

#         if animal == 'poultry' :
#             extra = self.df[(self.df["index_excel"] >= 1023) & (self.df["index_excel"] <= 1025) &
#                                     (self.df["nom"] == 'eggs')]

#             if not extra.empty:
#                 oeuf = extra[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur

#             C_fixé += oeuf * 13 / 100

#         if animal == 'ovines' or animal == 'caprines':
#             extra = self.df[(self.df["index_excel"] >= 1023) & (self.df["index_excel"] <= 1025) &
#                                     (self.df["nom"] == 'sheep and goat milk')]

#             if not extra.empty:
#                 lait = extra[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur

#             C_fixé += lait/2 * 8.7 / 100


#         # 4. Calculer les émissions de CO2 (la différence entre C_excreted et la somme des flux et CH4)
#         CO2_emissions = C_ingered - C_excreted - C_fixé - ch4_emissions  # Formule des émissions de CO2
#         #print(CO2_emissions, total_flux_to_cultures, ch4_emissions)
#         # 5. Ajouter ces émissions de CO2 à la matrice de flux, si nécessaire
#         # Vous pouvez ajouter un flux de CO2 à une cible spécifique si besoin, par exemple vers "CO2 atmospheric"
#         co2_idx = self.haber_bosch_labels.index("CO2 atmospheric")
#         self.carbon_matrix[animal_idx, co2_idx] = CO2_emissions

# def modify_flux_for_C02_emissions_human(self):
#     """Modéliser les émissions de CO2 en fonction des excréments et des flux associés"""


#     # Calculer les émissions de CO2 pour chaque animal
#     for human in Pop:  # Parcourir chaque animal de la liste 'betail'


#         human_idx = self.data_loader.labels.index(human)

#         # 1. Calculer la somme des flux allant des cultures vers l'animal
#         C_ingered = 0
#         for j in range(self.data_loader.n):
#             if self.haber_bosch_labels[j] in cultures + prairies + legumineuses + grains + straws + betail and self.carbon_matrix[j, human_idx] != 0:
#                     C_ingered += self.carbon_matrix[ j , human_idx]  # Ajouter le flux à la somme

#         # 2. Calculer la somme des flux allant de cet animal vers les cultures
#         C_excreted = 0
#         for j in range(self.data_loader.n):  # Parcourir chaque secteur cible
#             # Si la cible est une culture et que le flux est non nul
#             if self.haber_bosch_labels[j] in cultures + prairies + legumineuses + grains + straws and self.carbon_matrix[human_idx, j] != 0:
#                 C_excreted += self.carbon_matrix[human_idx, j]  # Ajouter le flux à la somme

#         # 3. Soustraire les émissions de CH4 pour cet animal
#         ch4_emissions = self.carbon_matrix[_idx, self.haber_bosch_labels.index("CH4 naturel")]

#         # 4. Calculer le carbone fixé dans l'animal

#         C_edible_excel = pd.read_excel(
#             "./data/GRAFS_data.xlsx",
#             sheet_name="Volatilisation",         # feuille qui contient la colonne "CH4 rate"
#             usecols=["Elevage", "%C edible"]  # adapte le nom de la colonne si besoin
#         )
#         C_edible = C_edible_excel.loc[C_edible_excel["Elevage"] == animal, "%C edible"].values[0]

#         C_non_edible_excel = pd.read_excel(
#             "./data/GRAFS_data.xlsx",
#             sheet_name="Volatilisation",         # feuille qui contient la colonne "CH4 rate"
#             usecols=["Elevage", "%C non edible"]  # adapte le nom de la colonne si besoin
#         )

#         C_non_edible = C_non_edible_excel.loc[C_non_edible_excel["Elevage"] == animal, "%C non edible"].values[0]

#         carcasse_tot = self.df[(self.df["index_excel"] >= 1017) & (self.df["index_excel"] <= 1022) &
#                                     (self.df["nom"] == animal)]


#         if not carcasse_tot.empty:
#             carcasse = carcasse_tot[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur


#         C_fixé = C_edible / 100 * carcasse + C_non_edible / 100 * carcasse

#         if animal == 'bovins' :
#             extra = self.df[(self.df["index_excel"] >= 1023) & (self.df["index_excel"] <= 1025) &
#                                     (self.df["nom"] == 'cow milk')]

#             if not extra.empty:
#                 lait = extra[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur

#             C_fixé += lait * 5.48/100 #pourcentage de carbone

#         if animal == 'poultry' :
#             extra = self.df[(self.df["index_excel"] >= 1023) & (self.df["index_excel"] <= 1025) &
#                                     (self.df["nom"] == 'eggs')]

#             if not extra.empty:
#                 oeuf = extra[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur

#             C_fixé += oeuf * 13 / 100

#         if animal == 'ovines' or animal == 'caprines':
#             extra = self.df[(self.df["index_excel"] >= 1023) & (self.df["index_excel"] <= 1025) &
#                                     (self.df["nom"] == 'sheep and goat milk')]

#             if not extra.empty:
#                 lait = extra[self.region].values[0]  # Accéder à la colonne région et prendre la première valeur

#             C_fixé += lait/2 * 8.7 / 100


#         # 4. Calculer les émissions de CO2 (la différence entre C_excreted et la somme des flux et CH4)
#         CO2_emissions = C_ingered - C_excreted - C_fixé - ch4_emissions  # Formule des émissions de CO2
#         #print(CO2_emissions, total_flux_to_cultures, ch4_emissions)
#         # 5. Ajouter ces émissions de CO2 à la matrice de flux, si nécessaire
#         # Vous pouvez ajouter un flux de CO2 à une cible spécifique si besoin, par exemple vers "CO2 atmospheric"
#         co2_idx = self.haber_bosch_labels.index("CO2 atmospheric")
#         self.carbon_matrix[animal_idx, co2_idx] = CO2_emissions
