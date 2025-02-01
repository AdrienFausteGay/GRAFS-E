import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LogNorm
from pulp import LpContinuous, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum

# Afficher toutes les colonnes
pd.set_option("display.max_columns", None)

# Afficher toutes les lignes
pd.set_option("display.max_rows", None)

pd.set_option("future.no_silent_downcasting", True)

from grafs_e.donnees import *

# current_dir = Path.cwd()  # Répertoire de travail actuel
# file_path = current_dir.parent / "data/full_grafs.xlsx"
# sheets_dict = pd.read_excel(file_path, sheet_name=None)


class DataLoader:
    def __init__(self, year, region):
        file_path = os.path.dirname(__file__)
        self.sheets_dict = pd.read_excel(os.path.join(file_path, "data/full_grafs.xlsx"), sheet_name=None)
        self.year = year
        self.region = region
        self.df = self.pre_process_df()
        self.data = self.df[["nom", self.region, "index_excel"]]
        self.data_path = os.path.join(file_path, "data")

    def pre_process_df(self):
        df = self.sheets_dict["pvar" + self.year].copy()
        df.loc[df.index[0], "Primary data, parameters, pre-treatments "] = "nom"
        df.columns = df.iloc[0]
        df["index_excel"] = df.index + 2
        return df

    def get_import_feed(self):
        df = self.sheets_dict["GRAFS" + self.year].copy()
        df.columns = df.iloc[0]
        correct_region = {"Pyrénées occid": "Pyrénées occidentales", "Pyrénées Orient": "Pyrénées Orientales"}
        region = self.region
        if region in correct_region.keys():
            region = correct_region[region]
        return df[region].iloc[32]


class CultureData:
    def __init__(self, data_loader, categories_mapping):
        self.df = data_loader.df
        self.region = data_loader.region
        self.data_path = data_loader.data_path
        self.categories_mapping = categories_mapping
        self.df_cultures = self.create_culture_dataframe()

    def create_culture_dataframe(self):
        df = self.df
        region = self.region
        data_path = self.data_path

        # Extraire les données de surface
        surface_data = df[(df["index_excel"] >= 259) & (df["index_excel"] <= 294)][["nom", region]]
        surface_dict = surface_data.set_index("nom")[region].to_dict()
        surface_dict["Rice"] = surface_dict.pop("rice")
        surface_dict["Forage cabbages"] = surface_dict.pop("Forage cabbages & roots")

        # Extraire les données de production végétale
        vege_prod_data = df[(df["index_excel"] >= 183) & (df["index_excel"] <= 218)][["nom", region]]
        vege_prod_dict = vege_prod_data.set_index("nom")[region].to_dict()

        # Extraire les données de teneur en azote
        N_content_data = df[(df["index_excel"] >= 335) & (df["index_excel"] <= 370)][["nom", region]]
        N_content_dict = N_content_data.set_index("nom")[region].to_dict()

        Rendement_data = df[(df["index_excel"] >= 221) & (df["index_excel"] <= 256)][["nom", region]]
        Rendement_dict = Rendement_data.set_index("nom")[region].to_dict()
        Rendement_dict["Forage cabbages"] = Rendement_dict.pop("Forage cabbages & roots")

        # Extraire les taux de surface avec épendage
        epend = pd.read_excel(os.path.join(data_path, "GRAFS_data.xlsx"), usecols=[0, 1], sheet_name="Surf N org")
        epend = epend.set_index("Culture").to_dict()["Surface recevant N organique maîtrisable"]

        # Créer un DataFrame combiné
        combined_data = {
            "Surface": surface_dict,
            "Production végétale": vege_prod_dict,
            "Teneur en azote": N_content_dict,
            "Rendement": Rendement_dict,
            "Taux épendage": epend,
        }

        combined_df = pd.DataFrame(combined_data)

        # Ajouter la colonne 'catégories' en mappant les cultures sur leurs catégories
        combined_df["catégories"] = combined_df.index.map(self.categories_mapping)

        combined_df["Rendement"] = combined_df["Rendement"] * 10

        return combined_df


class ElevageData:
    def __init__(self, data_loader):
        self.data = data_loader.data
        self.region = data_loader.region
        self.data_path = data_loader.data_path
        self.df_elevage = self.create_elevage_dataframe()

    def create_elevage_dataframe(self):
        df = self.data
        region = self.region
        data_path = self.data_path

        def add_data(nom, ligne, delta, keys):
            # Extraire les données supplémentaires
            additional_data = df.loc[[ligne + i * delta - 2 for i in range(6)], ["nom", region]]
            additional_dict = dict(zip(keys, additional_data[region].values))
            # Ajouter les nouvelles données au DataFrame existant dans l'ordre
            for key, value in additional_dict.items():
                if value <= 10**-5:
                    value = 0
                combined_df.loc[key, nom] = value

        # Production animale, attention, contrairement au reste, ici on est en kton carcasse
        production_data = df[(df["index_excel"] >= 1017) & (df["index_excel"] <= 1022)][["nom", region]]
        production_dict = production_data.set_index("nom")[region].to_dict()

        gas_em = pd.read_excel(os.path.join(data_path, "GRAFS_data.xlsx"), sheet_name="Volatilisation").set_index(
            "Elevage"
        )

        combined_data = {"Production": production_dict}
        combined_df = pd.DataFrame(combined_data)

        combined_df = combined_df.join(gas_em, how="left")

        add_data("prop comestible", 1092, 12, ["bovines", "ovines", "porcines", "poultry", "equine"])
        combined_df.loc["caprines", "prop comestible"] = combined_df["prop comestible"]["ovines"]
        combined_df["prop comestible"] = combined_df["prop comestible"] / 100
        add_data("prop non comestible", 1093, 12, ["bovines", "ovines", "porcines", "poultry", "equine"])
        combined_df.loc["caprines", "prop non comestible"] = combined_df["prop non comestible"]["ovines"]
        combined_df["prop non comestible"] = combined_df["prop non comestible"] / 100

        combined_df = combined_df.fillna(0)
        return combined_df


class FluxGenerator:
    def __init__(self, labels, region, year):
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        self.n = len(self.labels)
        self.adjacency_matrix = np.zeros((self.n, self.n))
        self.region = region
        self.year = year

    def generate_flux(self, source, target):
        for source_label, source_value in source.items():
            source_index = self.label_to_index.get(source_label)
            if source_index is None:
                continue
            for target_label, target_value in target.items():
                coefficient = source_value * target_value
                target_index = self.label_to_index.get(target_label)
                if target_index is not None:
                    if coefficient > 10**-7:
                        self.adjacency_matrix[source_index, target_index] += coefficient
                else:
                    print(f"Problème avec {target_label}")

    def get_coef(self, source_label, target_label):
        source_index = self.label_to_index.get(source_label)
        target_index = self.label_to_index.get(target_label)
        if source_index is not None and target_index is not None:
            return self.adjacency_matrix[source_index][target_index]
        else:
            return None


class NitrogenFlowModel:
    def __init__(
        self, data, year, region, categories_mapping, labels, cultures, legumineuses, prairies, betail, Pop, ext
    ):
        self.year = year
        self.region = region
        self.categories_mapping = categories_mapping
        self.labels = labels
        self.cultures = cultures
        self.legumineuses = legumineuses
        self.prairies = prairies
        self.betail = betail
        self.Pop = Pop
        self.ext = ext

        self.data_loader = data  # DataLoader(year, region)
        self.culture_data = CultureData(self.data_loader, categories_mapping)
        self.elevage_data = ElevageData(self.data_loader)
        self.flux_generator = FluxGenerator(labels, region, year)

        self.df_cultures = self.culture_data.df_cultures
        self.df_elevage = self.elevage_data.df_elevage
        self.adjacency_matrix = self.flux_generator.adjacency_matrix
        self.label_to_index = self.flux_generator.label_to_index

        self.compute_fluxes()

    def plot_heatmap(self):
        plt.figure(figsize=(10, 12), dpi=500)
        ax = plt.gca()

        # Créer la heatmap sans grille pour le moment
        norm = LogNorm(vmin=10**-4, vmax=self.adjacency_matrix.max())
        sns.heatmap(
            self.adjacency_matrix,
            xticklabels=range(1, len(self.labels)),
            yticklabels=range(1, len(self.labels)),
            cmap="plasma_r",
            annot=False,
            norm=norm,
            ax=ax,
            cbar_kws={"label": "ktN/year", "orientation": "horizontal", "pad": 0.02},
        )

        # Ajouter la grille en gris clair
        ax.grid(True, color="lightgray", linestyle="-", linewidth=0.5)

        # Déplacer les labels de l'axe x en haut
        ax.xaxis.set_ticks_position("top")  # Placer les ticks en haut
        ax.xaxis.set_label_position("top")  # Placer le label en haut

        # Rotation des labels de l'axe x
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        # Assurer que les axes sont égaux
        ax.set_aspect("equal", adjustable="box")
        # Ajouter des labels et un titre
        plt.xlabel("Target", fontsize=14, fontweight="bold")
        plt.ylabel("Source", fontsize=14, fontweight="bold")
        # plt.title(f'Heatmap of adjacency matrix for {region} in {year}')

        legend_labels = [f"{i + 1}: {label}" for i, label in enumerate(self.labels)]
        for i, label in enumerate(legend_labels):
            ax.text(
                1.05,
                1 - 1.1 * (i + 0.5) / len(legend_labels),
                label,
                transform=ax.transAxes,
                fontsize=10,
                va="center",
                ha="left",
                color="black",
                verticalalignment="center",
                horizontalalignment="left",
                linespacing=20,
            )  # Augmenter l'espacement entre les lignes

        # plt.subplots_adjust(bottom=0.2, right=0.85)  # Réduire l'espace vertical entre la heatmap et la colorbar
        # Afficher la heatmap
        plt.show()

    def plot_heatmap_interactive(self):
        """
        Génére une heatmap interactive (Plotly) :
        - Échelle 'log' simulée via log10(z).
        - Colorbar horizontale en bas.
        - Légende index -> label à droite sans chevauchement.
        - Axe X en haut et titre centré.
        """

        # 1) Préparation des labels numériques
        x_labels = list(range(1, len(self.labels) + 1))
        y_labels = list(range(1, len(self.labels) + 1))

        # Si vous ignorez la dernière ligne/colonne comme dans votre code :
        adjacency_subset = self.adjacency_matrix[: len(self.labels), : len(self.labels)]

        # 2) Gestion min/max et transformation log10
        cmin = max(1e-4, np.min(adjacency_subset[adjacency_subset > 0]))
        cmax = 100  # np.max(adjacency_subset)
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
        ticktext = [10**x for x in range(-4, 3, 1)]  # [f"{10**v:.2e}" for v in tickvals]
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
            width=1980,
            height=800,
            margin=dict(t=0, b=0, l=0, r=150),  # espace à droite pour la légende
        )

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
        legend_text = "<br>".join(f"{i + 1} : {lbl}" for i, lbl in enumerate(self.labels))
        fig.add_annotation(
            x=1.3,  # un peu à droite
            y=0.5,  # centré en hauteur
            xref="paper",
            yref="paper",
            showarrow=False,
            text=legend_text,
            align="left",
            valign="middle",
            font=dict(size=9),
            bordercolor="rgba(0,0,0,0)",
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(0,0,0,0)",
        )

        return fig

    def compute_fluxes(self):
        # Extraire les variables nécessaires
        df_cultures = self.df_cultures
        df_elevage = self.df_elevage
        adjacency_matrix = self.adjacency_matrix
        label_to_index = self.label_to_index
        region = self.region
        data_loader = self.data_loader
        flux_generator = self.flux_generator
        data = data_loader.data
        year = self.year

        # Calcul de l'azote disponible pour les cultures
        df_cultures["Azote disponible"] = df_cultures["Production végétale"] * df_cultures["Teneur en azote"] / 100

        # Gestion du cas particulier pour 'Straw'
        cereales = ["Wheat", "Rye", "Barley", "Oat", "Grain maize", "Other cereals"]
        somme_azote_produit_cereales = df_cultures["Azote disponible"][cereales].sum()
        somme_surface_cereales = df_cultures["Surface"][cereales].sum()
        df_cultures.loc["Straw", "Surface"] = (
            somme_surface_cereales * df_cultures.loc["Straw", "Azote disponible"] / somme_azote_produit_cereales
        )
        for cereal in cereales:
            df_cultures.loc[cereal, "Surface"] -= (
                df_cultures.loc["Straw", "Surface"] * df_cultures.loc[cereal, "Surface"] / somme_surface_cereales
            )
        df_cultures.loc["Straw", "Rendement"] = (
            df_cultures["Production végétale"]["Straw"] / df_cultures["Surface"]["Straw"] * 1000
        )

        # Flux depuis 'other sectors' vers les cibles sélectionnées
        selected_data = data[(data["index_excel"] >= 106) & (data["index_excel"] <= 139)]
        target = selected_data.set_index("nom")[region].to_dict()
        source = {"other sectors": 1}
        flux_generator.generate_flux(source, target)

        # Dépôt atmosphérique
        coef_surf = data[data["index_excel"] == 41][region].item()
        # Dépôt sur les prairies
        target_prairies = df_cultures.loc[
            df_cultures.index.isin(["Natural meadow ", "Non-legume temporary meadow", "Alfalfa and clover"]), "Surface"
        ].to_dict()
        source_atmosphere = {"atmospheric N2": coef_surf / 1e6}
        flux_generator.generate_flux(source_atmosphere, target_prairies)

        # Dépôt sur les terres arables
        Surf_reel = data.loc[data["index_excel"] == 23, region].item()
        Surf = df_cultures.loc[
            ~df_cultures.index.isin(["Natural meadow ", "Non-legume temporary meadow", "Alfalfa and clover"]), "Surface"
        ].sum()
        target_arable = (
            df_cultures.loc[
                ~df_cultures.index.isin(["Natural meadow ", "Non-legume temporary meadow", "Alfalfa and clover"]),
                "Surface",
            ]
            * Surf_reel
            / Surf
        ).to_dict()
        flux_generator.generate_flux(source_atmosphere, target_arable)

        # Fixation symbiotique
        selected_data = data[(data["index_excel"] >= 36) & (data["index_excel"] <= 38)]
        coefficients = selected_data.set_index("nom")[region].to_dict()
        target_fixation = {}
        for culture in df_cultures.index:
            if culture in self.legumineuses + ["Alfalfa and clover", "Natural meadow "]:
                if culture == "Natural meadow ":
                    coefficient = coefficients["N fixation coef for perm grassland"]
                elif culture == "Alfalfa and clover":
                    coefficient = coefficients["N fixation coef fodder for cropland"]
                else:
                    coefficient = coefficients["N fixation coef grain for cropland"]

                vege_prods = df_cultures.at[culture, "Production végétale"]
                teneur_en_azote = df_cultures.at[culture, "Teneur en azote"]
                target_fixation[culture] = vege_prods * teneur_en_azote * coefficient / 100

        source_fixation = {"atmospheric N2": 1}
        flux_generator.generate_flux(source_fixation, target_fixation)
        df_cultures["Fixation symbiotique"] = df_cultures.index.map(target_fixation).fillna(0)

        # Épandage de boue sur les champs
        FE_N_N02_em = 0.002
        FE_N_NH3_em = 0.118
        FE_N_N2_em = 0.425
        pop = data[data["index_excel"] == 5][region].item()
        N_boue = (
            data[data["nom"] == "Total per capita protein ingestion"][region].item() * pop
        )  # data[data["nom"] == "N Sludges to cropland"][region].item()
        prop_urb = data[data["nom"] == "Urban population"][region].item() / 100
        prop_recy_urb = data[data["nom"] == "N recycling rate of human excretion in urban area"][region].item() / 100
        prop_recy_rur = data[data["nom"] == "N recycling rate of human excretion in rural area"][region].item() / 100

        Norm = sum(df_cultures["Surface"] * df_cultures["Taux épendage"])
        # Création du dictionnaire target
        target_ependage = {
            culture: row["Surface"] * row["Taux épendage"] / Norm for culture, row in df_cultures.iterrows()
        }

        source_boue = {"urban": N_boue * prop_urb * prop_recy_urb, "rural": N_boue * (1 - prop_urb) * prop_recy_rur}

        flux_generator.generate_flux(source_boue, target_ependage)

        # Le reste est perdu dans l'environnement
        # Ajouter les fuites de NO2
        source = {"urban": N_boue * prop_urb * FE_N_N02_em, "rural": N_boue * (1 - prop_urb) * FE_N_N02_em}
        target = {"N2O emission": 1}
        flux_generator.generate_flux(source, target)

        # Ajouter les fuites de NH3
        source = {"urban": N_boue * prop_urb * FE_N_NH3_em, "rural": N_boue * (1 - prop_urb) * FE_N_NH3_em}
        target = {"NH3 volatilization": 1}
        flux_generator.generate_flux(source, target)

        # Ajouter les fuites de N2
        source = {"urban": N_boue * prop_urb * FE_N_N2_em, "rural": N_boue * (1 - prop_urb) * FE_N_N2_em}
        target = {"atmospheric N2": 1}
        flux_generator.generate_flux(source, target)

        # Le reste est perdu dans l'hydroshere
        target = {"hydro-system": 1}
        source = {
            "urban": N_boue * prop_urb * (1 - prop_recy_urb - FE_N_N02_em - FE_N_NH3_em - FE_N_N2_em),
            "rural": N_boue * (1 - prop_urb) * (1 - prop_recy_rur - FE_N_NH3_em - FE_N_N02_em),
        }
        # Remplir la matrice d'adjacence
        flux_generator.generate_flux(source, target)

        # Azote excrété sur prairies
        # Production d'azote commestible

        df_elevage["Azote comestible"] = df_elevage["Production"] * df_elevage["prop comestible"]
        df_elevage.loc["poultry", "Azote comestible"] += (
            data[data["index_excel"] == 1023][region].item() * data[data["index_excel"] == 1067][region].item() / 100
        )  # ajout des oeufs
        df_elevage.loc["bovines", "Azote comestible"] += (
            data[data["index_excel"] == 1024][region].item() * data[data["index_excel"] == 1068][region].item() / 100
        )  # ajout du lait de vache

        # Plus délicat pour les ovins/caprins car la production de lait est mélangée
        tete_ovins_femelle = data[data["index_excel"] == 1171][region].item()
        tete_caprins_femelle = data[data["index_excel"] == 1167][region].item()
        production_par_tete_caprins = 1000  # kg/tete vu sur internet
        production_par_tete_ovins = 300  # kg/tete vu sur internet
        df_elevage.loc["ovines", "Azote comestible"] += (
            data[data["index_excel"] == 1025][region].item()
            * data[data["index_excel"] == 1069][region].item()
            / 100
            * production_par_tete_ovins
            * tete_ovins_femelle
            / (production_par_tete_ovins * tete_ovins_femelle + production_par_tete_caprins * tete_caprins_femelle)
        )  # ajout du lait de brebis
        df_elevage.loc["caprines", "Azote comestible"] += (
            data[data["index_excel"] == 1025][region].item()
            * data[data["index_excel"] == 1069][region].item()
            / 100
            * production_par_tete_caprins
            * tete_caprins_femelle
            / (production_par_tete_ovins * tete_ovins_femelle + production_par_tete_caprins * tete_caprins_femelle)
        )  # ajout du lait de brebis

        df_elevage["Azote non comestible"] = df_elevage["Production"] * df_elevage["prop non comestible"]

        index = [1241 + j for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)]
        selected_data.loc[:, "nom"] = selected_data["nom"].apply(lambda x: x.split()[0])
        selected_data = selected_data.groupby("nom").agg({region: "sum", "index_excel": "first"}).reset_index()

        df_elevage["azote excrete"] = selected_data.set_index("nom")[region]
        df_elevage["Ingestion"] = (
            df_elevage["azote excrete"] + df_elevage["Azote comestible"] + df_elevage["Azote non comestible"]
        )

        index = [1250 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excretion sur prairie"] = selected_data

        index = [1251 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excretion en intérieur"] = selected_data

        index = [1252 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excretion en intérieur comme lisier"] = selected_data

        # On ajouter la catégorie other manure dans la catégorie liter manure
        index = [1253 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excretion en intérieur comme lisier"] += selected_data

        index = [1254 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excretion en intérieur comme fumier"] = selected_data

        # Calculer les poids pour chaque cible
        # Calcul de la surface totale pour les prairies
        total_surface = (
            df_cultures.loc["Alfalfa and clover", "Surface"]
            + df_cultures.loc["Non-legume temporary meadow", "Surface"]
            + df_cultures.loc["Natural meadow ", "Surface"]
        )

        # Création du dictionnaire target
        target = {
            "Alfalfa and clover": df_cultures.loc["Alfalfa and clover", "Surface"] / total_surface,
            "Non-legume temporary meadow": df_cultures.loc["Non-legume temporary meadow", "Surface"] / total_surface,
            "Natural meadow ": df_cultures.loc["Natural meadow ", "Surface"] / total_surface,
        }
        source = (
            df_elevage["azote excrete"]
            * df_elevage["% excretion sur prairie"]
            / 100
            * (1 - df_elevage["N-NH3 EM. outdoor"] - df_elevage["N-N2O EM. outdoor"] - df_elevage["N-N2 EM. outdoor"])
        ).to_dict()

        flux_generator.generate_flux(source, target)

        # Le reste est émit dans l'atmosphere
        # N2
        target = {"atmospheric N2": 1}
        source = (
            df_elevage["azote excrete"] * df_elevage["% excretion sur prairie"] / 100 * df_elevage["N-N2 EM. outdoor"]
        ).to_dict()

        flux_generator.generate_flux(source, target)

        # NH3
        # 1 % est volatilisée de manière indirecte sous forme de N2O
        target = {"NH3 volatilization": 0.99}
        source = (
            df_elevage["azote excrete"] * df_elevage["% excretion sur prairie"] / 100 * df_elevage["N-NH3 EM. outdoor"]
        ).to_dict()

        flux_generator.generate_flux(source, target)

        volat_N2O = (
            0.01
            * df_elevage["azote excrete"]
            * df_elevage["% excretion sur prairie"]
            / 100
            * df_elevage["N-NH3 EM. outdoor"]
        )
        # N2O
        target = {"N2O emission": 1}
        source = (
            volat_N2O
            + df_elevage["azote excrete"]
            * df_elevage["% excretion sur prairie"]
            / 100
            * df_elevage["N-N2O EM. outdoor"]
        ).to_dict()

        flux_generator.generate_flux(source, target)

        ## Epandage sur champs

        source = (
            df_elevage["azote excrete"]
            * df_elevage["% excretion en intérieur"]
            / 100
            * (
                df_elevage["% excretion en intérieur comme lisier"]
                / 100
                * (
                    1
                    - df_elevage["N-NH3 EM. manure indoor"]
                    - df_elevage["N-N2O EM. manure indoor"]
                    - df_elevage["N-N2 EM. manure indoor"]
                )
                + df_elevage["% excretion en intérieur comme fumier"]
                / 100
                * (
                    1
                    - df_elevage["N-NH3 EM. slurry indoor"]
                    - df_elevage["N-N2O EM. slurry indoor"]
                    - df_elevage["N-N2 EM. slurry indoor"]
                )
            )
        ).to_dict()

        flux_generator.generate_flux(source, target_ependage)

        # Le reste part dans l'atmosphere

        # N2
        target = {"atmospheric N2": 1}
        source = (
            df_elevage["azote excrete"]
            * df_elevage["% excretion en intérieur"]
            / 100
            * (
                df_elevage["% excretion en intérieur comme lisier"] / 100 * df_elevage["N-N2 EM. manure indoor"]
                + df_elevage["% excretion en intérieur comme fumier"] / 100 * df_elevage["N-N2 EM. slurry indoor"]
            )
        ).to_dict()

        flux_generator.generate_flux(source, target)

        # NH3
        # 1 % est volatilisée de manière indirecte sous forme de N2O
        target = {"NH3 volatilization": 0.99}
        source = (
            df_elevage["azote excrete"]
            * df_elevage["% excretion en intérieur"]
            / 100
            * (
                df_elevage["% excretion en intérieur comme lisier"] / 100 * df_elevage["N-NH3 EM. manure indoor"]
                + df_elevage["% excretion en intérieur comme fumier"] / 100 * df_elevage["N-NH3 EM. slurry indoor"]
            )
        ).to_dict()

        flux_generator.generate_flux(source, target)

        volat_N2O = (
            0.01
            * df_elevage["azote excrete"]
            * df_elevage["% excretion en intérieur"]
            / 100
            * (
                df_elevage["% excretion en intérieur comme lisier"] / 100 * df_elevage["N-NH3 EM. manure indoor"]
                + df_elevage["% excretion en intérieur comme fumier"] / 100 * df_elevage["N-NH3 EM. slurry indoor"]
            )
        )
        # N2O
        target = {"N2O emission": 1}
        source = (
            volat_N2O
            + df_elevage["azote excrete"]
            * df_elevage["% excretion en intérieur"]
            / 100
            * (
                df_elevage["% excretion en intérieur comme lisier"] / 100 * df_elevage["N-N2O EM. manure indoor"]
                + df_elevage["% excretion en intérieur comme fumier"] / 100 * df_elevage["N-N2O EM. slurry indoor"]
            )
        ).to_dict()

        flux_generator.generate_flux(source, target)

        # Azote synthétique
        # Calcul des besoins en azote par culture
        besoin_azote = {
            "Wheat": 3.5,
            "Rye": 2.3,
            "Barley": 2.5,
            "Oat": 2.2,
            "Grain maize": 2.2,
            "Other cereals": 2.6,
            "Straw": 3,
            "Rapeseed": 7,
            "Sunflower": 4.5,
            "Other oil crops": 5,
            "Forage maize": 1.3,
        }
        df_cultures["Besoin en azote"] = df_cultures.index.map(besoin_azote)
        df_cultures["Besoin total en azote/ha"] = df_cultures["Besoin en azote"] * df_cultures["Rendement"]

        # Fixer manuellement les besoins pour certaines cultures
        df_cultures.loc["Sugar beet", "Besoin total en azote/ha"] = 220
        df_cultures.loc["Potatoes", "Besoin total en azote/ha"] = 220
        df_cultures.loc["Other roots", "Besoin total en azote/ha"] = 220
        df_cultures.loc["Dry vegetables", "Besoin total en azote/ha"] = 50
        df_cultures.loc["Dry fruits", "Besoin total en azote/ha"] = 100
        df_cultures.loc["Squash and melons", "Besoin total en azote/ha"] = 180
        df_cultures.loc["Cabbage", "Besoin total en azote/ha"] = 300
        df_cultures.loc["Leaves vegetables", "Besoin total en azote/ha"] = 150
        df_cultures.loc["Fruits", "Besoin total en azote/ha"] = 100
        df_cultures.loc["Olives", "Besoin total en azote/ha"] = 80
        df_cultures.loc["Citrus", "Besoin total en azote/ha"] = 130
        df_cultures.loc["Hemp", "Besoin total en azote/ha"] = 120
        df_cultures.loc["Flax", "Besoin total en azote/ha"] = 60
        df_cultures.loc["Non-legume temporary meadow", "Besoin total en azote/ha"] = 150
        df_cultures.loc["Natural meadow ", "Besoin total en azote/ha"] = 150
        df_cultures.loc["Rice", "Besoin total en azote/ha"] = 125
        df_cultures.loc["Forage cabbages", "Besoin total en azote/ha"] = 300

        df_cultures = df_cultures.fillna(0)

        # Calcul de l'azote épendu par hectare
        def calculer_azote_ependu(culture):
            sources = self.betail + self.Pop + ["atmospheric N2", "other sectors"]
            adj_matrix_df = pd.DataFrame(adjacency_matrix, index=self.labels, columns=self.labels)
            return adj_matrix_df.loc[sources, culture].sum()

        df_cultures["Azote épendu"] = df_cultures.index.map(calculer_azote_ependu)
        df_cultures["Azote épendu/ha"] = df_cultures.apply(
            lambda row: row["Azote épendu"] / row["Surface"] * 10**6
            if row["Surface"] > 0 and row["Azote épendu"] > 0
            else 0,
            axis=1,
        )

        # Mécanisme d'héritage de l'azote en surplus des légumineuses
        df_cultures["Surplus Azote leg"] = 0.0
        df_cultures.loc[self.legumineuses, "Surplus Azote leg"] = (
            df_cultures.loc[self.legumineuses, "Azote épendu"] - df_cultures.loc[self.legumineuses, "Azote disponible"]
        )

        # Distribution du surplus aux céréales
        total_surplus_azote = df_cultures.loc[self.legumineuses, "Surplus Azote leg"].sum()
        total_surface_cereales = df_cultures.loc[
            (
                (df_cultures["catégories"] == "cereals (excluding rice)")
                | (df_cultures.index.isin(["Straw", "Forage maize"]))
            ),
            "Surface",
        ].sum()
        df_cultures["heritage legumineuse"] = 0.0
        df_cultures.loc[
            (
                (df_cultures["catégories"] == "cereals (excluding rice)")
                | (df_cultures.index.isin(["Straw", "Forage maize"]))
            ),
            "heritage legumineuse",
        ] = (
            df_cultures.loc[
                (
                    (df_cultures["catégories"] == "cereals (excluding rice)")
                    | (df_cultures.index.isin(["Straw", "Forage maize"]))
                ),
                "Surface",
            ]
            / total_surface_cereales
            * total_surplus_azote
        )

        # Génération des flux pour l'héritage des légumineuses
        source_leg = (
            df_cultures.loc[df_cultures["Surplus Azote leg"] > 0]["Surplus Azote leg"]
            / df_cultures["Surplus Azote leg"].sum()
        ).to_dict()
        target_leg = df_cultures["heritage legumineuse"].to_dict()
        flux_generator.generate_flux(source_leg, target_leg)

        # Calcul de l'azote à épendre
        df_cultures["Azote à épendre/ha"] = df_cultures.apply(
            lambda row: row["Besoin total en azote/ha"]
            - row["Azote épendu/ha"]
            - (row["heritage legumineuse"] / row["Surface"] * 1e6)
            if row["Surface"] > 0
            else row["Besoin total en azote/ha"] - row["Azote épendu/ha"],
            axis=1,
        )
        df_cultures["Azote à épendre/ha"] = df_cultures["Azote à épendre/ha"].apply(lambda x: max(x, 0))
        df_cultures["Azote à épendre (ktN)"] = df_cultures["Azote à épendre/ha"] * df_cultures["Surface"] / 1e6

        # Calcul de la quantité moyenne (kgN) d'azote synthétique épendu par hectare
        # Séparer les données en prairies et champs
        df_prairies = df_cultures[df_cultures.index.isin(prairies)].copy()
        df_champs = df_cultures[df_cultures.index.isin(cultures)].copy()

        moyenne_ponderee_prairies = (
            df_prairies["Azote à épendre/ha"] * df_prairies["Surface"]
        ).sum()  # / df_prairies['Surface'].sum()
        moyenne_ponderee_champs = (
            df_champs["Azote à épendre/ha"] * df_champs["Surface"]
        ).sum()  # / df_champs['Surface'].sum()

        moyenne_reel_champs = (
            data[data["index_excel"] == 27][region].item() * data[data["index_excel"] == 23][region].item()
        )
        moyenne_reel_prairies = (
            data[data["index_excel"] == 29][region].item() * data[data["index_excel"] == 22][region].item() / 10**6
        )

        df_prairies.loc[:, "Azote à épendre (ktN) corrigé"] = moyenne_reel_prairies
        df_champs.loc[:, "Azote à épendre (ktN) corrigé"] = (
            df_champs["Azote à épendre (ktN)"] * moyenne_reel_champs / moyenne_ponderee_champs
        )

        # Mise à jour de df_cultures

        df_cultures["Azote à épendre (ktN) corrigé"] = (
            df_champs["Azote à épendre (ktN) corrigé"]
            .combine_first(df_prairies["Azote à épendre (ktN) corrigé"])
            .reindex(df_cultures.index, fill_value=0)  # Remplissage des clés manquantes (les légumineuses) avec 0
        )

        ## Azote synthétique volatilisé par les terres
        # Est ce qu'il n'y a que l'azote synthétique qui est volatilisé ?
        coef_volat_NH3 = data[data["index_excel"] == 31][region].item() / 100
        coef_volat_N2O = 0.01

        # 1 % des emissions de NH3 du aux fert. synth sont volatilisées sous forme de N2O
        df_cultures["Azote Volatilisé N-NH3 (ktN)"] = (
            df_cultures["Azote à épendre (ktN) corrigé"] * 0.99 * coef_volat_NH3
        )
        df_cultures["Azote émit N-N2O (ktN)"] = df_cultures["Azote à épendre (ktN) corrigé"] * (
            coef_volat_N2O + 0.01 * coef_volat_NH3
        )
        df_cultures["Azote à épendre (ktN) corrigé"] = df_cultures["Azote à épendre (ktN) corrigé"] * (
            1 - coef_volat_NH3 - coef_volat_N2O
        )
        # La quantité d'azote réellement épendue est donc un peu plus faible car une partie est volatilisée

        source = {"Haber-Bosch": 1}
        target = df_cultures["Azote à épendre (ktN) corrigé"].to_dict()

        flux_generator.generate_flux(source, target)

        source = df_cultures["Azote Volatilisé N-NH3 (ktN)"].to_dict()
        target = {"NH3 volatilization": 1}

        flux_generator.generate_flux(source, target)

        source = df_cultures["Azote émit N-N2O (ktN)"].to_dict()
        target = {"N2O emission": 1}

        flux_generator.generate_flux(source, target)

        # A cela on ajoute les emissions indirectes de N2O lors de la fabrication des engrais
        epend_tot_synt = (
            df_cultures["Azote Volatilisé N-NH3 (ktN)"]
            + df_cultures["Azote émit N-N2O (ktN)"]
            + df_cultures["Azote à épendre (ktN) corrigé"]
        ).sum()
        coef_emis_N_N2O = data[data["index_excel"] == 32][region].item()
        target = {"N2O emission": 1}
        source = {"Haber-Bosch": epend_tot_synt * coef_emis_N_N2O}

        flux_generator.generate_flux(source, target)

        # Azote issu de la partie non comestible des carcasses
        source_non_comestible = df_elevage["Azote non comestible"].to_dict()
        target_other_sectors = {"other sectors": 1}
        flux_generator.generate_flux(source_non_comestible, target_other_sectors)

        # On va chercher les éventuelles corrections apportées par JLN (=0 si export, donc pas vraiment net...)
        import_feed_net = self.data_loader.get_import_feed()
        # Et la valeur net
        import_feed_net_tot = data[data["index_excel"] == 1009][region].item()

        df_elevage_comp = df_elevage.copy()
        df_elevage = df_elevage.loc[df_elevage["Ingestion"] > 10**-8]

        supp_export = 0
        if import_feed_net > df_elevage["Ingestion"].sum():
            supp_export = import_feed_net - df_elevage["Ingestion"].sum()  # On augmentera d'autant les exports
            import_feed_net = df_elevage["Ingestion"].sum()

        if len(df_elevage) > 0:
            # Dictionnaire enregistrant toutes les cultures présentes dans le régime d'un élevage
            all_cultures_regime_elevage = {}
            for elevage in df_elevage.index:
                cultures_name = set()
                for cultures_liste in regime_elevage[elevage].values():
                    cultures_name.update(cultures_liste)
                all_cultures_regime_elevage[elevage] = cultures_name

            # Initialisation du problème
            prob = LpProblem("Allocation_Azote_Animaux", LpMinimize)

            # Variables de décision pour les allocations
            x_vars = LpVariable.dicts(
                "x",
                [(culture, elevage) for culture in df_cultures.index for elevage in df_elevage.index],
                lowBound=0,
                cat="Continuous",
            )

            # Variable de depassement des importations
            E_vars = LpVariable.dicts(
                "E",
                [
                    (elevage, culture)
                    for elevage in df_elevage.index
                    for culture in all_cultures_regime_elevage[elevage]
                ],
                lowBound=0,
                cat="Continuous",
            )

            # Variables de déviation des régimes alimentaires
            delta_vars = LpVariable.dicts(
                "delta",
                [
                    (elevage, proportion)
                    for elevage in df_elevage.index
                    for proportion in regime_elevage[elevage].keys()
                ],
                lowBound=0,
                cat=LpContinuous,
            )

            # Variables de pénalité pour la concentration des allocations
            penalite_vars = LpVariable.dicts(
                "penalite",
                [(culture, elevage) for culture in df_cultures.index for elevage in df_elevage.index],
                lowBound=0,
                cat=LpContinuous,
            )

            # Variables de pénalité pour la distribution au sein des catégories
            penalite_culture_vars = LpVariable.dicts(
                "penalite_culture",
                [
                    (elevage, proportion, culture)
                    for elevage in df_elevage.index
                    for proportion in regime_elevage[elevage].keys()
                    for culture in regime_elevage[elevage][proportion]
                ],
                lowBound=0,
                cat=LpContinuous,
            )

            # Variables d'importation pour chaque élevage et catégorie
            I_vars = LpVariable.dicts(
                "I",
                [
                    (elevage, culture)
                    for elevage in df_elevage.index
                    for culture in all_cultures_regime_elevage[elevage]
                ],
                lowBound=0,
                cat="Continuous",
            )

            # Variables pour capturer les importations associées aux déviations négatives
            delta_import_vars = LpVariable.dicts(
                "delta_import",
                [
                    (elevage, proportion)
                    for elevage in df_elevage.index
                    for proportion in regime_elevage[elevage].keys()
                ],
                lowBound=0,
                cat=LpContinuous,
            )

            # Pondération pour le terme de pénalité
            poids_penalite_deviation = 10

            poids_penalite = 0  # Ajustez ce poids selon vos préférences

            # Poids pour le nouveau terme de pénalité
            poids_penalite_culture = 0.5  # À ajuster selon vos préférences

            # Définir un poids élevé pour pénaliser les importations
            if int(year) > 1960:
                poids_exces_import = 1
            else:
                poids_exces_import = 1000.0  # Ajustez ce poids selon vos préférences

            # poids pour évacuer un éventuel surplus d'import
            poids_export = 50  # Ajustez ce poids selon vos préférences

            poids_delta_import = (
                500.0  # Poids supplémentaire pour orienter les importations pour minimiser les fortes déviations
            )

            prob += (
                poids_penalite_deviation
                * lpSum(
                    delta_vars[(elevage, proportion)]
                    for elevage in df_elevage.index
                    for proportion in regime_elevage[elevage].keys()
                )
                + poids_delta_import
                * lpSum(
                    delta_import_vars[(elevage, proportion)]
                    for elevage in df_elevage.index
                    for proportion in regime_elevage[elevage].keys()
                )
                + poids_penalite
                * lpSum(
                    penalite_vars[(culture, elevage)] for culture in df_cultures.index for elevage in df_elevage.index
                )
                + poids_penalite_culture
                * lpSum(
                    penalite_culture_vars[(elevage, proportion, culture)]
                    for elevage in df_elevage.index
                    for proportion in regime_elevage[elevage].keys()
                    for culture in regime_elevage[elevage][proportion]
                )
                + poids_exces_import
                * lpSum(
                    E_vars[(elevage, culture)]
                    for elevage in df_elevage.index
                    for culture in all_cultures_regime_elevage[elevage]
                ),
                "Minimiser_Deviations_Penalties_Et_Excès_Importation",
            )

            for elevage in df_elevage.index:
                besoin = df_elevage.loc[elevage, "Ingestion"]
                prob += (
                    lpSum(x_vars[(culture, elevage)] for culture in df_cultures.index)
                    + lpSum(I_vars[(elevage, culture)] for culture in all_cultures_regime_elevage[elevage])
                    + lpSum(E_vars[(elevage, culture)] for culture in all_cultures_regime_elevage[elevage])
                    == besoin,
                    f"Besoin_{elevage}",
                )

            # Cette contrainte assure que la somme de l'azote alloué de chaque culture aux différents types d'élevage ne dépasse pas l'azote disponible pour cette culture.
            for culture in df_cultures.index:
                azote_disponible = df_cultures.loc[culture, "Azote disponible"]
                prob += (
                    lpSum(x_vars[(culture, elevage)] for elevage in df_elevage.index) <= azote_disponible,
                    f"Disponibilite_{culture}",
                )

            for elevage in df_elevage.index:
                cultures_autorisees = set()
                for cultures_liste in regime_elevage[elevage].values():
                    cultures_autorisees.update(cultures_liste)
                for culture in df_cultures.index:
                    if culture not in cultures_autorisees:
                        prob += x_vars[(culture, elevage)] == 0, f"Culture_Non_Autorisee_{culture}_{elevage}"
                        # Vérifier si la variable I_vars existe avant d'ajouter la contrainte
                        if (elevage, culture) in I_vars:
                            prob += I_vars[(elevage, culture)] == 0, f"Import_Non_Autorise_{elevage}_{culture}"
                        if (elevage, culture) in E_vars:
                            prob += (
                                E_vars[(elevage, culture)] == 0,
                                f"Import_excedentaire_Non_Autorise_{elevage}_{culture}",
                            )

            # Ces contraintes calculent les déviations entre les proportions effectives des catégories consommées par chaque élevage et les proportions initiales du régime alimentaire.
            for elevage in df_elevage.index:
                besoin = df_elevage.loc[elevage, "Ingestion"]
                for proportion_initiale, cultures_liste in regime_elevage[elevage].items():
                    # Azote total des cultures dans la liste
                    azote_cultures = lpSum(
                        x_vars[(culture, elevage)] for culture in cultures_liste if culture in df_cultures.index
                    ) + lpSum(I_vars[(elevage, culture)] for culture in cultures_liste)
                    proportion_effective = azote_cultures / besoin
                    # Déviation par rapport à la proportion initiale
                    delta_var = delta_vars[(elevage, proportion_initiale)]
                    prob += (
                        proportion_effective - proportion_initiale <= delta_var,
                        f"Deviation_Plus_{elevage}_{proportion_initiale}",
                    )
                    prob += (
                        proportion_initiale - proportion_effective <= delta_var,
                        f"Deviation_Moins_{elevage}_{proportion_initiale}",
                    )

            prob += (
                lpSum(
                    I_vars[(elevage, culture)]
                    for elevage in df_elevage.index
                    for culture in all_cultures_regime_elevage[elevage]
                )
                == import_feed_net,
                "Limite_Imports_Normaux",
            )

            # Calcul de l'allocation cible (par exemple, allocation uniforme)
            for culture in df_cultures.index:
                azote_disponible_culture = df_cultures.loc[culture, "Azote disponible"]
                allocation_cible = azote_disponible_culture / len(df_elevage.index)  # Allocation uniforme
                for elevage in df_elevage.index:
                    allocation_reelle = x_vars[(culture, elevage)]
                    # Pénalité est la valeur absolue de la différence entre l'allocation réelle et l'allocation cible
                    prob += (
                        allocation_reelle - allocation_cible <= penalite_vars[(culture, elevage)],
                        f"Penalite_Plus_{culture}_{elevage}",
                    )
                    prob += (
                        allocation_cible - allocation_reelle <= penalite_vars[(culture, elevage)],
                        f"Penalite_Moins_{culture}_{elevage}",
                    )

            # Pénaliser si on nourrit les animaux avec une seule culture dans chaque groupe de proportions
            for elevage in df_elevage.index:
                besoin = df_elevage.loc[elevage, "Ingestion"]
                for proportion, cultures_liste in regime_elevage[elevage].items():
                    # Allocation totale pour ce groupe de cultures
                    allocation_groupe = lpSum(
                        x_vars[(culture, elevage)] for culture in cultures_liste if culture in df_cultures.index
                    ) + lpSum(I_vars[(elevage, culture)] for culture in cultures_liste)
                    # Azote total disponible pour ce groupe de cultures
                    azote_total_groupe = df_cultures.loc[
                        df_cultures.index.isin(cultures_liste), "Azote disponible"
                    ].sum()
                    if azote_total_groupe > 0:
                        for culture in cultures_liste:
                            if culture in df_cultures.index:
                                azote_disponible_culture = df_cultures.loc[culture, "Azote disponible"]
                                # Calcul de l'allocation cible proportionnelle à la disponibilité
                                allocation_cible_culture = (
                                    azote_disponible_culture / azote_total_groupe
                                ) * allocation_groupe
                                # Allocation réelle
                                allocation_reelle_culture = x_vars[(culture, elevage)]
                                # Pénalités pour la déviation
                                prob += (
                                    allocation_reelle_culture - allocation_cible_culture
                                    <= penalite_culture_vars[(elevage, proportion, culture)],
                                    f"Penalite_Culture_Plus_{elevage}_{proportion}_{culture}",
                                )
                                prob += (
                                    allocation_cible_culture - allocation_reelle_culture
                                    <= penalite_culture_vars[(elevage, proportion, culture)],
                                    f"Penalite_Culture_Moins_{elevage}_{proportion}_{culture}",
                                )
                    else:
                        pass

            # Contrainte pour importer là où les déviations sont les plus importantes
            for elevage in df_elevage.index:
                for proportion, cultures_liste in regime_elevage[elevage].items():
                    # Total des importations pour cette proportion
                    azote_importe = lpSum(
                        I_vars[(elevage, culture)] + E_vars[(elevage, culture)]
                        for culture in cultures_liste
                        if culture in df_cultures.index
                    )
                    # Lier aux variables de déviation
                    prob += (
                        delta_import_vars[(elevage, proportion)]
                        >= azote_importe - delta_vars[(elevage, proportion)] * df_elevage.loc[elevage, "Ingestion"],
                        f"Delta_Import_Lien_{elevage}_{proportion}",
                    )

            # Résolution du problème
            prob.solve()

            # Vérification du statut de la solution
            # print("Status:", LpStatus[prob.status])

            if LpStatus[prob.status] == "Optimal":
                allocations = []
                for var in prob.variables():
                    if var.name.startswith("x") and var.varValue > 0:
                        # Nom de la variable : x_(culture, elevage)
                        chaine = str(var)
                        matches = re.findall(r"'([^']*)'", chaine)
                        parts = [match.replace("_", " ").strip() for match in matches]
                        culture = parts[0]
                        # Gestion du tiret dans le nom
                        if culture == "Non legume temporary meadow":
                            culture = "Non-legume temporary meadow"
                        if culture == "Natural meadow":
                            culture = "Natural meadow "
                        elevage = parts[1]
                        allocations.append(
                            {
                                "Culture": culture,
                                "Elevage": elevage,
                                "Azote_alloue": var.varValue,
                                "Type": "Culture Locale",
                            }
                        )
                    elif var.name.startswith("I") and var.varValue > 0:
                        # Nom de la variable : I_(elevage, culture)
                        chaine = str(var)
                        matches = re.findall(r"'([^']*)'", chaine)
                        parts = [match.replace("_", " ").strip() for match in matches]
                        elevage = parts[0]
                        culture = parts[1]
                        if culture == "Non legume temporary meadow":
                            culture = "Non-legume temporary meadow"
                        if culture == "Natural meadow":
                            culture = "Natural meadow "
                        allocations.append(
                            {
                                "Culture": culture,
                                "Elevage": elevage,
                                "Azote_alloue": var.varValue,
                                "Type": "Importation",
                            }
                        )

                    elif var.name.startswith("E") and var.varValue > 0:
                        # Nom de la variable : E_(elevage, culture)
                        chaine = str(var)
                        matches = re.findall(r"'([^']*)'", chaine)
                        parts = [match.replace("_", " ").strip() for match in matches]
                        elevage = parts[0]
                        culture = parts[1]
                        if culture == "Non legume temporary meadow":
                            culture = "Non-legume temporary meadow"
                        if culture == "Natural meadow":
                            culture = "Natural meadow "
                        allocations.append(
                            {
                                "Culture": culture,
                                "Elevage": elevage,
                                "Azote_alloue": var.varValue,
                                "Type": "Importation excedentaire",
                            }
                        )

                allocations_df = pd.DataFrame(allocations)

                # Filtrer les lignes en supprimant celles dont 'Azote_alloue' est très proche de zéro
                allocations_df = allocations_df[allocations_df["Azote_alloue"].abs() >= 1e-6]

                self.allocation_elevage = allocations_df

                # Extraction des déviations avec le signe
                deviations = []
                for elevage in df_elevage.index:
                    for proportion in regime_elevage[elevage].keys():
                        proportion_rounded = round(proportion, 5)
                        delta_var_key = (elevage, proportion_rounded)
                        deviation = delta_vars[delta_var_key].varValue
                        if deviation != 0:
                            # Récupérer la liste des cultures associées à cette proportion
                            cultures_liste = regime_elevage[elevage][proportion]
                            cultures_str = ", ".join(cultures_liste)

                            # Calcul de l'allocation totale (local et importée)
                            azote_cultures = (
                                sum(
                                    x_vars[(culture, elevage)].varValue
                                    for culture in cultures_liste
                                    if (culture, elevage) in x_vars
                                )
                                + sum(
                                    I_vars[(elevage, culture)].varValue
                                    for culture in cultures_liste
                                    if (elevage, culture) in I_vars
                                )
                                + sum(
                                    E_vars[(elevage, culture)].varValue
                                    for culture in cultures_liste
                                    if (elevage, culture) in E_vars
                                )
                            )
                            besoin_total = df_elevage.loc[elevage, "Ingestion"]

                            # Calcul de la proportion effective
                            proportion_effective = azote_cultures / besoin_total if besoin_total > 0 else 0

                            # Déterminer le signe
                            signe = 1 if proportion_effective > proportion else -1

                            deviations.append(
                                {
                                    "Élevage": elevage,
                                    "Proportion attendue (%)": proportion_rounded * 100,
                                    "Déviation (%)": signe * round(deviation, 4) * 100,  # Convertir en pourcentage
                                    "Cultures": cultures_str,
                                }
                            )
                deviations_df = pd.DataFrame(deviations)

                # Extraction des importations normales
                importations = []
                for elevage in df_elevage.index:
                    for culture in all_cultures_regime_elevage[elevage]:
                        if (elevage, culture) in I_vars:
                            import_value = I_vars[(elevage, culture)].varValue
                            if import_value > 0:
                                importations.append(
                                    {
                                        "Élevage": elevage,
                                        "Culture": culture,
                                        "Type": "Normal",
                                        "Azote importé": import_value,
                                    }
                                )

                # Extraction des imports excédentaires
                for elevage in df_elevage.index:
                    for culture in all_cultures_regime_elevage[elevage]:
                        if (elevage, culture) in E_vars:
                            excess_value = E_vars[(elevage, culture)].varValue
                            if excess_value > 0:
                                importations.append(
                                    {
                                        "Élevage": elevage,
                                        "Culture": culture,
                                        "Type": "Excédentaire",
                                        "Azote importé": excess_value,
                                    }
                                )

                # Convertir en DataFrame
                importations_df = pd.DataFrame(importations)

                # Calcul de la quantité d'azote importé non utilisée
                azote_importe_alloue = allocations_df[
                    allocations_df["Type"].isin(["Importation", "Importation excedentaire"])
                ]["Azote_alloue"].sum()

                # Mise à jour de df_cultures
                for idx, row in df_cultures.iterrows():
                    culture = row.name
                    azote_alloue = allocations_df[
                        (allocations_df["Culture"] == culture) & (allocations_df["Type"] == "Culture Locale")
                    ]["Azote_alloue"].sum()
                    df_cultures.loc[idx, "Azote disponible après feed"] = row["Azote disponible"] - azote_alloue
                    df_cultures.loc[idx, "Azote feed"] = azote_alloue
                # Correction des valeurs proches de zéro
                df_cultures["Azote disponible après feed"] = df_cultures["Azote disponible après feed"].apply(
                    lambda x: 0 if abs(x) < 1e-6 else x
                )

                # Mise à jour de df_elevage
                # Calcul de l'azote total alloué à chaque élevage
                azote_alloue_elevage = (
                    allocations_df.groupby(["Elevage", "Type"])["Azote_alloue"].sum().unstack(fill_value=0)
                )

                # Ajouter les colonnes d'azote alloué dans df_elevage
                df_elevage.loc[:, "Azote alloué cultures locales"] = df_elevage.index.map(
                    azote_alloue_elevage.get("Culture Locale", pd.Series(0, index=df_elevage.index))
                )
                df_elevage.loc[:, "Azote alloué importé"] = df_elevage.index.map(
                    lambda elevage: azote_alloue_elevage.get("Importation", pd.Series(0, index=df_elevage.index)).get(
                        elevage, 0
                    )
                    + azote_alloue_elevage.get("Importation excedentaire", pd.Series(0, index=df_elevage.index)).get(
                        elevage, 0
                    )
                )
                # df_elevage['Azote alloué importations'] = df_elevage.index.map(azote_alloue_elevage['Importation'])

                # Génération des flux pour les cultures locales
                allocations_locales = allocations_df[allocations_df["Type"] == "Culture Locale"]

                for elevage in df_elevage.index:
                    target = {elevage: 1}
                    source = (
                        allocations_locales[allocations_locales["Elevage"] == elevage]
                        .set_index("Culture")["Azote_alloue"]
                        .to_dict()
                    )
                    if source:
                        flux_generator.generate_flux(source, target)

                # Génération des flux pour les importations
                allocations_imports = allocations_df[
                    allocations_df["Type"].isin(["Importation", "Importation excedentaire"])
                ]

                for elevage in df_elevage.index:
                    target = {elevage: 1}
                    elevage_imports = allocations_imports[allocations_imports["Elevage"] == elevage]

                    # Initialisation d'un dictionnaire pour collecter les flux par catégorie
                    flux = {}

                    for _, row in elevage_imports.iterrows():
                        culture = row["Culture"]
                        azote_alloue = row["Azote_alloue"]

                        # Récupération de la catégorie de la culture
                        categorie = df_cultures.loc[culture, "catégories"]

                        # Construction du label source pour l'importation
                        label_source = f"{categorie} feed trade"

                        # Accumuler les flux par catégorie
                        if label_source in flux:
                            flux[label_source] += azote_alloue
                        else:
                            flux[label_source] = azote_alloue

                    # Génération des flux pour l'élevage
                    if sum(flux.values()) > 0:
                        flux_generator.generate_flux(flux, target)

                # Gestion de l'azote importé non utilisé
                unused_imports = import_feed_net_tot - import_feed_net

                if unused_imports > 10**-6:
                    # Répartition de l'azote importé inutilisé par catégorie
                    flux_unused = {}
                    for categorie in df_cultures["catégories"].unique():
                        # On exporte pas en feed des catégories dédiées aux humains
                        if categorie not in ["rice, fruits and vegetables", "roots"]:
                            label_source = f"{categorie} feed trade"

                            # Calculer la quantité inutilisée par catégorie proportionnellement aux catégories présentes dans df_cultures
                            total_per_categorie = df_cultures[df_cultures["catégories"] == categorie][
                                "Azote disponible après feed"
                            ].sum()

                            if total_per_categorie > 0:
                                flux_unused[label_source] = unused_imports * (
                                    total_per_categorie / df_cultures["Azote disponible après feed"].sum()
                                )

                    # Générer des flux pour rediriger les importations inutilisées vers leur catégorie d'origine
                    for label_source, azote_unused in flux_unused.items():
                        if azote_unused > 0:
                            target = {label_source: 1}
                            source = {label_source: azote_unused}
                            flux_generator.generate_flux(source, target)

        else:
            df_cultures["Azote disponible après feed"] = df_cultures["Azote disponible"]

        ## Exportation de feed

        # On ajoute aux exportations les importations supplémentaires calculées par l'optimisation
        # Pas sur sur de comment gérer ça...
        # On veut -export_feed + import_feed = import_feed_net_tot

        import_excedent_feed = lpSum(
            E_vars[(elevage, culture)].varValue
            for elevage in df_elevage.index
            for culture in all_cultures_regime_elevage[elevage]
        )

        # On redonne à df_elevage sa forme d'origine et à import_feed_net sa vraie valeur
        # Utiliser `infer_objects(copy=False)` pour éviter l'avertissement sur le downcasting
        df_elevage = df_elevage.combine_first(df_elevage_comp)

        # Remplir les valeurs manquantes avec 0
        df_elevage = df_elevage.fillna(0)

        # Inférer les types pour éviter le warning sur les colonnes object
        df_elevage = df_elevage.infer_objects(copy=False)

        export_feed = supp_export

        if import_feed_net_tot > 0:
            export_feed += import_excedent_feed
        else:
            export_feed += -import_feed_net_tot + import_excedent_feed

        if export_feed.value() > 0:
            # Liste des cultures prioritaires pour l'exportation
            export_prioritaire = [
                "Forage cabbages",
                "Natural meadow ",
                "Non-legume temporary meadow",
                "Alfalfa and clover",
                "Forage maize",
                "Straw",
            ]

            # Calcul de l'azote disponible dans les cultures prioritaires
            total_N_export_prio = df_cultures.loc[export_prioritaire, "Azote disponible après feed"].sum()

            # Initialisation du dictionnaire source
            source = {}

            # Si l'azote à exporter est inférieur ou égal à l'azote disponible dans les cultures prioritaires
            if export_feed.value() <= total_N_export_prio:
                # Distribution proportionnelle parmi les cultures prioritaires
                for culture in export_prioritaire:
                    azote_disponible = df_cultures.at[culture, "Azote disponible après feed"]
                    proportion = azote_disponible / total_N_export_prio
                    source[culture] = proportion * export_feed.value()
            else:
                # Exporter tout l'azote disponible des cultures prioritaires
                for culture in export_prioritaire:
                    source[culture] = df_cultures.at[culture, "Azote disponible après feed"]

                # Calcul de l'azote restant à exporter
                azote_restant = export_feed.value() - total_N_export_prio

                # Récupération des cultures utilisées dans le régime d'élevage, excluant les cultures déjà exportées
                cultures_regime = set()
                for elevage in regime_elevage.values():
                    for cultures_liste in elevage.values():
                        cultures_regime.update(cultures_liste)

                # Cultures supplémentaires à considérer pour l'exportation
                cultures_supplementaires = list(
                    set(df_cultures.index).intersection(cultures_regime) - set(export_prioritaire)
                )

                # Calcul de l'azote disponible dans les cultures supplémentaires
                total_N_cultures_supp = df_cultures.loc[cultures_supplementaires, "Azote disponible après feed"].sum()

                # Vérification si l'azote restant est inférieur ou égal à l'azote disponible dans les cultures supplémentaires
                if azote_restant <= total_N_cultures_supp:
                    # Distribution proportionnelle parmi les cultures supplémentaires
                    for culture in cultures_supplementaires:
                        azote_disponible = df_cultures.at[culture, "Azote disponible après feed"]
                        proportion = azote_disponible / total_N_cultures_supp
                        source[culture] = proportion * azote_restant
                else:
                    # Exporter tout l'azote disponible des cultures supplémentaires
                    for culture in cultures_supplementaires:
                        source[culture] = df_cultures.at[culture, "Azote disponible après feed"]

                    # Mise à jour de l'azote restant à exporter
                    azote_restant -= total_N_cultures_supp

                    # Si de l'azote reste encore à exporter, on considère toutes les autres cultures disponibles
                    if azote_restant > 0:
                        cultures_restantes = df_cultures.index.difference(
                            source.keys()
                        )  # TODO retirer les catégories non commestibles par les animaux (rice, roots, fruits and ..)
                        total_N_cultures_restantes = df_cultures.loc[
                            cultures_restantes, "Azote disponible après feed"
                        ].sum()

                        # Vérification si l'azote restant est inférieur ou égal à l'azote disponible dans les cultures restantes
                        if azote_restant <= total_N_cultures_restantes:
                            # Distribution proportionnelle parmi les cultures restantes
                            for culture in cultures_restantes:
                                azote_disponible = df_cultures.at[culture, "Azote disponible après feed"]
                                proportion = azote_disponible / total_N_cultures_restantes
                                source[culture] = azote_disponible * azote_restant / total_N_cultures_restantes
                        else:
                            # Exporter tout l'azote disponible des cultures restantes
                            for culture in cultures_restantes:
                                source[culture] = df_cultures.at[culture, "Azote disponible après feed"]

                            # L'azote total exporté est inférieur à l'azote demandé
                            print("Attention : L'azote disponible total est inférieur à l'azote à exporter.")

            # Mise à jour du DataFrame avec les quantités exportées
            df_cultures["Azote exporté feed"] = df_cultures.index.map(source).fillna(0)

            df_cultures["Azote disponible après feed et export feed"] = (
                df_cultures["Azote disponible après feed"] - df_cultures["Azote exporté feed"]
            )
            # Générer des flux par catégorie
            # flux = {}
            # for culture, proportion in source.items():
            #     categorie = df_cultures.loc[culture, "catégories"]
            #     label_target = f"{categorie} feed nitrogen import-export"
            #     flux[culture] = flux.get(culture, 0) + proportion * export_feed.value()

            # Envoi des flux vers les bonnes catégories
            for culture, flux_value in source.items():
                if flux_value > 0:
                    label_target = f"{df_cultures.loc[culture, 'catégories']} feed trade"
                    source = {culture: flux_value}
                    target = {label_target: 1}
                    flux_generator.generate_flux(source, target)

        if export_feed.value() == 0:
            df_cultures["Azote exporté feed"] = 0
            df_cultures["Azote disponible après feed et export feed"] = df_cultures["Azote disponible après feed"]

        ## Usage de l'azote végétal pour nourrir la population
        # Importation si nécessaire

        vege_cap = data[data["index_excel"] == 9][region].item()
        cons_vege = vege_cap * pop

        categories_presentes = df_cultures["catégories"].unique()

        # Initialisation du problème
        prob = LpProblem("Allocation_Azote_Humains", LpMinimize)

        # Variables d'allocation de l'azote des cultures aux humains
        y_vars = LpVariable.dicts("y", df_cultures.index, lowBound=0, cat=LpContinuous)

        # Variables de déviation du régime alimentaire humain
        delta_vars = LpVariable.dicts("delta_humain", regime_humains.keys(), lowBound=0, cat=LpContinuous)

        # Variables d'importation pour chaque catégorie
        I_vars = LpVariable.dicts("Import", regime_humains.keys(), lowBound=0, cat=LpContinuous)

        # Variables de pénalité pour la répartition des cultures
        penalite_culture_vars = LpVariable.dicts("penalite_culture", df_cultures.index, lowBound=0, cat=LpContinuous)

        # Poids pour les pénalités
        poids_deviation = 10
        if int(year) > 1960:
            poids_import = 0.05  # Ajustez selon les préférences => Aucune restriction à importer (sinon le riz passe à la trappe...)
        else:
            poids_import = 1000000  # Pas d'importation ou subsidiaire avant les années 60
        poids_penalite_culture = 1.0  # Ajustez selon les préférences

        # Fonction objectif
        prob += (
            (
                poids_deviation * lpSum(delta_vars[k] for k in regime_humains.keys())
                + poids_import * lpSum(I_vars[k] for k in regime_humains.keys())
                + poids_penalite_culture * lpSum(penalite_culture_vars[c] for c in df_cultures.index)
            ),
            "Minimiser_Deviations_Importations_PenalitesCultures",
        )

        for idx, row in df_cultures.iterrows():
            culture = row.name
            azote_restant = row["Azote disponible après feed et export feed"]
            prob += y_vars[culture] <= azote_restant, f"Disponibilite_{culture}"

        for k in regime_humains.keys():
            # Somme des allocations pour la catégorie k
            cultures_k = df_cultures[df_cultures["catégories"] == k].index
            allocation_categorie = lpSum(y_vars[c] for c in cultures_k)
            proportion_effective = (allocation_categorie + I_vars[k]) / cons_vege
            proportion_initiale = regime_humains[k]
            # Contraintes de déviation
            prob += proportion_effective - proportion_initiale <= delta_vars[k], f"Deviation_Plus_{k}"
            prob += proportion_initiale - proportion_effective <= delta_vars[k], f"Deviation_Moins_{k}"

        for k in regime_humains.keys():
            # Cultures disponibles dans la catégorie k
            cultures_k = df_cultures[df_cultures["catégories"] == k].index
            # Azote total disponible dans la catégorie k
            azote_total_categorie = df_cultures[df_cultures["catégories"] == k][
                "Azote disponible après feed et export feed"
            ].sum()
            # Allocation totale à la catégorie k (incluant importations)
            allocation_categorie = lpSum(y_vars[c] for c in cultures_k) + I_vars[k]
            if azote_total_categorie > 0:
                for c in cultures_k:
                    azote_disponible_culture = df_cultures[df_cultures.index == c][
                        "Azote disponible après feed et export feed"
                    ].values[0]
                    # Allocation cible proportionnelle à la disponibilité
                    allocation_cible_culture = (
                        (azote_disponible_culture / azote_total_categorie) * allocation_categorie
                        if azote_total_categorie > 0
                        else 0
                    )
                    # Contraintes de pénalité
                    prob += (
                        y_vars[c] - allocation_cible_culture <= penalite_culture_vars[c],
                        f"Penalite_Culture_Plus_{c}",
                    )
                    prob += (
                        allocation_cible_culture - y_vars[c] <= penalite_culture_vars[c],
                        f"Penalite_Culture_Moins_{c}",
                    )
            else:
                pass

        # Supposons que nous voulons limiter la consommation de 'Carottes' à 5% du besoin total
        # max_carottes = 0.05 * cons_vege
        # prob += y_vars['Carottes'] <= max_carottes, "Limite_Carottes"

        # Résolution du problème
        prob.solve()

        if LpStatus[prob.status] == "Optimal":
            # Extraction des allocations
            allocations = []
            for c in df_cultures.index:
                allocation = y_vars[c].varValue
                if allocation > 0:
                    allocations.append({"Culture": c, "Consommation humaine": allocation})
            allocations_df = pd.DataFrame(allocations)

            # Filtrer les lignes en supprimant celles dont 'Azote_alloue' est très proche de zéro
            allocations_df = allocations_df[allocations_df["Consommation humaine"].abs() >= 1e-7]

            self.allocation_humain = allocations_df

            # Importations nécessaires
            importations = []
            for k in regime_humains.keys():
                importation = I_vars[k].varValue
                if importation > 0:
                    importations.append({"Catégorie": k, "Azote_importe": importation})
            importations_df = pd.DataFrame(importations)
            # Gestion du cas particulier sans importation
            if importations == []:
                importations_df = pd.DataFrame(importations, columns=["Catégorie", "Azote_importe"])
                df_cultures["Azote importé"] = 0.0

            # Extraction des déviations
            deviations = []
            for categorie, proportion_initiale in regime_humains.items():
                deviation = delta_vars[categorie].varValue
                if deviation != 0:
                    # Calcul de la proportion effective
                    azote_categorie = (
                        sum(
                            y_vars[culture].varValue
                            for culture in df_cultures[df_cultures["catégories"] == categorie].index
                        )
                        + I_vars[categorie].varValue
                    )
                    besoin_total = (
                        sum(y_vars[c].varValue for c in df_cultures.index) + importations_df["Azote_importe"].sum()
                    )

                    proportion_effective = azote_categorie / besoin_total if besoin_total > 0 else 0

                    # Déterminer le signe
                    signe = 1 if proportion_effective > proportion_initiale else -1

                    # Récupérer les cultures de la catégorie
                    cultures_str = ", ".join(df_cultures[df_cultures["catégories"] == categorie].index)

                    deviations.append(
                        {
                            "Catégorie": categorie,
                            "Proportion attendue (%)": proportion_initiale * 100,
                            "Déviation (%)": signe * round(deviation, 4) * 100,  # Convertir en pourcentage
                            "Cultures": cultures_str,
                        }
                    )

            deviations_df = pd.DataFrame(deviations)
        else:
            print("Aucune solution optimale n'a été trouvée.")

        allocations_df.set_index("Culture", inplace=True)
        importations_df.set_index("Catégorie", inplace=True)
        if "Consommation humaine" in df_cultures.columns:
            df_cultures.drop("Consommation humaine", axis=1, inplace=True)
        df_cultures = df_cultures.join(allocations_df, how="left")

        df_cultures["Consommation humaine"] = df_cultures["Consommation humaine"].fillna(0)

        source = df_cultures["Consommation humaine"].to_dict()
        target = {"urban": prop_urb, "rural": 1 - prop_urb}
        flux_generator.generate_flux(source, target)

        # Puis on importe le manque
        if len(importations_df) > 0:
            if int(year) >= 1965:
                cons_vege_import = importations_df["Azote_importe"].sum()
                commerce_path = "C:/Users/faustega/Documents/These/informatique/Donnees/metabolisme/commerce FAO/FAOSTAT_data_fr_vege_import.csv"
                commerce = pd.read_csv(commerce_path)
                commerce = commerce.loc[commerce["Année"] == int(year), ["Produit", "Valeur"]]
                # On utilise le sarrasin pour autres céréales
                # Huile de palme pour autres huiles
                # Lentilles pour autres proteins crops (lupins indisponible)
                # Carottes et navets pour autres racines
                # Fruit sec: amandes
                # Poids frais pour green peas
                # pois chiche sec pour dry vegetable
                # Laitues et chicorée pour leaves vege
                # Olive = olives + huile d'olive
                # blé = blé + farine de blé
                # banane pour fruits
                corresp_dict = {
                    "Amandes, écalées": "Dry fruits",
                    "Avoine laminée": "Oat",
                    "Bananes": "Fruits",
                    "Blé": "Wheat",
                    "Cantaloup et autres melons": "Squash and melons",
                    "Carottes et navets": "Other roots",
                    "Choux": "Cabbage",
                    "Citrons et limes": "Citrus",
                    "Farine de blé et de méteil": "Wheat",  # Regroupé avec "blé"
                    "Fèves et féveroles, sèches": "Horse beans and faba beans",
                    "Graines de lin": "Flax",
                    "Haricots secs": "Dry beans",
                    "Huile de colza ou de canola, brute": "Rapeseed",
                    "Huile de graines de tournesol, brute": "Sunflower",
                    "Huile de palme": "Other oil crops",  # Règle spéciale pour huile de palme
                    "Huile d'olives": "Olives",  # Regroupé avec "Olives"
                    "Laitue et chicorée": "Leaves vegetables",
                    "Lentilles secs": "Other protein crops",
                    "Maïs": "Grain maize",
                    "Olives": "Olives",  # Regroupé avec "Huile d'olives"
                    "Orge": "Barley",
                    "Pois chiches, secs": "Dry vegetables",  # Pois chiches secs = dry vegetables
                    "Pois frais": "Green peas",  # Pois frais = green peas
                    "Pois secs": "Peas",
                    "Pommes de terre": "Potatoes",
                    "Riz, paddy (riz blanchi équivalent)": "Rice",
                    "Sarrasin, blé noir": "Other cereals",  # Utilisé pour "autres céréales"
                    "Seigle": "Rye",
                    "Fèves de soja": "Soybean",
                    "Sucre de betterave ou de canne brut (centrifugé uniquement)": "Sugar beet",
                }

                commerce["Produit"] = commerce["Produit"].map(corresp_dict).fillna(commerce["Produit"])
                commerce.index = commerce["Produit"]
                commerce = commerce.groupby(commerce.index).sum()
                commerce["catégories"] = commerce.index.map(categories_mapping)

                # On distribue au sein de chaque catégorie les besoins d'importation
                df_cultures["Azote importé"] = 0.0
                for categories in importations_df.index:
                    cat_commerce = commerce.loc[commerce["catégories"] == categories].copy()
                    cat_commerce["Ratio"] = (cat_commerce["Valeur"] / cat_commerce["Valeur"].sum()).astype(float)
                    # TODO on en fait quoi de ça ??
                    # target = {"urbain": prop_urb, "rural": 1-prop_urb}
                    # source = (cat_commerce["Ratio"]*importations_df.loc[importations_df.index==categories, "Azote_importe"].item()).to_dict()
                    # generateur_flux(source, target, adjacency_matrix, label_to_index)
                    azote_importe = importations_df.loc[importations_df.index == categories, "Azote_importe"].item()
                    df_cultures.loc[cat_commerce.index, "Azote importé"] += cat_commerce["Ratio"] * azote_importe
                    # df_cultures["Azote importé"] += cat_commerce["Ratio"]*importations_df.loc[importations_df.index==categories, "Azote_importe"].item()
                df_cultures["Azote importé"] = df_cultures["Azote importé"].fillna(0)
            else:
                import_ancien = {
                    "cereals (excluding rice)": "Wheat",
                    "rice": "Rice",
                    "oleaginous": "tournesol",
                    "leguminous": "Other oil crops",
                    "roots": "Potatoes",
                    "fruits and vegetables": "Leaves vegetables",
                }
                for categories in importations_df.index:
                    azote_importe = importations_df.loc[importations_df.index == categories, "Azote_importe"].item()
                    df_cultures["Azote importé"][import_ancien[categories]] = azote_importe
                df_cultures["Azote importé"] = df_cultures["Azote importé"].fillna(0)

        # Calculer l'azote exporté net
        df_cultures["Azote exporté net"] = (
            df_cultures["Azote disponible après feed et export feed"]
            - df_cultures["Consommation humaine"]
            - df_cultures["Azote importé"]
        )
        # On corrige les poussières proches de 0
        df_cultures["Azote exporté net"] = df_cultures["Azote exporté net"].apply(lambda x: 0 if abs(x) < 10**-6 else x)

        # Gestion des exports
        flux_exports = {}
        for culture, azote_exporte in df_cultures[df_cultures["Azote exporté net"] > 0]["Azote exporté net"].items():
            categorie = df_cultures.loc[culture, "catégories"]
            if categorie not in ["forages", "grasslands"]:
                label_target = f"{categorie} food trade"
                flux_exports[label_target] = flux_exports.get(label_target, 0) + azote_exporte

        # Envoi des flux d'exportation par catégorie
        for label_target, flux_value in flux_exports.items():
            if flux_value > 0:
                source = {
                    culture: df_cultures.loc[culture, "Azote exporté net"]
                    for culture in df_cultures.index
                    if df_cultures.loc[culture, "catégories"].split()[0] == label_target.split()[0]
                    and df_cultures.loc[culture, "Azote exporté net"] > 0
                }
                target = {label_target: 1}
                flux_generator.generate_flux(source, target)

        # Gestion des imports
        flux_imports = {}
        for culture, azote_importe in df_cultures[df_cultures["Azote importé"] > 0]["Azote importé"].items():
            categorie = df_cultures.loc[culture, "catégories"]
            if categorie not in ["forages", "grasslands"]:
                label_source = f"{categorie} food trade"
                flux_imports[label_source] = flux_imports.get(label_source, 0) + azote_importe

        # Envoi des flux d'importation par catégorie
        for label_source, flux_value in flux_imports.items():
            if flux_value > 0:
                source = {label_source: flux_value}
                target = {"urban": prop_urb, "rural": 1 - prop_urb}
                flux_generator.generate_flux(source, target)

        ## Usage de l'azote animal pour nourir la population, on pourrait améliorer en distinguant viande, oeufs et lait

        viande_cap = data[data["index_excel"] == 10][region].item()
        cons_viande = viande_cap * pop

        if cons_viande < df_elevage["Azote comestible"].sum():  # Il y a assez de viande locale
            target = {"urban": prop_urb * cons_viande, "rural": (1 - prop_urb) * cons_viande}
            source = (df_elevage["Azote comestible"] / df_elevage["Azote comestible"].sum()).to_dict()
            df_elevage["Azote animal exporté net"] = df_elevage["Azote comestible"] - df_elevage.index.map(
                source
            ) * sum(target.values())
            flux_generator.generate_flux(source, target)

        else:
            # On commence par consommer tout l'azote disponible
            target = {"urban": prop_urb, "rural": (1 - prop_urb)}
            source = df_elevage["Azote comestible"].to_dict()
            flux_generator.generate_flux(source, target)

            cons_viande_import = cons_viande - df_elevage["Azote comestible"].sum()
            commerce_path = "C:/Users/faustega/Documents/These/informatique/Donnees/metabolisme/commerce FAO/FAOSTAT_data_fr_viande_import.csv"
            commerce = pd.read_csv(commerce_path)
            if (
                int(year) < 1965
            ):  # Si on est avant 65, on se base sur les rations de 65. De toute façon ça concerne des import minoritaires...
                year = "1965"
            commerce = commerce.loc[commerce["Année"] == int(year), ["Produit", "Valeur"]]

            corresp_dict = {
                "Viande, bovine, fraîche ou réfrigérée": "bovines",
                "Viande ovine, fraîche ou réfrigérée": "ovines",
                "Viande, caprin, fraîche ou réfrigérée": "caprines",
                "Viande, cheval, fraîche ou réfrigérée": "equine",
                "Viande, porc, fraîche ou réfrigérée": "porcines",
                "Viande, poulet, fraîche ou réfrigérée": "poultry",
            }

            commerce["Produit"] = commerce["Produit"].map(corresp_dict).fillna(commerce["Produit"])
            commerce["Ratio"] = commerce["Valeur"] / commerce["Valeur"].sum()
            commerce.index = commerce["Produit"]

            target = {"urban": prop_urb * cons_viande_import, "rural": (1 - prop_urb) * cons_viande_import}
            source = {
                "animal trade": 1
            }  # commerce["Ratio"].to_dict() On peut distinguer les différents types d'azote importé
            flux_generator.generate_flux(source, target)
            # Et on reporte ce qu'il manque dans la colonne "Azote animal exporté net"
            df_elevage["Azote animal exporté net"] = -commerce["Ratio"] * (cons_viande_import)

        if cons_viande < df_elevage["Azote comestible"].sum():
            source = df_elevage["Azote animal exporté net"].to_dict()
            target = {"animal trade": 1}
            flux_generator.generate_flux(source, target)

        # Calcul des déséquilibres négatifs
        for label in cultures + legumineuses + prairies:
            node_index = label_to_index[label]
            row_sum = adjacency_matrix[node_index, :].sum()
            col_sum = adjacency_matrix[:, node_index].sum()
            imbalance = row_sum - col_sum  # Déséquilibre entre sorties et entrées
            if abs(imbalance) < 10**-4:
                imbalance = 0

            if (
                imbalance > 0
            ):  # Que conclure si il y a plus de sortie que d'entrée ? Que l'on détériore les réserves du sol ?
                # print(f"pb de balance avec {label}")
                # Plus de sorties que d'entrées, on augmente les entrées
                # new_adjacency_matrix[n, node_index] = imbalance  # Flux du nœud de balance vers la culture
                target = {label: imbalance}
                source = {"soil stock": 1}
                flux_generator.generate_flux(source, target)
            elif imbalance < 0:
                # Plus d'entrées que de sorties, on augmente les sorties
                # adjacency_matrix[node_index, n] = -imbalance  # Flux de la culture vers le nœud de balance
                if label != "Nartural meadow ":  # 70% de l'excès fini dans les ecosystèmes aquatiques
                    source = {label: -imbalance}
                    target = {"other losses": 0.2925, "hydro-system": 0.7, "N2O emission": 0.0075}
                elif (
                    imbalance * 10**6 / df_cultures[label]["Surface"] > 100
                ):  # Si c'est une prairie, l'azote est lessivé seulement au dela de 100 kgN/ha
                    source = {label: -imbalance - 100 * df_cultures[label]["Surface"] / 10**6}
                    target = {"other losses": 0.2925, "hydro-system": 0.7, "N20 emission": 0.0075}
                    flux_generator.generate_flux(source, target)
                    source = {label: 100 * df_cultures[label]["Surface"] / 10**6}
                    target = {label: 1}
                else:  # Autrement, l'azote reste dans le sol (cas particulier, est ce que cela a du sens, quid des autres cultures ?)
                    source = {label: -imbalance}
                    target = {"soil stock": 1}
                flux_generator.generate_flux(source, target)
            # Si imbalance == 0, aucun ajustement nécessaire

        # Calcul de imbalance dans df_cultures
        df_cultures["bilan"] = (
            df_cultures["Azote à épendre (ktN) corrigé"]
            + df_cultures["Azote épendu"]
            + df_cultures["heritage legumineuse"]
            - df_cultures["Surplus Azote leg"]
            - df_cultures["Azote disponible"]
            - df_cultures["Azote Volatilisé N-NH3 (ktN)"]
            - df_cultures["Azote émit N-N2O (ktN)"]
        )

        # On équilibre Haber-Bosch avec atmospheric N2 pour le faire entrer dans le système
        target = {"Haber-Bosch": adjacency_matrix[:, 50].sum()}
        source = {"atmospheric N2": 1}
        flux_generator.generate_flux(source, target)

        self.df_cultures = df_cultures
        self.df_elevage = df_elevage
        self.adjacency_matrix = adjacency_matrix

    def get_df_culture(self):
        return self.df_cultures

    def get_df_elevage(self):
        return self.df_elevage

    def get_allocation_elevage(self):
        return self.allocation_elevage

    def get_allocation_humain(self):
        return self.allocation_humain

    def get_transition_matrix(self):
        return self.adjacency_matrix

    def get_core_matrix(self):
        # Calcul de la taille du noyau
        core_size = len(self.adjacency_matrix) - len(self.ext)

        # Extraire la matrice principale (noyau)
        core_matrix = self.adjacency_matrix[:core_size, :core_size]

        # Calculer la somme des éléments sur chaque ligne
        row_sums = core_matrix.sum(axis=1)

        # Identifier les indices des lignes où la somme est non nulle
        non_zero_rows = row_sums != 0

        # Identifier les indices des colonnes à garder (les mêmes indices que les lignes non nulles)
        non_zero_columns = non_zero_rows

        # Filtrer les lignes et les colonnes avec une somme non nulle
        core_matrix_filtered = core_matrix[non_zero_rows, :][:, non_zero_columns]

        # Retourner la matrice filtrée
        self.core_matrix = core_matrix_filtered
        self.non_zero_rows = non_zero_rows
        return core_matrix_filtered

    def get_adjacency_matrix(self):
        return (self.core_matrix != 0).astype(int)

    def extract_input_output_matrixs(self, clean=True):
        # Fonction pour extraire la matrice entrée (C) et la matrice sortie (B) de la matrice complète.
        # Taille de la matrice coeur
        core_size = len(self.adjacency_matrix) - len(self.ext)
        n = len(self.adjacency_matrix)
        # Extraire la sous-matrice B (bloc haut-droit)
        B = self.adjacency_matrix[:core_size, core_size:n]

        # Extraire la sous-matrice C (bloc bas-gauche)
        C = self.adjacency_matrix[core_size:n, :core_size]

        if clean:
            C = C[:][:, self.non_zero_rows]
            B = B[self.non_zero_rows, :][:]

        return B, C


# # Créer une instance du modèle
# year = "2010"
# region = "Picardie"
# data = DataLoader(year, region)

# nitrogen_model = NitrogenFlowModel(
#     data = data,
#     year=year,
#     region=region,
#     categories_mapping=categories_mapping,
#     labels=labels,
#     cultures=cultures,
#     legumineuses=legumineuses,
#     prairies=prairies,
#     betail=betail,
#     Pop=Pop,
#     ext=ext
# )

# # Calculer les flux
# nitrogen_model.compute_fluxes()

# # Afficher la heatmap
# nitrogen_model.plot_heatmap()
