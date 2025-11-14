import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LogNorm
from pulp import (
    LpContinuous,
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
    LpStatus,
    LpBinary,
    LpAffineExpression,
)
import warnings
import math
import hashlib
import numbers

# Afficher toutes les colonnes
pd.set_option("display.max_columns", None)

# Afficher toutes les lignes
pd.set_option("display.max_rows", None)

pd.set_option("future.no_silent_downcasting", True)


class DataLoader:
    """Load input data files : project file and data file."""

    def __init__(self, project_path, data_path):
        self.df_data = pd.read_excel(data_path, sheet_name=None)
        self.data_path = data_path

        self.metadata = pd.read_excel(project_path, sheet_name=None)

        self.check_dup()

        # Creation df_cultures
        self.init_df_cultures = self.metadata["crops"].set_index("Culture")
        self.crops = list(self.init_df_cultures.index)

        # Creation df_elevage
        self.init_df_elevage = self.metadata["livestock"].set_index("Livestock")

        # Creation df_pop
        self.init_df_pop = self.metadata["pop"].set_index("Population")

        # Creation de df_excr
        self.init_df_excr = self.metadata["excretion"].set_index("Excretion")

        # Check que l'utilisateur a mis les bons compartiments dans excr
        # Sinon ajouter les lignes correspondantes avec 0 pour toutes les colonnes sauf "Livestock"

        suffixes = [" manure", " slurry", " grasslands excretion"]
        # Une seule list comprehension pour tout faire
        excretion_compartments = [
            i + suffix for i in self.init_df_elevage.index for suffix in suffixes
        ]

        # Verifier que tous les excretion compartments sont dans self.init_excr. Sinon ajouter les lignes manquantes. Initialiser à 0 sauf livestock qui doit correspondre au self.init_df_elevage manquant
        # Find the compartments that are missing from the dataframe
        missing_compartments = (
            set(excretion_compartments)
            - set(self.init_df_excr.index)
            - set(self.init_df_pop.index)
        )

        if missing_compartments:
            warning_message = (
                f"The following excretion compartments are missing from the 'excretion' sheet: {missing_compartments}. "
                "They have been added with default values of 0. Please verify your input data."
            )
            warnings.warn(warning_message)
            missing_data = {
                col: [0] * len(missing_compartments)
                for col in self.init_df_excr.columns
            }

            # Explicitly set the 'Origin compartment' column data type to 'object' (for strings)
            missing_data["Origin compartment"] = [""] * len(missing_compartments)
            missing_data["Type"] = [""] * len(missing_compartments)

            # Create the DataFrame with the correct data types from the start
            missing_df = pd.DataFrame(
                missing_data,
                index=list(missing_compartments),
                columns=self.init_df_excr.columns,
            )

            # Fill the 'Origin compartment' column for the new rows
            for compartment_name in missing_df.index:
                # Loop through the known livestock names to find the match
                for livestock_name in self.init_df_elevage.index:
                    if compartment_name.startswith(f"{livestock_name} "):
                        missing_df.loc[compartment_name, "Origin compartment"] = (
                            livestock_name
                        )
                        for suffix in suffixes:
                            if compartment_name.endswith(suffix):
                                missing_df.loc[compartment_name, "Type"] = (
                                    suffix.strip()
                                )
                                break  # Exit the inner suffix loop once the type is found
                        break  # Exit the inner loop once the match is found

            # Concatenate the new rows to the original DataFrame
            self.init_df_excr = pd.concat([self.init_df_excr, missing_df])

        # Sort the index for a clean and consistent order
        self.init_df_excr = self.init_df_excr.sort_index()

        # Creation df_prod
        self.init_df_prod = self.metadata["prod"].set_index("Product")

        # Creation df_energy
        self.init_df_energy = self.metadata["energy"].set_index("Facility")

        fixed_compartments = [
            "Haber-Bosch",
            "hydro-system",
            "atmospheric N2",
            "atmospheric NH3",
            "atmospheric N2O",
            "fishery products",
            "other sectors",
            "waste",
            "soil stock",
            "seeds",
            "hydrocarbures",
        ]

        trade = [
            i + " trade"
            for i in set(self.init_df_prod["Sub Type"])
            if i not in ["non edible meat", "grazing"]
        ]

        self.trade_labels = trade

        self.labels = (
            list(self.init_df_cultures.index)
            + list(self.init_df_elevage.index)
            + list(self.init_df_excr.index)
            + list(self.init_df_prod.index)
            + list(self.init_df_pop.index)
            + list(self.init_df_energy.index)
            + fixed_compartments
            + trade
        )

        self.label_to_index = {
            self.label: index for index, self.label in enumerate(self.labels)
        }
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        # creation de df_global
        self.init_df_global = self.metadata["global"].set_index("item")

        self.available_years = set(self.df_data["Input data"]["Year"])
        self.available_regions = set(self.df_data["Input data"]["Area"])

    # ──────────────────────────────────────────────────────────────────────

    def check_dup(self):
        dup = (
            self.df_data["Input data"]
            .groupby(["Area", "Year", "item", "category"])
            .size()
        )
        dup = dup[dup > 1]
        if not dup.empty:
            # choix : lever Warning / Erreur, ou agréger. Ici on lève une erreur pour forcer la correction
            duplicated_keys = dup.index.tolist()
            raise ValueError(
                f"Duplicate global metrics found for keys: {duplicated_keys}"
            )

    # Utilitaire : conversion robuste de "value" (virgules → points, etc.)
    @staticmethod
    def _to_num(x):
        return pd.to_numeric(
            pd.Series(x, dtype="object").astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).astype(float)

    def get_columns(
        self,
        area: str,
        year: int,
        init,
        categories_needed: tuple,
        overwrite=False,
        warn_if_nans=True,
    ):
        """
        Alimente les dataframes avec les colonnes demandées en allant chercher
        les valeurs dans la feuille 'Input data' de self.df_data.

        - `area` / `year` filtrent la feuille Input data
        - `categories_needed` : liste des catégories à récupérer
        """
        if "Input data" not in self.df_data:
            raise KeyError("La feuille 'Input data' est absente de data_path.")

        df_in = self.df_data["Input data"].copy()

        # Sélection de la zone/année
        df_in = df_in[
            (df_in["Area"] == area)
            & (df_in["Year"] == year)
            & (df_in["item"].isin(init.index))
        ]

        # On ne garde que les catégories d'intérêt
        df_in = df_in[df_in["category"].isin(categories_needed)]

        # Passage en large : index = Culture, colonnes = category
        wide = (
            df_in.pivot_table(
                index="item",
                columns="category",
                values="value",
                aggfunc="last",  # si doublons, on prend la dernière valeur
            ).reindex(init.index)  # aligne sur l'index des cultures du modèle
        )

        # S'assure que toutes les colonnes demandées existent (même si absentes)
        for col in categories_needed:
            if col not in wide.columns:
                wide[col] = np.nan

        wide = wide[list(categories_needed)]

        # Remplissage manquant par 0 (ou garde NaN si tu préfères)
        # wide = wide[categories_needed].fillna(0.0)

        # --- Merge ---
        # Option : écraser uniquement si overwrite=True, sinon n'écrase que là où added n'est pas null
        merged_df = init.copy()

        for col in categories_needed:
            series_added = wide[col]
            if col in merged_df.columns and merged_df[col].dtype != object:
                merged_df[col] = merged_df[col].astype(object)
            if overwrite:
                # On met la colonne telle quelle (même si NaN)
                merged_df[col] = series_added
            else:
                # On n'écrase que les positions où wide a une valeur non-nulle (non-NaN)
                mask_has_value = series_added.notna()
                if col not in merged_df.columns:
                    # créer la colonne si elle n'existe pas
                    merged_df[col] = np.nan
                    merged_df[col] = merged_df[col].astype(object)
                merged_df.loc[mask_has_value, col] = series_added.loc[mask_has_value]

        if warn_if_nans:
            nan_report = {}
            for col in categories_needed:
                missing = merged_df[merged_df[col].isna()].index.tolist()
                if missing:
                    nan_report[col] = missing

            if nan_report:
                # construir un message synthétique
                lines = []
                for col, miss in nan_report.items():
                    count = len(miss)
                    sample = ", ".join(miss[:5])
                    more = "" if count <= 5 else f", +{count - 5} autres..."
                    lines.append(f"- {col}: {count} missing item (ex.: {sample}{more})")
                msg = (
                    f"Warning: NaNs remain in the imported columns for year {year}, area {area}.\n"
                    + "\n".join(lines)
                    + "\nIf this is expected, set `warn_if_nans=False`. Otherwise check the 'Input data' sheet and the mappings."
                )
                warnings.warn(msg)

        merged_df = merged_df.fillna(0)
        return merged_df

    def generate_df_prod(self, area, year, prospect=False):
        categories_needed = (
            "Nitrogen Content (%)",
            "Origin compartment",
            "Type",
            "Sub Type",
            "Waste (%)",
            "Other uses (%)",
        )

        if prospect:
            categories_needed += ("Co-Production Ratio (%)",)
        else:
            categories_needed += ("Production (kton)",)

        df_prod = self.get_columns(
            area, year, self.init_df_prod, categories_needed=categories_needed
        )

        df_prod = df_prod[list(categories_needed)].copy()

        if prospect:
            df_prod = df_prod.fillna(0)
            self.df_prod = df_prod
            return df_prod

        # Calcul de l'azote disponible pour les cultures
        df_prod["Nitrogen Production (ktN)"] = (
            df_prod["Production (kton)"] * df_prod["Nitrogen Content (%)"] / 100
        )

        df_prod["Nitrogen Wasted (ktN)"] = (
            df_prod["Nitrogen Production (ktN)"] * df_prod["Waste (%)"] / 100
        )

        df_prod["Nitrogen for Other uses (ktN)"] = (
            df_prod["Nitrogen Production (ktN)"] * df_prod["Other uses (%)"] / 100
        )

        df_prod["Available Nitrogen Production (ktN)"] = (
            df_prod["Nitrogen Production (ktN)"]
            - df_prod["Nitrogen Wasted (ktN)"]
            - df_prod["Nitrogen for Other uses (ktN)"]
        )

        df_prod = df_prod.fillna(0)

        # for animal in df_elevage.index:
        #     # Calcul et mise à jour pour la viande comestible
        #     df_prod.loc[f'{animal} edible meat', 'Production (kton)'] = (
        #         df_elevage.loc[animal, 'Production (kton carcass)'] *
        #         df_elevage.loc[animal, '% edible']
        #     )
        #     df_prod.loc[f'{animal} edible meat', 'Nitrogen Production (ktN)'] = (
        #         df_prod.loc[f'{animal} edible meat', 'Production (kton)'] *
        #         df_prod.loc[f'{animal} edible meat', 'Nitrogen Content (%)']
        #     )

        #     # Calcul et mise à jour pour la viande non comestible
        #     df_prod.loc[f'{animal} non edible meat', 'Production (kton)'] = (
        #         df_elevage.loc[animal, 'Production (kton carcass)'] *
        #         df_elevage.loc[animal, '% non edible']
        #     )
        #     df_prod.loc[f'{animal} non edible meat', 'Nitrogen Production (ktN)'] = (
        #         df_prod.loc[f'{animal} non edible meat', 'Production (kton)'] *
        #         df_prod.loc[f'{animal} non edible meat', 'Nitrogen Content (%)']
        #     )

        df_prod = df_prod.fillna(0)
        self.df_prod = df_prod

    def generate_df_cultures(self, area, year, prospect=False):
        categories_needed = (
            "Area (ha)",
            "Spreading Rate (%)",
            "Seed input (ktN/ktN)",
            "Harvest Index",
            "Main Production",
            "Category",
            "BNF alpha",
            "BNF beta",
            "BGN",
            "Residue Nitrogen Content (%)",
        )
        if prospect:
            categories_needed += (
                "Maximum Yield (tFW/ha)",
                "Characteristic Fertilisation (kgN/ha)",
            )
        else:
            categories_needed += (
                "Raw Surface Synthetic Fertilizer Use (kgN/ha)",
                "Fertilization Need (kgN/qtl)",
                "Surface Fertilization Need (kgN/ha)",
            )
        df_cultures = self.get_columns(
            area, year, self.init_df_cultures, categories_needed=categories_needed
        )
        df_cultures = df_cultures[list(categories_needed)].copy()

        self.generate_df_prod(area, year, prospect)
        df_prod = self.df_prod.copy()

        if df_cultures["Residue Nitrogen Content (%)"].eq(0).all():
            df_cultures.loc[
                df_cultures["Category"] != "leguminous", "Residue Nitrogen Content (%)"
            ] = 0.5
            df_cultures.loc[
                df_cultures["Category"] == "leguminous", "Residue Nitrogen Content (%)"
            ] = 1.5

        df_cultures["Nitrogen Harvest Index"] = (
            df_cultures["Harvest Index"]
            * df_cultures["Main Production"].map(df_prod["Nitrogen Content (%)"])
            / 100
        ) / (
            df_cultures["Harvest Index"]
            * df_cultures["Main Production"].map(df_prod["Nitrogen Content (%)"])
            / 100
            + (1 - df_cultures["Harvest Index"])
            * df_cultures["Residue Nitrogen Content (%)"]
            / 100
        )

        if prospect:
            df_cultures["Ymax (kgN/ha)"] = (
                df_cultures["Maximum Yield (tFW/ha)"]
                * 1000
                * df_cultures["Main Production"].map(df_prod["Nitrogen Content (%)"])
                / 100
            )
            df_cultures = df_cultures.fillna(0)
            self.df_cultures = df_cultures
            return

        df_cultures["Main Crop Production (kton)"] = df_cultures["Main Production"].map(
            df_prod["Production (kton)"]
        )

        # Calcul du N produit (ktN) = production (kton) * N% / 100
        df_cultures["Main Nitrogen Production (ktN)"] = (
            df_cultures["Main Crop Production (kton)"]
            * df_cultures["Main Production"].map(df_prod["Nitrogen Content (%)"])
            / 100
        )

        df_cultures["Seeds Input (ktN)"] = (
            df_cultures["Seed input (ktN/ktN)"]
            * df_cultures["Main Nitrogen Production (ktN)"]
        )

        mask = df_cultures["Area (ha)"] != 0

        df_cultures.loc[mask, "Yield (qtl/ha)"] = (
            df_cultures.loc[mask, "Main Crop Production (kton)"]
            * 1e4
            / df_cultures.loc[mask, "Area (ha)"]
        )

        df_cultures.loc[mask, "Yield (kgN/ha)"] = (
            df_cultures.loc[mask, "Main Nitrogen Production (ktN)"]
            / df_cultures.loc[mask, "Area (ha)"]
            * 1e6
        )

        mask = df_cultures["Fertilization Need (kgN/qtl)"] > 0
        df_cultures["Surface Fertilization Need (kgN/ha)"] = df_cultures[
            "Surface Fertilization Need (kgN/ha)"
        ].astype("float64", copy=False)
        df_cultures.loc[mask, "Surface Fertilization Need (kgN/ha)"] = (
            df_cultures.loc[mask, "Fertilization Need (kgN/qtl)"]
            * df_cultures.loc[mask, "Yield (qtl/ha)"]
        ).astype("float64", copy=False)

        if df_cultures["Raw Surface Synthetic Fertilizer Use (kgN/ha)"].eq(0).all():
            df_cultures = df_cultures.drop(
                columns=["Raw Surface Synthetic Fertilizer Use (kgN/ha)"]
            )
        else:
            df_cultures = df_cultures.drop(
                columns=[
                    "Surface Fertilization Need (kgN/ha)",
                    "Fertilization Need (kgN/qtl)",
                ]
            )

        df_cultures = df_cultures.fillna(0)
        self.df_cultures = df_cultures

    def generate_df_elevage(self, area, year, prospect=False):
        categories_needed = (
            "Excreted indoor (%)",
            "Excreted indoor as manure (%)",
            "Excretion / LU (kgN)",
            "LU",
            "Diet",
        )
        df_elevage = self.get_columns(
            area, year, self.init_df_elevage, categories_needed=categories_needed
        )
        df_elevage = df_elevage[list(categories_needed)].copy()

        df_elevage["Excreted indoor as slurry (%)"] = (
            100 - df_elevage["Excreted indoor as manure (%)"]
        )
        df_elevage["Excreted on grassland (%)"] = (
            100 - df_elevage["Excreted indoor (%)"]
        )

        self.generate_df_prod(area, year, prospect)
        df_prod = self.df_prod.copy()
        if prospect:
            mask_animal = df_prod["Type"] == "animal"

            df_prod.loc[mask_animal, "Production (kton)"] = (
                df_prod.loc[mask_animal, "Co-Production Ratio (%)"]
                * df_prod.loc[mask_animal, "Origin compartment"].map(df_elevage["LU"])
                / 100
            )
            df_prod.loc[mask_animal, "Nitrogen Production (ktN)"] = (
                df_prod.loc[mask_animal, "Production (kton)"]
                * df_prod.loc[mask_animal, "Nitrogen Content (%)"]
                / 100
            )
            df_prod.loc[mask_animal, "Available Nitrogen Production (ktN)"] = (
                df_prod.loc[mask_animal, "Nitrogen Production (ktN)"]
                * (
                    1
                    - df_prod.loc[mask_animal, "Waste (%)"] / 100
                    - df_prod.loc[mask_animal, "Other uses (%)"] / 100
                )
            )
        df_prod = df_prod.fillna(0)
        self.df_prod = df_prod
        df_elevage["Edible Nitrogen (ktN)"] = (
            (
                df_prod.loc[df_prod["Sub Type"] == "edible meat", :].set_index(
                    "Origin compartment"
                )["Nitrogen Content (%)"]
                * df_prod.loc[df_prod["Sub Type"] == "edible meat", :].set_index(
                    "Origin compartment"
                )["Production (kton)"]
                / 100
            )
            .groupby("Origin compartment")
            .sum()
        )
        df_elevage["Non Edible Nitrogen (ktN)"] = (
            (
                df_prod.loc[df_prod["Sub Type"] == "non edible meat", :].set_index(
                    "Origin compartment"
                )["Nitrogen Content (%)"]
                * df_prod.loc[df_prod["Sub Type"] == "non edible meat", :].set_index(
                    "Origin compartment"
                )["Production (kton)"]
                / 100
            )
            .groupby("Origin compartment")
            .sum()
        )
        df_elevage["Dairy Nitrogen (ktN)"] = (
            (
                df_prod.loc[df_prod["Sub Type"] == "dairy", :].set_index(
                    "Origin compartment"
                )["Nitrogen Content (%)"]
                * df_prod.loc[df_prod["Sub Type"] == "dairy", :].set_index(
                    "Origin compartment"
                )["Production (kton)"]
                / 100
            )
            .groupby("Origin compartment")
            .sum()
        )

        df_elevage["Excreted nitrogen (ktN)"] = (
            df_elevage["Excretion / LU (kgN)"] * df_elevage["LU"] / 1e6
        )

        df_elevage = df_elevage.fillna(0)
        df_elevage["Ingestion (ktN)"] = (
            df_elevage["Excreted nitrogen (ktN)"]
            + df_elevage["Edible Nitrogen (ktN)"]
            + df_elevage["Non Edible Nitrogen (ktN)"]
            + df_elevage["Dairy Nitrogen (ktN)"]
        )

        self.df_elevage = df_elevage

    def generate_df_excr(self, area, year, prospect=False):
        _ = self.generate_df_elevage(area, year, prospect)
        categories_needed = (
            "N-NH3 EM (%)",
            "N-N2 EM (%)",
            "N-N2O EM (%)",
            "Type",
            "Origin compartment",
            "Nitrogen Content (%)",
        )
        df_excr = self.get_columns(
            area, year, self.init_df_excr, categories_needed=categories_needed
        )
        df_excr = df_excr[list(categories_needed)].copy()

        # Calculate the total N excreted indoors
        indoor_excretion_total = (
            self.df_elevage["Excreted nitrogen (ktN)"]
            * self.df_elevage["Excreted indoor (%)"]
            / 100
        )

        # 1. Calculate the 'manure' excretion for each livestock type
        manure_excretion = (
            indoor_excretion_total
            * self.df_elevage["Excreted indoor as manure (%)"]
            / 100
        )

        # 2. Calculate the 'slurry' excretion for each livestock type
        slurry_excretion = indoor_excretion_total * (
            1 - self.df_elevage["Excreted indoor as manure (%)"] / 100
        )

        # 3. Calculate the 'grasslands excretion' for each livestock type
        grasslands_excretion = self.df_elevage["Excreted nitrogen (ktN)"] * (
            1 - self.df_elevage["Excreted indoor (%)"] / 100
        )

        # Combine the results into a single DataFrame or Series for assignment
        # This will be used to create the new column in df_excr
        excretion_values = (
            pd.DataFrame(
                {
                    "manure": manure_excretion,
                    "slurry": slurry_excretion,
                    "grasslands excretion": grasslands_excretion,
                }
            )
            .stack()
            .reset_index()
        )

        excretion_values.columns = ["Livestock", "Excretion_Type", "value"]
        excretion_values["Excretion"] = (
            excretion_values["Livestock"] + " " + excretion_values["Excretion_Type"]
        )
        excretion_values = excretion_values.set_index("Excretion")["value"]

        # Now, assign the calculated values to the 'Excretion (ktN)' column in df_excr
        df_excr["Excretion (ktN)"] = excretion_values

        df_excr["Excretion after volatilization (ktN)"] = (
            df_excr["Excretion (ktN)"]
            * (
                100
                - df_excr["N-NH3 EM (%)"]
                - df_excr["N-N2 EM (%)"]
                - df_excr["N-N2O EM (%)"]
            )
            / 100
        )

        df_excr["Excretion as NH3 (ktN)"] = (
            df_excr["Excretion (ktN)"] * df_excr["N-NH3 EM (%)"] / 100
        )
        df_excr["Excretion as N2 (ktN)"] = (
            df_excr["Excretion (ktN)"] * df_excr["N-N2 EM (%)"] / 100
        )
        df_excr["Excretion as N2O (ktN)"] = (
            df_excr["Excretion (ktN)"] * df_excr["N-N2O EM (%)"] / 100
        )

        df_excr = df_excr.fillna(0)
        self.df_excr = df_excr

    def generate_df_pop(self, area, year):
        categories_needed = (
            "Inhabitants",
            "N-NH3 EM excretion (%)",
            "N-N2 EM excretion (%)",
            "N-N2O EM excretion (%)",
            "Total ingestion per capita (kgN)",
            "Fishery ingestion per capita (kgN)",
            "Excretion recycling (%)",
            "Diet",
        )
        df_pop = self.get_columns(
            area, year, self.init_df_pop, categories_needed=categories_needed
        )
        df_pop = df_pop[list(categories_needed)].copy()
        # if carbon:
        #     return df_pop.fillna(0)

        # TODO a calculer à la fin de compute_fluxes()
        # df_pop["Plant Ingestion (ktN)"] = df_pop["Inhabitants"]*df_pop["Vegetal ingestion per capita (kgN)"] / 1e6
        # df_pop["Animal Ingestion (ktN)"] = df_pop["Inhabitants"]*df_pop["Animal ingestion per capita (kgN)"] / 1e6
        df_pop["Ingestion (ktN)"] = (
            df_pop["Inhabitants"] * df_pop["Total ingestion per capita (kgN)"] / 1e6
        )
        df_pop["Fishery Ingestion (ktN)"] = (
            df_pop["Inhabitants"] * df_pop["Fishery ingestion per capita (kgN)"] / 1e6
        )

        df_pop["Excretion after volatilization (ktN)"] = (
            df_pop["Ingestion (ktN)"] * df_pop["Excretion recycling (%)"] / 100
        )

        df_pop = df_pop.fillna(0)
        self.df_pop = df_pop

    def generate_df_energy(self, area, year):
        categories_needed = (
            "Target Energy Production (GWh)",
            "Diet",
            "Type",
        )
        df_energy = self.get_columns(
            area, year, self.init_df_energy, categories_needed=categories_needed
        )
        df_energy = df_energy[list(categories_needed)].copy()
        self.df_energy = df_energy

    def get_global_metrics(self, area, year, carbon=False, prospect=False):
        input_df = self.df_data["Input data"].copy()
        mask_global = (
            (input_df["category"] == "Global")
            & (input_df["Year"] == year)
            & (input_df["Area"] == area)
        )
        global_df = (
            input_df.loc[mask_global, ["item", "value"]].copy().set_index("item")
        )

        global_df = global_df.combine_first(self.init_df_global)

        required_items = [
            "Total Synthetic Fertilizer Use on crops (ktN)",
            "Total Synthetic Fertilizer Use on grasslands (ktN)",
            "Atmospheric deposition coef (kgN/ha)",
            "coefficient N-NH3 volatilization synthetic fertilization (%)",
            "coefficient N-N2O emission synthetic fertilization (%)",
            "Weight diet",
            "Weight import",
            "Weight distribution",
            "Weight fair local split",
            "Weight energy production",
            "Weight energy inputs",
            "Enforce animal share",
            "Green waste nitrogen content (%)",
        ]

        # The list of required items
        if prospect:
            required_items += [
                "Weight synthetic fertilizer",
                "Weight synthetic distribution",
            ]
        elif carbon:
            required_items += [
                "Total Haber-Bosch methan input (kgC/kgN)",
                "Green waste C/N",
            ]

        weight_diet = global_df.loc["Weight diet", "value"]
        weight_import = global_df.loc["Weight import", "value"]
        non_zero_weights = [w for w in [weight_diet, weight_import] if w > 0] + [0]
        # Weight distribution is given in option and can be computed from other weights
        if "Weight distribution" not in global_df.index:
            global_df.loc["Weight distribution", "value"] = min(non_zero_weights) / 10
        # Weight distribution is given in option and can be computed from other weights
        if "Weight fair local split" not in global_df.index:
            global_df.loc["Weight fair local split", "value"] = (
                min(non_zero_weights) / 20
            )
        if (prospect) and ("Weight synthetic distribution" not in global_df.index):
            weight_synth = global_df.loc["Weight synthetic fertilizer", "value"]
            global_df.loc["Weight synthetic distribution"] = weight_synth / 10

        if carbon:
            if "Green waste C/N" not in global_df.index:
                global_df.loc["Green waste C/N", "value"] = 10

        if len(self.init_df_energy) == 0:
            global_df.loc["Green waste nitrogen content (%)", "value"] = 0
            global_df.loc["Weight energy production", "value"] = 0
            global_df.loc["Weight energy inputs", "value"] = 0
        # Check for the presence of each required item
        missing_items = [item for item in required_items if item not in global_df.index]

        if missing_items:
            raise KeyError(
                f"❌ The following required global metrics were not found for year {year} and area {area}: "
                f"{', '.join(missing_items)}. Please check the input data."
            )

        self.global_df = global_df

    def load_diets_for_area_year(
        self, area: str, year: int, tol: float = 1e-6
    ) -> dict[str, pd.DataFrame]:
        """
        Charge et valide les régimes pour une zone et une année.
        Retourne un dict mapping consumer -> DataFrame(columns=['Proportion','Products']) où Products est une list.

        Exigences :
        - self.df_data['Diet'] doit exister avec colonnes similaires à ['Diet ID','Proportion','Products']
        - self.df_data['Input data'] doit contenir des lignes avec category == 'Diet', item==consumer, value==Diet ID
        - df_prod.index doit contenir les produits référencés dans 'Products'
        - Tous les index de self.df_elevage et self.df_pop doivent être présents comme consumer pour cette zone/année

        Si la somme des proportions d'un diet_id n'est pas ~1 : on émet un warning et on renormalise.
        Si un produit référencé dans un régime n'existe pas dans df_prod.index : ValueError.
        Si un consumer présent dans self.df_elevage / self.df_pop n'a pas de mapping diet pour (area,year) : ValueError.
        """

        # --- 0) Récupérations robustes et vérifications initiales -----------------
        if "Diet" not in self.df_data:
            raise ValueError("No 'Diet' sheet found in self.df_data")

        diet_table = self.df_data["Diet"].copy()
        # input_df = self.df_data.get("Input data", None)
        # if input_df is None:
        #     raise ValueError("No 'Input data' sheet found in self.df_data")

        # s'assurer que Proportion est numérique
        diet_table["Proportion"] = pd.to_numeric(
            diet_table["Proportion"], errors="coerce"
        )

        # --- 1) Construire diet_by_id : diet_id -> DataFrame(Proportion, Products:list) ---
        diet_by_id = {}
        for diet_id, block in diet_table.groupby("Diet ID"):
            # Forcer la liste de produits pour chaque ligne
            rows = []
            for _, row in block.iterrows():
                prop = row["Proportion"]
                prod_cell = row["Products"]
                if pd.isna(prod_cell):
                    products = []
                elif isinstance(prod_cell, (list, tuple)):
                    products = [p.strip() for p in prod_cell]
                else:
                    # split sur virgule
                    products = [
                        p.strip() for p in str(prod_cell).split(",") if p.strip() != ""
                    ]
                rows.append(
                    {
                        "Proportion": float(prop) if not pd.isna(prop) else np.nan,
                        "Products": products,
                    }
                )
            df_rows = pd.DataFrame(rows)

            # vérif somme proportions
            total = df_rows["Proportion"].sum(skipna=True)
            if not np.isfinite(total) or abs(total - 1.0) > tol:
                warnings.warn(
                    f"Diet '{diet_id}': proportions sum to {total:.6f} (tol={tol}). Renormalizing to sum=1.",
                    UserWarning,
                )
                # renormalisation en gardant NaN comme 0 (ou lever si tjs NaN)
                # remplacer NaN par 0 puis renormaliser si total > 0 sinon erreur
                df_rows["Proportion"] = df_rows["Proportion"].fillna(0.0)
                total2 = df_rows["Proportion"].sum()
                if total2 <= 0:
                    raise ValueError(
                        f"Diet '{diet_id}' has non-positive total proportion after fillna: {total2}"
                    )
                df_rows["Proportion"] = df_rows["Proportion"] / total2

            # stocker
            diet_by_id[diet_id] = df_rows.reset_index(drop=True)

        # --- 2) validation : tous les produits référencés existent dans df_prod.index --
        prod_index = set(self.init_df_prod.index.astype(str))
        excr_index = set(self.init_df_excr.index.astype(str))
        excr_index.add("waste")
        missing_products = []
        for diet_id, df_d in diet_by_id.items():
            for prod_list in df_d["Products"]:
                for prod in prod_list:
                    if prod not in prod_index and prod not in excr_index:
                        missing_products.append((diet_id, prod))
        if missing_products:
            lines = "\n".join(
                [
                    f"diet {did}: missing product '{p}'"
                    for did, p in missing_products[:20]
                ]
            )
            raise ValueError(
                "Some products referenced in diets are missing from df_prod.index:\n"
                + lines
            )

        # --- 3) Récupérer mapping consumer -> diet_id pour (area, year) depuis Input data ---
        # On suppose Input data a colonnes: Area, Year, category, item, value  (value = Diet ID)
        # on sélectionne les lignes où category == 'Diet' (insensible à la casse)
        # et Area==area and Year==year
        # input_df2 = input_df.copy()

        # item and value
        # item = consumer (e.g. 'bovines'), value = diet id
        # col_item = "item"
        # col_value = "value"

        # filter relevant rows
        # mask = (
        #     (input_df2["Area"] == area)
        #     & (input_df2["Year"] == year)
        #     & (input_df2["category"] == "Diet")
        # )

        # mapping_rows = input_df2.loc[mask, [col_item, col_value]].copy()
        # if mapping_rows.empty:
        #     raise ValueError(
        #         f"No Diet mapping found in Input data for area={area}, year={year}"
        #     )

        # # build dict consumer -> diet_id (string)
        consumer_to_diet = {}
        # for _, r in mapping_rows.iterrows():
        #     consumer = str(r[col_item]).strip()
        #     diet_id_val = r[col_value]
        #     if pd.isna(diet_id_val):
        #         raise ValueError(
        #             f"Empty diet id for consumer '{consumer}' in Input data for {area}/{year}"
        #         )
        #     consumer_to_diet[consumer] = str(diet_id_val).strip()

        for index, row in self.df_elevage.iterrows():
            consumer_to_diet[index] = row["Diet"]
        for index, row in self.df_pop.iterrows():
            consumer_to_diet[index] = row["Diet"]
        for index, row in self.df_energy.iterrows():
            consumer_to_diet[index] = row["Diet"]

        # # --- 4) Vérifier que chaque index de df_elevage et df_pop a un mapping ---
        # # on normalise casse pour comparaison facile : on compare en minuscules des deux côtés
        # consumers_expected = set(
        #     [
        #         c.lower()
        #         for c in list(self.init_df_elevage.index) + list(self.init_df_pop.index)
        #     ]
        #     + ["methanizer"]
        # )
        # consumers_found = set([k.lower() for k in consumer_to_diet.keys()])

        # missing_consumers = sorted(list(consumers_expected - consumers_found))
        # if missing_consumers:
        #     raise ValueError(
        #         "Missing diet mapping for the following consumers (indexes in df_elevage, df_pop or 'mathanizer') for "
        #         f"{area}/{year}:\n" + "\n".join(missing_consumers)
        #     )

        # --- 5) Construire diet_by_consumer : consumer -> expanded DataFrame with proportions and products ---
        diet_by_consumer = {}
        for consumer, diet_id in consumer_to_diet.items():
            if diet_id not in diet_by_id:
                raise ValueError(
                    f"Diet id '{diet_id}' referenced for consumer '{consumer}' not present in Diet table"
                )
            df_d = diet_by_id[diet_id].copy()
            # ajouter colonne consumer, diet_id utile
            df_d["Consumer"] = consumer
            df_d["DietID"] = diet_id
            # Products restent listes
            diet_by_consumer[consumer] = df_d.reset_index(drop=True)

        diet_by_consumer = pd.concat(diet_by_consumer.values(), ignore_index=True)

        # option: stocker dans self
        self.diet_by_id = diet_by_id
        self.diet_by_consumer = diet_by_consumer
        self.consumer_to_diet = consumer_to_diet

        return diet_by_consumer

    def generate_input_data(self, area, year, prospect=False):
        self.generate_df_prod(area, year, prospect)
        self.generate_df_cultures(area, year, prospect)
        self.generate_df_elevage(area, year, prospect)
        self.generate_df_excr(area, year, prospect)
        self.generate_df_pop(area, year)
        self.generate_df_energy(area, year)
        self.get_global_metrics(area, year, prospect=prospect)


class FluxGenerator:
    """This class generates and manages the transition matrix of fluxes between various sectors (e.g., agriculture, livestock, industry, trade).

    The transition matrix is used to model the flow of resources or interactions between sectors, where each entry in the matrix represents the relationship or flow between a source sector and a target sector.

    Args:
        labels (list): A list of labels representing the sectors (e.g., ['agriculture', 'livestock', 'industry', 'trade']) in the model. These labels are used to index the transition matrix and identify the sectors in the flux calculations.

    Attributes:
        labels (list): The list of labels (sectors) that are used to define the transition matrix.
        label_to_index (dict): A dictionary mapping each label (sector) to its corresponding index in the adjacency matrix.
        n (int): The number of sectors (i.e., the length of the labels list).
        adjacency_matrix (numpy.ndarray): A square matrix of size n x n representing the fluxes between sectors. Each element in the matrix holds the transition coefficient between a source and a target sector.

    Methods:
        generate_flux(source, target):
            Generates and updates the transition matrix by calculating flux coefficients between the source and target sectors. The coefficients are based on the provided `source` and `target` dictionaries.

        get_coef(source_label, target_label):
            Retrieves the transition coefficient for the flux between two sectors (identified by their labels) from the transition matrix.
    """

    def __init__(self, labels):
        """Initializes the FluxGenerator with a list of sector labels.

        Args:
            labels (list): List of labels representing sectors in the model.
        """
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        self.n = len(self.labels)
        self.adjacency_matrix = np.zeros((self.n, self.n))

    def generate_flux(self, source, target, erase=False):
        """Generates and updates the transition matrix by calculating the flux coefficients between the source and target sectors.

        Args:
            source (dict): A dictionary representing the source sector, where keys are sector labels and values are the corresponding flux values.
            target (dict): A dictionary representing the target sector, where keys are sector labels and values are the corresponding flux values.
            erase (bool): a boolean to add flows to existing flows or to erase previously stored flow and replace it with a new value

        This method updates the adjacency matrix by computing the flux between all pairs of source and target sectors.
        A flux coefficient is calculated as the product of the corresponding values from the `source` and `target` dictionaries.
        If the coefficient exceeds a small threshold (10^-7), it is added to the matrix at the corresponding position.
        """
        for source_label, source_value in source.items():
            source_index = self.label_to_index.get(source_label)
            if source_index is None:
                print(f"{source_label} not found in label_to_index")
            for target_label, target_value in target.items():
                coefficient = source_value * target_value
                target_index = self.label_to_index.get(target_label)
                if target_index is not None:
                    if coefficient > 10**-7:
                        if erase:
                            self.adjacency_matrix[source_index, target_index] = (
                                coefficient
                            )
                        else:
                            self.adjacency_matrix[source_index, target_index] += (
                                coefficient
                            )
                else:
                    print(f"{target_label} not found in label_to_index")

    def get_coef(self, source_label, target_label):
        """Retrieves the transition coefficient between two sectors from the adjacency matrix.

        Args:
            source_label (str): The label of the source sector.
            target_label (str): The label of the target sector.

        Returns:
            float or None: The transition coefficient between the source and target sectors. Returns None if either sector is not found in the matrix.
        """
        source_index = self.label_to_index.get(source_label)
        target_index = self.label_to_index.get(target_label)
        if source_index is not None and target_index is not None:
            return self.adjacency_matrix[source_index][target_index]
        else:
            return None


class NitrogenFlowModel:
    """This class models the nitrogen flow in an agricultural system, calculating fluxes and nitrogen dynamics
    for different sectors (e.g., crops, livestock) over a given year and area.

    The model incorporates various processes, including crop production, animal production, nitrogen emissions,
    and fertilization, and computes the corresponding nitrogen fluxes between sectors using transition matrices.

    This class provides methods to compute fluxes, generate heatmaps, and access matrices such as the transition matrix,
    core matrix, and adjacency matrix.

    For a detailed explanation of the model's methodology and mathematical foundations, please refer to the associated
    scientific paper: Tracking Nitrogen Dynamics: A Disaggregated Approach for French Agro-Food Systems, 2025, Adrien Fauste-Gay (pre-print).

    Args:
        data (DataLoader): An instance of the DataLoader class to load and preprocess the data.
        year (str): The year for which to compute the nitrogen flow model.
        area (str): The area for which to compute the nitrogen flow model.
        labels (list, optional): A list of labels representing the sectors in the model. Defaults to labels.

    Attributes:
        year (str): The year for which the nitrogen flow model is computed.
        area (str): The area for which the nitrogen flow model is computed.
        labels (list): The list of labels representing the sectors in the model.
        data_loader (DataLoader): The instance of the DataLoader used to load the data.
        culture_data (CultureData): An instance of the CultureData class for crop data.
        elevage_data (ElevageData): An instance of the ElevageData class for livestock data.
        flux_generator (FluxGenerator): An instance of the FluxGenerator class to generate flux coefficients.
        df_cultures (pandas.DataFrame): The dataframe containing crop data.
        df_elevage (pandas.DataFrame): The dataframe containing livestock data.
        adjacency_matrix (numpy.ndarray): The matrix representing nitrogen fluxes between sectors.
        label_to_index (dict): A dictionary mapping sector labels to matrix indices.

    Methods:
        compute_fluxes():
            Computes the nitrogen fluxes between the sectors of the model. This method populates the adjacency matrix
            with flux coefficients based on the interactions between crops, livestock, emissions, and fertilization.

        plot_heatmap():
            Generates a heatmap visualization of the transition matrix for nitrogen fluxes.

        plot_heatmap_interactive():
            Generates an interactive heatmap for the nitrogen fluxes.

        get_df_culture():
            Returns the dataframe containing crop data.

        get_df_elevage():
            Returns the dataframe containing livestock data.

        get_transition_matrix():
            Returns the transition matrix representing nitrogen fluxes between sectors.

        get_core_matrix():
            Returns the core matrix representing the primary fluxes in the system.

        get_adjacency_matrix():
            Returns the adjacency matrix, which is used to represent the complete set of nitrogen fluxes between sectors.

        extract_input_output_matrixs():
            Extracts and returns the input-output matrices of nitrogen fluxes.

        imported_nitrogen():
            Returns the total amount of imported nitrogen in the system.

        net_imported_plant():
            Returns the net imported nitrogen for plants.

        net_imported_animal():
            Returns the net imported nitrogen for livestock.

        total_plant_production():
            Returns the total plant production in the model.

        stacked_plant_production():
            Returns a stacked representation of plant production data.
    """

    def __init__(self, data, area, year, debug=False, prospective=False):
        """Initializes the NitrogenFlowModel class with the necessary data and model parameters.

        Args:
            data (DataLoader): An instance of the DataLoader class to load and preprocess the data.
            year (str): The year for which to compute the nitrogen flow model.
            area (str): The area for which to compute the nitrogen flow model.
            debug (bool): print a report on objective model terms
        """
        self.year = year
        self.area = area
        self.debug = debug
        self.prospective = prospective

        self.data_loader = data
        self.labels = data.labels
        self.label_to_index = data.label_to_index
        self.index_to_label = data.index_to_label

        self.flux_generator = FluxGenerator(self.labels)

        self.data_loader.generate_input_data(self.area, self.year, self.prospective)

        self.df_cultures = self.data_loader.df_cultures
        self.df_elevage = self.data_loader.df_elevage
        self.df_excr = self.data_loader.df_excr
        self.df_prod = self.data_loader.df_prod
        self.df_pop = self.data_loader.df_pop
        self.df_energy = self.data_loader.df_energy
        self.df_global = self.data_loader.global_df
        self.diets = data.load_diets_for_area_year(self.area, self.year)

        self._build_energy_power_map()

        self.adjacency_matrix = self.flux_generator.adjacency_matrix

        self.compute_fluxes()

    @staticmethod
    def _slug(s: str, maxlen: int = 40) -> str:
        s = re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")
        if len(s) <= maxlen:
            return s
        h = hashlib.md5(s.encode()).hexdigest()[:8]
        return s[: maxlen - 9] + "_" + h

    def _build_energy_power_map(self):
        """
        Construit un dict {facility: {item: power_MWh_per_ktN}}
        à partir de l'onglet 'Energy power' (MWh/tFW) + %N.
        Pour 'waste', on utilise df_global['Green waste nitrogen content (%)'].
        """
        from collections import defaultdict

        power_map = defaultdict(dict)

        tbl_name = "Energy power"
        if tbl_name not in self.data_loader.df_data:
            self.energy_power_map = power_map
            return power_map

        tbl = self.data_loader.df_data[tbl_name].copy()

        # Filtres optionnels si présents
        if "Area" in tbl.columns:
            tbl = tbl[(tbl["Area"].isna()) | (tbl["Area"] == self.area)]
        if "Year" in tbl.columns:
            tbl = tbl[(tbl["Year"].isna()) | (tbl["Year"] == self.year)]

        required = {"Facility", "Items", "Energy Power (MWh/tFW)"}
        missing = required - set(tbl.columns)
        if missing:
            raise KeyError(f"'Energy power' sheet missing columns: {missing}")

        # %N pour les wastes
        green_waste_n_pct = float(
            self.df_global.loc["Green waste nitrogen content (%)", "value"]
        )

        for _, r in tbl.iterrows():
            fac = str(r["Facility"]).strip()
            items_cell = r["Items"]
            pw_tf = float(r["Energy Power (MWh/tFW)"])  # MWh/tFW

            if not fac or not np.isfinite(pw_tf):
                continue

            # split multi-items "a, b, c"
            if pd.isna(items_cell):
                items = []
            elif isinstance(items_cell, (list, tuple)):
                items = [str(x).strip() for x in items_cell]
            else:
                items = [s.strip() for s in str(items_cell).split(",") if s.strip()]

            for it in items:
                # convertir MWh/tFW -> MWh/ktN avec le %N
                if it in self.df_prod.index:
                    n_pct = float(self.df_prod.loc[it, "Nitrogen Content (%)"])
                elif it in self.df_excr.index:
                    n_pct = float(self.df_excr.loc[it, "Nitrogen Content (%)"])
                elif it == "waste":
                    n_pct = green_waste_n_pct
                else:
                    warnings.warn(
                        f"[Energy power] Item '{it}' inconnu ; ignoré pour facility '{fac}'."
                    )
                    continue

                if n_pct <= 0:
                    warnings.warn(
                        f"[Energy power] %N=0 pour '{it}' ; MWh/tFW pas convertible -> ignoré."
                    )
                    continue

                mwh_per_ktn = pw_tf * 1000 / (n_pct / 100.0)
                power_map[fac][it] = mwh_per_ktn

        self.energy_power_map = power_map
        return power_map

    def _conv_MWh_per_ktN(self, facility: str, item: str) -> float:
        """
        Retourne le pouvoir énergétique spécifique (MWh/ktN) pour (facility,item).
        0.0 si non défini (ce qui forcera énergie=0 pour cet item dans cette facility).
        """
        pm = getattr(self, "energy_power_map", {}) or {}
        return float(pm.get(facility, {}).get(item, 0.0))

    # def _secants_for_yield(
    #     self, Ymax: float, Fstar: float, breaks: list[float]
    # ) -> list[tuple[float, float]]:
    #     """
    #     Construit des sécantes (m,b) pour majorer Y(F) = Ymax * (1 - exp(-F/Fstar)).
    #     Chaque paire (m,b) sert dans la contrainte: y <= m * F + b.

    #     Paramètres
    #     ----------
    #     Ymax : float
    #         Rendement frais maximum (tFW/ha).
    #     Fstar : float
    #         Fertilisation caractéristique (kgN/ha).
    #     breaks : list[float]
    #         Liste de points de rupture en F (kgN/ha), ex: [0, 0.25F*, 0.5F*, 1.0F*, 1.5F*, 2.0F*, 3.0F*].

    #     Retour
    #     ------
    #     List[Tuple[m, b]]
    #         Sécantes entre points consécutifs (Fi, Yi) et (Fj, Yj), plus éventuellement
    #         d'autres protections que tu ajoutes ailleurs (ex: cap y <= Ymax).

    #     Notes
    #     -----
    #     - La fonction est concave et croissante; les sécantes donnent donc une
    #     sur-approximation (borne supérieure) parfaite pour un LP.
    #     - On ajoute *en plus* côté modèle un cap explicite y <= Ymax (à garder).
    #     """

    #     # Cas limites
    #     if Ymax <= 0 or Fstar <= 0:
    #         return [(0.0, 0.0)]  # y <= 0

    #     # Nettoyage/sécurisation des breaks
    #     B = sorted(set(float(b) for b in breaks if b is not None and b >= 0.0))
    #     if not B or B[0] > 1e-12:
    #         B = [0.0] + B
    #     # Déduplique les éventuels très proches
    #     B_unique = [B[0]]
    #     for x in B[1:]:
    #         if x - B_unique[-1] > 1e-9:
    #             B_unique.append(x)
    #     B = B_unique

    #     # Points (F, Y(F))
    #     def Y(F):
    #         return Ymax * (1.0 - math.exp(-F / Fstar))

    #     pts = [(F, Y(F)) for F in B]

    #     segs: list[tuple[float, float]] = []
    #     for (Fi, Yi), (Fj, Yj) in zip(pts[:-1], pts[1:]):
    #         if Fj <= Fi + 1e-12:
    #             continue
    #         m = (Yj - Yi) / (Fj - Fi)
    #         b = Yi - m * Fi
    #         segs.append((m, b))

    #     # Sécurité : si aucun segment produit (ex: un seul break), on place la tangente en 0
    #     if not segs:
    #         # Tangente en F=0 : dY/dF|0 = Ymax/Fstar => y <= (Ymax/Fstar)*F
    #         segs.append((Ymax / Fstar, 0.0))

    #     return segs

    # @staticmethod
    # def yield_model(Ymax: float, Fstar: float, F: float) -> float:
    #     """Formule du rendement (exponentielle)"""
    #     return Ymax * (1.0 - np.exp(-F / Fstar))

    # def _secants_for_yield_2(
    #     self, Ymax: float, Fstar: float, breaks: list[float]
    # ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    #     """
    #     Calcule les sécantes (majoration concave) et les tangentes (minoration convexe)
    #     pour encadrer la courbe de rendement Y(F) = Ymax * (1 - exp(-F/Fstar)).

    #     Paramètres
    #     ----------
    #     Ymax : float
    #         Rendement frais maximum (tFW/ha).
    #     Fstar : float
    #         Fertilisation caractéristique (kgN/ha).
    #     breaks : list[float]
    #         Liste de points de rupture en F (kgN/ha).

    #     Retour
    #     ------
    #     Tuple(List[Tuple[m_sec, b_sec]], List[Tuple[m_tan, b_tan]])
    #         - La première liste contient les coefficients des sécantes (borne supérieure: y <= m*F + b).
    #         - La seconde liste contient les coefficients des tangentes (borne inférieure: y >= m'*F + b').
    #     """

    #     # --- Initialisation et Nettoyage ---

    #     # Cas limites
    #     if Ymax <= 0 or Fstar <= 0:
    #         return [(0.0, 0.0)], [(0.0, 0.0)]

    #     # Nettoyage/sécurisation des breaks (comme dans votre code)
    #     B = sorted(set(float(b) for b in breaks if b is not None and b >= 0.0))
    #     if not B or B[0] > 1e-12:
    #         B = [0.0] + B
    #     B_unique = [B[0]]
    #     for x in B[1:]:
    #         if x - B_unique[-1] > 1e-9:
    #             B_unique.append(x)
    #     B = B_unique

    #     # Points (F, Y(F))
    #     pts = [(F, self.yield_model(Ymax, Fstar, F)) for F in B]

    #     # Dérivée de Y(F) : dY/dF = (Ymax / Fstar) * exp(-F / Fstar)
    #     def dY_dF(F: float) -> float:
    #         return (Ymax / Fstar) * math.exp(-F / Fstar)

    #     # --- 1. Calcul des Sécantes (Majoration Concave / Borne Supérieure) ---
    #     # La sécante entre (Fi, Yi) et (Fj, Yj) est toujours au-dessus de la courbe.
    #     segs_upper: list[tuple[float, float]] = []
    #     for (Fi, Yi), (Fj, Yj) in zip(pts[:-1], pts[1:]):
    #         if Fj <= Fi + 1e-12:
    #             continue
    #         m_sec = (Yj - Yi) / (Fj - Fi)  # Pente de la sécante
    #         b_sec = Yi - m_sec * Fi        # Ordonnée à l'origine
    #         segs_upper.append((m_sec, b_sec))

    #     # --- 2. Calcul des Tangentes (Minoration Convexe / Borne Inférieure) ---
    #     # La tangente en chaque point (Fi, Yi) est toujours au-dessous de la courbe.
    #     segs_lower: list[tuple[float, float]] = []

    #     # On ajoute une tangente pour CHAQUE point de rupture
    #     for Fi, Yi in pts:
    #         m_tan = dY_dF(Fi)           # Pente de la tangente au point Fi
    #         b_tan = Yi - m_tan * Fi     # Ordonnée à l'origine (Y = m*F + b => b = Y - m*F)
    #         segs_lower.append((m_tan, b_tan))

    #     # Sécurité (si B n'a qu'un ou deux points proches)
    #     if not segs_upper:
    #         # La tangente en 0 est aussi la sécante par défaut
    #         m_tan_0 = dY_dF(0.0)
    #         segs_upper.append((m_tan_0, 0.0))
    #         if not segs_lower:
    #             segs_lower.append((m_tan_0, 0.0)) # La tangente en 0 est déjà la première dans segs_lower

    #     return segs_upper, segs_lower

    @staticmethod
    def _Y_func(F, Ymax, Fstar):
        if Fstar <= 0:
            return 0.0
        return Ymax * (1.0 - math.exp(-F / Fstar))

    def _pre_lp_supply(self, prob):
        """
        Mode prospectif :
        - y_c (tFW/ha) via Y(F) (sécantes) avec 'Maximum Yield (tFW/ha)' et 'Characteristic fertilisation (kgN/ha)'
        - F total (kgN/ha) alimenté par dépôt + organique épandu + graines + BNF + synthèse_eff
        - Synthèse s_c (ktN) avec pertes NH3/N2O
        - BNF = affine(yFW)
        - Graines seeds_ktN = ratio * P_c (ktN)
        - Q_p (ktN) = ratios "Co-Production Ratio (%)" en frais × %N produit
        - Variables d'excès synthétique
        """
        if not self.prospective:
            return
        import pandas as pd
        from pulp import LpVariable, lpSum

        df_cu, df_pr, df_ex, df_pop, df_gl = (
            self.df_cultures,
            self.df_prod,
            self.df_excr,
            self.df_pop,
            self.df_global,
        )

        # ---------- A) Dépôts / organique épandu (kgN/ha) ----------
        if "Atmospheric deposition (kgN/ha)" in df_cu.columns:
            depo_kg_ha = {
                c: float(df_cu.at[c, "Atmospheric deposition (kgN/ha)"] or 0.0)
                for c in df_cu.index
            }
        else:
            depo_val = float(df_gl.loc["Atmospheric deposition coef (kgN/ha)", "value"])
            depo_kg_ha = {c: depo_val for c in df_cu.index}

        denom = (df_cu["Area (ha)"] * df_cu["Spreading Rate (%)"] / 100.0).sum()
        share = {
            c: (df_cu.at[c, "Area (ha)"] * df_cu.at[c, "Spreading Rate (%)"] / 100.0)
            / max(1e-9, denom)
            for c in df_cu.index
        }

        tot_boue = (
            df_pop["Ingestion (ktN)"] * df_pop["Excretion recycling (%)"] / 100.0
        ).sum()
        excr_fields = df_ex.loc[
            df_ex["Type"].isin(["manure", "slurry"]),
            "Excretion after volatilization (ktN)",
        ].sum()

        excr_meadow = df_ex.loc[
            df_ex["Type"].isin(["grasslands excretion"]),
            "Excretion after volatilization (ktN)",
        ].sum()
        share_meadow = self.target_grass

        boue_kg_ha, excr_kg_ha = {}, {}
        for c in df_cu.index:
            A = float(df_cu.at[c, "Area (ha)"]) or 0.0
            boue_k = share.get(c, 0.0) * tot_boue
            excr_k = share.get(c, 0.0) * excr_fields
            boue_kg_ha[c] = (boue_k * 1e6 / A) if A > 0 else 0.0
            excr_kg_ha[c] = (excr_k * 1e6 / A) if A > 0 else 0.0
            excr_kg_ha[c] += (
                (share_meadow.get(c, 0.0) * excr_meadow * 1e6 / A) if A > 0 else 0.0
            )

        org_only_kg_ha = {c: boue_kg_ha[c] + excr_kg_ha[c] for c in df_cu.index}
        O_base_kg_ha = {c: org_only_kg_ha[c] for c in df_cu.index}

        # store
        self._pros_vars = {}
        self._pros_vars["depo_kg_ha"] = depo_kg_ha
        self._pros_vars["org_only_kg_ha"] = org_only_kg_ha
        self._pros_vars["O_base_kg_ha"] = O_base_kg_ha

        # ---------- B) Variables principales ----------
        f_c = {
            c: LpVariable(f"f_{self._slug(c)}", lowBound=0) for c in df_cu.index
        }  # kgN/ha
        y_c = {
            c: LpVariable(f"yFW_{self._slug(c)}", lowBound=0) for c in df_cu.index
        }  # tFW/ha
        s_c = {
            c: LpVariable(f"s_{self._slug(c)}", lowBound=0) for c in df_cu.index
        }  # ktN
        self._pros_vars["f_c"] = f_c
        self._pros_vars["y_c"] = y_c
        self._pros_vars["s_c"] = s_c

        # NOUVELLES VARIABLES pour la linéarisation SOS2
        w_c_pts = {}  # Poids (w_i) pour chaque point de rupture
        b_c_segs = {}  # Sélecteur (b_k) pour chaque segment

        YCOL, FCOL = "Ymax (kgN/ha)", "Characteristic Fertilisation (kgN/ha)"

        # ---------- C) Linéarisation Y(F) ----------
        # Nouvelle version SOS2
        for c in df_cu.index:
            Ymax = float(df_cu.at[c, YCOL]) if YCOL in df_cu.columns else 0.0
            Fst = float(df_cu.at[c, FCOL]) if FCOL in df_cu.columns else 0.0

            if Ymax <= 0 or Fst <= 0:
                # Si pas de courbe, on fixe y_c = 0 et f_c = 0
                prob += (y_c[c] == 0, f"no_yield_{self._slug(c)}")
                prob += (f_c[c] == 0, f"no_fert_{self._slug(c)}")
                continue

            # 1. Définir les points de rupture (F, Y)
            B = [
                0.0,
                0.25 * Fst,
                0.5 * Fst,
                1.0 * Fst,
                1.5 * Fst,
                2.0 * Fst,
                3.0 * Fst,
                4.0 * Fst,
                5.0 * Fst,
                100.0 * Fst,
            ]
            # Nettoyage des breaks (similaire à votre fonction)
            B_unique = sorted(set(float(b) for b in B if b is not None and b >= 0.0))

            # pts contient la liste des (F_i, Y_i)
            pts = [(F, self._Y_func(F, Ymax, Fst)) for F in B_unique]
            F_pts = [p[0] for p in pts]
            Y_pts = [p[1] for p in pts]

            num_breaks = len(pts)
            num_segs = num_breaks - 1

            if num_segs <= 0:
                # Cas bizarre (un seul point), on fixe à ce point
                prob += (y_c[c] == Y_pts[0], f"fixed_yield_{self._slug(c)}")
                prob += (f_c[c] == F_pts[0], f"fixed_fert_{self._slug(c)}")
                continue

            # 2. Créer les variables w_i et b_k pour cette culture 'c'
            w_c_pts[c] = LpVariable.dicts(
                f"w_{self._slug(c)}", range(num_breaks), lowBound=0
            )
            b_c_segs[c] = LpVariable.dicts(
                f"b_{self._slug(c)}", range(num_segs), cat=LpBinary
            )

        for c in w_c_pts.keys():  # Boucle sur les cultures qui ont des variables w/b
            # Récupérer les variables et les points
            w_vars = w_c_pts[c]
            b_vars = b_c_segs[c]

            # Recalculer les points (ou les stocker)
            Ymax = float(df_cu.at[c, YCOL])
            Fst = float(df_cu.at[c, FCOL])
            F_MAX_ANCORAGE = 1000  # kgN/ha, fertilisation maximale avant de considérer qu'on est à Ymax
            B = [0.0, 0.25 * Fst, 0.5 * Fst, 1.0 * Fst, 1.5 * Fst, 2.0 * Fst, 3.0 * Fst]
            B_unique = sorted(set(float(b) for b in B if b is not None and b >= 0.0))

            pts = []
            for F in B_unique:
                # On ajoute les points calculés (F_i, Y_i)
                Y = self._Y_func(F, Ymax, Fst)
                pts.append((F, Y))

            F_last_calc = pts[-1][0] if pts else 0.0
            F_anchor = max(F_last_calc * 1.5, F_MAX_ANCORAGE)

            if abs(pts[-1][1] - Ymax) > 1e-6:
                pts.append((F_anchor, Ymax))

            pts_final = sorted(list(set(pts)))
            F_pts = [p[0] for p in pts_final]
            Y_pts = [p[1] for p in pts_final]

            num_breaks = len(pts)
            num_segs = num_breaks - 1

            # ----- LES 5 CONTRAINTES CLÉS -----

            # 1. Sélection d'un seul segment
            prob += (
                lpSum(b_vars[k] for k in range(num_segs)) == 1,
                f"SOS2_select_seg_{self._slug(c)}",
            )

            # 2. Somme des poids de la combinaison convexe
            prob += (
                lpSum(w_vars[i] for i in range(num_breaks)) == 1,
                f"SOS2_sum_weights_{self._slug(c)}",
            )

            # 3. Reconstitution de f_c
            prob += (
                f_c[c] == lpSum(w_vars[i] * F_pts[i] for i in range(num_breaks)),
                f"SOS2_calc_f_{self._slug(c)}",
            )

            # 4. Reconstitution de y_c
            prob += (
                y_c[c] == lpSum(w_vars[i] * Y_pts[i] for i in range(num_breaks)),
                f"SOS2_calc_y_{self._slug(c)}",
            )

            # 5. Lien "SOS2" (forcer les w_i à 0 sauf autour du segment b_k choisi)
            # w_0 ne peut être actif que si b_0 l'est
            prob += (w_vars[0] <= b_vars[0], f"SOS2_link_w{0}_{self._slug(c)}")

            # Les w_i intermédiaires ne peuvent être actifs que si b_{i-1} ou b_i l'est
            for i in range(1, num_segs):
                prob += (
                    w_vars[i] <= b_vars[i - 1] + b_vars[i],
                    f"SOS2_link_w{i}_{self._slug(c)}",
                )

            # Le dernier poids w_N ne peut être actif que si le dernier segment b_{N-1} l'est
            prob += (
                w_vars[num_breaks - 1] <= b_vars[num_segs - 1],
                f"SOS2_link_w{num_breaks - 1}_{self._slug(c)}",
            )

            # La contrainte y_c <= Ymax est maintenant implicite,
            # mais la garder ne fait pas de mal (elle peut aider le solveur)
            prob += (y_c[c] <= Ymax, f"cap_Ymax__{self._slug(c)}")

        self._pros_vars["w_c_pts"] = w_c_pts
        self._pros_vars["b_c_segs"] = b_c_segs

        # ---------- D) P_c (ktN) à partir de yFW ----------
        main_of = df_cu["Main Production"].astype(str).to_dict()
        # NC = (
        #     df_pr["Nitrogen Content (%)"]
        #     if "Nitrogen Content (%)" in df_pr.columns
        #     else pd.Series(0.0, index=df_pr.index)
        # )
        P_c = {
            c: LpVariable(f"PmainN_{self._slug(c)}", lowBound=0) for c in df_cu.index
        }
        for c in df_cu.index:
            A = float(df_cu.at[c, "Area (ha)"]) or 0.0
            # main = main_of.get(c, "")
            # nc_main = float(NC.get(main, 0.0))
            prob += (
                P_c[c] == y_c[c] * (A / 1e6),
                f"link_PmainN__{self._slug(c)}",
            )
        self._pros_vars["P_c"] = P_c

        # ---------- E) Q_p (ktN) via ratios FW ----------
        Q_p = {
            p: LpVariable(f"Q_{self._slug(p)}", lowBound=0)
            for p in df_pr.index
            if str(df_pr.at[p, "Type"]).strip().lower() == "plant"
        }
        origin = (
            df_pr["Origin compartment"].astype(str)
            if "Origin compartment" in df_pr.columns
            else pd.Series("", index=df_pr.index)
        )
        has_share = "Co-Production Ratio (%)" in df_pr.columns
        for p in Q_p:
            c = origin.at[p]
            if c not in y_c:
                continue
            s_fw = 1.0 if (p == main_of.get(c, "")) else 0.0
            if has_share:
                val = df_pr.at[p, "Co-Production Ratio (%)"]
                if pd.notna(val):
                    s_fw = float(val) / 100.0
            A = float(df_cu.at[c, "Area (ha)"]) or 0.0
            # nc_p = float(NC.get(p, 0.0))
            prob += (
                Q_p[p] == s_fw * y_c[c] * (A / 1e6),  # * (nc_p / 100.0),
                f"prodN_from_FW__{self._slug(p)}",
            )
        self._pros_vars["Q_p"] = Q_p

        # ---------- F) BNF affine(yFW) & split use/transfer ----------
        bnf_c = {
            c: LpVariable(f"bnf_{self._slug(c)}", lowBound=0) for c in df_cu.index
        }  # kgN/ha
        for c in df_cu.index:
            HI = float(df_cu.at[c, "Nitrogen Harvest Index"] or 1.0)
            a = float(df_cu.at[c, "BNF alpha"] or 0.0)
            b = float(df_cu.at[c, "BNF beta"] or 0.0)
            BGN = float(df_cu.at[c, "BGN"] or 0.0)
            a_c = a * BGN / max(1e-9, HI)
            b_c = b * BGN
            prob += bnf_c[c] == a_c * y_c[c] + b_c, f"bnf_affine__{self._slug(c)}"
        self._pros_vars["bnf_c"] = bnf_c

        # ---------- G) Graines : seeds_ktN = r_seed * P_c ----------
        seeds_ktN = {
            c: LpVariable(f"seeds_{self._slug(c)}", lowBound=0) for c in df_cu.index
        }
        for c in df_cu.index:
            r_seed = float(df_cu.at[c, "Seed input (ktN/ktN)"] or 0.0)
            prob += seeds_ktN[c] == r_seed * P_c[c], f"seeds_link__{self._slug(c)}"
        self._pros_vars["seeds_ktN"] = seeds_ktN

        # ---------- H) Synthèse avec pertes NH3/N2O ----------
        a = (
            float(
                df_gl.loc[
                    "coefficient N-NH3 volatilization synthetic fertilization (%)",
                    "value",
                ]
            )
            / 100.0
        )
        b = (
            float(
                df_gl.loc[
                    "coefficient N-N2O emission synthetic fertilization (%)", "value"
                ]
            )
            / 100.0
        )
        eff_syn = max(0.0, 1.0 - a - b)

        for c in df_cu.index:
            A = float(df_cu.at[c, "Area (ha)"]) or 0.0
            seeds_gha = (seeds_ktN[c] * 1e6 / A) if A > 0 else 0.0
            synth_gha = (eff_syn * s_c[c] * 1e6 / A) if A > 0 else 0.0
            # seule la part efficace (eff_syn*s_c) nourrit F
            prob += (
                f_c[c]
                == depo_kg_ha[c] + O_base_kg_ha[c] + bnf_c[c] + seeds_gha + synth_gha,
                f"Feff_def__{self._slug(c)}",
            )
            # légumineuses : pas de synthé + borne
            if str(df_cu.at[c, "Category"]).strip().lower() == "leguminous":
                prob += s_c[c] == 0, f"no_synth_on_legumes__{self._slug(c)}"
                # prob += (
                #     f_c[c] == O_base_kg_ha[c] + bnf_c[c] + seeds_gha,
                #     f"f_cap_legume__{self._slug(c)}",
                # )

        # ---------- I) Excès synthétique (pénalisation) ----------
        is_grass = df_cu["Category"].isin(["natural meadows", "temporary meadows"])
        S_crops = lpSum(s_c[c] for c in df_cu.index if not bool(is_grass.loc[c]))
        S_grass = lpSum(s_c[c] for c in df_cu.index if bool(is_grass.loc[c]))
        exc_crops = LpVariable("excess_synth_crops", lowBound=0)
        exc_grass = LpVariable("excess_synth_grass", lowBound=0)
        th_crops = float(
            df_gl.loc["Total Synthetic Fertilizer Use on crops (ktN)", "value"]
        )
        th_grass = float(
            df_gl.loc["Total Synthetic Fertilizer Use on grasslands (ktN)", "value"]
        )
        prob += exc_crops >= S_crops - th_crops
        prob += exc_grass >= S_grass - th_grass

        self._pros_vars.update(
            S_crops=S_crops,
            S_grass=S_grass,
            exc_crops=exc_crops,
            exc_grass=exc_grass,
            th_crops=th_crops,
            th_grass=th_grass,
        )

        # --- J) pénalité de répartition autour de F* ---
        # Poids lu dans df_global (0 si absent)
        w_syn_dist = self.df_global.loc["Weight synthetic distribution", "value"]

        if w_syn_dist > 0:
            from pulp import LpVariable, lpSum

            # Variables d’écart absolu: f_c[c] - F*_c = dev_plus - dev_minus, dev_* >= 0
            dev_plus = {}
            dev_minus = {}

            is_non_legume = ~self.df_cultures["Category"].astype(str).str.lower().eq(
                "leguminous"
            )

            for c in self.df_cultures.index:
                A = float(self.df_cultures.at[c, "Area (ha)"] or 0.0)
                Fst = (
                    float(self.df_cultures.at[c, FCOL])
                    if FCOL in self.df_cultures.columns
                    else 0.0
                )

                if not bool(is_non_legume.loc[c]):
                    continue

                if A <= 0 or Fst <= 0:
                    continue  # rien à faire si pas d’aire ou pas de F*

                # 1) variables
                dev_plus[c] = LpVariable(f"devp_{self._slug(c)}", lowBound=0)
                dev_minus[c] = LpVariable(f"devm_{self._slug(c)}", lowBound=0)

                # 2) égalité d’écart (kgN/ha)
                # f_c est déjà en kgN/ha (fertilisation totale, orga + minérale)
                prob += (
                    f_c[c] - Fst == (dev_plus[c] - dev_minus[c]) * Fst,
                    f"dev_abs_link_{self._slug(c)}",
                )

            # stocke pour _extra_objective
            self._pros_vars["devF_rel_pos"] = dev_plus
            self._pros_vars["devF_rel_neg"] = dev_minus

    # ── HOOK 2 ─────────────────────────────────────────────────────────────────
    def _rhs_for_product(self, p, default_rhs, is_plant):
        if not self.prospective:
            return default_rhs
        Q_p = self._pros_vars.get("Q_p", {})
        if is_plant and p in Q_p:
            # pertes/other uses côté df_prod (tu les remets à jour en post_solve)
            wasted = (
                float(self.df_prod.at[p, "Waste (%)"])
                if "Waste (%)" in self.df_prod.columns
                else 0.0
            )
            other = (
                float(self.df_prod.at[p, "Other uses (%)"])
                if "Other uses (%)" in self.df_prod.columns
                else 0.0
            )
            return Q_p[p] * (1.0 - wasted / 100.0 - other / 100.0)
        return default_rhs

    # ── HOOK 3 ─────────────────────────────────────────────────────────────────
    def _extra_objective(self):
        if not self.prospective:
            return 0
        W_SYN = float(self.df_global.loc["Weight synthetic fertilizer", "value"])
        th_crops = self._pros_vars["th_crops"]
        th_grass = self._pros_vars["th_grass"]
        exc_crops = self._pros_vars["exc_crops"]
        exc_grass = self._pros_vars["exc_grass"]
        # normalisation par les seuils pour avoir un ordre de grandeur ~0-1
        term = (exc_crops / max(1e-9, th_crops)) + (exc_grass / max(1e-9, th_grass))

        # Terme de distribution de l'azote synthétique
        W_DIS = float(self.df_global.loc["Weight synthetic distribution", "value"])
        distribution_term = 0
        if W_DIS > 0:
            dev_pos = self._pros_vars.get("devF_rel_pos", {})
            dev_neg = self._pros_vars.get("devF_rel_neg", {})
            distribution_term = lpSum(dev_pos[c] + dev_neg[c] for c in dev_pos.keys())

        return W_SYN * term + W_DIS * distribution_term

    # ── HOOK 4 ─────────────────────────────────────────────────────────────────
    def _post_solve_supply(self):
        if not self.prospective:
            return

        df_cu = self.df_cultures.copy()
        df_pr = self.df_prod.copy()

        # Vars LP
        s_c = self._pros_vars["s_c"]  # ktN
        P_c = self._pros_vars["P_c"]  # ktN (prod principale en N)
        Q_p = self._pros_vars["Q_p"]  # ktN (prod par produit)
        bnf_c = self._pros_vars["bnf_c"]  # kgN/ha
        seeds_ktN = self._pros_vars["seeds_ktN"]  # ktN
        depo_kg_ha = self._pros_vars["depo_kg_ha"]  # kgN/ha

        # === Production par produit (ktN) & disponibles ===
        if "Nitrogen Production (ktN)" not in df_pr.columns:
            df_pr["Nitrogen Production (ktN)"] = 0.0
        for p, var in Q_p.items():
            df_pr.at[p, "Nitrogen Production (ktN)"] = float(var.varValue or 0.0)

        if "Waste (%)" in df_pr.columns and "Other uses (%)" in df_pr.columns:
            df_pr["Nitrogen Wasted (ktN)"] = (
                df_pr["Nitrogen Production (ktN)"] * df_pr["Waste (%)"] / 100.0
            )
            df_pr["Nitrogen for Other uses (ktN)"] = (
                df_pr["Nitrogen Production (ktN)"] * df_pr["Other uses (%)"] / 100.0
            )
            df_pr["Available Nitrogen Production (ktN)"] = (
                df_pr["Nitrogen Production (ktN)"]
                - df_pr["Nitrogen Wasted (ktN)"]
                - df_pr["Nitrogen for Other uses (ktN)"]
            )

        # Colonnes garanties
        cols = [
            "Adjusted Total Synthetic Fertilizer Use (ktN)",
            "Adjusted Surface Synthetic Fertilizer Use (kgN/ha)",
            "Synthetic to field (ktN)",
            "Volatilized Nitrogen N-NH3 (ktN)",
            "Volatilized Nitrogen N-N2O (ktN)",
            "BNF (kgN/ha)",
            "BNF (ktN)",
            "Organic Fertilization (ktN)",
            "Atmospheric deposition (ktN)",
            "Seeds Input (ktN)",
            "Seeds Input (kgN/ha)",
            "Surface Non Synthetic Fertilizer Use (kgN/ha)",
            "Total Non Synthetic Fertilizer Use (ktN)",
            "Harvested Production (ktN)",
            # Nouvelles colonnes bilan récolte & surplus
            "Yield (kgN/ha)",
            "Yield (qtl/ha)",
            "Main Nitrogen Production (ktN)",
            "Production (kton)",
        ]
        for c in cols:
            if c not in df_cu.columns:
                df_cu[c] = 0.0

        # Boucle cultures
        for c in df_cu.index:
            A = float(df_cu.at[c, "Area (ha)"]) or 0.0

            # Synthétique & pertes
            syn_ktN = float(s_c[c].varValue or 0.0)

            # BNF & composantes non-synth
            bnf_gha = float(bnf_c[c].varValue or 0.0)
            seeds_k = float(seeds_ktN[c].varValue or 0.0)
            seeds_gha = (seeds_k * 1e6 / A) if A > 0 else 0.0
            depo_gha = float(depo_kg_ha[c] or 0.0)

            # Écritures de base
            df_cu.at[c, "Adjusted Total Synthetic Fertilizer Use (ktN)"] = syn_ktN
            df_cu.at[c, "Adjusted Surface Synthetic Fertilizer Use (kgN/ha)"] = (
                (syn_ktN * 1e6 / A) if A > 0 else 0.0
            )

            df_cu.at[c, "BNF (kgN/ha)"] = bnf_gha
            df_cu.at[c, "BNF (ktN)"] = bnf_gha * A / 1e6

            df_cu.at[c, "Atmospheric deposition (ktN)"] = depo_gha * A / 1e6
            df_cu.at[c, "Seeds Input (ktN)"] = seeds_k
            df_cu.at[c, "Seeds Input (kgN/ha)"] = seeds_gha

            # Production principale
            mainN_k = float(P_c[c].varValue or 0.0)
            Yield_kgN_ha = (mainN_k * 1e6 / A) if A > 0 else 0.0
            Yield_qtl_ha = (
                Yield_kgN_ha
                / 100
                / df_pr.loc[df_pr["Origin compartment"] == c, "Nitrogen Content (%)"]
                * 100.0
            ).item()  # 1 qtl = 100 kg
            df_cu.at[c, "Main Nitrogen Production (ktN)"] = mainN_k
            df_cu.at[c, "Yield (kgN/ha)"] = Yield_kgN_ha
            df_cu.at[c, "Yield (qtl/ha)"] = Yield_qtl_ha

            df_cu.at[c, "Harvested Production (ktN)"] = df_pr.loc[
                df_pr["Origin compartment"] == c, "Nitrogen Production (ktN)"
            ].sum()

        # On recalcule la production brute
        df_pr["Production (kton)"] = np.where(
            df_pr["Nitrogen Content (%)"] != 0,
            df_pr["Nitrogen Production (ktN)"] / (df_pr["Nitrogen Content (%)"] / 100),
            0,
        )

        self.df_cultures = df_cu
        self.df_prod = df_pr
        # self._recompute_soil_budget_unified()

    def _build_crops_livestock_to_product(self):
        if not self.prospective:
            # Calcul de la production totale récoltée d'azote par culture
            # Étape 1 : Calculer la somme de la production d'azote par "Origin compartment"
            nitrogen_production_sum = (
                self.df_prod.loc[self.df_prod["Type"] == "plant"]
                .groupby("Origin compartment")["Nitrogen Production (ktN)"]
                .sum()
            )

            # Étape 2 : Mettre à jour la colonne "Nitrogen Production (ktN)" dans df_cultures
            # Pandas aligne automatiquement les index de la série nitrogen_production_sum
            # avec l'index de df_cultures.
            self.df_cultures["Harvested Production (ktN)"] = nitrogen_production_sum

            # BNF
            HI_safe = self.df_cultures["Nitrogen Harvest Index"].replace(0, np.nan)

            target_fixation = (
                (
                    (
                        self.df_cultures["BNF alpha"]
                        * self.df_cultures["Yield (kgN/ha)"]
                        / HI_safe
                        + self.df_cultures["BNF beta"]
                    )
                    * self.df_cultures["BGN"]
                    * self.df_cultures["Area (ha)"]
                    / 1e6
                )
                .fillna(0)
                .to_dict()
            )
            self.df_cultures["BNF (ktN)"] = self.df_cultures.index.map(
                target_fixation
            ).fillna(0)

            # Depot atmospherique
            self.df_cultures["Atmospheric deposition (ktN)"] = (
                self.df_cultures["Area (ha)"]
                * self.df_global.loc["Atmospheric deposition coef (kgN/ha)", "value"]
                / 1e6
            )

    def _available_expr(self, p):
        """
        Expression (ou constante) de la production disponible de p (ktN),
        c'est-à-dire après Waste(%) et Other uses(%).
        - Prospectif + végétal: (1 - waste - other) * Q_p[p]   (variable LP)
        - Sinon: df_prod["Available Nitrogen Production (ktN)"] (constante)
        """
        df_pr = self.df_prod
        waste = (
            float(df_pr.at[p, "Waste (%)"]) / 100.0
            if "Waste (%)" in df_pr.columns
            else 0.0
        )
        other = (
            float(df_pr.at[p, "Other uses (%)"]) / 100.0
            if "Other uses (%)" in df_pr.columns
            else 0.0
        )

        if getattr(self, "prospective", False) and p in self._pros_vars.get("Q_p", {}):
            return (1.0 - waste - other) * self._pros_vars["Q_p"][
                p
            ]  # lpAffineExpression
        # fallback (historique ou non-végétal)
        if not self.prospective or df_pr.at[p, "Type"] == "animal":
            return float(df_pr.at[p, "Available Nitrogen Production (ktN)"])
        # très rare: si colonne absente, recompose
        prod = (
            float(df_pr.at[p, "Nitrogen Production (ktN)"])
            if "Nitrogen Production (ktN)" in df_pr.columns
            else 0.0
        )
        return prod * (1.0 - waste - other)

    def _prod_expr(self, p):
        """
        Expression (ou constante) de la production TOTALE (ktN), avant Waste/Other.
        Utile si une contrainte travaille 'sur la prod' et non sur la 'disponible'.
        """
        if getattr(self, "prospective", False) and p in self._pros_vars.get("Q_p", {}):
            return self._pros_vars["Q_p"][p]
        return (
            float(self.df_prod.at[p, "Nitrogen Production (ktN)"])
            if "Nitrogen Production (ktN)" in self.df_prod.columns
            else 0.0
        )

    def _recompute_soil_budget_unified(self):
        """
        Bilan azoté par culture dans l'esprit GRAFS (Julia Le Noë).

        Principes:
        - Surplus = Entrées AU CHAMP - Azote récolté (produits agrégés par culture).
        - Résidus & racines : flux internes vers le sol (calculés pour tracer les flux),
        mais non comptés comme "sorties" dans le surplus.
        - Partition du surplus:
            * Grandes cultures (non prairies): 70% lixiviation par défaut,
            une petite part N2O, le reste en stockage sol (ΔSOM-N).
            * Prairies: on stocke d'abord jusqu'à 100 kgN/ha; l'excédent est partitionné
            (70% lixiviation par défaut, faible N2O, reste stockage sol).
        - Mining (appauvrissement) = max(-Surplus, 0) : prélèvement net au stock du sol
        pour équilibrer la récolte quand les entrées sont insuffisantes.

        Colonnes créées (ktN sauf mention contraire):
        * Synthetic to field, Volatilized NH3/N2O (pré-champ)
        * Harvested Production (ktN)  (somme des produits d'origine "culture")
        * Inputs to field (ktN) = dep + BNF + excreta + digestat + seeds + synthetic_to_field
        * Surplus (ktN) = Inputs to field - Harvested N
        * Leached to hydro-system (ktN)   [surplus>0]
        * Surplus N2O (ktN)               [surplus>0]
        * Soil storage from surplus (ktN) [surplus>0]
        * Mining from soil (ktN) = max(-Surplus, 0)
        * Net Soil stock (ktN) = Soil storage from surplus - Mining
        * Surface indicators (kgN/ha)

        """

        import numpy as np

        df_cu = self.df_cultures.copy()
        df_pr = self.df_prod.copy()

        # --------- 0) Petits utilitaires sur df_global (coeffs paramétrables) ----------
        def _get_pct(name, default):
            try:
                return float(self.df_global.loc[name, "value"]) / 100.0
            except Exception:
                return default

        def _get_val(name, default):
            try:
                return float(self.df_global.loc[name, "value"])
            except Exception:
                return default

        # Pertes "pré-champ" sur engrais de synthèse (déjà dans ton modèle)
        coef_volat_NH3 = _get_pct(
            "coefficient N-NH3 volatilization synthetic fertilization (%)", 0.0
        )
        coef_N2O_dir = _get_pct(
            "coefficient N-N2O emission synthetic fertilization (%)", 0.0
        )
        eff_syn = max(0.0, 1.0 - coef_volat_NH3 - coef_N2O_dir)

        # Partitions du surplus (défauts cohérents GRAFS)
        f_leach_arable = _get_pct("share of surplus leached on arable (%)", 0.70)  # 70%
        f_leach_prairie = _get_pct(
            "share of surplus leached on grassland excess (%)", 0.70
        )
        f_n2o_arable = _get_pct(
            "share of surplus to N2O on arable (%)", 0.0075
        )  # 0.75%
        f_n2o_prairie = _get_pct(
            "share of surplus to N2O on grassland excess (%)", 0.0025
        )  # 0.25%
        prairie_store_thr = _get_val(
            "grassland surplus first stored (kgN/ha)", 100.0
        )  # 100 kgN/ha

        # --------- 1) Synthétique : pertes pré-champ & "to field" ----------
        syn_ktN = (
            df_cu["Adjusted Total Synthetic Fertilizer Use (ktN)"]
            .astype(float)
            .fillna(0.0)
        )

        # 1% des NH3 => N2O (ta règle), le reste NH3 reste NH3
        df_cu["Volatilized Nitrogen N-NH3 (ktN)"] = syn_ktN * 0.99 * coef_volat_NH3
        df_cu["Volatilized Nitrogen N-N2O (ktN)"] = syn_ktN * (
            coef_N2O_dir + 0.01 * coef_volat_NH3
        )
        df_cu["Synthetic to field (ktN)"] = syn_ktN * eff_syn

        # Surface (kgN/ha) pour info
        df_cu["Adjusted Surface Synthetic Fertilizer Use (kgN/ha)"] = 0.0
        area = df_cu["Area (ha)"].astype(float).fillna(0.0)
        mask_area = area > 0
        df_cu.loc[mask_area, "Adjusted Surface Synthetic Fertilizer Use (kgN/ha)"] = (
            syn_ktN[mask_area] * 1e6 / area[mask_area]
        )

        # --------- 2) Azote récolté par culture (somme des produits) ----------
        # # Si besoin, reconstituer le "Main Nitrogen Production (ktN)" depuis df_prod
        # if (df_cu.get("Main Nitrogen Production (ktN)", 0) == 0).all():
        #     main_map = df_pr.set_index("Product")["Nitrogen Production (ktN)"]
        #     df_cu["Main Nitrogen Production (ktN)"] = (
        #         df_cu["Main Production"].map(main_map).fillna(0.0)
        #     )

        harvested_by_culture = df_pr.groupby("Origin compartment")[
            "Nitrogen Production (ktN)"
        ].sum()
        df_cu["Harvested Production (ktN)"] = (
            df_cu.index.to_series().map(harvested_by_culture).fillna(0.0)
        )

        # --------- 4) Entrées au champ (Inputs to field) ----------
        inputs_to_field = (
            df_cu["Atmospheric deposition (ktN)"].astype(float).fillna(0.0)
            + df_cu["BNF (ktN)"].astype(float).fillna(0.0)
            + df_cu["Excreta Fertilization (ktN)"].astype(float).fillna(0.0)
            + df_cu["Digestat Fertilization (ktN)"].astype(float).fillna(0.0)
            + df_cu["Seeds Input (ktN)"].astype(float).fillna(0.0)
            + df_cu["Synthetic to field (ktN)"].astype(float).fillna(0.0)
        )
        df_cu["Inputs to field (ktN)"] = inputs_to_field

        # --------- 5) Surplus (coeur GRAFS) ----------
        surplus = inputs_to_field - df_cu["Harvested Production (ktN)"]
        df_cu["Surplus (ktN)"] = surplus

        pos_surplus = np.maximum(surplus.values, 0.0)
        neg_surplus = np.maximum(-surplus.values, 0.0)  # mining

        # --------- 6) Partition du surplus : prairies vs autres ----------
        # Init colonnes
        df_cu["Leached to hydro-system (ktN)"] = 0.0
        df_cu["Surplus N2O (ktN)"] = 0.0
        df_cu["Soil storage from surplus (ktN)"] = 0.0
        df_cu["Mining from soil (ktN)"] = neg_surplus  # direct

        categories = df_cu["Category"].astype(str).str.lower().fillna("")

        is_pasture = categories.isin(
            [
                "natural meadows",
                "temporary meadows",
            ]
        )

        # --- Prairies: stockage d'abord jusqu'au seuil (kg/ha), puis partition du reste
        # Conversion ktN <-> kgN/ha selon surface
        surplus_ktN = pos_surplus
        surplus_kg_ha = np.zeros_like(surplus_ktN)
        with np.errstate(divide="ignore", invalid="ignore"):
            surplus_kg_ha[mask_area.values] = (
                surplus_ktN[mask_area.values] * 1e6 / area[mask_area].values
            )

        store_first_kg_ha = np.minimum(surplus_kg_ha, prairie_store_thr)
        store_first_ktN = np.zeros_like(surplus_ktN)
        store_first_ktN[mask_area.values] = (
            store_first_kg_ha[mask_area.values] * area[mask_area].values / 1e6
        )

        remainder_ktN = np.maximum(surplus_ktN - store_first_ktN, 0.0)

        # Appliquer partition prairies vs arables
        # Arables (non prairies)
        arable_idx = (~is_pasture).values
        df_cu.loc[arable_idx, "Leached to hydro-system (ktN)"] = (
            f_leach_arable * surplus_ktN[arable_idx]
        ).astype(float)
        df_cu.loc[arable_idx, "Surplus N2O (ktN)"] = (
            f_n2o_arable * surplus_ktN[arable_idx]
        ).astype(float)
        df_cu.loc[arable_idx, "Soil storage from surplus (ktN)"] = (
            surplus_ktN[arable_idx]
            - df_cu.loc[arable_idx, "Leached to hydro-system (ktN)"].values
            - df_cu.loc[arable_idx, "Surplus N2O (ktN)"].values
        ).astype(float)

        # Prairies
        past_idx = is_pasture.values
        df_cu.loc[past_idx, "Leached to hydro-system (ktN)"] = (
            f_leach_prairie * remainder_ktN[past_idx]
        ).astype(float)
        df_cu.loc[past_idx, "Surplus N2O (ktN)"] = (
            f_n2o_prairie * remainder_ktN[past_idx]
        ).astype(float)
        df_cu.loc[past_idx, "Soil storage from surplus (ktN)"] = (
            store_first_ktN[past_idx]
            + remainder_ktN[past_idx]
            - df_cu.loc[past_idx, "Leached to hydro-system (ktN)"].values
            - df_cu.loc[past_idx, "Surplus N2O (ktN)"].values
        ).astype(float)

        # --------- 7) Bilan de stock de sol (Δ stock) ----------
        df_cu["Soil stock (ktN)"] = (
            +df_cu["Soil storage from surplus (ktN)"].values
            - df_cu["Mining from soil (ktN)"].values
        ).astype(float)

        # --------- 8) Indicateurs surfaciques utiles ----------
        df_cu["Surface Surplus (kgN/ha)"] = 0.0
        df_cu.loc[mask_area, "Surface Surplus (kgN/ha)"] = (
            df_cu.loc[mask_area, "Surplus (ktN)"] * 1e6 / area[mask_area]
        ).astype(float)

        df_cu["Surface Inputs to field (kgN/ha)"] = 0.0
        df_cu.loc[mask_area, "Surface Inputs to field (kgN/ha)"] = (
            df_cu.loc[mask_area, "Inputs to field (ktN)"] * 1e6 / area[mask_area]
        ).astype(float)

        # (On garde tes colonnes d'usage si besoin)
        df_cu["Total Non Synthetic Fertilizer Use (ktN)"] = (
            df_cu["Excreta Fertilization (ktN)"].fillna(0.0)
            + df_cu["Digestat Fertilization (ktN)"].fillna(0.0)
            + df_cu["Seeds Input (ktN)"].fillna(0.0)
            + df_cu["BNF (ktN)"].fillna(0.0)
            + df_cu["Atmospheric deposition (ktN)"].fillna(0.0)
        )
        df_cu["Surface Non Synthetic Fertilizer Use (kgN/ha)"] = 0.0
        df_cu.loc[mask_area, "Surface Non Synthetic Fertilizer Use (kgN/ha)"] = (
            df_cu.loc[mask_area, "Total Non Synthetic Fertilizer Use (ktN)"]
            * 1e6
            / area[mask_area]
        ).astype(float)

        # --------- 9) Construction des flux ----------

        # Flux des cultures vers les productions végétales :
        for index, row in df_pr.iterrows():
            # Création du dictionnaire target
            source = {row["Origin compartment"]: 1}

            # Création du dictionnaire source
            target = {index: row["Nitrogen Production (ktN)"]}
            self.flux_generator.generate_flux(source, target)

        # Flux des produits vers Waste et other sectors:
        for index, row in df_pr.iterrows():
            source = {index: row["Nitrogen Wasted (ktN)"]}

            target = {"waste": 1}
            self.flux_generator.generate_flux(source, target)

            source = {index: row["Nitrogen for Other uses (ktN)"]}
            target = {"other sectors": 1}
            self.flux_generator.generate_flux(source, target)

        # Seeds input
        target = df_cu["Seeds Input (ktN)"].to_dict()
        source = {"seeds": 1}
        self.flux_generator.generate_flux(source, target)

        # BNF
        source = {"atmospheric N2": 1}
        target = df_cu["BNF (ktN)"].to_dict()
        self.flux_generator.generate_flux(source, target)

        # Soil Stock/pertes

        source = df_cu["Soil storage from surplus (ktN)"].to_dict()
        target = {"soil stock": 1}
        self.flux_generator.generate_flux(source, target)

        source = {"soil stock": 1}
        target = df_cu["Mining from soil (ktN)"].to_dict()
        self.flux_generator.generate_flux(source, target)

        # Excreta fertilization
        # Deja fait dans compute_fluxes

        # Depot atmospherique
        # Deja fait dans compute_fluxes

        # Synthétique + pertes
        source = {"Haber-Bosch": 1}
        target = df_cu["Adjusted Total Synthetic Fertilizer Use (ktN)"].to_dict()

        self.flux_generator.generate_flux(source, target)

        source = df_cu["Volatilized Nitrogen N-NH3 (ktN)"].to_dict()
        target = {"atmospheric NH3": 1}

        self.flux_generator.generate_flux(source, target)

        source = df_cu["Volatilized Nitrogen N-N2O (ktN)"].to_dict()
        target = {"atmospheric N2O": 1}

        self.flux_generator.generate_flux(source, target)

        # A cela on ajoute les emissions indirectes de N2O lors de la fabrication des engrais
        epend_tot_synt = df_cu["Adjusted Total Synthetic Fertilizer Use (ktN)"].sum()

        coef_emis_N_N2O = (
            self.df_global.loc[
                "coefficient N-N2O indirect emission synthetic fertilization (%)"
            ].item()
            / 100
        )
        target = {"atmospheric N2O": 1}
        source = {"Haber-Bosch": epend_tot_synt * coef_emis_N_N2O}

        self.flux_generator.generate_flux(source, target)

        # Et les fuites liées aux surplus de fertilisation minérale

        source = df_cu["Leached to hydro-system (ktN)"].to_dict()
        target = {"hydro-system": 1}
        self.flux_generator.generate_flux(source, target)

        source = df_cu["Surplus N2O (ktN)"].to_dict()
        target = {"atmospheric N2O": 1}
        self.flux_generator.generate_flux(source, target)

        # Sauvegarde
        self.df_cultures = df_cu
        return df_cu

    def plot_heatmap(self):
        """
        Generates a static heatmap to visualize the nitrogen flux transition matrix.

        The heatmap displays the nitrogen fluxes between sectors using a logarithmic color scale. It provides a visual representation
        of the relative magnitudes of nitrogen fluxes, where each cell in the matrix corresponds to a transition from one sector
        (source) to another (target).

        Features of the heatmap:
        - **Logarithmic scale** for better visualization of fluxes with wide value ranges (`LogNorm` is used with `vmin=10^-4` and `vmax` set to the maximum value in the adjacency matrix).
        - **Color palette** is reversed "plasma" (`plasma_r`), with the color bar indicating flux magnitudes in kton/year.
        - **Grid**: A light gray grid is added for visual separation of cells.
        - **Labels**: Axis labels are moved to the top of the heatmap, with tick labels rotated for better readability.
        - **Legend**: Each sector is labeled with its corresponding index (e.g., "1: Agriculture"), positioned next to the heatmap.

        Args:
            None

        Returns:
            None: Displays the heatmap plot on screen.

        Note:
            This method uses `matplotlib` and `seaborn` for visualization. Make sure these libraries are installed.
        """
        plt.figure(figsize=(15, 18), dpi=500)
        ax = plt.gca()

        # Créer la heatmap sans grille pour le moment
        norm = LogNorm(vmin=10**-4, vmax=self.adjacency_matrix.max())
        sns.heatmap(
            self.adjacency_matrix,
            xticklabels=range(1, len(self.labels) + 1),
            yticklabels=range(1, len(self.labels) + 1),
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
        # plt.title(f'Heatmap of adjacency matrix for {area} in {year}')

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

        adj = np.array(self.adjacency_matrix)  # ou .copy()
        adjacency_subset = adj[: len(self.labels), : len(self.labels)].copy()

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
        ticktext = [
            10**x for x in range(-4, 3, 1)
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
        """Computes the nitrogen fluxes for all sectors in the model.

        This method populates the adjacency matrix with flux coefficients based on sector interactions. These interactions
        include crop production, livestock production, nitrogen emissions, and fertilization. The coefficients are used to
        model the flow of nitrogen between sectors over the specified year and area.

        The computation involves complex mathematical processes, which are detailed in the associated scientific methodology
        paper: Tracking Nitrogen Dynamics: A Disaggregated Approach for French Agro-Food Systems, 2025, Adrien Fauste-Gay (pre-print).

        For an in-depth explanation of the model's functioning, please refer to the accompanying paper.
        """
        from pulp import lpSum

        # Extraire les variables nécessaires
        df_elevage = self.df_elevage
        df_excr = self.df_excr
        df_prod = self.df_prod
        df_pop = self.df_pop
        df_energy = self.df_energy
        df_global = self.df_global
        diets = self.diets
        label_to_index = self.label_to_index
        flux_generator = self.flux_generator

        self._build_crops_livestock_to_product()

        # Flux des animaux vers les compartiments d'excretion
        for index, row in df_excr.iterrows():
            # Création du dictionnaire target
            source = {row["Origin compartment"]: 1}

            # Création du dictionnaire source
            target = {index: row["Excretion (ktN)"]}
            flux_generator.generate_flux(source, target)

        ## Dépôt atmosphérique
        source = {"atmospheric N2O": 0.1, "atmospheric NH3": 0.9}
        target = (
            df_global.loc["Atmospheric deposition coef (kgN/ha)"].item()
            * self.df_cultures["Area (ha)"]
            / 1e6
        ).to_dict()  # Dépôt proportionnel aux surface
        flux_generator.generate_flux(source, target)

        ## Consommation de produits de la mer

        source = {"fishery products": 1}
        target = df_pop["Fishery Ingestion (ktN)"].to_dict()
        flux_generator.generate_flux(source, target)

        ## Épandage de boue sur les champs
        Norm = (
            self.df_cultures["Area (ha)"] * self.df_cultures["Spreading Rate (%)"] / 100
        ).sum()
        # Création du dictionnaire target
        target_epandage = {
            culture: row["Area (ha)"] * row["Spreading Rate (%)"] / 100 / Norm
            for culture, row in self.df_cultures.iterrows()
        }

        source_boue = (
            df_pop["Ingestion (ktN)"] * df_pop["Excretion recycling (%)"] / 100
        ).to_dict()

        # On le comptabilise dans une colonne
        target_series = pd.Series(target_epandage)
        total_boue_ktN = sum(source_boue.values())

        self.df_cultures["Excreta Fertilization (ktN)"] = (
            self.df_cultures.index.map(target_series).fillna(0) * total_boue_ktN
        )

        flux_generator.generate_flux(source_boue, target_epandage)

        # Le reste est perdu dans l'environnement
        source = (
            (df_pop["Ingestion (ktN)"] * df_pop["N-N2O EM excretion (%)"]) / 100
        ).to_dict()
        target = {"atmospheric N2O": 1}
        flux_generator.generate_flux(source, target)

        # Ajouter les fuites de NH3
        source = (
            (df_pop["Ingestion (ktN)"] * df_pop["N-NH3 EM excretion (%)"]) / 100
        ).to_dict()
        target = {"atmospheric NH3": 1}
        flux_generator.generate_flux(source, target)

        # Ajouter les fuites de N2
        source = (
            (df_pop["Ingestion (ktN)"] * df_pop["N-N2 EM excretion (%)"]) / 100
        ).to_dict()
        target = {"atmospheric N2": 1}
        flux_generator.generate_flux(source, target)

        # Le reste est perdu dans l'hydroshere
        target = {"hydro-system": 1}
        source = (
            df_pop["Ingestion (ktN)"]
            * (
                100
                - df_pop["Excretion recycling (%)"]
                - df_pop["N-N2O EM excretion (%)"]
                - df_pop["N-N2 EM excretion (%)"]
                - df_pop["N-NH3 EM excretion (%)"]
            )
            / 100
        ).to_dict()

        # Remplir la matrice d'adjacence
        flux_generator.generate_flux(source, target)

        ## Excretions animales

        # Azote excrété sur prairies

        # Calculer les poids pour chaque cible
        # Calcul de la surface totale pour les prairies

        total_surface_grasslands = self.df_cultures.loc[
            self.df_cultures["Category"].isin(["natural meadows", "temporary meadows"]),
            "Area (ha)",
        ].sum()

        # Création du dictionnaire target
        target_grass = (
            self.df_cultures.loc[
                self.df_cultures["Category"].isin(
                    ["natural meadows", "temporary meadows"]
                ),
                "Area (ha)",
            ]
            / total_surface_grasslands
        ).to_dict()

        self.target_grass = target_grass

        # Source
        source = df_excr.loc[
            df_excr["Type"] == "grasslands excretion",
            "Excretion after volatilization (ktN)",
        ].to_dict()

        # flux_generator.generate_flux(source, target_grass)

        # On ajoute la fertilisation par excretat dans df_cultures
        total_excr_grasslands_ktN = sum(source.values())
        target_series = pd.Series(target_grass)
        self.df_cultures["Excreta Fertilization (ktN)"] += (
            self.df_cultures.index.map(target_series).fillna(0)
            * total_excr_grasslands_ktN
        )

        # Le reste est émit dans l'atmosphere
        # N2
        target = {"atmospheric N2": 1}
        source = df_excr.loc[
            df_excr["Type"] == "grasslands excretion", "Excretion as N2 (ktN)"
        ].to_dict()

        flux_generator.generate_flux(source, target)

        # NH3
        # 1 % est volatilisée de manière indirecte sous forme de N2O
        target = {"atmospheric NH3": 0.99, "atmospheric N2O": 0.01}
        source = df_excr.loc[
            df_excr["Type"] == "grasslands excretion", "Excretion as NH3 (ktN)"
        ].to_dict()

        flux_generator.generate_flux(source, target)

        # N2O
        target = {"atmospheric N2O": 1}
        source = df_excr.loc[
            df_excr["Type"] == "grasslands excretion", "Excretion as N2O (ktN)"
        ].to_dict()

        flux_generator.generate_flux(source, target)

        ## Epandage sur champs

        source = df_excr.loc[
            df_excr["Type"].isin(["manure", "slurry"]),
            "Excretion after volatilization (ktN)",
        ].to_dict()

        # flux_generator.generate_flux(source, target_epandage)

        # On l'enregistre dans la colonne excretion fertilization. On reprendra ce calcul après pour les flux vers les méthaniseurs
        total_excr_fields_ktN = sum(source.values())
        target_series = pd.Series(target_epandage)
        self.df_cultures["Excreta Fertilization (ktN)"] += (
            self.df_cultures.index.map(target_series).fillna(0) * total_excr_fields_ktN
        )
        # Le reste part dans l'atmosphere

        # N2
        target = {"atmospheric N2": 1}

        source = df_excr.loc[
            df_excr["Type"].isin(["manure", "slurry"]), "Excretion as N2 (ktN)"
        ].to_dict()
        flux_generator.generate_flux(source, target)

        # NH3
        # 1 % est volatilisée de manière indirecte sous forme de N2O
        target = {"atmospheric NH3": 0.99, "atmospheric N2O": 0.01}
        source = df_excr.loc[
            df_excr["Type"].isin(["manure", "slurry"]), "Excretion as NH3 (ktN)"
        ].to_dict()
        flux_generator.generate_flux(source, target)

        # N2O
        target = {"atmospheric N2O": 1}
        source = df_excr.loc[
            df_excr["Type"].isin(["manure", "slurry"]), "Excretion as N2O (ktN)"
        ].to_dict()
        flux_generator.generate_flux(source, target)

        if not self.prospective:
            ## Azote synthétique
            # Calcul de l'azote à épendre (bilan azoté)

            self.df_cultures["Total Non Synthetic Fertilizer Use (ktN)"] = (
                self.df_cultures["Excreta Fertilization (ktN)"]
                + self.df_cultures["Seeds Input (ktN)"]
                + self.df_cultures["BNF (ktN)"]
                + self.df_cultures["Atmospheric deposition (ktN)"]
            )

            # Séparer les données en prairies et champs
            # On commence par un bilan sur les prairies pour avoir un bilan complet des prairies et en déduire un bilan azoté non synthétique complet sur les culturess
            df_prairies = self.df_cultures[
                self.df_cultures["Category"].isin(
                    ["natural meadows", "temporary meadows"]
                )
            ].copy()

            df_prairies["Surface Non Synthetic Fertilizer Use (kgN/ha)"] = (
                df_prairies.apply(
                    lambda row: row["Total Non Synthetic Fertilizer Use (ktN)"]
                    / row["Area (ha)"]
                    * 10**6
                    if row["Area (ha)"] > 0
                    and row["Total Non Synthetic Fertilizer Use (ktN)"] > 0
                    else 0,
                    axis=1,
                )
            )

            if (
                "Raw Surface Synthetic Fertilizer Use (kgN/ha)"
                not in df_prairies.columns
            ):
                df_prairies["Raw Surface Synthetic Fertilizer Use (kgN/ha)"] = (
                    df_prairies.apply(
                        lambda row: row["Surface Fertilization Need (kgN/ha)"]
                        - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"]
                        if row["Area (ha)"] > 0
                        else 0,
                        axis=1,
                    )
                )
                df_prairies["Raw Surface Synthetic Fertilizer Use (kgN/ha)"] = (
                    df_prairies["Raw Surface Synthetic Fertilizer Use (kgN/ha)"].apply(
                        lambda x: max(x, 0)
                    )
                )

            df_prairies["Raw Total Synthetic Fertilizer Use (ktN)"] = (
                df_prairies["Raw Surface Synthetic Fertilizer Use (kgN/ha)"]
                * df_prairies["Area (ha)"]
                / 1e6
            )

            moyenne_ponderee_prairies = (
                df_prairies["Raw Total Synthetic Fertilizer Use (ktN)"]
            ).sum()

            moyenne_reel_prairies = df_global.loc[
                df_global.index == "Total Synthetic Fertilizer Use on grasslands (ktN)"
            ]["value"].item()

            if moyenne_ponderee_prairies != 0:
                df_prairies.loc[:, "Adjusted Total Synthetic Fertilizer Use (ktN)"] = (
                    df_prairies["Raw Total Synthetic Fertilizer Use (ktN)"]
                    * moyenne_reel_prairies
                    / moyenne_ponderee_prairies
                )
            else:
                if len(df_prairies) > 0:
                    df_prairies.loc[
                        :, "Adjusted Total Synthetic Fertilizer Use (ktN)"
                    ] = 0
                warnings.warn("No Synthetic fertilizer need for grasslands.")

            # Bouclage du bilan des Cultures n'étant pas des prairies ou des légumineuses
            df_champs = self.df_cultures[
                ~self.df_cultures["Category"].isin(
                    [
                        "natural meadows",
                        "temporary meadows",
                        "leguminous",
                    ]  # Les légumineuses sont interdit de HB
                )
            ].copy()

            df_champs["Surface Non Synthetic Fertilizer Use (kgN/ha)"] = (
                df_champs.apply(
                    lambda row: row["Total Non Synthetic Fertilizer Use (ktN)"]
                    / row["Area (ha)"]
                    * 10**6
                    if row["Area (ha)"] > 0
                    and row["Total Non Synthetic Fertilizer Use (ktN)"] > 0
                    else 0,
                    axis=1,
                )
            )

            if "Raw Surface Synthetic Fertilizer Use (kgN/ha)" not in df_champs.columns:
                df_champs["Raw Surface Synthetic Fertilizer Use (kgN/ha)"] = (
                    df_champs.apply(
                        lambda row: row["Surface Fertilization Need (kgN/ha)"]
                        - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"]
                        if row["Area (ha)"] > 0
                        else row["Surface Fertilization Need (kgN/ha)"]
                        - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"],
                        axis=1,
                    )
                )
                df_champs["Raw Surface Synthetic Fertilizer Use (kgN/ha)"] = df_champs[
                    "Raw Surface Synthetic Fertilizer Use (kgN/ha)"
                ].apply(lambda x: max(x, 0))

            df_champs["Raw Total Synthetic Fertilizer Use (ktN)"] = (
                df_champs["Raw Surface Synthetic Fertilizer Use (kgN/ha)"]
                * df_champs["Area (ha)"]
                / 1e6
            )

            moyenne_ponderee_champs = (
                df_champs["Raw Total Synthetic Fertilizer Use (ktN)"]
            ).sum()

            moyenne_reel_champs = df_global.loc[
                df_global.index == "Total Synthetic Fertilizer Use on crops (ktN)"
            ]["value"].item()

            if moyenne_ponderee_champs != 0:
                df_champs.loc[:, "Adjusted Total Synthetic Fertilizer Use (ktN)"] = (
                    df_champs["Raw Total Synthetic Fertilizer Use (ktN)"]
                    * moyenne_reel_champs
                    / moyenne_ponderee_champs
                )
                self.gamma = moyenne_reel_champs / moyenne_ponderee_champs
            else:
                if len(df_champs) > 0:
                    df_champs.loc[
                        :, "Adjusted Total Synthetic Fertilizer Use (ktN)"
                    ] = 0
                warnings.warn("No Synthetic fertilizer need for grasslands.")
                self.gamma = None

            # Mise à jour de df_cultures
            df_calc = pd.concat([df_prairies, df_champs], axis=0, sort=False)
            self.df_cultures = df_calc.combine_first(self.df_cultures).fillna(0)

            self.df_cultures["Adjusted Surface Synthetic Fertilizer Use (kgN/ha)"] = 0.0

            mask = self.df_cultures["Area (ha)"] != 0
            self.df_cultures.loc[
                mask, "Adjusted Surface Synthetic Fertilizer Use (kgN/ha)"
            ] = (
                self.df_cultures.loc[
                    mask, "Adjusted Total Synthetic Fertilizer Use (ktN)"
                ]
                / self.df_cultures.loc[mask, "Area (ha)"]
                * 1e6
            ).astype(float)

        ## Modèle d'allocation de l'azote aux consommateurs
        # Filtre les données d'ingestion animale et les stocke dans df_cons
        df_cons = df_elevage.loc[
            df_elevage["Ingestion (ktN)"] > 10**-8, ["Ingestion (ktN)"]
        ].copy()
        df_cons["Type"] = "Animal"

        # Filtre les données d'ingestion humaine (attention à ne pas ajouter la consommation de produits de la mer)
        df_pop_ingestion = df_pop.copy()
        df_pop_ingestion["Ingestion (ktN)"] = (
            df_pop_ingestion["Ingestion (ktN)"]
            - df_pop_ingestion["Fishery Ingestion (ktN)"]
        )
        df_pop_ingestion = df_pop_ingestion.loc[
            df_pop["Ingestion (ktN)"] > 10**-8, ["Ingestion (ktN)"]
        ]
        df_pop_ingestion["Type"] = "Human"

        # Ajoute les données humaines aux données animales
        df_cons = pd.concat([df_cons, df_pop_ingestion])

        # Dictionary to store all the cultures consumed by each consumer
        all_cultures_regime = {}

        # Group the DataFrame by the 'Consumer' column
        grouped_by_consumer = diets.groupby("Consumer")

        # Iterate through the groups and process the data
        for cons, group_df in grouped_by_consumer:
            # `group_df` is a DataFrame containing all rows for the current consumer
            cultures_name = set()

            # Iterate over the 'Products' column in the current group
            for product_list in group_df["Products"]:
                # The `.update()` method efficiently adds all items from an iterable
                cultures_name.update(product_list)

            # Store the set of unique cultures in the dictionary
            all_cultures_regime[cons] = cultures_name

        pairs = []
        for index, row in diets.iterrows():
            if (
                len(df_cons.loc[df_cons.index == row["Consumer"], "Ingestion (ktN)"])
                != 0
            ):
                pairs.append(
                    (row["Consumer"], row["Proportion"], tuple(row["Products"]))
                )

        proportion_animal = {}
        for cons, prop, products in pairs:
            proportion_animal[cons] = proportion_animal.get(cons, 0.0)

            if df_prod.loc[products[0], "Type"] == "animal":
                proportion_animal[cons] += prop

        # Initialisation du problème
        prob = LpProblem("Allocation_Azote_Animaux", LpMinimize)
        if self.prospective:
            self._pre_lp_supply(prob)

        # Variables de décision pour les allocations

        # Créez une liste de tous les tuples (produit, consommateur) valides
        valid_pairs = [
            (prod, cons) for cons, _, products_list in pairs for prod in products_list
        ]

        # Créez les variables d'allocation en utilisant ces paires
        x_vars = LpVariable.dicts("x", valid_pairs, lowBound=0, cat="Continuous")

        # Variables de déviation des régimes alimentaires
        delta_vars = LpVariable.dicts(
            "delta",
            [(cons, tuple(products)) for cons, proportion, products in pairs],
            lowBound=0,
            cat=LpContinuous,
        )

        # Variables de pénalité pour la distribution au sein des catégories
        penalite_culture_vars = LpVariable.dicts(
            "penalite_culture",
            [
                (cons, proportion, prod)
                for cons, proportion, products_list in pairs
                for prod in products_list
            ],
            lowBound=0,
            cat=LpContinuous,
        )

        # Variables d'importation pour chaque élevage et catégorie
        I_vars = LpVariable.dicts(
            "I",
            valid_pairs,
            lowBound=0,
            cat="Continuous",
        )

        # Pondération pour le terme de pénalité
        poids_penalite_deviation = df_global.loc[df_global.index == "Weight diet"][
            "value"
        ].item()
        poids_import_brut = df_global.loc[df_global.index == "Weight import"][
            "value"
        ].item()

        # Poids pour équilibrer la distribution des cultures dans les categories
        poids_penalite_culture = df_global.loc[
            df_global.index == "Weight distribution"
        ]["value"].item()

        # Contrainte sur l'écart aux diètes
        for cons, proportion_initiale, products_list in pairs:
            # Récupère le besoin total en azote pour ce consommateur
            besoin = df_cons.loc[cons, "Ingestion (ktN)"]

            azote_cultures = lpSum(
                x_vars.get((prod_i, cons), 0) for prod_i in products_list
            ) + lpSum(I_vars.get((prod_i, cons), 0) for prod_i in products_list)

            # Assure que le besoin n'est pas nul avant de calculer la proportion effective
            if besoin > 0:
                proportion_effective = azote_cultures / besoin

                # Récupère la variable de déviation
                delta_var = delta_vars[(cons, tuple(products_list))]

                # Ajoute les contraintes pour la déviation
                prob += (
                    proportion_effective - proportion_initiale <= delta_var,
                    f"Deviation_Plus_{cons}_{products_list}",
                )
                prob += (
                    proportion_initiale - proportion_effective <= delta_var,
                    f"Deviation_Moins_{cons}_{products_list}",
                )

        # Contrainte sur la distribution d'un produit à tous ses consommateurs.
        # 0) Construire les parts de référence s_ref[(p, c)]
        from collections import defaultdict

        raw_ref = defaultdict(
            float
        )  # somme des contributions "proportion / len(products_list)" pour chaque (p,c)

        for cons, proportion, products_list in pairs:
            if len(products_list) == 0:
                continue
            share_each = proportion / len(products_list)
            for p in products_list:
                raw_ref[(p, cons)] += share_each

        # Normalisation par produit : somme_c s_ref[(p,c)] = 1 si p apparaît ; sinon, on ignore
        s_ref = {}
        for p in df_prod.index:
            s = sum(raw_ref[(p, c)] for c in df_cons.index)
            if s > 0:
                for c in df_cons.index:
                    s_ref[(p, c)] = raw_ref[(p, c)] / s
            else:
                # Si le produit p n'apparaît dans aucun groupe de diète, on peut
                # soit répartir uniformément, soit mettre 0; ici on répartit uniformément sur les consommateurs qui ont p
                consumers_with_p = [c for (prod, c) in valid_pairs if prod == p]
                if consumers_with_p:
                    for c in consumers_with_p:
                        s_ref[(p, c)] = 1.0 / len(consumers_with_p)

        # 1) variables de déviation ABSOLUE par (p,c)
        gamma_fair_abs = LpVariable.dicts(
            "gamma_fair_abs",
            [(p, c) for (p, c) in valid_pairs],
            lowBound=0,
            cat=LpContinuous,
        )

        # 2) contraintes |x_{p,c} - x_cible| <= gamma_fair_abs
        for p, c in valid_pairs:
            prodN = self._available_expr(p)
            x_cible = s_ref.get((p, c), 0.0) * prodN

            x_local = x_vars[(p, c)]
            # valeur absolue via 2 inégalités
            prob += (
                x_local - x_cible <= gamma_fair_abs[(p, c)],
                f"FairAbs_plus_{p}_{c}",
            )
            prob += (
                x_cible - x_local <= gamma_fair_abs[(p, c)],
                f"FairAbs_moins_{p}_{c}",
            )

        # 3) terme objectif normalisé par produit: sum_p ( (1/max(1,Prod_p)) * sum_c gamma_{p,c} )
        def prod_scale(p: str) -> float:
            df_pr = df_prod
            # 1) si tu as une valeur historique dispo :
            if not self.prospective or df_prod.at[p, "Type"] == "animal":
                base = float(df_pr.at[p, "Available Nitrogen Production (ktN)"] or 0.0)
            elif df_prod.at[p, "Type"] == "plant":
                c = df_prod.at[p, "Origin compartment"]
                base = float(
                    self.df_cultures.at[c, "Maximum Yield (tFW/ha)"]
                    * self.df_cultures.at[c, "Area (ha)"]
                    / 1e3
                    * df_prod.at[p, "Co-Production Ratio (%)"]
                    * df_prod.at[p, "Nitrogen Content (%)"]
                    / 100
                )
            # 2) borne minimale 1.0 pour éviter divisions explosives
            return max(1.0, base)

        fair_term = lpSum(
            (
                lpSum(
                    gamma_fair_abs[(p, c)]
                    for c in df_cons.index
                    if (p, c) in gamma_fair_abs
                )
                / prod_scale(p)
            )
            for p in df_prod.index
        )

        # 4) ajoute-le à l’objectif (éventuellement moyenner par nb de produits)
        w_fair = df_global.loc[df_global.index == "Weight fair local split"][
            "value"
        ].item()

        # Variable de surplus
        # Surplus (à exporter) par produit p
        U_vars = LpVariable.dicts(
            "U", list(df_prod.index), lowBound=0, cat=LpContinuous
        )

        W_ENERGY_PROD = float(df_global.loc["Weight energy production", "value"])
        W_ENERGY_INPUT = float(df_global.loc["Weight energy inputs", "value"])

        energy_vars = {}  # fac -> {"prod": {p: var}, "excr": {e: var}, "waste": var}
        I_energy_vars = {}  # (product, facility) -> var  (imports autorisés pour bioraffineries)

        energy_dev_terms = []  # déviations d'énergie (|E - cible|/cible par fac)
        delta_fac = {}  # Penalité déviation diète
        fair_term_energy_parts = []  # parts fair-share "produits -> infrastructures"
        penalite_energy_terms = []  # pénalités intra-groupes des diètes d'infras

        # Pour les bilans
        energy_prod_vars_by_p = {p: [] for p in df_prod.index}
        energy_excr_vars_by_e = {e: [] for e in df_excr.index}

        # Production par infra
        energy_E_GWh_expr = {}

        pairs_fac_all = []
        Nhat_by_fac = {}

        for facility, row in df_energy.iterrows():
            TARGET_GWh = float(row["Target Energy Production (GWh)"])
            # IMPORTANT : la diète d'une infrastructure se récupère par le NOM DE L'INFRA, pas l'ID de diète
            fac_diet_df = diets[diets["Consumer"] == facility].copy()

            # Items autorisés par la diète de l’infrastructure
            fac_prod_items = set()
            fac_excr_items = set()

            allow_waste = False
            for _, r in fac_diet_df.iterrows():
                for it in r["Products"]:
                    if it in df_prod.index:
                        fac_prod_items.add(it)
                    elif it in df_excr.index:
                        fac_excr_items.add(it)
                    elif it == "waste":
                        allow_waste = True
                    else:
                        # produit inconnu → on l'ignore (on n'assimile plus à "waste")
                        pass

            # Variables d’allocation vers l’infra (seulement pour les items autorisés)
            x_fac_prod = LpVariable.dicts(
                f"x_{facility}_prod", list(fac_prod_items), lowBound=0, cat=LpContinuous
            )
            x_fac_excr = LpVariable.dicts(
                f"x_{facility}_excr", list(fac_excr_items), lowBound=0, cat=LpContinuous
            )
            # Ne créer 'waste' QUE si la diète l'autorise explicitement
            N_waste_fac = (
                LpVariable(f"N_{facility}_waste", lowBound=0, cat=LpContinuous)
                if allow_waste
                else None
            )

            energy_vars[facility] = {
                "prod": x_fac_prod,
                "excr": x_fac_excr,
                "waste": N_waste_fac,
            }

            # IMPORTS autorisés UNIQUEMENT pour les bioraffineries, et UNIQUEMENT si le produit est dans la diète
            if str(row["Type"]).lower() == "bioraffinery":
                allowed_import_products = {
                    it
                    for _, r in fac_diet_df.iterrows()
                    for it in r["Products"]
                    if it in df_prod.index
                }
                for p in allowed_import_products:
                    I_energy_vars[(p, facility)] = LpVariable(
                        f"I_energy_{p}_{facility}", lowBound=0, cat=LpContinuous
                    )

            # -- Énergie produite (MWh) puis conversion en GWh
            # Énergie (MWh), puis conversion GWh
            # PRODUITS → facility
            E_MWh_products = (
                lpSum(
                    x_fac_prod[p] * self._conv_MWh_per_ktN(facility, p)
                    for p in fac_prod_items
                )
                if len(fac_prod_items)
                else 0
            )

            # EXCRETA → facility
            E_MWh_excreta = (
                lpSum(
                    x_fac_excr[e] * self._conv_MWh_per_ktN(facility, e)
                    for e in fac_excr_items
                )
                if len(fac_excr_items)
                else 0
            )

            # WASTE → facility (si 'waste' est autorisé dans la diète)
            E_MWh_waste = (
                N_waste_fac * self._conv_MWh_per_ktN(facility, "waste")
                if isinstance(N_waste_fac, LpVariable)
                else 0
            )

            # (si tu as des imports de produits vers les facilities)
            E_MWh_products_import = (
                lpSum(
                    I_energy_vars[(p, facility)] * self._conv_MWh_per_ktN(facility, p)
                    for (p, f) in I_energy_vars
                    if f == facility
                )
                if len(I_energy_vars)
                else 0
            )

            E_GWh_total_fac = (
                E_MWh_products + E_MWh_excreta + E_MWh_waste + E_MWh_products_import
            ) / 1000.0

            # #Contrainte pour interdire de donner 0 à une infrastructure énergétique (si son besoin est >0). Evite les solutions corners où le terme de diète est nul
            # prob += E_GWh_total_fac >=

            energy_E_GWh_expr[facility] = E_GWh_total_fac

            # Total N envoyé vers la facility (ktN)
            N_to_fac_total = 0
            if len(x_fac_prod):
                N_to_fac_total += lpSum(x_fac_prod.values())
            if len(x_fac_excr):
                N_to_fac_total += lpSum(x_fac_excr.values())
            if N_waste_fac is not None:
                N_to_fac_total += N_waste_fac
            # Imports dédiés à CETTE facility (bioraffinerie)
            N_to_fac_total += (
                lpSum(
                    I_energy_vars[(p, facility)]
                    for (p, f) in I_energy_vars
                    if f == facility
                )
                if len(I_energy_vars)
                else 0
            )

            # Déviation d'énergie normalisée
            dev_fac = LpVariable(f"{facility}_energy_dev", lowBound=0, cat=LpContinuous)
            if TARGET_GWh > 0:
                prob += dev_fac >= (E_GWh_total_fac - TARGET_GWh) / TARGET_GWh
                prob += dev_fac >= (TARGET_GWh - E_GWh_total_fac) / TARGET_GWh
            else:
                prob += dev_fac == 0
            energy_dev_terms.append(dev_fac)

            # --- DIETE DE LA FACILITE : paires LOCALES ---
            pairs_fac_fac = []
            for _, r in fac_diet_df.iterrows():
                prop = float(r["Proportion"])
                prop = prop / 100.0 if prop > 1.0 else prop
                pairs_fac_fac.append((facility, prop, tuple(r["Products"])))

            # variables delta SEULEMENT pour les paires de CETTE facility
            for k, (_, _, pL) in enumerate(pairs_fac_fac):
                key = (facility, tuple(pL))  # clé logique inchangée
                delta_fac[key] = LpVariable(
                    f"delta_{facility}_{k}",  # nom court pour le LP
                    lowBound=0,
                    cat=LpContinuous,
                )

            # --- 1) Puissance moyenne attendue selon la diète déclarée (MWh/ktN) ---
            P_avg = 0.0
            w_sum = 0.0
            for _, r in diets[diets["Consumer"] == facility].iterrows():
                prop = float(r["Proportion"])
                prop = prop / 100.0 if prop > 1.0 else prop
                items = list(r["Products"])
                if not items:
                    continue
                share_each = prop / len(items)
                for it in items:
                    # puissance par item pour CETTE infra, en MWh/ktN (tu l’as dans energy_power_map)
                    p_item = float(self.energy_power_map.get(facility, {}).get(it, 0.0))
                    P_avg += share_each * p_item
                    w_sum += share_each

            P_avg = P_avg / w_sum if w_sum > 0 else 0.0

            # --- 2) Nhat_fac: intrant attendu (ktN) pour atteindre Target (GWh) ---
            Target_GWh = float(
                df_energy.loc[facility, "Target Energy Production (GWh)"]
            )
            Nhat_fac = (
                (Target_GWh * 1000.0 / max(P_avg, 1e-6)) if Target_GWh > 0 else 0.0
            )

            Nhat_by_fac[facility] = Nhat_fac

            # --- Lier delta_fac à l'écart absolu entre l'allocation du groupe et sa cible ---
            for _, prop, prod_list in pairs_fac_fac:
                # somme N envoyée au groupe (local + excréta + waste éventuel + imports énergie)
                N_group = 0
                allowed_items = []
                for it in prod_list:
                    if it in x_fac_prod:
                        allowed_items.append(it)
                        N_group += x_fac_prod[it]
                    elif it in x_fac_excr:
                        allowed_items.append(it)
                        N_group += x_fac_excr[it]
                    elif it == "waste" and (N_waste_fac is not None):
                        allowed_items.append(it)
                        N_group += N_waste_fac
                    if (it, facility) in I_energy_vars:
                        allowed_items.append(it)
                        N_group += I_energy_vars[(it, facility)]
                    # sinon: item non utilisable par cette infra → ignoré

                # cible (en ktN) pour ce groupe: part de la diète * total N vers l'infrastructure
                cible_group = prop * Nhat_fac

                dv = delta_fac[(facility, tuple(prod_list))]
                # |N_group - cible_group| <= dv
                if cible_group > 0:
                    prob += (
                        (N_group - cible_group) / cible_group <= dv,
                        f"EnergyDietDev_plus_{facility}_{hash(tuple(prod_list)) % 10**6}",
                    )
                    prob += (
                        (cible_group - N_group) / cible_group <= dv,
                        f"EnergyDietDev_moins_{facility}_{hash(tuple(prod_list)) % 10**6}",
                    )

            # --- pénalités intra-groupe : n'utiliser QUE les paires de CETTE facility ---
            penalite_culture_fac = LpVariable.dicts(
                f"penalite_culture_{facility}",
                [(facility, prop, it) for _, prop, L in pairs_fac_fac for it in L],
                lowBound=0,
                cat=LpContinuous,
            )
            for _, prop, L in pairs_fac_fac:
                allowed_items = []
                for it in L:
                    if (
                        (it in x_fac_prod)
                        or (it in x_fac_excr)
                        or (it == "waste" and (N_waste_fac is not None))
                        or ((it, facility) in I_energy_vars)
                    ):
                        allowed_items.append(it)
                k = max(1, len(allowed_items))
                cible_item = (prop * N_to_fac_total) / k
                for it in allowed_items:
                    if it in x_fac_prod:
                        alloc_it = x_fac_prod[it]
                    elif it in x_fac_excr:
                        alloc_it = x_fac_excr[it]
                    elif it == "waste":
                        alloc_it = N_waste_fac
                    elif (it, facility) in I_energy_vars:
                        alloc_it = I_energy_vars[(it, facility)]
                    pv = penalite_culture_fac[(facility, prop, it)]
                    prob += alloc_it - cible_item <= pv
                    prob += cible_item - alloc_it <= pv

            # (facultatif) si tu veux toujours un warning « power manquant »
            # reconstitue un petit conteneur global si besoin :
            pairs_fac_all.extend(pairs_fac_fac)

            # -- Fair-share par PRODUIT (comme pour les consommateurs)
            # On construit les parts de ref de CETTE infra et de TOUTES les entités (consommateurs + toutes infras)
            raw_ref_fac = defaultdict(float)
            for _, prop, L in pairs_fac_fac:
                if len(L) == 0:
                    continue
                share_each = prop / len(L)
                for p in L:
                    if p in df_prod.index:
                        raw_ref_fac[p] += share_each

            # Denominateur: parts consommateurs + parts de TOUTES les infras
            # Pour éviter la double boucle, on cumule d'abord toutes les raw_ref_fac dans un dict global
            # → plus simple: on calcule les cibles en considérant "autres consommateurs + cette infra"
            # (approx. équitable et linéaire). Si tu veux strictement toutes les infras, tu peux
            # pré-agréger sur le tour complet; on reste simple ici.
            gamma_fair_abs_fac = LpVariable.dicts(
                f"gamma_fair_abs_{facility}",
                list(fac_prod_items),
                lowBound=0,
                cat=LpContinuous,
            )

            for p in fac_prod_items:
                s_den = sum(raw_ref[(p, c)] for c in df_cons.index) + raw_ref_fac.get(
                    p, 0.0
                )
                if s_den <= 0:
                    continue
                s_ref_fac_p = raw_ref_fac.get(p, 0.0) / s_den
                prodN = self._available_expr(p)  # si tu as Y_var, remplace ici
                x_cible_fac = s_ref_fac_p * prodN
                prob += (
                    x_fac_prod[p] - x_cible_fac <= gamma_fair_abs_fac[p],
                    f"FairAbs_{facility}_plus_{p}",
                )
                prob += (
                    x_cible_fac - x_fac_prod[p] <= gamma_fair_abs_fac[p],
                    f"FairAbs_{facility}_moins_{p}",
                )

            # ajoute au terme fair global (même normalisation que pour les produits)
            fair_term_energy_parts.append(
                lpSum(gamma_fair_abs_fac[p] / prod_scale(p) for p in fac_prod_items)
            )

            n_items_fac = max(1, sum(len(L) for _, _, L in pairs_fac_fac))
            penalite_energy_terms.append(
                (
                    facility,
                    (poids_penalite_culture / n_items_fac)
                    * lpSum(penalite_culture_fac.values()),
                )
            )

            # -- Pour les bilans produits et la contrainte globale d'excréta
            for p in fac_prod_items:
                energy_prod_vars_by_p[p].append(x_fac_prod[p])
            for e in fac_excr_items:
                energy_excr_vars_by_e[e].append(x_fac_excr[e])

        # -- Contrainte GLOBALE d'excréta (somme de toutes les infras) <= excrétion dispo
        for e in df_excr.index:
            if len(energy_excr_vars_by_e[e]) == 0:
                continue
            avail_excr = float(df_excr.loc[e, "Excretion after volatilization (ktN)"])
            prob += (lpSum(energy_excr_vars_by_e[e]) <= avail_excr, f"Excreta_cap_{e}")

        # -- Bilan produit : ajouter la somme des prises par TOUTES les infras
        for p in df_prod.index:
            # consommateurs qui ont p dans leur régime
            consumers_with_p = [c for (prod, c) in valid_pairs if prod == p]

            # inputs (locaux) de p vers les infrastructures énergie
            x_to_energy_local = lpSum(
                energy_vars[fac]["prod"][p]
                for fac in df_energy.index
                if ("prod" in energy_vars.get(fac, {}))
                and (p in energy_vars[fac]["prod"])
            )

            # bilan local : feed/food locaux + énergie locale + surplus == prod locale
            rhs_const = self._available_expr(p)
            is_plant = df_prod.loc[p, "Type"] == "plant"
            rhs = self._rhs_for_product(p, rhs_const, is_plant)  # ← HOOK 2

            prob += (
                lpSum(x_vars[(p, c)] for c in consumers_with_p)
                + x_to_energy_local
                + U_vars[p]
                == rhs,
                f"Bilan_{p}",
            )

        # -- Ajouts à l’objectif (agrégés sur toutes les infras)
        #    * Production d’énergie (déviation normalisée) avec W_ENERGY_PROD
        #    * Diète des infras (deltas) avec W_ENERGY_INPUT
        #    * Fair-share infrastructures (on additionne à ton fair_term)
        #    * Pénalités intra-groupe des infras (comme ta penalite_culture_vars)
        fair_term_energy = (
            lpSum(fair_term_energy_parts) if fair_term_energy_parts else 0
        )
        penalite_culture_energy_term = (
            lpSum(t for _, t in penalite_energy_terms) if penalite_energy_terms else 0
        )

        energy_dev_total = lpSum(energy_dev_terms) if energy_dev_terms else 0

        imp_norm = df_cons["Ingestion (ktN)"].sum() + sum(Nhat_by_fac.values())

        I_food = lpSum(
            I_vars[(p, c)]
            for (p, c) in I_vars
            if c in df_cons.index and df_cons.loc[c, "Type"] == "Human"
        )
        I_feed = lpSum(
            I_vars[(p, c)]
            for (p, c) in I_vars
            if c in df_cons.index and df_cons.loc[c, "Type"] == "Animal"
        )

        I_food_normed = I_food / imp_norm
        I_feed_normed = I_feed / imp_norm

        # 2) Imports Energy -> normaliser comme la diète énergie, par la somme des Nhat_fac
        I_energy = lpSum(I_energy_vars.values()) if len(I_energy_vars) else 0

        # imp_energy_norm = max(1e-9, sum(Nhat_by_fac.values()))

        I_energy_normed = I_energy / imp_norm

        # Terme d'import co-normalisé
        raw_import_term = I_food_normed + I_feed_normed + I_energy_normed

        # Fonction objectif
        objective = (
            (poids_penalite_deviation / max(1, len(pairs))) * lpSum(delta_vars.values())
            + (poids_penalite_culture / max(1, len(df_prod)))
            * (lpSum(penalite_culture_vars.values()) + penalite_culture_energy_term)
            + poids_import_brut * raw_import_term
            + w_fair * ((fair_term + fair_term_energy) / max(1, len(df_prod)))
            + W_ENERGY_INPUT * lpSum(delta_fac.values())
            + (W_ENERGY_PROD / max(1, len(df_energy.index))) * energy_dev_total
            + self._extra_objective()
        )

        prob += objective

        # Les besoins en feed sont complétés par la prod locale, l'importation de feed (donnees GRAFS) et un eventuel import excedentaire
        for cons in df_cons.index:
            besoin = df_cons.loc[cons, "Ingestion (ktN)"]
            prob += (
                lpSum(x_vars[(prod_i, cons)] for prod_i in all_cultures_regime[cons])
                + lpSum(I_vars[(prod_i, cons)] for prod_i in all_cultures_regime[cons])
                == besoin,
                f"Besoin_{cons}",
            )

            # Ajout de la contrainte pour respecter la différence entre consommation animale et végétale
            if df_global.loc[df_global.index == "Enforce animal share"]["value"].item():
                # Créez une variable ou utilisez une valeur pour la consommation animale
                cons_animale_totale = lpSum(
                    x_vars[(prod_i, cons)] + I_vars[(prod_i, cons)]
                    for prod_i in all_cultures_regime[cons]
                    if df_prod.loc[df_prod.index == prod_i, "Type"].item() == "animal"
                )
                # Ajoutez la contrainte pour imposer une proportion ou un ratio
                prob += (
                    cons_animale_totale == besoin * proportion_animal[cons],
                    f"Contrainte_part_animale_{cons}",
                )

        # Interdiction d'importer des prairies naturelles
        prob += (
            lpSum(
                I_vars[(prod, cons)]
                for cons in df_cons.loc[df_cons["Type"] == "Animal"].index
                for prod in df_prod.loc[df_prod["Sub Type"] == "grazing"].index
                if prod in all_cultures_regime[cons]
            )
            == 0,
            "Pas_d_import_prairies_nat",
        )

        # # Crée une fonction pour obtenir l'azote disponible après perte et autres usages, cela évitera la redondance
        # def get_nitrogen_production(prod_i, df_prod):
        #     return self._available_expr(prod_i)

        # Fusionne les deux boucles pour traiter tous les consommateurs
        for cons, proportion, products_list in pairs:
            besoin = df_cons.loc[cons, "Ingestion (ktN)"]

            # Calcule l'azote total disponible pour ce groupe de cultures
            # azote_total_groupe = lpSum(get_nitrogen_production(p, df_prod) for p in products_list)
            # azote_total_groupe = sum(
            #     get_nitrogen_production(p, df_prod) for p in products_list
            # )

            # Ajoute les contraintes de pénalité si l'allocation du groupe n'est pas nulle
            if besoin > 0:
                for prod_i in products_list:
                    # Récupère la production d'azote pour le produit actuel
                    # azote_disponible_prod_i = get_nitrogen_production(prod_i, df_prod)

                    ingestion_totale = df_cons.loc[cons, "Ingestion (ktN)"]
                    allocation_groupe_cible = proportion * ingestion_totale
                    # Calcule l'allocation cible proportionnelle à la disponibilité
                    allocation_cible_culture = (
                        allocation_groupe_cible / len(products_list)
                    )  # (azote_disponible_prod_i / azote_total_groupe) * allocation_groupe_cible
                    # Allocation réelle
                    allocation_reelle_culture = x_vars.get(
                        (prod_i, cons), 0
                    ) + I_vars.get((prod_i, cons), 0)

                    # Pénalités pour la déviation
                    # On utilise try-except pour éviter les erreurs si la variable n'existe pas
                    try:
                        penalite_var = penalite_culture_vars[(cons, proportion, prod_i)]
                        prob += (
                            (allocation_reelle_culture - allocation_cible_culture)
                            <= penalite_var * allocation_cible_culture,
                            f"Penalite_Culture_Plus_{cons}_{proportion}_{prod_i}",
                        )
                        prob += (
                            (allocation_cible_culture - allocation_reelle_culture)
                            <= penalite_var * allocation_cible_culture,
                            f"Penalite_Culture_Moins_{cons}_{proportion}_{prod_i}",
                        )
                    except KeyError:
                        # La variable n'existe pas, on ignore l'ajout des contraintes
                        pass

        # (Option MILP) binaire no-swap import/surplus par produit — formulation stricte
        use_milp_no_swap = True  # mets False pour rester LP 100%
        if use_milp_no_swap:
            # binaire par produit : 1 => mode "import" (I>0 autorisé, U=0), 0 => mode "surplus" (U>0 autorisé, I=0)
            y_vars = LpVariable.dicts("y", list(df_prod.index), 0, 1, cat="Binary")

            # --------- 1) Besoin max importable par produit p (constantes) ----------
            # (consommateurs + infrastructures énergie capables d’utiliser p)
            M_imp = {}
            for p in df_prod.index:
                # consommateurs
                cons_need = sum(
                    float(df_cons.loc[c, "Ingestion (ktN)"])
                    for c in df_cons.index
                    if p in all_cultures_regime.get(c, set())
                )
                # énergie (via cibles et conversion MWh/ktN, si la fac peut utiliser p)
                energy_need = 0.0
                for fac in df_energy.index:
                    fac_has_p = any(
                        p in r["Products"]
                        for _, r in diets[diets["Consumer"] == fac].iterrows()
                    )
                    if fac_has_p:
                        target_gwh = float(
                            df_energy.loc[fac, "Target Energy Production (GWh)"]
                        )
                        if target_gwh > 0:
                            conv = self._conv_MWh_per_ktN(fac, p)  # MWh/ktN
                            if conv > 0:
                                energy_need += (target_gwh * 1000.0) / conv
                M_imp[p] = cons_need + energy_need + 1e-6  # Big-M import

            # --------- 2) Agrégat des imports (consommateurs + énergie) -------------
            def I_total_p(p: str):
                return lpSum(
                    I_vars[(p, c)] for c in df_cons.index if (p, c) in I_vars
                ) + lpSum(
                    I_energy_vars[(p, fac)]
                    for fac in df_energy.index
                    if (p, fac) in I_energy_vars
                )

            # --------- 3) Big-M constant sur le surplus possible --------------------
            # Borne CONSTANTE >= surplus maximal plausible (prospectif compatible).
            # - Historique / produits animaux : dispo historique.
            # - Prospectif / végétal : Ymax*Area -> ktFW, *N%, *co-prod, *(1-waste-other).
            def _surplus_upperbound_const(p: str) -> float:
                try:
                    is_animal = str(df_prod.at[p, "Type"]).strip().lower() == "animal"
                    if (not getattr(self, "prospective", False)) or is_animal:
                        base = float(
                            df_prod.at[p, "Available Nitrogen Production (ktN)"] or 0.0
                        )
                        return max(1e-6, base, M_imp.get(p, 0.0))
                    # végétal en mode prospectif
                    c = df_prod.at[p, "Origin compartment"]  # index culture
                    Ymax_tFW_ha = float(
                        self.df_cultures.at[c, "Maximum Yield (tFW/ha)"]
                    )
                    Area_ha = float(self.df_cultures.at[c, "Area (ha)"])
                    Npct = (
                        float(df_prod.at[p, "Nitrogen Content (%)"]) / 100.0
                        if "Nitrogen Content (%)" in df_prod.columns
                        else 0.0
                    )
                    copct = (
                        float(df_prod.at[p, "Co-Production Ratio (%)"]) / 100.0
                        if "Co-Production Ratio (%)" in df_prod.columns
                        else 1.0
                    )
                    waste = (
                        float(df_prod.at[p, "Waste (%)"]) / 100.0
                        if "Waste (%)" in df_prod.columns
                        else 0.0
                    )
                    other = (
                        float(df_prod.at[p, "Other uses (%)"]) / 100.0
                        if "Other uses (%)" in df_prod.columns
                        else 0.0
                    )
                    # tFW -> ktFW : /1000 ; puis * N% ; puis * co-pro ; puis (1 - waste - other)
                    ub = (
                        (Ymax_tFW_ha * Area_ha / 1000.0)
                        * Npct
                        * copct
                        * max(0.0, 1.0 - waste - other)
                    )
                    if not (ub > 0.0 and math.isfinite(ub)):
                        ub = 0.0
                    # au minimum la demande totale pour ne pas bloquer
                    return max(1e-6, ub, M_imp.get(p, 0.0))
                except Exception:
                    # fallback robuste : au moins la demande
                    return max(1.0, M_imp.get(p, 0.0))

            M_sur_max = {p: _surplus_upperbound_const(p) for p in df_prod.index}

            # --------- 4) Contraintes "either-or" strictes --------------------------
            for p in df_prod.index:
                # Si y_p = 1 (mode import)  -> autorise imports (≤ M_imp), force U_p = 0
                prob += I_total_p(p) <= M_imp[p] * y_vars[p], f"NoSwap_I_{p}"
                prob += U_vars[p] <= M_sur_max[p] * (1 - y_vars[p]), f"NoSwap_U_{p}"
                # => y_p = 1  ⇒  U_p ≤ 0  (car (1 - y_p)=0) ; y_p = 0  ⇒  I_total_p ≤ 0.
                # Pas de produit binaire × expression : M_sur_max et M_imp sont des constantes.

        # prob.writeLP("model.lp")

        # 2) activer les logs CBC (et voir l’infeasibility si c’est le cas)
        # from pulp import PULP_CBC_CMD
        # prob.solve(PULP_CBC_CMD(msg=True))

        # Résolution du problème
        prob.solve()
        self._post_solve_supply()

        if prob.status == -1:
            raise Exception("Allocation model infeasible. Please check input data.")

        # from IPython import embed

        # embed()

        # print(LpStatus.get(prob.status, "Unknown"))

        # df_cultures = self.df_cultures
        df_prod = self.df_prod

        if self.debug:
            from pulp import value, lpSum
            import math, pprint

            def V(expr):
                # Valeur sûre (float); 0.0 si None / non-évaluable
                try:
                    return float(value(expr)) if expr is not None else 0.0
                except Exception:
                    return 0.0

            # ---------------- Normaliseurs identiques à l'objectif ----------------
            n_pairs = max(1, len(pairs))
            n_prod = max(1, len(df_prod))
            n_energy = max(1, len(df_energy.index))

            # ---------------- Reconstruire les termes de l'objectif ----------------
            expr_diet = (poids_penalite_deviation / n_pairs) * lpSum(
                delta_vars.values()
            )
            expr_cult = (poids_penalite_culture / n_prod) * (
                lpSum(penalite_culture_vars.values()) + penalite_culture_energy_term
            )
            expr_import = poids_import_brut * raw_import_term
            expr_fair = w_fair * ((fair_term + fair_term_energy) / n_prod)
            expr_energy_prod = W_ENERGY_PROD * energy_dev_total / n_energy
            expr_energy_diets = W_ENERGY_INPUT * lpSum(delta_fac.values())

            # ------------------ Termes prospectifs (si activé) -------------------
            expr_syn_excess = 0
            expr_syn_distribution = 0

            if self.prospective:
                # (i) Excès d'engrais synthétiques, normalisé par les seuils
                try:
                    W_SYN = float(
                        self.df_global.loc["Weight synthetic fertilizer", "value"]
                    )
                except Exception:
                    W_SYN = 0.0

                th_crops = self._pros_vars.get("th_crops", 0.0)
                th_grass = self._pros_vars.get("th_grass", 0.0)
                exc_crops = self._pros_vars.get("exc_crops", 0)  # var LP
                exc_grass = self._pros_vars.get("exc_grass", 0)  # var LP

                term_syn_crops = (exc_crops / max(1e-9, th_crops)) if th_crops else 0
                term_syn_grass = (exc_grass / max(1e-9, th_grass)) if th_grass else 0
                expr_syn_excess = W_SYN * (term_syn_crops + term_syn_grass)

                # (ii) Répartition autour de F* (déviations relatives |f-F*|/F* via dev+/-)
                try:
                    W_DIS = float(
                        self.df_global.loc["Weight synthetic distribution", "value"]
                    )
                except Exception:
                    W_DIS = 0.0

                dev_pos = self._pros_vars.get("devF_rel_pos", {})  # {c: var+}
                dev_neg = self._pros_vars.get("devF_rel_neg", {})  # {c: var-}
                if W_DIS > 0 and len(dev_pos):
                    expr_syn_distribution = W_DIS * lpSum(
                        (dev_pos[c] + dev_neg.get(c, 0)) for c in dev_pos.keys()
                    )

            # ------------------------- Valeurs pondérées --------------------------
            w_diet = V(expr_diet)
            w_cult = V(expr_cult)
            w_import = V(expr_import)
            w_fair = V(expr_fair)
            w_energy_inputs = V(expr_energy_diets)
            w_energy_prod = V(expr_energy_prod)

            w_syn_excess = V(expr_syn_excess) if self.prospective else 0.0
            w_syn_distribution = V(expr_syn_distribution) if self.prospective else 0.0

            # Somme totale reconstituée (même formalisme que l'objectif)
            obj_val = V(
                expr_diet
                + expr_cult
                + expr_import
                + expr_fair
                + expr_energy_diets
                + expr_energy_prod
                + (expr_syn_excess if self.prospective else 0)
                + (expr_syn_distribution if self.prospective else 0)
            )

            def share(x):
                return (
                    (100.0 * x / obj_val)
                    if (obj_val and not math.isclose(obj_val, 0.0))
                    else None
                )

            # -------------------- Détailler par infrastructure --------------------
            #  a) production d'énergie : variable unique {fac}_energy_dev
            per_fac_energy_prod = {}
            for fac in df_energy.index:
                v = next(
                    (
                        var
                        for var in prob.variables()
                        if var.name == f"{fac}_energy_dev"
                    ),
                    None,
                )
                if v is not None:
                    per_fac_energy_prod[fac] = V((W_ENERGY_PROD / n_energy) * v)

            #  b) inputs d'énergie : somme des deltas par facility, / n_pairs_fac
            per_fac_energy_inputs = {}
            if len(df_energy.index):
                for fac, _row in df_energy.iterrows():
                    n_pairs_fac = max(1, len(diets[diets["Consumer"] == fac]))
                    sum_deltas_fac = lpSum(
                        [
                            var
                            for var in prob.variables()
                            if var.name.startswith(f"delta_{fac}_")
                        ]
                    )
                    per_fac_energy_inputs[fac] = V(
                        W_ENERGY_INPUT * (sum_deltas_fac / n_pairs_fac)
                    )

            # ----------------------------- Familles ------------------------------
            families = {
                "diet_deviation": {
                    "weighted": w_diet,
                    "share_%": share(w_diet),
                    "weight": float(poids_penalite_deviation),
                },
                "intra_group_distribution_ALL": {
                    "weighted": w_cult,
                    "share_%": share(w_cult),
                    "weight": float(poids_penalite_culture),
                    "normalizer": {"n_products": n_prod},
                    "note": "Inclut pénalités produits + pénalités des infrastructures (terme unique).",
                },
                "import_brut": {
                    "weighted": w_import,
                    "share_%": share(w_import),
                    "weight": float(poids_import_brut),
                    "note": "raw_import_term est déjà normalisé (inchangé).",
                },
                "fair_total": {
                    "weighted": w_fair,
                    "share_%": share(w_fair),
                    "weight": float(w_fair),
                },
                "energy_inputs_total": {
                    "weighted": w_energy_inputs,
                    "share_%": share(w_energy_inputs),
                    "weight": float(W_ENERGY_INPUT),
                    "per_facility": per_fac_energy_inputs,
                },
                "energy_production_total": {
                    "weighted": w_energy_prod,
                    "share_%": share(w_energy_prod),
                    "weight": float(W_ENERGY_PROD),
                    "per_facility": per_fac_energy_prod,
                },
            }

            # Familles prospectives (ajout conditionnel)
            if self.prospective:
                # Détails composants pour l’excès synthétique
                W_SYN = (
                    float(self.df_global.loc["Weight synthetic fertilizer", "value"])
                    if "Weight synthetic fertilizer" in self.df_global.index
                    else 0.0
                )
                th_crops = self._pros_vars.get("th_crops", 0.0)
                th_grass = self._pros_vars.get("th_grass", 0.0)
                exc_crops = self._pros_vars.get("exc_crops", 0)
                exc_grass = self._pros_vars.get("exc_grass", 0)

                comp_crops = V(
                    W_SYN * ((exc_crops / max(1e-9, th_crops)) if th_crops else 0)
                )
                comp_grass = V(
                    W_SYN * ((exc_grass / max(1e-9, th_grass)) if th_grass else 0)
                )

                families["synthetic_fertilizer_excess_total"] = {
                    "weighted": w_syn_excess,
                    "share_%": share(w_syn_excess),
                    "weight": float(W_SYN),
                    "components": {
                        "crops_weighted": comp_crops,
                        "grasslands_weighted": comp_grass,
                    },
                    "normalizer": {
                        "th_crops(ktN)": float(th_crops) if th_crops else 0.0,
                        "th_grass(ktN)": float(th_grass) if th_grass else 0.0,
                    },
                }

                W_DIS = (
                    float(self.df_global.loc["Weight synthetic distribution", "value"])
                    if "Weight synthetic distribution" in self.df_global.index
                    else 0.0
                )
                families["synthetic_distribution_total"] = {
                    "weighted": w_syn_distribution,
                    "share_%": share(w_syn_distribution),
                    "weight": float(W_DIS),
                    "note": "Somme des déviations relatives |f - F*|/F* via variables dev+ / dev-.",
                }

            # ------------------------------- Rapport ------------------------------
            solver_info = {
                "status_code": prob.status,
                "status": LpStatus.get(prob.status, "Unknown"),
            }

            weights_used = {
                "Weight diet": float(poids_penalite_deviation),
                "Weight distribution": float(poids_penalite_culture),
                "Weight fair local split": float(w_fair),
                "Weight import brut": float(poids_import_brut),
                "Weight energy inputs": float(W_ENERGY_INPUT),
                "Weight energy production": float(W_ENERGY_PROD),
            }
            if self.prospective:
                weights_used.update(
                    {
                        "Weight synthetic fertilizer": float(
                            self.df_global.loc["Weight synthetic fertilizer", "value"]
                        )
                        if "Weight synthetic fertilizer" in self.df_global.index
                        else 0.0,
                        "Weight synthetic distribution": float(
                            self.df_global.loc["Weight synthetic distribution", "value"]
                        )
                        if "Weight synthetic distribution" in self.df_global.index
                        else 0.0,
                    }
                )

            sanity_terms = [
                share(w_diet),
                share(w_cult),
                share(w_import),
                share(w_fair),
                share(w_energy_inputs),
                share(w_energy_prod),
            ]
            if self.prospective:
                sanity_terms += [share(w_syn_excess), share(w_syn_distribution)]

            report = {
                "solver": solver_info,
                "objective_total": obj_val,
                "families": families,
                "weights_used": weights_used,
                "sanity": {
                    "sum_shares_%": sum(s for s in sanity_terms if s is not None)
                },
            }

            pprint.pp(report, sort_dicts=False)

        # Warning si un élément de la diète des energy facilities n'a pas de pouvoir énergétique
        for fac, _, items in pairs_fac_all:  # ou ta structure équivalente
            for it in items:
                if self._conv_MWh_per_ktN(fac, it) == 0.0:
                    warnings.warn(
                        f"[Energy power] No power for ({fac}, {it}) → energy=0."
                    )

        # ---------- A) Table des allocations (y compris infrastructures énergie)
        allocations = []

        # 1) allocations locales (produits -> consommateurs)
        for (prod_i, cons), var in x_vars.items():
            v = float(var.varValue or 0.0)
            if v <= 1e-6:
                continue
            Type = (
                "Local culture Feed"
                if cons in df_elevage.index
                else "Local culture Food"
            )
            allocations.append(
                {
                    "Product": prod_i,
                    "Consumer": cons,
                    "Allocated Nitrogen": v,
                    "Type": Type,
                }
            )

        # 2) importations (produit -> consommateur)
        for (prod_i, cons), var in I_vars.items():
            v = float(var.varValue or 0.0)
            if v <= 1e-6:
                continue
            Type = "Imported Feed" if cons in df_elevage.index else "Imported Food"
            allocations.append(
                {
                    "Product": prod_i,
                    "Consumer": cons,
                    "Allocated Nitrogen": v,
                    "Type": Type,
                }
            )

        # 3) infrastructures énergie : produits/excréta/waste -> infrastructure
        for fac in df_energy.index:
            fac_type = str(df_energy.loc[fac, "Type"])
            # produits
            for p, var in energy_vars.get(fac, {}).get("prod", {}).items():
                v = float(var.varValue or 0.0)
                if v > 1e-6:
                    allocations.append(
                        {
                            "Product": p,
                            "Consumer": fac,
                            "Allocated Nitrogen": v,
                            "Type": "Local culture Energy",
                        }
                    )
            # excréta
            for e, var in energy_vars.get(fac, {}).get("excr", {}).items():
                v = float(var.varValue or 0.0)
                if v > 1e-6:
                    allocations.append(
                        {
                            "Product": e,
                            "Consumer": fac,
                            "Allocated Nitrogen": v,
                            "Type": "Excretion Energy",
                        }
                    )
            # waste agrégé
            from pulp import value

            wvar = energy_vars.get(fac, {}).get("waste", None)
            if wvar is not None:  # wvar est un LpVariable
                v = float(value(wvar) or 0.0)
                if v > 1e-6:
                    allocations.append(
                        {
                            "Product": "waste",
                            "Consumer": fac,
                            "Allocated Nitrogen": v,
                            "Type": "Waste Energy",
                        }
                    )
        for (p, fac), var in I_energy_vars.items():
            v = float(var.varValue or 0.0)
            if v <= 1e-6:
                continue
            allocations.append(
                {
                    "Product": p,
                    "Consumer": fac,
                    "Allocated Nitrogen": v,
                    "Type": "Imported Energy",
                }
            )

        allocations_df = pd.DataFrame(allocations)
        allocations_df = allocations_df[
            allocations_df["Allocated Nitrogen"].abs() >= 1e-6
        ]
        self.allocations_df = allocations_df  # ← remplace self.allocations_df existant

        # ---------- B) Déviations : consommateurs "classiques"
        deviations = []
        for cons, proportion, products_list in pairs:
            consumer_type = df_cons.loc[cons, "Type"]
            besoin_total = float(df_cons.loc[cons, "Ingestion (ktN)"])
            # allocation totale du groupe
            alloc_tot = sum(
                x_vars.get((p, cons), 0).varValue or 0 for p in products_list
            ) + sum(I_vars.get((p, cons), 0).varValue or 0 for p in products_list)
            prop_eff = (alloc_tot / besoin_total) if besoin_total > 0 else 0.0
            delta_var_key = (cons, tuple(products_list))
            if delta_var_key in delta_vars:
                deviation_value = float(delta_vars[delta_var_key].varValue or 0)
                signe = 1 if prop_eff > proportion else -1
                deviations.append(
                    {
                        "Consumer": cons,
                        "Type": consumer_type,
                        "Expected Proportion (%)": round(proportion, 5) * 100,
                        "Deviation (%)": signe * round(deviation_value, 4) * 100,
                        "Proportion Allocated (%)": (
                            round(proportion, 5) + signe * round(deviation_value, 4)
                        )
                        * 100,
                        "Product": ", ".join(products_list),
                    }
                )

        # ---------- C) Déviations : infrastructures énergie (par DIET associée)
        energy_dev_rows = []
        for fac in df_energy.index:
            fac_type = str(df_energy.loc[fac, "Type"])
            fac_diet_df = diets[diets["Consumer"] == fac].copy()

            # N total vers la fac (local + excréta + waste + imports)
            N_total = 0.0
            N_total += sum(
                float(v.varValue or 0.0) for v in energy_vars[fac]["prod"].values()
            )
            N_total += sum(
                float(v.varValue or 0.0) for v in energy_vars[fac]["excr"].values()
            )
            if energy_vars[fac]["waste"] is not None:
                N_total += float(value(energy_vars[fac]["waste"]))
            N_total += sum(
                float(v.varValue or 0.0)
                for (p, f), v in I_energy_vars.items()
                if f == fac
            )

            for _, row in fac_diet_df.iterrows():
                prop = float(row["Proportion"])
                prop = prop / 100.0 if prop > 1.0 else prop
                prods = list(row["Products"])
                label = ", ".join(prods)

                N_group = 0.0
                for it in prods:
                    if it in energy_vars[fac]["prod"]:
                        N_group += float(energy_vars[fac]["prod"][it].varValue or 0.0)
                    elif it in energy_vars[fac]["excr"]:
                        N_group += float(energy_vars[fac]["excr"][it].varValue or 0.0)
                    elif it == "waste" and (energy_vars[fac]["waste"] is not None):
                        N_group += float(value(energy_vars[fac]["waste"]))
                    if (it, fac) in I_energy_vars:
                        N_group += float(I_energy_vars[(it, fac)].varValue or 0.0)

                alloc_pct = 100.0 * N_group / N_total if N_total > 1e-6 else 0.0
                exp_pct = 100.0 * prop
                dev_pct = alloc_pct - exp_pct

                energy_dev_rows.append(
                    {
                        "Consumer": fac,
                        "Type": "Energy",
                        "Expected Proportion (%)": exp_pct,
                        "Deviation (%)": dev_pct,
                        "Proportion Allocated (%)": alloc_pct,
                        "Product": label,
                    }
                )

        deviations_df = pd.DataFrame(deviations)
        deviations_energy_df = pd.DataFrame(energy_dev_rows)
        self.deviations_df = pd.concat(
            [deviations_df, deviations_energy_df], ignore_index=True
        )

        # Extraction des importations (consommateurs ET bioraffineries)
        importations = []

        # (a) Imports vers les consommateurs (humains & animaux)
        for cons in df_cons.index:
            for prod_i in all_cultures_regime[cons]:
                if (prod_i, cons) in I_vars:
                    import_value = float(I_vars[(prod_i, cons)].varValue or 0.0)
                    if import_value <= 0:
                        continue
                    Type = "Feed" if cons in df_elevage.index else "Food"
                    importations.append(
                        {
                            "Consumer": cons,
                            "Product": prod_i,
                            "Type": Type,
                            "Imported Nitrogen (ktN)": import_value,
                        }
                    )

        # Imports dédiés aux bioraffineries (produits -> facility)
        for (p, fac), var in I_energy_vars.items():
            v = float(var.varValue or 0.0)
            if v > 0:
                importations.append(
                    {
                        "Consumer": fac,
                        "Product": p,
                        "Type": "Energy (Bioraffinery)",
                        "Imported Nitrogen (ktN)": v,
                    }
                )

        # Convertir en DataFrame
        if importations is None or len(importations) == 0:
            self.importations_df = pd.DataFrame(
                columns=["Consumer", "Product", "Type", "Imported Nitrogen (ktN)"]
            )
        else:
            self.importations_df = pd.DataFrame(importations)

        # ---------- D) Tableau énergie (source -> target) + parts par infrastructure
        rows_energy = []
        for fac in df_energy.index:
            # produits
            for p, var in energy_vars[fac]["prod"].items():
                N = float(var.varValue or 0.0)
                if N <= 1e-6:
                    continue
                # MWh/ktN = (MWh/tFW) * 1000 / (%N/100)
                mwh_per_ktn = self._conv_MWh_per_ktN(fac, p)
                E_gwh = N * (mwh_per_ktn / 1000.0)
                rows_energy.append(
                    {
                        "source": p,
                        "target": fac,
                        "allocation (ktN)": N,
                        "energy production (GWh)": E_gwh,
                    }
                )
            # Imports produits vers la facility : source = "<Sub Type> trade"
            for (p, f), var in I_energy_vars.items():
                if f != fac:
                    continue
                N = float(var.varValue or 0.0)
                if N <= 1e-6:
                    continue
                E_gwh = N * self._conv_MWh_per_ktN(fac, p) / 1000
                cat_trade = f"{df_prod.loc[p, 'Sub Type']} trade"
                rows_energy.append(
                    {
                        "source": cat_trade,
                        "target": fac,
                        "allocation (ktN)": N,
                        "energy production (GWh)": E_gwh,
                    }
                )

            # excréta
            for e, var in energy_vars[fac]["excr"].items():
                N = float(var.varValue or 0.0)
                if N <= 1e-6:
                    continue
                mwh_per_ktn = self._conv_MWh_per_ktN(fac, e)
                E_gwh = N * (mwh_per_ktn / 1000.0)
                rows_energy.append(
                    {
                        "source": e,
                        "target": fac,
                        "allocation (ktN)": N,
                        "energy production (GWh)": E_gwh,
                    }
                )

            # waste agrégé
            wvar = energy_vars[fac]["waste"]
            if wvar is not None:
                N = float(value(wvar))
                if N > 1e-6:
                    E_gwh = N * self._conv_MWh_per_ktN(fac, "waste") / 1000
                    rows_energy.append(
                        {
                            "source": "waste",
                            "target": fac,
                            "allocation (ktN)": N,
                            "energy production (GWh)": E_gwh,
                        }
                    )

        df_energy_flows = pd.DataFrame(rows_energy)
        if not df_energy_flows.empty:
            # parts par infrastructure (target)
            by_t = df_energy_flows.groupby("target", as_index=False).agg(
                total_alloc=("allocation (ktN)", "sum"),
                total_energy=("energy production (GWh)", "sum"),
            )
            df_energy_flows = df_energy_flows.merge(by_t, on="target", how="left")
            df_energy_flows["allocation share (%)"] = (
                100.0
                * df_energy_flows["allocation (ktN)"]
                / df_energy_flows["total_alloc"]
            ).fillna(0.0)
            df_energy_flows["energy production share (%)"] = (
                100.0
                * df_energy_flows["energy production (GWh)"]
                / df_energy_flows["total_energy"]
            ).fillna(0.0)
            df_energy_flows = df_energy_flows.drop(
                columns=["total_alloc", "total_energy"]
            )
            # Ajouter Type & Diet pour information
            df_energy_flows["Type"] = df_energy_flows["target"].map(df_energy["Type"])
            df_energy_flows["Diet"] = df_energy_flows["target"].map(df_energy["Diet"])
        else:
            df_energy_flows = pd.DataFrame(
                columns=[
                    "source",
                    "target",
                    "allocation (ktN)",
                    "allocation share (%)",
                    "energy production (GWh)",
                    "energy production share (%)",
                    "Type",
                    "Diet",
                ]
            )

        # Remplacer df_energy par la vue "source->target" demandée
        self.df_energy_flows = df_energy_flows.copy()

        # Mise à jour de df_energy
        df_energy["Energy Production (GWh)"] = 0.0

        for fac, E_expr_GWh in energy_E_GWh_expr.items():
            # evaluate the PuLP expression at the optimum
            E_GWh = float(value(E_expr_GWh))
            # store in GWh
            df_energy.loc[fac, "Energy Production (GWh)"] = E_GWh

        df_energy["Nitrogen Input to Energy (ktN)"] = 0.0
        for fac in df_energy.index:
            N_input = 0.0
            for p, var in energy_vars[fac]["prod"].items():
                v = float(var.varValue or 0.0)
                if v > 1e-6:
                    N_input += v
            for e, var in energy_vars[fac]["excr"].items():
                v = float(var.varValue or 0.0)
                if v > 1e-6:
                    N_input += v
            wvar = energy_vars[fac]["waste"]
            if wvar is not None:
                v = float(value(wvar) or 0.0)
                if v > 1e-6:
                    N_input += v
            for (p, f), var in I_energy_vars.items():
                if f == fac:
                    v = float(var.varValue or 0.0)
                    if v > 1e-6:
                        N_input += v
            df_energy.loc[fac, "Nitrogen Input to Energy (ktN)"] = N_input

        # ---------- E) Mettre à jour df_excr : "Excretion to Energy (ktN)" / "Excretion to soil (ktN)"
        df_excr["Excretion to Energy (ktN)"] = 0.0
        for fac in df_energy.index:
            for e, var in energy_vars[fac]["excr"].items():
                v = float(var.varValue or 0.0)
                if v > 1e-6:
                    df_excr.loc[e, "Excretion to Energy (ktN)"] += v
        df_excr["Excretion to soil (ktN)"] = (
            df_excr["Excretion after volatilization (ktN)"]
            - df_excr["Excretion to Energy (ktN)"]
        )

        # ---------- F) Mettre à jour df_prod : "Nitrogen For Energy (ktN)" + Export recalculé
        df_prod["Nitrogen For Energy (ktN)"] = 0.0
        for fac in df_energy.index:
            for p, var in energy_vars[fac]["prod"].items():
                v = float(var.varValue or 0.0)
                if v > 1e-6:
                    df_prod.loc[p, "Nitrogen For Energy (ktN)"] += v

        # (re)calcul food/feed depuis allocations_df
        df_prod["Nitrogen For Feed (ktN)"] = 0.0
        df_prod["Nitrogen For Food (ktN)"] = 0.0
        if not allocations_df.empty:
            feed_by_prod = (
                allocations_df.loc[allocations_df["Type"] == "Local culture Feed"]
                .groupby("Product")["Allocated Nitrogen"]
                .sum()
            )
            food_by_prod = (
                allocations_df.loc[allocations_df["Type"] == "Local culture Food"]
                .groupby("Product")["Allocated Nitrogen"]
                .sum()
            )
            df_prod.loc[feed_by_prod.index, "Nitrogen For Feed (ktN)"] = (
                feed_by_prod.values
            )
            df_prod.loc[food_by_prod.index, "Nitrogen For Food (ktN)"] = (
                food_by_prod.values
            )

        df_prod["Nitrogen Exported (ktN)"] = (
            df_prod["Available Nitrogen Production (ktN)"]
            - df_prod["Nitrogen For Feed (ktN)"]
            - df_prod["Nitrogen For Food (ktN)"]
            - df_prod["Nitrogen For Energy (ktN)"]
        )
        # Petites corrections numériques
        for col in [
            "Nitrogen Exported (ktN)",
            "Nitrogen For Feed (ktN)",
            "Nitrogen For Food (ktN)",
            "Nitrogen For Energy (ktN)",
        ]:
            df_prod[col] = df_prod[col].where(df_prod[col].abs() >= 1e-6, 0.0)

        # Mise à jour de df_elevage
        # Calcul de l'azote total alloué à chaque consommateur
        azote_alloue_elevage = (
            allocations_df.groupby(["Consumer", "Type"])["Allocated Nitrogen"]
            .sum()
            .unstack(fill_value=0)
        )

        azote_alloue_elevage = azote_alloue_elevage.loc[
            azote_alloue_elevage.index.get_level_values("Consumer").isin(
                df_elevage.index
            )
        ]

        # Ajouter les colonnes d'azote alloué dans df_elevage
        df_elevage.loc[:, "Consummed nitrogen from local feed (ktN)"] = (
            df_elevage.index.map(
                azote_alloue_elevage.get(
                    "Local culture Feed", pd.Series(0, index=df_elevage.index)
                )
            )
        )
        df_elevage.loc[:, "Consummed Nitrogen from imported feed (ktN)"] = (
            df_elevage.index.map(
                lambda elevage: azote_alloue_elevage.get(
                    "Imported Feed", pd.Series(0, index=df_elevage.index)
                ).get(elevage, 0)
                + azote_alloue_elevage.get(
                    "Excess feed imports", pd.Series(0, index=df_elevage.index)
                ).get(elevage, 0)
            )
        )

        # Génération des flux pour les produits locaux
        allocations_locales = allocations_df[
            allocations_df["Type"].isin(
                ["Local culture Food", "Local culture Feed", "Local culture Energy"]
            )
        ]

        for cons in df_cons.index:
            target = {cons: 1}
            source = (
                allocations_locales[allocations_locales["Consumer"] == cons]
                .set_index("Product")["Allocated Nitrogen"]
                .to_dict()
            )
            if source:
                flux_generator.generate_flux(source, target)

        # Génération des flux pour les importations
        allocations_imports = allocations_df[
            allocations_df["Type"].isin(["Imported Feed", "Imported Food"])
        ]

        for cons in df_cons.index:
            target = {cons: 1}
            cons_vege_imports = allocations_imports[
                allocations_imports["Consumer"] == cons
            ]

            # Initialisation d'un dictionnaire pour collecter les flux par catégorie
            flux = {}

            for _, row in cons_vege_imports.iterrows():
                prod_i = row["Product"]
                azote_alloue = row["Allocated Nitrogen"]

                # Récupération de la catégorie de la culture
                categorie = df_prod.loc[prod_i, "Sub Type"]

                # Construction du label source pour l'importation
                label_source = f"{categorie} trade"

                # Accumuler les flux par catégorie
                if label_source in flux:
                    flux[label_source] += azote_alloue
                else:
                    flux[label_source] = azote_alloue

            # Génération des flux pour l'élevage
            if sum(flux.values()) > 0:
                flux_generator.generate_flux(flux, target)

        # Export
        # Le surplus est exporté (ou perdu pour les pailles et prairies permanentes)
        for idx, row in df_prod.iterrows():
            prod = row.name
            categorie = row["Sub Type"]
            nitrogen_value = row["Nitrogen Exported (ktN)"]

            source = {prod: nitrogen_value}

            if categorie not in ["grazing", "non edible meat"]:
                target = {f"{categorie} trade": 1}
            elif categorie == "non edible meat":
                target = {"other sectors": 1}
            else:
                target = {"soil stock": 1}

            flux_generator.generate_flux(source, target)

        df_elevage["Net animal nitrogen exports (ktN)"] = (
            df_prod.loc[df_prod["Type"] == "animal"]
            .groupby("Origin compartment")["Nitrogen Exported (ktN)"]
            .sum()
            .sub(
                self.importations_df.merge(
                    df_prod.loc[df_prod["Type"] == "animal"],
                    left_on="Product",
                    right_index=True,
                )
                .groupby("Origin compartment")["Imported Nitrogen (ktN)"]
                .sum(),
                fill_value=0,
            )
            .reindex(df_elevage.index, fill_value=0)
        )

        # ---------- G) Flux vers les infrastructures énergie
        # source (produits/excréta/waste) -> infrastructure
        if not self.df_energy_flows.empty:
            for fac in df_energy.index:
                target = {fac: 1}
                # agrège N(allocation) par source pour cette infra
                src = self.df_energy_flows.loc[
                    self.df_energy_flows["target"] == fac,
                    ["source", "allocation (ktN)"],
                ]
                if not src.empty:
                    source = src.set_index("source")["allocation (ktN)"].to_dict()
                    if source:
                        flux_generator.generate_flux(source, target)

        # ---------- H) Sorties des infrastructures :
        # - Si Type == "Methanizer" : tout l'azote va vers l'épandage (digestats)
        # - Sinon (ex. "Bioraffinery") : vers "hydrocarbures"
        if not self.df_energy_flows.empty:
            for fac in df_energy.index:
                fac_type = str(df_energy.loc[fac, "Type"])
                N_alloc_fac = float(
                    self.df_energy_flows.loc[
                        self.df_energy_flows["target"] == fac, "allocation (ktN)"
                    ].sum()
                )
                if N_alloc_fac <= 1e-6:
                    continue
                source = {fac: N_alloc_fac}
                if fac_type.lower() == "methanizer":
                    flux_generator.generate_flux(source, target_epandage)
                else:
                    flux_generator.generate_flux(source, {"hydrocarbures": 1})

        # ---------- I) Mise à jour des flux digestats et excretion to soil
        # Excretion réelles vers le sol (près départ vers digestat)
        source = df_excr.loc[
            df_excr["Type"].isin(["manure", "slurry"]),
            "Excretion to soil (ktN)",
        ].to_dict()

        # épandage excretat
        flux_generator.generate_flux(source, target_epandage)

        # Excretion de prairie
        source_grass_excr = df_excr.loc[
            df_excr["Type"].isin(["grasslands excretion"]), "Excretion to soil (ktN)"
        ].to_dict()

        flux_generator.generate_flux(source_grass_excr, target_grass)

        # On l'enregistre dans la colonne excretion fertilization. On reprendra ce calcul après pour les flux vers les méthaniseurs
        total_excr_fields_ktN = sum(source.values())
        target_series = pd.Series(target_epandage)

        total_excr_grass_ktN = sum(source_grass_excr.values())
        target_grass_series = pd.Series(target_grass)

        # On recalcule la fertilisation par excretion
        self.df_cultures["Excreta Fertilization (ktN)"] = 0.0

        # Excretion sur prairies :
        self.df_cultures["Excreta Fertilization (ktN)"] += (
            self.df_cultures.index.map(target_grass_series).fillna(0)
            * total_excr_grass_ktN
        )

        # Epandage excretion et boue
        self.df_cultures["Excreta Fertilization (ktN)"] += self.df_cultures.index.map(
            target_series
        ).fillna(0) * (total_excr_fields_ktN + total_boue_ktN)

        # Epandage digestat
        source = df_energy.loc[
            df_energy["Type"].isin(["Methanizer"]),
            "Nitrogen Input to Energy (ktN)",
        ].to_dict()

        # flux_generator.generate_flux(source, target_epandage)

        total_digestat_fields_ktN = sum(source.values())
        target_series = pd.Series(target_epandage)
        self.df_cultures["Digestat Fertilization (ktN)"] = (
            self.df_cultures.index.map(target_series).fillna(0)
            * total_digestat_fields_ktN
        )

        # ---------- J) Bilan sol
        self._recompute_soil_budget_unified()

        # Calcul de imbalance dans df_cultures
        self.df_cultures["Balance (ktN)"] = (
            self.df_cultures["Inputs to field (ktN)"]
            - self.df_cultures["Harvested Production (ktN)"]
        )

        # On équilibre Haber-Bosch avec atmospheric N2 pour le faire entrer dans le système
        target = {
            "Haber-Bosch": self.adjacency_matrix[label_to_index["Haber-Bosch"], :].sum()
        }
        source = {"atmospheric N2": 1}
        flux_generator.generate_flux(source, target)

        mask = df_elevage["Ingestion (ktN)"] > 0
        df_elevage.loc[mask, "Conversion factor (%)"] = (
            (
                df_elevage.loc[mask, "Edible Nitrogen (ktN)"]
                + df_elevage.loc[mask, "Dairy Nitrogen (ktN)"]
            )
            * 100
            / df_elevage.loc[mask, "Ingestion (ktN)"]
        )

        # On ajoute une ligne total à df_cultures et df_elevage et df_prod
        colonnes_a_exclure = [
            "Spreading Rate (%)",
            "Nitrogen Content (%)",
            "Seed input (ktN/ktN)",
            "Category",
            "Main Production",
            "Harvest Index",
            "Nitrogen Harvest Index",
            "Characteristic Fertilisation (kgN/ha)",
            "Maximum Yield (tFW/ha)",
            "Ymax (kgN/h)",
            "Residue Nitrogen Content (%)",
            "Nitrogen Harvest IndexFan coef a",
            "Fan coef b",
            "BGN",
            "BNF alpha",
            "BNF beta",
            "Fertilization Need (kgN/qtl)",
            "Surface Fertilization Need (kgN/ha)",
            "Yield (qtl/ha)",
            "Yield (kgN/ha)",
            "Surface Non Synthetic Fertilizer Use (kgN/ha)",
            "Raw Surface Synthetic Fertilizer Use (ktN/ha)",
        ]
        colonnes_a_sommer = self.df_cultures.columns.difference(colonnes_a_exclure)
        total = self.df_cultures[colonnes_a_sommer].sum()
        total.name = "Total"
        self.df_cultures_display = pd.concat([self.df_cultures, total.to_frame().T])
        self.df_cultures_display = self.df_cultures_display.loc[
            self.df_cultures_display["Area (ha)"] != 0
        ]

        colonnes_a_exclure = [
            "Excreted indoor (%)",
            "Excreted indoor as manure (%)",
            "Excreted indoor as slurry (%)",
            "Excreted on grassland (%)",
            "N-N2 EM manure (%)",
            "N-N2 EM outdoor (%)",
            "N-N2 EM slurry (%)",
            "N-N2O EM manure (%)",
            "N-N2O EM outdoor (%)",
            "N-N2O EM slurry (%)",
            "N-NH3 EM manure (%)",
            "N-NH3 EM outdoor (%)",
            "N-NH3 EM slurry (%)",
            "Conversion factor (%)",
            "Excretion / LU (kgN)",
            "Diet",
        ]
        colonnes_a_sommer = df_elevage.columns.difference(colonnes_a_exclure)
        total = df_elevage[colonnes_a_sommer].sum()
        total.name = "Total"
        self.df_elevage_display = pd.concat([df_elevage, total.to_frame().T])
        self.df_elevage_display = self.df_elevage_display.loc[
            self.df_elevage_display["LU"] != 0
        ]

        colonnes_a_exclure = [
            "Type",
            "Sub Type",
            "Nitrogen Content (%)",
            "Origin compartment",
            "Carbon Content (%)",
            "Waste (%)",
            "Other uses (%)",
        ]
        colonnes_a_sommer = df_prod.columns.difference(colonnes_a_exclure)
        total = df_prod[colonnes_a_sommer].sum()
        total.name = "Total"
        self.df_prod_display = pd.concat([df_prod, total.to_frame().T])
        self.df_prod_display = self.df_prod_display.loc[
            self.df_prod_display["Nitrogen Production (ktN)"] != 0
        ]

        colonnes_a_exclure = [
            "Type",
            "Nitrogen Content (%)",
            "Origin compartment",
            "N-NH3 EM (%)",
            "N-N2 EM (%)",
            "N-N2O EM (%)",
        ]
        colonnes_a_sommer = df_excr.columns.difference(colonnes_a_exclure)
        total = df_excr[colonnes_a_sommer].sum()
        total.name = "Total"
        self.df_excr_display = pd.concat([df_excr, total.to_frame().T])
        self.df_excr_display = self.df_excr_display.loc[
            self.df_excr_display["Excretion to soil (ktN)"] != 0
        ]

        colonnes_a_exclure = [
            "Type",
            "Diet",
            "Share CO2 (%)",
        ]
        colonnes_a_sommer = df_energy.columns.difference(colonnes_a_exclure)
        total = df_energy[colonnes_a_sommer].sum()
        total.name = "Total"
        self.df_energy_display = pd.concat([df_energy, total.to_frame().T])
        self.df_energy_display = self.df_energy_display.loc[
            self.df_energy_display["Target Energy Production (GWh)"] != 0
        ]

        # self.df_cultures = df_cultures
        self.df_elevage = df_elevage
        self.df_prod = df_prod
        self.df_excr = df_excr
        self.df_energy = df_energy
        # self.adjacency_matrix = adjacency_matrix

    def get_df_culture(self):
        """
        Returns the DataFrame containing crop-related data.

        :return: A pandas DataFrame with crop data used in the nitrogen model.
        :rtype: pandas.DataFrame
        """
        return self.df_cultures

    def get_df_elevage(self):
        """
        Returns the DataFrame containing livestock-related data.

        :return: A pandas DataFrame with livestock data used in the nitrogen model.
        :rtype: pandas.DataFrame
        """
        return self.df_elevage

    def get_df_prod(self):
        """
        Returns the DataFrame containing product-related data.

        :return: A pandas DataFrame with product data used in the nitrogen model.
        :rtype: pandas.DataFrame
        """
        return self.df_prod

    def get_transition_matrix(self):
        """
        Returns the full nitrogen transition matrix.

        This matrix represents all nitrogen fluxes between sectors, including core and external processes.

        :return: A 2D NumPy array representing nitrogen fluxes between all sectors.
        :rtype: numpy.ndarray
        """
        return self.adjacency_matrix

    def imported_nitrogen(self):
        """
        Calculates the total amount of nitrogen imported into the system.

        Includes nitrogen in imported food, feed, and excess feed.

        :return: Total imported nitrogen (in ktN).
        :rtype: float
        """
        return self.allocations_df.loc[
            self.allocations_df["Type"].isin(["Imported Food", "Imported Feed"]),
            "Allocated Nitrogen",
        ].sum()

    def net_imported_plant(self):
        """
        Computes the net nitrogen imports of plant.

        Calculated as the difference between total nitrogen imports and plant sector availability after local uses (feed and food).

        :return: Net nitrogen import for plant-based products (in ktN).
        :rtype: float
        """
        return (
            self.importations_df["Imported Nitrogen (ktN)"].sum()
            - self.df_prod.loc[
                (self.df_prod["Type"] == "plant")
                & ~(self.df_prod["Sub Type"].isin(["grazing"])),
                "Nitrogen Exported (ktN)",
            ].sum()
        )

    def net_imported_animal(self):
        """
        Returns the net nitrogen export for animal sectors.

        :return: Total nitrogen exported via animal products (in ktN).
        :rtype: float
        """
        return -self.df_elevage["Net animal nitrogen exports (ktN)"].sum()

    def total_plant_production(self):
        """
        Computes the Harvested Production from all crop categories.

        :return: Total nitrogen produced by crops (in ktN).
        :rtype: float
        """
        return self.df_prod["Nitrogen Production (ktN)"].sum()

    def stacked_plant_production(self):
        """
        Returns the vector of nitrogen production by crop category.

        :return: A pandas Series of nitrogen production per crop.
        :rtype: pandas.Series
        """
        return self.df_prod["Nitrogen Production (ktN)"]

    def production(self, category):
        """
        Returns the nitrogen production from category crops.

        :return: Total nitrogen from cereals (in ktN).
        :rtype: float
        """
        return self.df_cultures.loc[
            self.df_cultures["Category"] == category,
            "Harvested Production (ktN)",
        ].sum()

    def production_r(self, category):
        """
        Returns the share of nitrogen production from cereals relative to total plant production.

        :return: Percentage of total plant nitrogen production from cereals.
        :rtype: float
        """
        return (
            self.df_cultures.loc[
                self.df_cultures["Category"] == category,
                "Harvested Production (ktN)",
            ].sum()
            * 100
            / self.total_plant_production()
        )

    def animal_production(self):
        """
        Returns the total edible nitrogen produced by livestock sectors.

        :return: Total nitrogen in edible animal products (in ktN).
        :rtype: float
        """
        return (
            self.df_elevage["Edible Nitrogen (ktN)"].sum()
            + self.df_elevage["Dairy Nitrogen (ktN)"].sum()
        )

    def emissions(self):
        """
        Computes the total nitrogen emissions from the system.

        Includes N₂O emissions, atmospheric N₂ release, and NH₃ volatilization, with unit conversions.

        :return: A pandas Series with nitrogen emission quantities.
        :rtype: pandas.Series
        """
        return pd.Series(
            {
                "atmospheric N2O": np.round(
                    self.adjacency_matrix[
                        :, self.data_loader.label_to_index["atmospheric N2O"]
                    ].sum()
                    * (14 * 2 + 16)
                    / (14 * 2),
                    2,
                ),
                "atmospheric N2": np.round(
                    self.adjacency_matrix[
                        :, self.data_loader.label_to_index["atmospheric N2"]
                    ].sum(),
                    2,
                ),
                "atmospheric NH3": np.round(
                    self.adjacency_matrix[
                        :, self.data_loader.label_to_index["atmospheric NH3"]
                    ].sum()
                    * 17
                    / 14,
                    2,
                ),
            },
            name="Emission",
        ).to_frame()["Emission"]

    def surfaces(self):
        """
        Returns the cultivated area per crop.

        :return: A pandas Series with area per crop (in hectares).
        :rtype: pandas.Series
        """
        return self.df_cultures["Area (ha)"]

    def surfaces_tot(self):
        """
        Returns the total cultivated area in the model.

        :return: Total area (in hectares).
        :rtype: float
        """
        return self.df_cultures["Area (ha)"].sum()

    def Ftot(self, culture):
        area = self.df_cultures.loc[
            self.df_cultures.index == culture, "Area (ha)"
        ].item()
        if area == 0:  # Vérification pour éviter la division par zéro
            return 0
        return (
            self.adjacency_matrix[:, self.data_loader.label_to_index[culture]].sum()
            * 1e6
            / area
        )

    def Y(self, culture):
        """
        Computes the total nitrogen yield of a given crop (main prod + secondary prod).

        Yield is calculated as nitrogen production (kgN) per hectare for the specified crop.

        :param culture: The name of the crop (index of `df_cultures`).
        :type culture: str
        :return: Nitrogen yield in kgN/ha.
        :rtype: float
        """
        area = self.df_cultures.loc[
            self.df_cultures.index == culture, "Area (ha)"
        ].item()
        if area == 0:  # Vérification pour éviter la division par zéro
            return 0
        return (
            self.df_cultures.loc[
                self.df_cultures.index == culture, "Harvested Production (ktN)"
            ].item()
            * 1e6
            / area
        )

    def LU_density(self):
        """
        Calculates the livestock unit density over the agricultural area.

        :return: Livestock unit per hectare (LU/ha).
        :rtype: float
        """
        return np.round(
            self.df_elevage["LU"].sum() / self.df_cultures["Area (ha)"].sum(), 2
        )

    def NH3_vol(self):
        """
        Returns the total NH₃ volatilization in the system.

        :return: NH₃ emissions in kt.
        :rtype: float
        """
        return self.emissions()["atmospheric NH3"]

    def N2O_em(self):
        """
        Returns the total N₂O emissions in the system.

        :return: N₂O emissions in kt.
        :rtype: float
        """
        return self.emissions()["atmospheric N2O"]

    def export_flux_to_csv(self, filename="flux_data.xlsx"):
        """
        Exporte les flux de la matrice de transition vers un fichier CSV.
        Chaque ligne contient (source, target, value) pour chaque flux non nul.

        :param filename: Nom du fichier CSV de sortie.
        :type filename: str
        """
        # Créer une liste pour stocker les flux
        flux_data = []

        # Parcourir la matrice pour extraire les flux non nuls
        for i, source in enumerate(self.labels):
            for j, target in enumerate(self.labels):
                value = self.adjacency_matrix[i, j]
                if value != 0:  # Ne prendre que les flux non nuls
                    flux_data.append(
                        {"origine": source, "destination": target, "valeur": value}
                    )

        # Créer un DataFrame à partir des données collectées
        df_flux = pd.DataFrame(flux_data)

        # Exporter le DataFrame vers un fichier CSV
        df_flux.to_excel(filename, index=False)

    # =========================
    # Helpers génériques (dans la classe)
    # =========================
    def _find_label_indices(self, *keywords, casefold=True):
        """
        Retourne la liste d'indices de labels dont le nom contient TOUS les mots-clés.
        Exemple: _find_label_indices("atmospheric", "N2O")
        """
        if not hasattr(self.data_loader, "label_to_index"):
            return []
        out = []
        for label, idx in self.data_loader.label_to_index.items():
            name = label.casefold() if casefold else label
            if all(k.casefold() in name for k in keywords):
                out.append(idx)
        return out

    def _indices_for_crops(self):
        """
        Indices (dans le graphe) correspondant aux cultures présentes dans df_cultures.index.
        S'ils n'existent pas dans le graphe, on ignore.
        """
        if not hasattr(self.data_loader, "label_to_index"):
            return []
        idxs = []
        for crop in self.df_cultures.index:
            if crop in self.data_loader.label_to_index:
                idxs.append(self.data_loader.label_to_index[crop])
        return idxs

    def _sum_from_to(self, src_indices, dst_indices):
        """Somme des flux du graphe de src->dst (ktN)."""
        if (
            self.adjacency_matrix is None
            or len(src_indices) == 0
            or len(dst_indices) == 0
        ):
            return 0.0
        return float(self.adjacency_matrix[np.ix_(src_indices, dst_indices)].sum())

    def _safe_series(self, s, dtype=float):
        """Assure une Series float, NaN->0."""
        if s is None:
            return pd.Series(dtype=dtype)
        s = pd.Series(s).copy()
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return s

    def _yield_area_map(self):
        """
        Renvoie deux Series alignées sur df_cultures.index :
        - prod_k : Harvested Production (ktN) par culture (>=0)
        - area_ha: Area (ha)
        + calcule une valeur de repli pour les cultures à prod_k == 0 (moyenne par catégorie puis globale).
        """
        area_ha = self._safe_series(self.df_cultures.get("Area (ha)", 0.0))
        prod_k = self._safe_series(
            self.df_cultures.get(
                "Harvested Production (ktN)",
                self.df_cultures.get("Nitrogen Production (ktN)", 0.0),
            )
        )

        # Replis pour prod_k == 0 → moyenne de la catégorie puis moyenne globale
        zero_mask = prod_k <= 0
        if zero_mask.any():
            # moyennes par catégorie
            if "Category" in self.df_cultures.columns:
                cat_means = (
                    prod_k.groupby(self.df_cultures["Category"])
                    .apply(lambda x: x[x > 0].mean())
                    .to_dict()
                )
                for idx in prod_k.index[zero_mask]:
                    cat = self.df_cultures.at[idx, "Category"]
                    if pd.notna(cat) and cat in cat_means and pd.notna(cat_means[cat]):
                        prod_k.at[idx] = cat_means[cat]
            # repli global
            if (prod_k <= 0).any():
                global_mean = prod_k[prod_k > 0].mean()
                prod_k[(prod_k <= 0) | prod_k.isna()] = (
                    global_mean if pd.notna(global_mean) else 1e-9
                )

        # éviter les 0 stricts
        prod_k = prod_k.replace(0, 1e-9)
        return prod_k, area_ha

    def tot_fert(self):
        """
        Total des apports d'azote vers les cultures, ventilés par grandes origines (ktN).
        Catégories retournées (si trouvées dans le graphe) :
        - "Haber-Bosch"
        - "Atmospheric deposition" = N2O + NH3 vers cultures
        - "atmospheric N2"         = "biological N2" node → cultures (fixation)
        - "Animal excretion"       = animaux (df_elevage.index) → cultures
        - "Human excretion"        = urban/rural → cultures
        - "Seeds"                  = seed/other sectors → cultures
        - "Mining"                 = soil stock/mining → cultures
        """
        crops = self._indices_for_crops()

        # --- sources simples par mots-clés
        idx_haber = self._find_label_indices("haber", "bosch")
        idx_n2o = self._find_label_indices("atmospheric", "n2o")
        idx_nh3 = self._find_label_indices("atmospheric", "nh3")
        idx_n2 = [
            i for i in self._find_label_indices("atmospheric", "n2") if i not in idx_n2o
        ]  # exclure N2O
        # graines/autres secteurs (mots-clés souples)
        idx_seeds = list(set(self._find_label_indices("seeds")))
        # stock de sol / mining
        idx_mining = list(
            set(
                self._find_label_indices("mining")
                + self._find_label_indices("soil", "stock")
            )
        )

        # --- animaux -> cultures (utilise df_elevage.index)
        animal_src = []
        if hasattr(self, "df_elevage") and self.df_elevage is not None:
            for animal in self.df_elevage.index:
                animal_src += self._find_label_indices(animal)
            animal_src = sorted(set(animal_src))

        # Populations => cultures
        pop_src = []
        if hasattr(self, "df_pop") and self.df_pop is not None:
            for pop in self.df_pop.index:
                pop_src += self._find_label_indices(pop)
            pop_src = sorted(set(pop_src))

        # --- sommes des flux vers cultures (ktN)
        out = {
            "Haber-Bosch": self._sum_from_to(idx_haber, crops),
            "Atmospheric deposition": self._sum_from_to(idx_n2o, crops)
            + self._sum_from_to(idx_nh3, crops),
            "atmospheric N2": self._sum_from_to(idx_n2, crops),
            "Animal excretion": self._sum_from_to(animal_src, crops),
            "Human excretion": self._sum_from_to(pop_src, crops),
            "Seeds": self._sum_from_to(idx_seeds, crops),
            "Mining": self._sum_from_to(idx_mining, crops),
        }
        # Nettoyage: garder uniquement > 0 (ou laisser zéro si tu préfères)
        return pd.Series(out, dtype=float)

    def rel_fert(self):
        df = self.tot_fert()
        s = df.sum()
        return (df * 100.0 / s) if s > 0 else df * 0

    def primXsec(self):
        """Part des sources 'secondaires' (%) : excrétions + atmosphérique + seeds."""
        df = self.tot_fert()
        num = (
            df.get("Human excretion", 0)
            + df.get("Animal excretion", 0)
            + df.get("atmospheric N2", 0)
            + df.get("Atmospheric deposition", 0)
            + df.get("Seeds", 0)
        )
        den = df.sum()
        return float(num * 100.0 / den) if den > 0 else 0.0

    def NUE(self):
        """NUE végétale (%) = Production végétale / apports totaux."""
        df = self.tot_fert()
        den = df.sum()
        num = float(
            self.df_cultures.get("Harvested Production (ktN)", pd.Series()).sum()
        )
        return float(num * 100.0 / den) if den > 0 else 0.0

    def env_footprint(self):
        """
        Empreinte 'land footprint' (ha) ventilée :
        - Local Food, Local Feed
        - Import Food, Import Feed
        - Import Livestock
        - Export Livestock (négatif)
        - Export Plant     (négatif)
        Retour: pandas.Series (float, ha)
        """
        import numpy as np
        import pandas as pd

        # =======================
        # 0) Préparations
        # =======================
        # Surfaces/productions par culture (index = df_cultures.index)
        prod_k, area_ha = (
            self._yield_area_map()
        )  # Series alignées sur df_cultures.index

        # Helper: surface équivalente par culture, zéro si prod_k == 0
        def surface_eq(n_k_series: pd.Series) -> pd.Series:
            n_k = (
                self._safe_series(n_k_series)
                .reindex(prod_k.index)
                .fillna(0.0)
                .astype(float)
            )
            # éviter les divisions par 0 : où prod_k<=0 → 0 ha
            ratio = np.divide(
                n_k.values,
                prod_k.values,
                out=np.zeros_like(n_k.values, dtype=float),
                where=(prod_k.values > 0),
            )
            return pd.Series(ratio, index=prod_k.index) * area_ha

        # Helper: projeter une série indexée par PRODUIT vers les CULTURES via df_prod["Origin compartment"]
        def products_to_crops(n_by_product: pd.Series) -> pd.Series:
            if (
                not hasattr(self, "df_prod")
                or self.df_prod is None
                or self.df_prod.empty
            ):
                return pd.Series(0.0, index=prod_k.index)
            if n_by_product is None or n_by_product.empty:
                return pd.Series(0.0, index=prod_k.index)

            # mapping produit -> culture d'origine
            p2c = self.df_prod[
                "Origin compartment"
            ].to_dict()  # keys = product label (index), val = crop label
            tmp = (
                pd.DataFrame({"N": n_by_product.astype(float)})
                .assign(origin=lambda d: d.index.map(p2c))
                .dropna(subset=["origin"])
            )
            by_crop = tmp.groupby("origin")["N"].sum()
            return by_crop.reindex(prod_k.index).fillna(0.0)

        # Prods auxiliaires depuis df_prod (groupby 'Origin compartment')
        if (
            hasattr(self, "df_prod")
            and self.df_prod is not None
            and not self.df_prod.empty
        ):
            by_origin = self.df_prod.groupby("Origin compartment").sum(
                numeric_only=True
            )

            food_k = (
                self._safe_series(by_origin.get("Nitrogen For Food (ktN)", 0.0))
                .reindex(prod_k.index)
                .fillna(0.0)
            )
            feed_k = (
                self._safe_series(by_origin.get("Nitrogen For Feed (ktN)", 0.0))
                .reindex(prod_k.index)
                .fillna(0.0)
            )
            exported_k = (
                self._safe_series(by_origin.get("Nitrogen Exported (ktN)", 0.0))
                .reindex(prod_k.index)
                .fillna(0.0)
            )
        else:
            food_k = pd.Series(0.0, index=prod_k.index)
            feed_k = pd.Series(0.0, index=prod_k.index)
            exported_k = pd.Series(0.0, index=prod_k.index)

        # =======================
        # 1) Locaux (Food/Feed)
        # =======================
        local_surface_food = surface_eq(food_k).sum()
        local_surface_feed = surface_eq(feed_k).sum()

        # =======================
        # 2) Imports Food/Feed (allocations_df)
        # =======================
        import_food_k = pd.Series(0.0, index=prod_k.index)
        import_feed_k = pd.Series(0.0, index=prod_k.index)

        if (
            hasattr(self, "allocations_df")
            and self.allocations_df is not None
            and not self.allocations_df.empty
        ):
            alloc = self.allocations_df.copy()
            keep_cols = [
                c
                for c in ["Type", "Product", "Allocated Nitrogen", "Consumer"]
                if c in alloc.columns
            ]
            alloc = alloc[keep_cols]

            # Sommes par PRODUIT :
            imp_food_prod = (
                alloc[alloc.get("Type", "") == "Imported Food"]
                .groupby("Product")["Allocated Nitrogen"]
                .sum()
                if "Type" in alloc.columns
                and "Product" in alloc.columns
                and "Allocated Nitrogen" in alloc.columns
                else pd.Series(dtype=float)
            )
            imp_feed_prod = (
                alloc[alloc.get("Type", "") == "Imported Feed"]
                .groupby("Product")["Allocated Nitrogen"]
                .sum()
                if "Type" in alloc.columns
                and "Product" in alloc.columns
                and "Allocated Nitrogen" in alloc.columns
                else pd.Series(dtype=float)
            )

            # → projeter PRODUITS → CULTURES
            import_food_k = products_to_crops(imp_food_prod)
            import_feed_k = products_to_crops(imp_feed_prod)

        total_food_import = surface_eq(import_food_k).sum()
        total_feed_import = surface_eq(import_feed_k).sum()

        # =======================
        # 3) Élevage import/export (via allocations 'Consumer' = animal)
        # =======================
        import_animal_ha = 0.0
        export_animal_ha = 0.0
        if (
            hasattr(self, "df_elevage")
            and self.df_elevage is not None
            and not self.df_elevage.empty
            and hasattr(self, "allocations_df")
            and self.allocations_df is not None
            and not self.allocations_df.empty
        ):
            elev = self.df_elevage.fillna(0.0)
            edible = self._safe_series(elev.get("Edible Nitrogen (ktN)", 0.0))
            net_an = self._safe_series(
                elev.get("Net animal nitrogen exports (ktN)", 0.0)
            )

            frac_import = pd.Series(0.0, index=elev.index, dtype=float)
            frac_export = pd.Series(0.0, index=elev.index, dtype=float)
            nonzero = edible > 0
            frac_import[nonzero] = (-net_an[nonzero] / edible[nonzero]).clip(
                lower=0.0, upper=1.0
            )
            frac_export[nonzero] = (net_an[nonzero] / edible[nonzero]).clip(
                lower=0.0, upper=1.0
            )

            alloc = self.allocations_df
            if (
                "Consumer" in alloc.columns
                and "Product" in alloc.columns
                and "Allocated Nitrogen" in alloc.columns
            ):
                for animal in elev.index:
                    sub = alloc[alloc["Consumer"] == animal]
                    if sub.empty:
                        continue
                    # N alloué par PRODUIT pour cet animal → projeter vers CULTURES
                    nk_prod = sub.groupby("Product")["Allocated Nitrogen"].sum()
                    nk_crop = products_to_crops(nk_prod)  # <- la clé du bug

                    import_animal_ha += surface_eq(
                        nk_crop * float(frac_import.get(animal, 0.0))
                    ).sum()
                    export_animal_ha += surface_eq(
                        nk_crop * float(frac_export.get(animal, 0.0))
                    ).sum()

        # =======================
        # 4) Export végétal (plantes)
        # =======================
        export_surface = surface_eq(exported_k).sum()

        # =======================
        # 5) Résultat (ha)
        # =======================
        return pd.Series(
            {
                "Local Food": float(local_surface_food),
                "Local Feed": float(local_surface_feed),
                "Import Food": float(total_food_import),
                "Import Feed": float(total_feed_import),
                "Import Livestock": float(import_animal_ha),
                "Export Livestock": -float(export_animal_ha),  # exports en négatif
                "Export Plant": -float(export_surface),  # exports en négatif
            }
        )

    def net_footprint(self):
        """
        Computes the net nitrogen land footprint of the area (in Mha).

        Aggregates all imports and exports to yield a net balance of nitrogen-dependent land use.

        :return: Net land footprint (in million hectares).
        :rtype: float
        """
        df = self.env_footprint()
        df_total_import = df.loc[
            ["Import Food", "Import Feed", "Import Livestock"]
        ].sum(axis=0)
        df_total_export = df.loc[["Export Plant", "Export Livestock"]].sum(axis=0)
        net_import_export = df_total_import + df_total_export
        return np.round(net_import_export / 1e6, 2)

    def check_balance(self):
        """
        Vérifie la balance des flux (sommes lignes et colonnes) de la matrice de transition M.
        Optimisé pour un affichage rapide en utilisant une seule instruction print finale.
        """
        M = self.get_transition_matrix()

        # Prépare les calculs en une seule fois (vectorisation)
        # Calcule la somme de chaque ligne (flux sortant)
        row_sums = M.sum(axis=1)
        # Calcule la somme de chaque colonne (flux entrant)
        col_sums = M.sum(axis=0)

        output_lines = []

        for i in range(len(M)):
            label = self.index_to_label[i]

            # Ajout des informations à la liste
            output_lines.append(label)
            output_lines.append(f"Flux sortant (Somme Ligne): {row_sums[i]:.6f}")
            output_lines.append(f"Flux entrant (Somme Colonne): {col_sums[i]:.6f}")
            output_lines.append("===")

        # Effectue un seul appel d'impression avec toutes les lignes jointes par un saut de ligne
        print("\n".join(output_lines))


# A reprendre
# def general_reg(self, cons):
#     """
#     Give the diet for a consumer.
#     """
#     # Filtrer les données pour "urban" et effectuer les transformations
#     df_filtered = self.allocations_df.loc[self.allocations_df["Consumer"] == cons, ["Product", "Allocated Nitrogen"]]

#     df_filtered = df_filtered.groupby("Product", as_index=False)["Allocated Nitrogen"].sum()

#     # Créer la colonne "Product" en joignant "Product" avec le premier mot de "Type"
#     # df_filtered["Product"] = df_filtered["Product"] + " " + df_filtered["Type"].str.split().str[0]

#     # Utiliser self.prod pour l'indexation, mettre 0 si l'index est absent dans df_filtered
#     # Nous devons parcourir self.prod et vérifier la présence dans df_filtered['Product']
#     df_filtered.set_index("Product", inplace=True)

#     # Ajouter les produits absents de df_filtered avec une valeur de 0
#     missing_products = set(self.df_prod) - set(df_filtered.index)
#     for missing in missing_products:
#         df_filtered.loc[missing] = 0.0  # Si nécessaire, ajustez les valeurs à insérer pour "Allocated Nitrogen" et "Type"

#     # Réorganiser l'ordre en fonction de self.prod
#     df_filtered = df_filtered.reindex(self.df_prod, fill_value=0)

#     return df_filtered["Allocated Nitrogen"]
