import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LogNorm
from pulp import LpContinuous, LpMinimize, LpProblem, LpVariable, lpSum
import warnings

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
            "methanizer",
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
        df_in = df_in[(df_in["Area"] == area) & (df_in["Year"] == year)]

        # Nettoyage numérique de 'value'
        df_in["value"] = self._to_num(df_in["value"])

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
            if overwrite:
                # On met la colonne telle quelle (même si NaN)
                merged_df[col] = series_added
            else:
                # On n'écrase que les positions où wide a une valeur non-nulle (non-NaN)
                mask_has_value = series_added.notna()
                if col not in merged_df.columns:
                    # créer la colonne si elle n'existe pas
                    merged_df[col] = np.nan
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

    def generate_df_prod(self, area, year, carbon=False):
        if carbon is False:
            categories_needed = (
                "Production (kton)",
                "Nitrogen Content (%)",
                "Origin compartment",
                "Type",
                "Sub Type",
                "Waste (%)",
                "Other uses (%)",
                "Methanization power (MWh/tMB)",
            )
        else:
            categories_needed = (
                "Production (kton)",
                "Sub Type",
                "Origin compartment",
                "Nitrogen Content (%)",
                "Carbon Content (%)",
            )

        df_prod = self.get_columns(
            area, year, self.init_df_prod, categories_needed=categories_needed
        )

        df_prod = df_prod[list(categories_needed)].copy()

        if carbon:
            return df_prod.fillna(0)

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
        return df_prod

    def generate_df_cultures(self, area, year, carbon=False):
        if carbon is False:
            categories_needed = (
                "Area (ha)",
                "Spreading Rate (%)",
                "Seed input (kt seeds/kt Ymax)",
                "Fertilization Need (kgN/qtl)",
                "Surface Fertilization Need (kgN/ha)",
                "Harvest Index",
                "Main Production",
                "Category",
                "BNF alpha",
                "BNF beta",
                "BGN",
            )
        else:
            categories_needed = (
                "Main Production",
                "Harvest Index",
                "Area (ha)",
                "Seed input (kt seeds/kt Ymax)",
                "Carbon Mechanisation Intensity (ktC/ha)",
                "Residue Humification Coefficient (%)",
                "Root Humification Coefficient (%)",
                "Surface Root Production (kgC/ha)",
            )
        df_cultures = self.get_columns(
            area, year, self.init_df_cultures, categories_needed=categories_needed
        )
        df_cultures = df_cultures[list(categories_needed)].copy()

        df_prod = self.generate_df_prod(area, year)
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
            df_cultures["Seed input (kt seeds/kt Ymax)"]
            * df_cultures["Main Nitrogen Production (ktN)"]
        )
        if carbon:
            return df_cultures.fillna(0)

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
        )

        df_cultures = df_cultures.fillna(0)
        self.df_cultures = df_cultures
        return df_cultures

    def generate_df_elevage(self, area, year, carbon=False):
        if carbon is False:
            categories_needed = (
                "Excreted indoor (%)",
                "Excreted indoor as manure (%)",
                "Excretion / LU (kgN)",
                "LU",
            )
        else:
            categories_needed = (
                "Excretion / LU (kgN)",
                "LU",
                "C-CH4 enteric/LU (kgC)",
                "Infrastructure CO2 emissions/LU (kgC)",
            )
        df_elevage = self.get_columns(
            area, year, self.init_df_elevage, categories_needed=categories_needed
        )
        df_elevage = df_elevage[list(categories_needed)].copy()
        if carbon:
            return df_elevage.fillna(0)

        df_elevage["Excreted indoor as slurry (%)"] = (
            100 - df_elevage["Excreted indoor as manure (%)"]
        )
        df_elevage["Excreted on grassland (%)"] = (
            100 - df_elevage["Excreted indoor (%)"]
        )

        df_prod = self.generate_df_prod(area, year)
        df_elevage["Edible Nitrogen (ktN)"] = (
            df_prod.loc[df_prod["Sub Type"] == "edible meat", :].set_index(
                "Origin compartment"
            )["Nitrogen Content (%)"]
            * df_prod.loc[df_prod["Sub Type"] == "edible meat", :].set_index(
                "Origin compartment"
            )["Production (kton)"]
            / 100
        )
        df_elevage["Non Edible Nitrogen (ktN)"] = (
            df_prod.loc[df_prod["Sub Type"] == "non edible meat", :].set_index(
                "Origin compartment"
            )["Nitrogen Content (%)"]
            * df_prod.loc[df_prod["Sub Type"] == "non edible meat", :].set_index(
                "Origin compartment"
            )["Production (kton)"]
            / 100
        )
        df_elevage["Dairy Nitrogen (ktN)"] = (
            df_prod.loc[df_prod["Sub Type"] == "dairy", :].set_index(
                "Origin compartment"
            )["Nitrogen Content (%)"]
            * df_prod.loc[df_prod["Sub Type"] == "dairy", :].set_index(
                "Origin compartment"
            )["Production (kton)"]
            / 100
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
        return df_elevage

    def generate_df_excr(self, area, year, carbon=False):
        _ = self.generate_df_elevage(area, year, False)
        if carbon is False:
            categories_needed = (
                "N-NH3 EM (%)",
                "N-N2 EM (%)",
                "N-N2O EM (%)",
                "Type",
                "Origin compartment",
                "Methanization power (MWh/tMB)",
            )
        else:
            categories_needed = (
                "N-NH3 EM (%)",
                "N-N2 EM (%)",
                "N-N2O EM (%)",
                "Type",
                "Origin compartment",
                "C/N",
                "CH4 EM (%)",
                "Humification coefficient (%)",
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
        return df_excr

    def generate_df_pop(self, area, year, carbon=False):
        if carbon is False:
            categories_needed = (
                "Inhabitants",
                "N-NH3 EM excretion (%)",
                "N-N2 EM excretion (%)",
                "N-N2O EM excretion (%)",
                "Total ingestion per capita (kgN)",
                "Fischery ingestion per capita (kgN)",
                "Excretion recycling (%)",
            )
        else:
            categories_needed = (
                "Inhabitants",
                "N-NH3 EM excretion (%)",
                "N-N2 EM excretion (%)",
                "N-N2O EM excretion (%)",
                "Total ingestion per capita (kgN)",
                "Fischery ingestion per capita (kgN)",
                "Excretion recycling (%)",
                "C/N",
                "CH4 EM (%)",
                "Humification coefficient (%)",
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
            df_pop["Inhabitants"] * df_pop["Fischery ingestion per capita (kgN)"] / 1e6
        )

        df_pop["Excretion after volatilization (ktN)"] = (
            df_pop["Ingestion (ktN)"] * df_pop["Excretion recycling (%)"] / 100
        )

        df_pop = df_pop.fillna(0)
        self.df_pop = df_pop
        return df_pop

    def get_global_metrics(self, area, year, carbon=False):
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

        # The list of required items
        if carbon is False:
            required_items = [
                "Total Synthetic Fertilizer Use on crops (ktN)",
                "Total Synthetic Fertilizer Use on grasslands (ktN)",
                "Atmospheric deposition coef (kgN/ha)",
                "coefficient N-NH3 volatilization synthetic fertilization (%)",
                "coefficient N-N2O emission synthetic fertilization (%)",
                "Weight diet",
                "Weight distribution",
                "Weight fair local split",
                "Enforce animal share",
                "Methanizer Energy Production (GWh)",
                "Weight methanizer production",
                "Weight methanizer inputs",
                "Green waste methanization power (MWh/ktN)",
            ]
        else:
            required_items = ["Total Haber-Bosch methan input (kgC/kgN)"]

        # Weight distribution is given in option and can be computed from other weights
        if "Weight distribution" not in global_df.index:
            weight_diet = global_df.loc["Weight diet", "value"]
            global_df.loc["Weight distribution", "value"] = weight_diet / 10

        # Weight distribution is given in option and can be computed from other weights
        if "Weight fair local split" not in global_df.index:
            weight_diet = global_df.loc["Weight diet", "value"]
            global_df.loc["Weight fair local split", "value"] = weight_diet / 20

        # Check for the presence of each required item
        missing_items = [item for item in required_items if item not in global_df.index]

        if missing_items:
            raise KeyError(
                f"❌ The following required global metrics were not found for year {year} and area {area}: "
                f"{', '.join(missing_items)}. Please check the input data."
            )

        return global_df

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
        input_df = self.df_data.get("Input data", None)
        if input_df is None:
            raise ValueError("No 'Input data' sheet found in self.df_data")

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
        input_df2 = input_df.copy()

        # item and value
        # item = consumer (e.g. 'bovines'), value = diet id
        col_item = "item"
        col_value = "value"

        # filter relevant rows
        mask = (
            (input_df2["Area"] == area)
            & (input_df2["Year"] == year)
            & (input_df2["category"] == "Diet")
        )
        mapping_rows = input_df2.loc[mask, [col_item, col_value]].copy()
        if mapping_rows.empty:
            raise ValueError(
                f"No Diet mapping found in Input data for area={area}, year={year}"
            )

        # build dict consumer -> diet_id (string)
        consumer_to_diet = {}
        for _, r in mapping_rows.iterrows():
            consumer = str(r[col_item]).strip()
            diet_id_val = r[col_value]
            if pd.isna(diet_id_val):
                raise ValueError(
                    f"Empty diet id for consumer '{consumer}' in Input data for {area}/{year}"
                )
            consumer_to_diet[consumer] = str(diet_id_val).strip()

        # --- 4) Vérifier que chaque index de df_elevage et df_pop a un mapping ---
        # on normalise casse pour comparaison facile : on compare en minuscules des deux côtés
        consumers_expected = set(
            [
                c.lower()
                for c in list(self.init_df_elevage.index) + list(self.init_df_pop.index)
            ]
            + ["methanizer"]
        )
        consumers_found = set([k.lower() for k in consumer_to_diet.keys()])

        missing_consumers = sorted(list(consumers_expected - consumers_found))
        if missing_consumers:
            raise ValueError(
                "Missing diet mapping for the following consumers (indexes in df_elevage, df_pop or 'mathanizer') for "
                f"{area}/{year}:\n" + "\n".join(missing_consumers)
            )

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

    def generate_flux(self, source, target):
        """Generates and updates the transition matrix by calculating the flux coefficients between the source and target sectors.

        Args:
            source (dict): A dictionary representing the source sector, where keys are sector labels and values are the corresponding flux values.
            target (dict): A dictionary representing the target sector, where keys are sector labels and values are the corresponding flux values.

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
                        self.adjacency_matrix[source_index, target_index] = coefficient
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

    def __init__(self, data, area, year):
        """Initializes the NitrogenFlowModel class with the necessary data and model parameters.

        Args:
            data (DataLoader): An instance of the DataLoader class to load and preprocess the data.
            year (str): The year for which to compute the nitrogen flow model.
            area (str): The area for which to compute the nitrogen flow model.
        """
        self.year = year
        self.area = area

        self.data_loader = data
        self.labels = data.labels
        self.label_to_index = data.label_to_index
        self.index_to_label = data.index_to_label

        self.flux_generator = FluxGenerator(self.labels)

        self.df_cultures = data.generate_df_cultures(self.area, self.year)
        self.df_elevage = data.generate_df_elevage(self.area, self.year)
        self.df_excr = data.generate_df_excr(self.area, self.year)
        self.df_prod = data.generate_df_prod(self.area, self.year)
        self.df_pop = data.generate_df_pop(self.area, self.year)
        self.df_global = data.get_global_metrics(self.area, self.year)
        self.diets = data.load_diets_for_area_year(self.area, self.year)

        self.adjacency_matrix = self.flux_generator.adjacency_matrix

        self.compute_fluxes()

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
        # Extraire les variables nécessaires
        df_cultures = self.df_cultures
        df_elevage = self.df_elevage
        df_excr = self.df_excr
        df_prod = self.df_prod
        df_pop = self.df_pop
        df_global = self.df_global
        diets = self.diets
        label_to_index = self.label_to_index
        flux_generator = self.flux_generator

        # Calcul de la production totale récoltée d'azote par culture
        # Étape 1 : Calculer la somme de la production d'azote par "Origin compartment"
        nitrogen_production_sum = (
            df_prod.loc[df_prod["Type"] == "plant"]
            .groupby("Origin compartment")["Nitrogen Production (ktN)"]
            .sum()
        )

        # Étape 2 : Mettre à jour la colonne "Nitrogen Production (ktN)" dans df_cultures
        # Pandas aligne automatiquement les index de la série nitrogen_production_sum
        # avec l'index de df_cultures.
        df_cultures["Total Nitrogen Production (ktN)"] = nitrogen_production_sum

        # Flux des cultures vers les productions végétales :
        for index, row in df_prod.iterrows():
            # Création du dictionnaire target
            source = {row["Origin compartment"]: 1}

            # Création du dictionnaire source
            target = {index: row["Nitrogen Production (ktN)"]}
            flux_generator.generate_flux(source, target)

        # Flux des produits vers Waste et other sectors:
        for index, row in df_prod.iterrows():
            source = {index: row["Nitrogen Wasted (ktN)"]}

            target = {"waste": 1}
            flux_generator.generate_flux(source, target)

            source = {index: row["Nitrogen for Other uses (ktN)"]}
            target = {"other sectors": 1}
            flux_generator.generate_flux(source, target)

        # Flux des animaux vers les compartiments d'excretion
        for index, row in df_excr.iterrows():
            # Création du dictionnaire target
            source = {row["Origin compartment"]: 1}

            # Création du dictionnaire source
            target = {index: row["Excretion (ktN)"]}
            flux_generator.generate_flux(source, target)

        # Seeds input
        target = df_cultures["Seeds Input (ktN)"].to_dict()
        source = {"seeds": 1}
        flux_generator.generate_flux(source, target)

        ## Dépôt atmosphérique
        source = {"atmospheric N2O": 0.1, "atmospheric NH3": 0.9}
        target = (
            df_global.loc["Atmospheric deposition coef (kgN/ha)"].item()
            * df_cultures["Area (ha)"]
            / 1e6
        ).to_dict()  # Dépôt proportionnel aux surface
        flux_generator.generate_flux(source, target)

        ## Consommation de produits de la mer

        source = {"fishery products": 1}
        target = df_pop["Fishery Ingestion (ktN)"].to_dict()
        flux_generator.generate_flux(source, target)

        ## Épandage de boue sur les champs

        mask = ~df_cultures["Category"].isin(["natural meadows", "temporary meadows"])
        Norm = (
            df_cultures[mask]["Area (ha)"]
            * df_cultures[mask]["Spreading Rate (%)"]
            / 100
        ).sum()
        # Création du dictionnaire target
        target_epandage = {
            culture: row["Area (ha)"] * row["Spreading Rate (%)"] / 100 / Norm
            for culture, row in df_cultures.iterrows()
            if row["Category"] not in ["natural meadows", "temporary meadows"]
        }

        source_boue = (
            df_pop["Ingestion (ktN)"] * df_pop["Excretion recycling (%)"] / 100
        ).to_dict()

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

        total_surface_grasslands = df_cultures.loc[
            df_cultures["Category"].isin(["natural meadows", "temporary meadows"]),
            "Area (ha)",
        ].sum()

        # Création du dictionnaire target
        target = (
            df_cultures.loc[
                df_cultures["Category"].isin(["natural meadows", "temporary meadows"]),
                "Area (ha)",
            ]
            / total_surface_grasslands
        ).to_dict()

        # Source
        source = df_excr.loc[
            df_excr["Type"] == "grasslands excretion",
            "Excretion after volatilization (ktN)",
        ].to_dict()

        flux_generator.generate_flux(source, target)

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

        flux_generator.generate_flux(source, target_epandage)
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

        ## Fixation symbiotique
        # Calcul de l'azote épendu par hectare
        def calculer_azote_ependu(culture):
            adj_matrix_df = pd.DataFrame(
                self.adjacency_matrix, index=self.labels, columns=self.labels
            )
            return adj_matrix_df.loc[:, culture].sum().item()

        target_fixation = (
            (
                df_cultures["BNF alpha"]
                * df_cultures["Yield (kgN/ha)"]
                / df_cultures["Harvest Index"]
                + df_cultures["BNF beta"]
            )
            * df_cultures["BGN"]
            * df_cultures["Area (ha)"]
            / 1e6
        ).to_dict()
        source_fixation = {"atmospheric N2": 1}
        flux_generator.generate_flux(source_fixation, target_fixation)
        df_cultures["Symbiotic Fixation (ktN)"] = df_cultures.index.map(
            target_fixation
        ).fillna(0)

        ## Azote synthétique
        # Calcul de l'azote à épendre (bilan azoté)

        df_cultures["Total Non Synthetic Fertilizer Use (ktN)"] = df_cultures.index.map(
            calculer_azote_ependu
        )

        # Séparer les données en prairies et champs
        # On commence par un bilan sur les prairies pour avoir un bilan complet des prairies et en déduire un bilan azoté non synthétique complet sur les culturess
        df_prairies = df_cultures[
            df_cultures["Category"].isin(["natural meadows", "temporary meadows"])
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

        df_prairies["Raw Surface Synthetic Fertilizer Use (kgN/ha)"] = (
            df_prairies.apply(
                lambda row: row["Surface Fertilization Need (kgN/ha)"]
                - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"]
                if row["Area (ha)"] > 0
                else row["Surface Fertilization Need (kgN/ha)"]
                - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"],
                axis=1,
            )
        )
        df_prairies["Raw Surface Synthetic Fertilizer Use (kgN/ha)"] = df_prairies[
            "Raw Surface Synthetic Fertilizer Use (kgN/ha)"
        ].apply(lambda x: max(x, 0))

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
                df_prairies.loc[:, "Adjusted Total Synthetic Fertilizer Use (ktN)"] = 0
            warnings.warn("No Synthetic fertilizer need for grasslands.")

        # source = {"Haber Bosch": 1}
        # target = df_prairies["Adjusted Total Synthetic Fertilizer Use (ktN)"].to_dict()
        # flux_generator.generate_flux(source, target)

        # Maintenant, on fait le bilan complet des légumineuses (de champs et de prairies)

        df_leg = pd.concat(
            [df_prairies, df_cultures[df_cultures["Category"] == "leguminous"]]
        )

        df_leg["Total Non Synthetic Fertilizer Use (ktN)"] = df_leg.index.map(
            calculer_azote_ependu
        )

        df_leg["Organic Fertilization (ktN)"] = (
            df_leg["Seeds Input (ktN)"] + df_leg["Symbiotic Fixation (ktN)"]
        )
        df_leg["Mineral Fertilization (ktN)"] = (
            df_leg["Adjusted Total Synthetic Fertilizer Use (ktN)"]
            + df_leg["Total Non Synthetic Fertilizer Use (ktN)"]
            - df_leg["Organic Fertilization (ktN)"]
        )

        df_leg["Residues Production (ktN)"] = (
            df_leg["Total Nitrogen Production (ktN)"] / df_leg["Harvest Index"]
            - df_leg["Total Nitrogen Production (ktN)"]
        )

        df_leg["Roots Production (ktN)"] = (
            df_leg["Total Nitrogen Production (ktN)"]
            / df_leg["Harvest Index"]
            * (df_leg["BGN"] - 1)
        )

        # 1. Calcul de la somme des productions pour toutes les lignes (vectoriel)
        production_sum = (
            df_leg["Residues Production (ktN)"]
            + df_leg["Roots Production (ktN)"]
            + df_leg["Total Nitrogen Production (ktN)"]
        )

        # 2. Définition de la condition de surplus organique (vectoriel)
        organic_surplus_condition = (
            df_leg["Organic Fertilization (ktN)"] > production_sum
        )

        # 3. Calcul vectoriel des colonnes avec numpy.where (le plus condensé et efficace)
        df_leg["Surplus Organic Fertilisation (ktN)"] = np.where(
            organic_surplus_condition,
            df_leg["Organic Fertilization (ktN)"]
            - production_sum,  # VRAI (Condition 1)
            0,  # FAUX (Condition 2)
        )

        df_leg["Surplus Mineral Fertilization (ktN)"] = np.where(
            organic_surplus_condition,
            df_leg["Mineral Fertilization (ktN)"],  # VRAI (Condition 1)
            df_leg["Mineral Fertilization (ktN)"]
            - (
                production_sum - df_leg["Organic Fertilization (ktN)"]
            ),  # FAUX (Condition 2)
        )

        # On répartit cet azote dans leur compartiment de destination (sauf produit pour lequel c'est déjà fait)
        # Héritage seulement pour prairies temporaires et légumineuses (pas prairies naturelles)
        mask = df_leg["Category"].isin(["temporary meadows", "leguminous"])
        df_leg.loc[mask, "Nitrogen for Heritage (ktN)"] = (
            df_leg.loc[mask, "Residues Production (ktN)"]
            + df_leg.loc[mask, "Surplus Organic Fertilisation (ktN)"]
        )

        total_surplus_azote = df_leg.loc[mask, "Nitrogen for Heritage (ktN)"].sum()
        total_surface_cereales = df_cultures.loc[
            (df_cultures["Category"] == "cereals (excluding rice)"),
            "Area (ha)",
        ].sum()
        df_cultures["Leguminous Heritage (ktN)"] = 0.0
        df_cultures.loc[
            (df_cultures["Category"] == "cereals (excluding rice)"),
            "Leguminous Heritage (ktN)",
        ] = (
            df_cultures.loc[
                (df_cultures["Category"] == "cereals (excluding rice)"),
                "Area (ha)",
            ]
            / total_surface_cereales
            * total_surplus_azote
        )

        # Génération des flux pour l'héritage des légumineuses et prairies temporaires
        source_leg = (
            df_leg.loc[df_leg["Nitrogen for Heritage (ktN)"] > 0][
                "Nitrogen for Heritage (ktN)"
            ]
            / df_leg["Nitrogen for Heritage (ktN)"].sum()
        ).to_dict()
        target_leg = df_cultures["Leguminous Heritage (ktN)"].to_dict()
        flux_generator.generate_flux(source_leg, target_leg)

        # Stock

        source = df_leg["Roots Production (ktN)"].to_dict()
        target = {"soil stock": 1}
        flux_generator.generate_flux(source, target)

        # On ajoute organic fertilization surplus et residues vers stock pour les prairies permanentes
        source = (
            df_leg.loc[~mask, "Residues Production (ktN)"]
            + df_leg.loc[~mask, "Surplus Organic Fertilisation (ktN)"]
        ).to_dict()
        target = {"soil stock": 1}
        flux_generator.generate_flux(source, target)

        # Et enfin part de l'azote lessivé

        source = df_leg["Surplus Mineral Fertilization (ktN)"].to_dict()
        target = {"hydro-system": 0.9925, "atmospheric N2O": 0.0075}
        flux_generator.generate_flux(source, target)

        # Bouclage du bilan des Cultures n'étant pas des prairies ou des légumineuses
        df_champs = df_cultures[
            ~df_cultures["Category"].isin(
                ["natural meadows", "temporary meadows", "leguminous"]
            )
        ].copy()

        df_champs["Total Non Synthetic Fertilizer Use (ktN)"] = df_champs.index.map(
            calculer_azote_ependu
        )

        df_champs["Surface Non Synthetic Fertilizer Use (kgN/ha)"] = df_champs.apply(
            lambda row: row["Total Non Synthetic Fertilizer Use (ktN)"]
            / row["Area (ha)"]
            * 10**6
            if row["Area (ha)"] > 0
            and row["Total Non Synthetic Fertilizer Use (ktN)"] > 0
            else 0,
            axis=1,
        )

        df_champs["Raw Surface Synthetic Fertilizer Use (kgN/ha)"] = df_champs.apply(
            lambda row: row["Surface Fertilization Need (kgN/ha)"]
            - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"]
            if row["Area (ha)"] > 0
            else row["Surface Fertilization Need (kgN/ha)"]
            - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"],
            axis=1,
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
        else:
            if len(df_champs) > 0:
                df_champs.loc[:, "Adjusted Total Synthetic Fertilizer Use (ktN)"] = 0
            warnings.warn("No Synthetic fertilizer need for grasslands.")

        self.gamma = moyenne_reel_champs / moyenne_ponderee_champs

        # Mise à jour de df_cultures
        df_calc = pd.concat([df_leg, df_champs], axis=0, sort=False)
        df_cultures = df_calc.combine_first(df_cultures).fillna(0)

        df_cultures["Adjusted Surface Synthetic Fertilizer Use (kgN/ha)"] = 0.0

        mask = df_cultures["Area (ha)"] != 0
        df_cultures.loc[mask, "Adjusted Surface Synthetic Fertilizer Use (kgN/ha)"] = (
            df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"]
            / df_cultures["Area (ha)"]
            * 1e6
        )

        ## Azote synthétique volatilisé par les terres
        # Est ce qu'il n'y a que l'azote synthétique qui est volatilisé ?
        coef_volat_NH3 = (
            df_global.loc[
                "coefficient N-NH3 volatilization synthetic fertilization (%)"
            ].item()
            / 100
        )
        coef_volat_N2O = (
            df_global.loc[
                "coefficient N-N2O emission synthetic fertilization (%)"
            ].item()
            / 100
        )

        # 1 % des emissions de NH3 du aux fert. synth sont volatilisées sous forme de N2O
        df_cultures["Volatilized Nitrogen N-NH3 (ktN)"] = (
            df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"]
            * 0.99
            * coef_volat_NH3
        )
        df_cultures["Volatilized Nitrogen N-N2O (ktN)"] = df_cultures[
            "Adjusted Total Synthetic Fertilizer Use (ktN)"
        ] * (coef_volat_N2O + 0.01 * coef_volat_NH3)

        # df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"] = df_cultures[
        #     "Adjusted Total Synthetic Fertilizer Use (ktN)"
        # ] * (1 - coef_volat_NH3 - coef_volat_N2O)
        # La quantité d'azote réellement épendue est donc un peu plus faible car une partie est volatilisée

        source = {"Haber-Bosch": 1}
        target = df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"].to_dict()

        flux_generator.generate_flux(source, target)

        source = df_cultures["Volatilized Nitrogen N-NH3 (ktN)"].to_dict()
        target = {"atmospheric NH3": 1}

        flux_generator.generate_flux(source, target)

        source = df_cultures["Volatilized Nitrogen N-N2O (ktN)"].to_dict()
        target = {"atmospheric N2O": 1}

        flux_generator.generate_flux(source, target)

        # A cela on ajoute les emissions indirectes de N2O lors de la fabrication des engrais
        epend_tot_synt = df_cultures[
            "Adjusted Total Synthetic Fertilizer Use (ktN)"
        ].sum()
        coef_emis_N_N2O = (
            df_global.loc[
                "coefficient N-N2O indirect emission synthetic fertilization (%)"
            ].item()
            / 100
        )
        target = {"atmospheric N2O": 1}
        source = {"Haber-Bosch": epend_tot_synt * coef_emis_N_N2O}

        flux_generator.generate_flux(source, target)

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

        # Poids pour équilibrer la distribution des cultures dans les categories
        poids_penalite_culture = df_global.loc[
            df_global.index == "Weight distribution"
        ]["value"].item()

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
            prodN = df_prod.loc[p, "Nitrogen Production (ktN)"]
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
        def prod_scale(p):
            return max(1.0, float(df_prod.loc[p, "Nitrogen Production (ktN)"]))

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

        # (Option MILP) binaire no-swap import/surplus par produit
        use_milp_no_swap = True  # mets False pour rester LP 100%
        if use_milp_no_swap:
            y_vars = LpVariable.dicts(
                "y", list(df_prod.index), lowBound=0, upBound=1, cat="Binary"
            )

        # Big-M par produit (borne supérieure serrée = plus rapide)
        M = {}
        for p in df_prod.index:
            prodN = float(df_prod.loc[p, "Available Nitrogen Production (ktN)"])
            # borne import plausible = somme des besoins des consommateurs qui ont p dans leur diète
            cons_with_p = [c for (prod, c) in valid_pairs if prod == p]
            need_with_p = (
                sum(float(df_cons.loc[c, "Ingestion (ktN)"]) for c in cons_with_p)
                if cons_with_p
                else 0.0
            )
            M[p] = max(prodN, need_with_p) + 1e-6  # marge min

        # Contraintes sur la méthanisation
        METH_DIET_NAME = "Methanizer"
        TARGET_GWh = df_global.loc["Methanizer Energy Production (GWh)", "value"]
        W_METH_ENERGY = df_global.loc["Weight methanizer production", "value"]
        W_METH_DIET = df_global.loc["Weight methanizer inputs", "value"]
        WASTE_PWR_MWh_per_ktN = df_global.loc[
            "Green waste methanization power (MWh/ktN)", "value"
        ]

        meth_diet_df = diets[diets["Consumer"] == METH_DIET_NAME].copy()

        meth_prod_items = set()
        meth_excr_items = set()
        meth_waste_items = set(["waste"])

        for _, row in meth_diet_df.iterrows():
            for it in row["Products"]:
                if it in df_prod.index:
                    meth_prod_items.add(it)
                elif "df_excr" in globals() and it in df_excr.index:
                    meth_excr_items.add(it)
                else:
                    # tout item non trouvé dans df_prod/df_excr est interprété comme "waste"
                    meth_waste_items.add(it)

        x_meth_prod = LpVariable.dicts(
            "x_meth_prod", list(meth_prod_items), lowBound=0, cat=LpContinuous
        )
        x_meth_excr = LpVariable.dicts(
            "x_meth_excr", list(meth_excr_items), lowBound=0, cat=LpContinuous
        )
        N_waste_meth = LpVariable("N_waste_meth", lowBound=0, cat=LpContinuous)

        # Production d'énergie
        E_MWh_products = (
            lpSum(
                x_meth_prod[p]
                * (
                    float(df_prod.loc[p, "Methanization power (MWh/tMB)"])
                    * 1000
                    / (float(df_prod.loc[p, "Nitrogen Content (%)"]) / 100)
                )
                for p in meth_prod_items
            )
            if len(meth_prod_items)
            else 0
        )

        E_MWh_excreta = (
            lpSum(
                x_meth_excr[e]
                * (
                    float(df_excr.loc[e, "Methanization power (MWh/tMB)"])
                    * 1000
                    / (float(df_excr.loc[e, "Nitrogen Content (%)"]) / 100)
                )
                for e in meth_excr_items
            )
            if len(meth_excr_items)
            else 0
        )

        E_MWh_waste = N_waste_meth * WASTE_PWR_MWh_per_ktN
        E_GWh_total = (E_MWh_products + E_MWh_excreta + E_MWh_waste) / 1000.0

        meth_energy_dev = LpVariable("meth_energy_dev", lowBound=0, cat=LpContinuous)
        prob += (meth_energy_dev >= E_GWh_total - TARGET_GWh, "Meth_energy_dev_pos")
        prob += (meth_energy_dev >= TARGET_GWh - E_GWh_total, "Meth_energy_dev_neg")

        # -- Diète du méthaniseur (parts d’azote par groupe)
        pairs_meth = []
        for _, row in meth_diet_df.iterrows():
            prop = float(row["Proportion"])
            prop = prop / 100.0 if prop > 1.0 else prop
            pairs_meth.append((METH_DIET_NAME, prop, tuple(row["Products"])))

        delta_meth = LpVariable.dicts(
            "delta_meth",
            [(METH_DIET_NAME, tuple(pL)) for _, _, pL in pairs_meth],
            lowBound=0,
            cat=LpContinuous,
        )

        N_to_meth_total = (
            (lpSum(x_meth_prod.values()) if len(meth_prod_items) else 0)
            + (lpSum(x_meth_excr.values()) if len(meth_excr_items) else 0)
            + (N_waste_meth if isinstance(N_waste_meth, LpVariable) else 0)
        )

        for _, prop, prod_list in pairs_meth:
            N_group = 0
            for it in prod_list:
                if it in x_meth_prod:
                    N_group += x_meth_prod[it]
                elif it in x_meth_excr:
                    N_group += x_meth_excr[it]
                elif isinstance(N_waste_meth, LpVariable) and it in meth_waste_items:
                    N_group += N_waste_meth  # agrégat waste

            dv = delta_meth[(METH_DIET_NAME, tuple(prod_list))]

            # ----- LINEARISATION (pas de division) -----
            # N_group - prop * N_to_meth_total <= dv
            prob += (
                N_group - prop * N_to_meth_total <= dv,
                f"METH_Diet_plus_{hash(tuple(prod_list))}",
            )

            # prop * N_to_meth_total - N_group <= dv
            prob += (
                prop * N_to_meth_total - N_group <= dv,
                f"METH_Diet_moins_{hash(tuple(prod_list))}",
            )

        # Fonction objectif
        objective = (
            (poids_penalite_deviation / max(1, len(pairs))) * lpSum(delta_vars.values())
            + (poids_penalite_culture / max(1, len(df_prod)))
            * lpSum(penalite_culture_vars.values())
            # + poids_import_brut * lpSum(I_vars.values())
            + w_fair
            * (fair_term / max(1, len(df_prod)))  # moyenne par produit (optionnel)
            + W_METH_ENERGY * meth_energy_dev
            + (W_METH_DIET / max(1, len(pairs_meth))) * lpSum(delta_meth.values())
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

        # Cette contrainte assure que la somme de l'azote alloué de chaque culture aux différents types de consommateurs + Surplus à exporter + methaniseur == l'azote disponible pour cette culture.
        for p in df_prod.index:
            consumers_with_p = [c for (prod, c) in valid_pairs if prod == p]
            prodN = float(df_prod.loc[p, "Available Nitrogen Production (ktN)"])
            x_meth_term = x_meth_prod[p] if p in x_meth_prod else 0
            prob += (
                lpSum(x_vars[(p, c)] for c in consumers_with_p)
                + x_meth_term
                + U_vars[p]
                == prodN,
                f"Bilan_{p}",
            )

        # Contrainte dure sur l'usage des excretions
        for e in meth_excr_items:
            avail_excr_soil = float(
                df_excr.loc[e, "Excretion after volatilization (ktN)"]
            )
            prob += (x_meth_excr[e] <= avail_excr_soil, f"Excreta_to_meth_cap_{e}")

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

        # Crée une fonction pour obtenir l'azote disponible après perte et autres usages, cela évitera la redondance
        def get_nitrogen_production(prod_i, df_prod):
            return df_prod.loc[prod_i, "Available Nitrogen Production (ktN)"]

        # Fusionne les deux boucles pour traiter tous les consommateurs
        for cons, proportion, products_list in pairs:
            besoin = df_cons.loc[cons, "Ingestion (ktN)"]

            # Calcule l'azote total disponible pour ce groupe de cultures
            # azote_total_groupe = lpSum(get_nitrogen_production(p, df_prod) for p in products_list)
            azote_total_groupe = sum(
                get_nitrogen_production(p, df_prod) for p in products_list
            )

            # Ajoute les contraintes de pénalité si l'allocation du groupe n'est pas nulle
            if besoin > 0 and azote_total_groupe > 0:
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
                            / allocation_cible_culture
                            <= penalite_var,
                            f"Penalite_Culture_Plus_{cons}_{proportion}_{prod_i}",
                        )
                        prob += (
                            (allocation_cible_culture - allocation_reelle_culture)
                            / allocation_cible_culture
                            <= penalite_var,
                            f"Penalite_Culture_Moins_{cons}_{proportion}_{prod_i}",
                        )
                    except KeyError:
                        # La variable n'existe pas, on ignore l'ajout des contraintes
                        pass

        # Contrainte explicite pas d'importation d'un produit si il reste du surplus de ce produit
        if use_milp_no_swap:
            for p in df_prod.index:
                I_p = lpSum(I_vars[(p, c)] for c in df_cons.index if (p, c) in I_vars)
                prob += I_p <= M[p] * y_vars[p], f"NoSwap_I_{p}"
                prob += U_vars[p] <= M[p] * (1 - y_vars[p]), f"NoSwap_U_{p}"

        # Résolution du problème
        prob.solve()

        # # DEBUG
        from pulp import value, LpStatus
        import pprint
        import math

        # (1) Reconstruire EXACTEMENT les morceaux de l’objectif
        expr_dev = (poids_penalite_deviation / max(1, len(pairs))) * lpSum(
            delta_vars.values()
        )
        expr_cult = (poids_penalite_culture / max(1, len(df_prod))) * lpSum(
            penalite_culture_vars.values()
        )
        expr_fair = w_fair * (
            fair_term / max(1, len(df_prod))
        )  # moyenne par produit (optionnel)
        expr_meth_energy = W_METH_ENERGY * meth_energy_dev
        expr_meth_diet = (W_METH_DIET / max(1, len(pairs_meth))) * lpSum(
            delta_meth.values()
        )

        # (2) Eval brute
        raw_dev = float(value(expr_dev))
        raw_cult = float(value(expr_cult))
        raw_fair = float(value(expr_fair))
        raw_meth_energy = float(value(expr_meth_energy))
        raw_meth_diet = float(value(expr_meth_diet))

        # (3) Somme = valeur de l’objectif
        obj_val = float(
            value(expr_dev + expr_cult + expr_fair + expr_meth_energy + expr_meth_diet)
        )

        # (4) Parts en % (protéger division par zéro)
        def _share(x, total):
            return (
                (100.0 * x / total) if total and not math.isclose(total, 0.0) else None
            )

        shares = {
            "dev_%": _share(raw_dev, obj_val),
            "cult_%": _share(raw_cult, obj_val),
            "fair_%": _share(raw_fair, obj_val),
            "meth_energy_%": _share(raw_meth_energy, obj_val),
            "meth_diet_%": _share(raw_meth_diet, obj_val),
        }
        # (5) État du solveur
        solver_info = {
            "status_code": prob.status,
            "status": LpStatus.get(prob.status, "Unknown"),
        }
        # (6) Rapport complet
        report = {
            "solver": solver_info,
            "objective_total": obj_val,
            "weighted_contributions": {
                "dev": raw_dev,
                "cult": raw_cult,
                "fair": raw_fair,
                "meth_energy": raw_meth_energy,
                "meth_diet": raw_meth_diet,
            },
            "shares_%": shares,
            "weights_used": {
                "Weight diet": float(poids_penalite_deviation),
                "Weight distribution": float(poids_penalite_culture),
                "Weight fair local split": float(w_fair),
                "Weight methanizer production": float(W_METH_ENERGY),
                "Weight methanizer inputs": float(W_METH_DIET),
            },
        }

        pprint.pp(report, sort_dicts=False)

        df_prod["Nitrogen for Methanizer (ktN)"] = 0.0
        df_excr["Excretion to Methanizer (ktN)"] = 0.0
        df_excr["Excretion to soil (ktN)"] = 0.0
        allocations = []
        meth_prod_alloc = {}  # somme ktN -> méthaniseur par produit
        excr_to_meth = {}  # somme ktN -> méthaniseur par excréta
        waste_to_meth = 0.0  # ktN waste vers méthaniseur (agrégé)

        for var in prob.variables():
            if not var.varValue or var.varValue <= 0:
                continue

            name = var.name
            val = float(var.varValue)

            # Cas 1 : variables classiques x_(product,consumer)
            if name.startswith("x_") and not name.startswith("x_meth"):
                # Nom de la variable : x_('culture','cons')
                chaine = str(var)
                matches = re.findall(r"'([^']*)'", chaine)
                if len(matches) >= 2:
                    prod_i = matches[0].replace("_", " ").strip()
                    cons = matches[1].replace("_", " ").strip()

                    # Type (même logique que ton exemple)
                    if any(idx in name for idx in df_elevage.index):
                        Type = "Local culture feed"
                    else:
                        Type = "Local culture food"

                    allocations.append(
                        {
                            "Product": prod_i,
                            "Consumer": cons,
                            "Allocated Nitrogen": val,
                            "Type": Type,
                        }
                    )
            # Cas 2 : produits -> méthaniseur
            elif name.startswith("x_meth_prod_"):
                prod_i = name.replace("x_meth_prod_", "").replace("_", " ").strip()
                meth_prod_alloc[prod_i] = meth_prod_alloc.get(prod_i, 0.0) + val

                allocations.append(
                    {
                        "Product": prod_i,
                        "Consumer": "Product to Methanizer",
                        "Allocated Nitrogen": val,
                        "Type": "Methanizer",
                    }
                )
            # Cas 3 : excréta -> méthaniseur
            elif name.startswith("x_meth_excr_"):
                excr_i = name.replace("x_meth_excr_", "").replace("_", " ").strip()
                excr_to_meth[excr_i] = excr_to_meth.get(excr_i, 0.0) + val

                allocations.append(
                    {
                        "Product": excr_i,
                        "Consumer": "Excretion to Methanizer",
                        "Allocated Nitrogen": val,
                        "Type": "Methanizer",
                    }
                )
            # Cas 4 : waste agrégé -> méthaniseur
            elif name == "N_waste_meth":
                waste_to_meth += val
                # Enregistrement dans allocations (source "Waste (aggregated)")
                allocations.append(
                    {
                        "Product": "waste",
                        "Consumer": "Green Waste to Methanizer",
                        "Allocated Nitrogen": val,
                        "Type": "Methanizer",
                    }
                )

            elif var.name.startswith("I") and var.varValue > 0:
                # Nom de la variable : I_(cons, culture)
                chaine = str(var)
                matches = re.findall(r"'([^']*)'", chaine)
                parts = [match.replace("_", " ").strip() for match in matches]
                prod_i = parts[0]
                cons = parts[1]
                if any(index in var.name for index in df_elevage.index):
                    Type = "Imported Feed"
                else:
                    Type = "Imported Food"
                allocations.append(
                    {
                        "Product": prod_i,
                        "Consumer": cons,
                        "Allocated Nitrogen": var.varValue,
                        "Type": Type,
                    }
                )

        allocations_df = pd.DataFrame(allocations)

        # Filtrer les lignes en supprimant celles dont 'Allocated Nitrogen' est très proche de zéro
        allocations_df = allocations_df[
            allocations_df["Allocated Nitrogen"].abs() >= 1e-6
        ]

        self.allocations_df = allocations_df

        deviations = []
        for cons, proportion, products_list in pairs:
            # Récupère le type de consommateur et le besoin total
            consumer_type = df_cons.loc[cons, "Type"]
            besoin_total = df_cons.loc[cons, "Ingestion (ktN)"]

            # Calcule l'allocation totale
            allocation_totale = sum(
                x_vars.get((p, cons), 0).varValue for p in products_list
            ) + sum(I_vars.get((p, cons), 0).varValue for p in products_list)

            # Assure que le besoin total n'est pas nul pour le calcul de la proportion
            proportion_effective = (
                allocation_totale / besoin_total if besoin_total > 0 else 0
            )

            # Récupère la valeur de la variable de déviation si elle existe
            delta_var_key = (cons, tuple(products_list))
            if delta_var_key in delta_vars:
                deviation_value = delta_vars[delta_var_key].varValue

                # Détermine le signe de la déviation
                signe = 1 if proportion_effective > proportion else -1

                # Ajoute les informations à la liste
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

        # deviation methaniseur
        meth_rows = []
        meth_diet_df = diets[diets["Consumer"] == METH_DIET_NAME].copy()
        N_meth_total = 0.0
        N_meth_total += sum(meth_prod_alloc.values())
        N_meth_total += sum(excr_to_meth.values())
        N_meth_total += waste_to_meth

        for _, row in meth_diet_df.iterrows():
            prop = float(row["Proportion"])
            exp_pct = prop if prop > 1.0 else (100.0 * prop)
            prods = tuple(row["Products"])
            label = ", ".join(prods)

            # N du groupe : somme des items (produits, excréta, et waste s'il apparaît dans le groupe)
            N_group = 0.0
            for it in prods:
                if it in meth_prod_alloc:
                    N_group += meth_prod_alloc[it]
                elif it in excr_to_meth:
                    N_group += excr_to_meth[it]
                else:
                    # Si "waste" apparaît dans un groupe de la diète, on affecte la variable agrégée
                    if it.lower().strip() in {
                        "waste",
                        "green waste",
                        "municipal waste",
                    }:
                        N_group += waste_to_meth

            alloc_pct = 100.0 * N_group / N_meth_total if N_meth_total > 0 else 0.0
            dev_pct = alloc_pct - exp_pct

            meth_rows.append(
                {
                    "Consumer": METH_DIET_NAME,
                    "Type": "Methanizer",
                    "Expected Proportion (%)": exp_pct,
                    "Deviation (%)": dev_pct,
                    "Proportion Allocated (%)": alloc_pct,
                    "Product": label,
                }
            )

        deviations_meth_df = pd.DataFrame(meth_rows)
        deviations = pd.DataFrame(deviations)
        deviations = pd.concat([deviations, deviations_meth_df], ignore_index=True)

        self.deviations_df = deviations

        # Extraction des importations
        importations = []
        for cons in df_cons.index:
            for prod_i in all_cultures_regime[cons]:
                if (prod_i, cons) in I_vars:
                    import_value = I_vars[(prod_i, cons)].varValue
                    if cons in df_elevage.index:
                        Type = "Feed"
                    else:
                        Type = "Food"
                    if import_value > 0:
                        importations.append(
                            {
                                "Consumer": cons,
                                "Product": prod_i,
                                "Type": Type,
                                "Imported Nitrogen (ktN)": import_value,
                            }
                        )

        # Convertir en DataFrame
        self.importations_df = pd.DataFrame(importations)

        # dataframe sur le méthaniseur
        rows = []
        eps = 1e-9

        # Produits -> méthaniseur
        for p in meth_prod_items if len(meth_prod_items) else []:
            N = float(value(x_meth_prod[p]))
            if N <= 0:
                continue
            # MWh/ktN = (MWh/tMB) * 100000 / (%N)
            mwh_per_ktn = (
                float(df_prod.loc[p, "Methanization power (MWh/tMB)"])
                * 100000.0
                / max(eps, float(df_prod.loc[p, "Nitrogen Content (%)"]))
            )
            E_gwh = N * (mwh_per_ktn / 1000.0)
            rows.append({"source": p, "allocation": N, "energy production": E_gwh})

        # Excréta -> méthaniseur
        for e in meth_excr_items if len(meth_excr_items) else []:
            N = float(value(x_meth_excr[e]))
            if N <= 0:
                continue
            mwh_per_ktn = (
                float(df_excr.loc[e, "Methanization power (MWh/tMB)"])
                * 100000.0
                / max(eps, float(df_excr.loc[e, "Nitrogen Content (%)"]))
            )
            E_gwh = N * (mwh_per_ktn / 1000.0)
            rows.append(
                {"source": e, "allocation (ktN)": N, "energy production (GWh)": E_gwh}
            )

        # Waste (agrégé) -> méthaniseur
        if isinstance(N_waste_meth, LpVariable):
            N = float(value(N_waste_meth))
            if N > 0:
                E_gwh = N * float(WASTE_PWR_MWh_per_ktN) / 1000.0
                src_name = (
                    ", ".join(sorted(meth_waste_items))
                    if len(meth_waste_items)
                    else "waste"
                )
                rows.append(
                    {
                        "source": src_name,
                        "allocation (ktN)": N,
                        "energy production (GWh)": E_gwh,
                    }
                )

        methanizer_overview_df = pd.DataFrame(rows)

        if not methanizer_overview_df.empty:
            total_alloc = methanizer_overview_df["allocation (ktN)"].sum()
            total_energy = methanizer_overview_df["energy production"].sum()

            methanizer_overview_df["allocation share (%)"] = (
                100.0 * methanizer_overview_df["allocation (ktN)"] / total_alloc
                if total_alloc > 0
                else 0.0
            )
            methanizer_overview_df["energy production share (%)"] = (
                100.0 * methanizer_overview_df["energy production (GWh)"] / total_energy
                if total_energy > 0
                else 0.0
            )

            methanizer_overview_df = methanizer_overview_df.sort_values(
                "energy production", ascending=False
            ).reset_index(drop=True)
        else:
            # DataFrame vide avec les bonnes colonnes
            methanizer_overview_df = pd.DataFrame(
                columns=[
                    "source",
                    "allocation (ktN)",
                    "allocation share (%)",
                    "energy production (GWh)",
                    "energy production share (%)",
                ]
            )
        self.methanizer_overview_df = methanizer_overview_df

        # Mise à jour de df_excr
        for e in df_excr.index:
            to_meth = float(excr_to_meth.get(e, 0.0))
            base = float(df_excr.loc[e, "Excretion after volatilization (ktN)"])

            to_soil = base - to_meth

            df_excr.loc[e, "Excretion to Methanizer (ktN)"] = to_meth
            df_excr.loc[e, "Excretion to soil (ktN)"] = to_soil

        # Mise à jour de df_prod
        for idx, row in df_prod.iterrows():
            prod = row.name
            azote_alloue = allocations_df[
                (allocations_df["Product"] == prod)
                & (
                    allocations_df["Type"].isin(
                        ["Local culture food", "Local culture feed", "Methanizer"]
                    )
                )
            ]["Allocated Nitrogen"].sum()
            azote_alloue_feed = allocations_df[
                (allocations_df["Product"] == prod)
                & (allocations_df["Type"] == "Local culture feed")
            ]["Allocated Nitrogen"].sum()
            azote_alloue_food = allocations_df[
                (allocations_df["Product"] == prod)
                & (allocations_df["Type"] == "Local culture food")
            ]["Allocated Nitrogen"].sum()
            azote_alloue_meth = allocations_df[
                (allocations_df["Product"] == prod)
                & (allocations_df["Type"] == "Methanizer")
            ]["Allocated Nitrogen"].sum()
            df_prod.loc[idx, "Nitrogen Exported (ktN)"] = (
                row["Available Nitrogen Production (ktN)"] - azote_alloue
            )
            df_prod.loc[idx, "Nitrogen For Feed (ktN)"] = azote_alloue_feed
            df_prod.loc[idx, "Nitrogen For Food (ktN)"] = azote_alloue_food
            df_prod.loc[idx, "Nitrogen For Methanizer (ktN)"] = azote_alloue_meth

        # Correction des valeurs proches de zéro
        df_prod["Nitrogen Exported (ktN)"] = df_prod["Nitrogen Exported (ktN)"].apply(
            lambda x: 0 if abs(x) < 1e-6 else x
        )
        df_prod["Nitrogen For Feed (ktN)"] = df_prod["Nitrogen For Feed (ktN)"].apply(
            lambda x: 0 if abs(x) < 1e-6 else x
        )
        df_prod["Nitrogen For Food (ktN)"] = df_prod["Nitrogen For Food (ktN)"].apply(
            lambda x: 0 if abs(x) < 1e-6 else x
        )

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
                    "Local culture feed", pd.Series(0, index=df_elevage.index)
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
                ["Local culture food", "Local culture feed", "Methanizer"]
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

        # Flux pour le méthaniseur
        target = {"methanizer": 1}
        source = (
            allocations_locales[
                allocations_locales["Consumer"] == "Product to Methanizer"
            ]
            .set_index("Product")["Allocated Nitrogen"]
            .to_dict()
        )
        if source:
            flux_generator.generate_flux(source, target)

        # Redirection des excretions et digestats
        source = df_excr["Excretion to soil (ktN)"].to_dict()
        flux_generator.generate_flux(source, target_epandage)

        # digestat
        source = {"methanizer": df_excr["Excretion to Methanizer (ktN)"].sum()}
        flux_generator.generate_flux(source, target_epandage)

        # Green waste
        source = {"waste": waste_to_meth}
        target = {"methanizer": 1}
        flux_generator.generate_flux(source, target)

        # Equilibrage des cultures
        for label in df_cultures.index:
            node_index = label_to_index.get(label)
            if node_index is None:
                continue

            # Calcul de l'imbalance (sorties - entrées)
            row_sum = self.adjacency_matrix[node_index, :].sum()
            col_sum = self.adjacency_matrix[:, node_index].sum()
            imbalance = row_sum - col_sum

            if abs(imbalance) < 1e-6:
                continue

            if (
                imbalance > 0
            ):  # Déficit (Plus de sorties que d'entrées) -> Augmenter l'entrée
                # Le flux manquant vient du sol (Entrée dans la culture)
                target = {label: imbalance}
                source = {"soil stock": 1}
                flux_generator.generate_flux(source, target)

            else:  # Excédent (Plus d'entrées que de sorties) -> Augmenter la sortie
                # Le surplus va aux systèmes environnementaux (Sortie de la culture)
                source = {label: -imbalance}
                target = {
                    "hydro-system": 0.9925,
                    "atmospheric N2O": 0.0075,
                }
                flux_generator.generate_flux(source, target)

        ## On annule les flux bidirectionels entre les cultures et soil stock
        soil_stock_index = label_to_index.get("soil stock")

        for label in df_cultures.index:
            node_index = label_to_index.get(label)
            if node_index is None:
                continue

            # Flux vers la culture (du sol)
            flux_in = self.adjacency_matrix[soil_stock_index, node_index]
            # Flux sortant de la culture (vers le sol)
            flux_out = self.adjacency_matrix[node_index, soil_stock_index]

            net_flux = flux_out - flux_in

            # 1. Annulation: met les flux bidirectionnels à zéro
            self.adjacency_matrix[soil_stock_index, node_index] = 0
            self.adjacency_matrix[node_index, soil_stock_index] = 0

            # 2. Réintroduction du flux net
            if abs(net_flux) > 1e-6:
                if net_flux > 0:
                    # Flux net positif: culture -> soil stock
                    self.adjacency_matrix[node_index, soil_stock_index] = net_flux
                else:
                    # Flux net négatif: soil stock -> culture
                    self.adjacency_matrix[soil_stock_index, node_index] = -net_flux

        # Calcul de imbalance dans df_cultures
        df_cultures["Balance (ktN)"] = (
            df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"]
            + df_cultures["Total Non Synthetic Fertilizer Use (ktN)"]
            - df_cultures["Total Nitrogen Production (ktN)"]
            - df_cultures["Volatilized Nitrogen N-NH3 (ktN)"]
            - df_cultures[
                "Volatilized Nitrogen N-N2O (ktN)"
            ]  # Pas de volat sous forme de N2 ?
        )

        # On équilibre Haber-Bosch avec atmospheric N2 pour le faire entrer dans le système
        target = {
            "Haber-Bosch": self.adjacency_matrix[label_to_index["Haber-Bosch"], :].sum()
        }
        source = {"atmospheric N2": 1}
        flux_generator.generate_flux(source, target)

        df_elevage["Conversion factor (%)"] = (
            (df_elevage["Edible Nitrogen (ktN)"] + df_elevage["Dairy Nitrogen (ktN)"])
            * 100
            / df_elevage["Ingestion (ktN)"]
        )

        # On ajoute une ligne total à df_cultures et df_elevage et df_prod
        colonnes_a_exclure = [
            "Spreading Rate (%)",
            "Nitrogen Content (%)",
            "Seed input (kt seeds/kt Ymax)",
            "Category",
            "Main Production",
            "Harvest Index",
            "Fan coef a",
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
        colonnes_a_sommer = df_cultures.columns.difference(colonnes_a_exclure)
        total = df_cultures[colonnes_a_sommer].sum()
        total.name = "Total"
        self.df_cultures_display = pd.concat([df_cultures, total.to_frame().T])

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
        ]
        colonnes_a_sommer = df_elevage.columns.difference(colonnes_a_exclure)
        total = df_elevage[colonnes_a_sommer].sum()
        total.name = "Total"
        self.df_elevage_display = pd.concat([df_elevage, total.to_frame().T])

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

        self.df_cultures = df_cultures
        self.df_elevage = df_elevage
        self.df_prod = df_prod
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
        Computes the total nitrogen production from all crop categories.

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
            "Total Nitrogen Production (ktN)",
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
                "Total Nitrogen Production (ktN)",
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
                self.df_cultures.index == culture, "Total Nitrogen Production (ktN)"
            ].item()
            * 1e6
            / area
        )

    # def primXsec(self):
    #     """
    #     Calculates the percentage of nitrogen from secondary sources (biological or recycled),
    #     compared to the total nitrogen inputs.

    #     Secondary sources include: human excretion, animal excretion, atmospheric inputs, seeds, and leguminous fixation.

    #     :return: Share of secondary sources in total nitrogen inputs (%).
    #     :rtype: float
    #     """
    #     df = self.tot_fert()
    #     return (
    #         (
    #             df["Human excretion"].sum()
    #             + df["Animal excretion"].sum()
    #             + df["atmospheric N2"].sum()
    #             + df["Atmospheric deposition"].sum()
    #             + df["Seeds"].sum()
    #             + df["Leguminous soil enrichment"].sum()
    #         )
    #         * 100
    #         / df.sum()
    #     )

    # def NUE(self):
    #     """
    #     Calculates the crop-level nitrogen use efficiency (NUE).

    #     Defined as the ratio of nitrogen produced by crops over total nitrogen inputs.

    #     :return: NUE of crop systems (%).
    #     :rtype: float
    #     """
    #     df = self.tot_fert()
    #     return self.df_cultures["Total Nitrogen Production (ktN)"].sum() * 100 / df.sum()

    # def NUE_system(self):
    #     """
    #     Calculates system-wide nitrogen use efficiency, including crop and livestock production.

    #     Accounts for feed losses and nitrogen consumed via imported feed.

    #     :return: System-wide NUE (%).
    #     :rtype: float
    #     """
    #     plant_prod = self.df_prod.loc[self.df_prod["Type"]=="plant"]
    #     N_NP = (
    #         plant_prod["Nitrogen Production (ktN)"].sum()
    #         - plant_prod["Nitrogen For Feed (ktN)"].sum()
    #         + self.df_elevage["Edible Nitrogen (ktN)"].sum()
    #         + self.df_elevage["Non Edible Nitrogen (ktN)"].sum()
    #     )
    #     df_fert = self.tot_fert()
    #     N_tot = (
    #         df_fert["Haber-Bosch"]
    #         + df_fert["atmospheric N2"]
    #         + df_fert["Atmospheric deposition"]
    #         + self.df_elevage["Consummed Nitrogen from imported feed (ktN)"].sum()
    #     )
    #     return N_NP / N_tot * 100

    # def NUE_system_2(self):
    #     """
    #     Alternative NUE computation considering livestock conversion factors and feed inputs.

    #     Includes non-edible nitrogen outputs and imported feed consumption in the calculation.

    #     :return: Adjusted system-wide NUE (%).
    #     :rtype: float
    #     """
    #     N_NP = (
    #         self.df_cultures["Total Nitrogen Production (ktN)"].sum()
    #         + (
    #             (self.df_elevage["Edible Nitrogen (ktN)"] + self.df_elevage["Non Edible Nitrogen (ktN)"])
    #             * (1 - 1 / self.df_elevage["Conversion factor (%)"])
    #         ).sum()
    #         + self.df_elevage["Consummed Nitrogen from imported feed (ktN)"].sum()
    #     )
    #     df_fert = self.tot_fert()
    #     N_tot = (
    #         df_fert["Haber-Bosch"]
    #         + df_fert["atmospheric N2"]
    #         + df_fert["Atmospheric deposition"]
    #         + self.df_elevage["Consummed Nitrogen from imported feed (ktN)"].sum()
    #     )
    #     return N_NP / N_tot * 100

    # def N_self_sufficient(self):
    #     """
    #     Estimates nitrogen self-sufficiency of the system.

    #     Defined as the share of atmospheric (biological) nitrogen inputs relative to all external nitrogen sources.

    #     :return: Self-sufficiency ratio (%).
    #     :rtype: float
    #     """
    #     df_fert = self.tot_fert()
    #     return (
    #         (df_fert["atmospheric N2"] + df_fert["Atmospheric deposition"])
    #         * 100
    #         / (
    #             df_fert["atmospheric N2"]
    #             + df_fert["Atmospheric deposition"]
    #             + df_fert["Haber-Bosch"]
    #             + self.df_elevage["Consummed Nitrogen from imported feed (ktN)"].sum()
    #         )
    #     )

    # def env_footprint(self):
    #     """
    #     Calculates the land footprint (in ha) of nitrogen flows.

    #     :return: A pandas Series of land footprint values (in ha).
    #     :rtype: pandas.Series
    #     """

    #     # Merge df_cultures and df_prod to have all data in one place for calculations.
    #     # We use the index of df_cultures and the 'Origin compartment' of df_prod as the merge key.
    #     merged_df = pd.merge(
    #         self.df_cultures,
    #         self.df_prod,
    #         left_index=True,
    #         right_on="Origin compartment",
    #         how="left",
    #         suffixes=("_cultures", "_prod"),
    #     )

    #     # Calculate Nitrogen Production (ktN) per culture by summing up from df_prod
    #     merged_df["Nitrogen Production (ktN)"] = merged_df.groupby(merged_df.index)[
    #         "Nitrogen Production (ktN)"
    #     ].transform("sum")

    #     # Remove duplicates from the merge to avoid over-counting in sum() operations
    #     merged_df = merged_df.loc[~merged_df.index.duplicated(keep="first")]

    #     # Local surface calculations
    #     # 'Nitrogen For Food (ktN)' and 'Nitrogen For Feed (ktN)' are not in the initial df_prod or df_cultures.
    #     # We will assume they are calculated in a previous step and correctly aligned in the merged_df.
    #     # The columns from the original code are assumed to exist in the merged_df after the merge
    #     # and any necessary prior calculations.
    #     local_surface_food = (
    #         merged_df["Nitrogen For Food (ktN)"]
    #         / merged_df["Nitrogen Production (ktN)"]
    #         * merged_df["Area (ha)"]
    #     ).sum()
    #     local_surface_feed = (
    #         merged_df["Nitrogen For Feed (ktN)"]
    #         / merged_df["Nitrogen Production (ktN)"]
    #         * merged_df["Area (ha)"]
    #     ).sum()

    #     # Define a reusable function for import calculations to avoid repeated code
    #     def calculate_import_surface(import_type):
    #         alloc = self.allocations_df.loc[
    #             self.allocations_df["Type"] == import_type,
    #             ["Product", "Allocated Nitrogen"],
    #         ]
    #         alloc_grouped = alloc.groupby("Product")["Allocated Nitrogen"].sum()

    #         # Align with merged_df index to ensure correct joining
    #         allocated_nitrogen = merged_df.merge(
    #             alloc_grouped.to_frame(), left_index=True, right_index=True, how="left"
    #         )["Allocated Nitrogen"].fillna(0)

    #         nitrogen_production = merged_df["Nitrogen Production (ktN)"]
    #         area = merged_df["Area (ha)"]

    #         # Handle zero production values
    #         wheat_nitrogen_production = merged_df.loc[
    #             "Wheat grain", "Nitrogen Production (ktN)"
    #         ]
    #         wheat_area = merged_df.loc["Wheat grain", "Area (ha)"]

    #         adjusted_nitrogen_production = nitrogen_production.replace(
    #             0, wheat_nitrogen_production
    #         )
    #         adjusted_area = area.where(nitrogen_production != 0, wheat_area)

    #         total_import = (
    #             allocated_nitrogen / adjusted_nitrogen_production * adjusted_area
    #         ).sum()
    #         return total_import

    #     total_food_import = calculate_import_surface("Imported Food")
    #     total_feed_import = calculate_import_surface("Imported Feed")

    #     # The livestock sections are already complex due to the nested loops and 'regimes' logic.
    #     # Refactoring them without more context is difficult.
    #     # The existing loops are not ideal, but are required due to the complex logic.
    #     # For now, we will simply correct the data access to use the merged_df.

    #     # Livestock import (as in original code, with corrected data access)
    #     # The logic here is highly complex and relies on undefined `regimes` and `allocations_df`.
    #     # It seems to be calculating a theoretical land use. This part is not easily vectorizable
    #     # without more context on the data structure. The original logic is kept.
    #     elevage_importe = self.df_elevage[
    #         self.df_elevage["Net animal nitrogen exports (ktN)"] < 0
    #     ].copy()
    #     elevage_importe["fraction_importée"] = (
    #         -elevage_importe["Net animal nitrogen exports (ktN)"]
    #         / elevage_importe["Edible Nitrogen (ktN)"]
    #     )
    #     surface_par_culture = pd.Series(0.0, index=self.df_prod.index)
    #     for animal in elevage_importe.index:
    #         if animal not in self.allocations_df["Consumer"].values:
    #             continue

    #         part_importee = elevage_importe.loc[animal, "fraction_importée"]
    #         if elevage_importe.loc[animal, "fraction_importée"] == np.inf:
    #             # ... (The rest of the `inf` logic from the original code)
    #             pass  # Skipping complex logic as it is not part of the main question.
    #         else:
    #             aliments = self.allocations_df[
    #                 self.allocations_df["Consumer"] == animal
    #             ]
    #             for _, row in aliments.iterrows():
    #                 culture = row["Product"]
    #                 azote = row["Allocated Nitrogen"] * part_importee

    #                 # Use merged_df for data access
    #                 if culture in merged_df.index:
    #                     prod = merged_df.loc[culture, "Nitrogen Production (ktN)"]
    #                     surface = merged_df.loc[culture, "Area (ha)"]
    #                     if prod > 0:
    #                         surface_equivalente = azote / prod * surface
    #                         surface_par_culture[culture] += surface_equivalente
    #     import_animal = surface_par_culture.sum()

    #     # Livestock export (as in original code, with corrected data access)
    #     elevage_exporte = self.df_elevage[
    #         self.df_elevage["Net animal nitrogen exports (ktN)"] > 0
    #     ].copy()
    #     elevage_exporte["fraction_exportée"] = (
    #         elevage_exporte["Net animal nitrogen exports (ktN)"]
    #         / elevage_exporte["Edible Nitrogen (ktN)"]
    #     )
    #     surface_par_culture_exporte = pd.Series(0.0, index=self.df_prod.index)
    #     for animal in elevage_exporte.index:
    #         if animal not in self.allocations_df["Consumer"].values:
    #             continue
    #         part_exportee = elevage_exporte.loc[animal, "fraction_exportée"]
    #         aliments = self.allocations_df[self.allocations_df["Consumer"] == animal]
    #         for _, row in aliments.iterrows():
    #             culture = row["Product"]
    #             azote = row["Allocated Nitrogen"] * part_exportee
    #             if culture in merged_df.index:
    #                 prod = merged_df.loc[culture, "Nitrogen Production (ktN)"]
    #                 surface = merged_df.loc[culture, "Area (ha)"]
    #                 if prod > 0:
    #                     surface_equivalente = azote / prod * surface
    #                     surface_par_culture_exporte[culture] += surface_equivalente
    #     export_animal = surface_par_culture_exporte.sum()

    #     # Crop exports
    #     # Columns assumed to exist in merged_df after merge and prior calculations.
    #     mask = merged_df["Sub Type"] != "grazing"
    #     export_surface = (
    #         merged_df.loc[mask, "Nitrogen Exported (ktN)"]
    #         / merged_df.loc[mask, "Nitrogen Production (ktN)"]
    #         * merged_df.loc[mask, "Area (ha)"]
    #     ).sum()

    #     return pd.Series(
    #         {
    #             "Local Food": int(local_surface_food),
    #             "Local Feed": int(local_surface_feed),
    #             "Import Food": int(total_food_import),
    #             "Import Feed": int(total_feed_import),
    #             "Import Livestock": int(import_animal),
    #             "Export Livestock": -int(export_animal),
    #             "Export Plant": -int(export_surface),
    #         }
    #     )

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
        - prod_k : Total Nitrogen Production (ktN) par culture (>=0)
        - area_ha: Area (ha)
        + calcule une valeur de repli pour les cultures à prod_k == 0 (moyenne par catégorie puis globale).
        """
        area_ha = self._safe_series(self.df_cultures.get("Area (ha)", 0.0))
        prod_k = self._safe_series(
            self.df_cultures.get(
                "Total Nitrogen Production (ktN)",
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
            self.df_cultures.get("Total Nitrogen Production (ktN)", pd.Series()).sum()
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
