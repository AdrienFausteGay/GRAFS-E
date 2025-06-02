import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyomo.contrib.appsi.solvers as _appsi_solvers  # active les solveurs APPSI
import seaborn as sns
from matplotlib.colors import LogNorm
from pyomo.environ import minimize
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress
from sklearn.metrics import r2_score
from tqdm import tqdm

import grafs_e.graphes_objet as gr
from grafs_e.donnees import *
from grafs_e.N_class import DataLoader, FluxGenerator, NitrogenFlowModel

_ = _appsi_solvers


class scenario:
    def __init__(self, scenario_path, dataloader=None):
        self.data_path = os.path.join(os.getcwd(), "src", "grafs_e", "data")
        if dataloader is None:
            self.dataloader = DataLoader()
        else:
            self.dataloader = dataloader
        self.scenario_path = scenario_path

    def historic_trend(self, region, excel_line):
        L = []
        for i in annees_disponibles:
            df = self.dataloader.pre_process_df(i, region)
            L.append(df.loc[df["index_excel"] == excel_line, region].item())
        return L

    @staticmethod
    def LU_excretion(dataloader, region, t=None):
        L = {}
        livestock = {
            "bovines": [(1150, 1164), (1196, 1210)],
            "caprines": [(1166, 1168), (1212, 1214)],
            "ovines": [(1170, 1172), (1216, 1218)],
            "porcines": [(1174, 1178), (1220, 1224)],
            "poultry": [(1180, 1190), (1226, 1236)],
            "equines": [(1192, 1193), (1238, 1239)],
        }
        for i in annees_disponibles:
            df = dataloader.pre_process_df(i, region)
            L[i] = {}
            for type in livestock.keys():
                heads = df.loc[df["index_excel"].between(livestock[type][0][0], livestock[type][0][1]), region]
                heads_type = df.loc[df["index_excel"].between(livestock[type][0][0], livestock[type][0][1]), "nom"]
                # Récupérer les coefficients LU correspondant à chaque type
                Lu = heads_type.map(lu_coefficients)
                # Calcul total LU
                LU = np.dot(heads, Lu)
                excr_cap = df.loc[df["index_excel"].between(livestock[type][1][0], livestock[type][1][1]), region]

                if heads.sum() == 0:
                    L[i][type] = 0
                else:
                    L[i][type] = np.dot(heads, excr_cap) / LU
        if t != None:
            return [entry[t] for entry in L.values()]
        return L

    @staticmethod
    def livestock_LU(dataloader, region, t=None):
        LU = {}
        livestock = {
            "bovines": [(1150, 1164), (0, 15)],
            "caprines": [(1166, 1168), (15, 18)],
            "ovines": [(1170, 1172), (18, 21)],
            "porcines": [(1174, 1178), (21, 26)],
            "poultry": [(1180, 1190), (26, 37)],
            "equines": [(1192, 1193), (37, 39)],
        }
        lu_list = list(lu_coefficients.values())
        for i in annees_disponibles:
            df = dataloader.pre_process_df(i, region)
            LU[i] = {}
            for type in livestock.keys():
                heads = df.loc[df["index_excel"].between(livestock[type][0][0], livestock[type][0][1]), region]
                lu_coef = lu_list[livestock[type][1][0] : livestock[type][1][1]]
                LU[i][type] = np.dot(heads, lu_coef)
        if t != None:
            return [entry[t] for entry in LU.values()]
        return LU

    @staticmethod
    def LU_prod(dataloader, region, t=None):
        LU_prod = {}
        index = {
            "bovines": [1017, 1024],
            "ovines": [1018, 1025],
            "caprines": [1019],
            "equines": [1022],
            "porcines": [1020],
            "poultry": [1021, 1023],
        }
        LU = scenario.livestock_LU(dataloader, region)
        for i in annees_disponibles:
            LU_prod[i] = {}
            df = dataloader.pre_process_df(i, region)
            for type in index.keys():
                if LU[i][type] == 0:
                    LU_prod[i][f"{type} productivity"] = 0
                    if type in ["bovines", "ovines", "caprines", "poultry"]:
                        LU_prod[i][f"{type} dairy productivity"] = 0
                else:
                    LU_prod[i][f"{type} productivity"] = (
                        df.loc[df["index_excel"] == index[type][0], region].item() / LU[i][type] * 1e6
                    )
                    if type in ["bovines", "ovines", "caprines", "poultry"]:
                        if type in ["ovines", "caprines"]:
                            LU_prod[i][f"{type} dairy productivity"] = (
                                df.loc[df["index_excel"] == index["ovines"][1], region].item()
                                * LU[i][type]
                                / (LU[i]["ovines"] + LU[i]["caprines"]) ** 2
                                * 1e6
                            )
                        else:
                            LU_prod[i][f"{type} dairy productivity"] = (
                                df.loc[df["index_excel"] == index[type][1], region].item() / LU[i][type] * 1e6
                                if LU[i][type] > 0
                                else 0
                            )
        if t != None:
            return [entry[t] for entry in LU_prod.values()]
        return LU_prod

    def extrapolate_recent_trend(self, data, future_year, alpha=7.0, seuil_bas=0, seuil_haut=None):
        """
        Extrapole une courbe historique en donnant plus de poids aux années récentes,
        sans modèle prédéfini (linéaire/exponentiel/polynôme).

        Paramètres :
        -----------
        - future_year : année-cible jusqu'à laquelle prolonger
        - alpha : coefficient pour l'exponentiel (importance des points récents)
        - seuil_bas : borne inférieure (optionnel)
        - seuil_haut : borne supérieure (optionnel)

        Retourne :
        ----------
        - extended_years : np.array des années depuis la plus récente jusqu'à future_year
        - extended_values : np.array des valeurs extrapolées
        - slope : la pente moyenne calculée à la fin de l'historique
        """

        x = np.array([int(i) for i in annees_disponibles])
        y = data

        # Calcul du poids exponentiel basé sur la distance à x[-1]
        # distances normalisées [0..1], 0 = point le plus récent, 1 = point le plus ancien

        dist = x[-1] - x
        if dist.max() > 0:
            dist = dist / dist.max()

        weights = np.exp(-alpha * dist)

        # Calcul des deltas et des pentes associées
        # delta_i = (y[i+1]-y[i]) / (x[i+1]-x[i])
        deltas = []
        delta_weights = []
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            slope_i = (y[i + 1] - y[i]) / dx
            # Pondérer par le poids du 2e point (ou la moyenne w[i], w[i+1] au choix)
            w_slope = weights[i + 1]
            deltas.append(slope_i)
            delta_weights.append(w_slope)

        deltas = np.array(deltas)
        delta_weights = np.array(delta_weights)

        # Pente moyenne pondérée par w[i+1]
        slope = np.average(deltas, weights=delta_weights)

        # Extrapolation "an par an" depuis x[-1] jusqu'à future_year
        year_start = int(x[-1]) + 1
        year_end = int(future_year)
        extended_years = np.arange(year_start, year_end + 1, 1, dtype=float)

        # On démarre à la dernière valeur historique
        current_val = y[-1]
        extended_values = []

        for yr in extended_years:
            # On avance d'une année => + slope * 1 an
            # (Si besoin, gérer un delta en jours ou fraction, mais ici 1 an)
            new_val = current_val + slope * (yr - (yr - 1))

            # Application des seuils
            if seuil_bas is not None:
                new_val = max(new_val, seuil_bas)
            if seuil_haut is not None:
                new_val = min(new_val, seuil_haut)

            extended_values.append(new_val)
            current_val = new_val

        return extended_years, extended_values

    def get_import_net(self, region):
        L = []
        for yr in annees_disponibles:
            df = self.dataloader.sheets_dict["GRAFS" + yr].copy()
            df.columns = df.iloc[0]
            correct_region = {
                "Pyrénées occid": "Pyrénées occidentales",
                "Pyrénées Orient": "Pyrénées Orientales",
            }
            if region in correct_region.keys():
                region = correct_region[region]
            L.append(df[region].iloc[78])
        return L

    def logistic_urb_pop(self, region, year):
        int_year = [int(i) for i in annees_disponibles]
        prop_urb = np.array(self.historic_trend(region, 6))
        logit = np.log(prop_urb / (100 - prop_urb))
        slope, intercept, r_value, p_value, std_err = linregress(int_year, logit)
        return 100.0 / (1.0 + np.exp(-(intercept + slope * int(year))))

    def generate_crop_tab(self, region, function_name="Linear"):
        df = self.dataloader.pre_process_df(annees_disponibles[-1], region)

        cultures_df = df.loc[df["index_excel"].isin(range(259, 294)), ("nom", region)]
        cultures_df[region] = cultures_df[region] * 100 / cultures_df[region].sum()

        df_insert = pd.DataFrame(cultures_df.values, columns=["Crops", "Area proportion (%)"])
        df_insert.loc[len(df_insert)] = ["Natural meadow ", None]
        df_insert.loc[df_insert["Crops"] == "Forage cabbages & roots", "Crops"] = "Forage cabbages"
        df_insert.loc[df_insert["Crops"] == "rice", "Crops"] = "Rice"

        df_insert["Enforce Area"] = False

        df_insert.index = df_insert["Crops"]

        Y_pros = Y(self.dataloader)

        def fit_and_store(culture: str, region: str) -> dict:
            """
            Calibre le modèle désigné par function_name pour <culture>.
            Ne renvoie que les paramètres utiles à ce modèle,
            plus le R² et le nom de la culture.
            """
            F, Yr = Y_pros.get_Y(culture, region)

            # --- Cas sans données ---
            empty_data = len(F) == 0 or len(Yr) == 0

            if function_name == "linear":
                a, b, _ = Y_pros.fit_Y_lin(culture, region)
                r2 = 0 if empty_data else r2_score(Yr, Y_pros.Y_th_lin(F, a, b))
                return {
                    "culture": culture,
                    "a": round(a, 2),
                    "b": round(b, 2),
                    "r2": round(r2, 2),
                }

            elif function_name == "linear2":
                a, xb, c, _ = Y_pros.fit_Y_lin_2_scipy(culture, region)
                r2 = 0 if empty_data else r2_score(Yr, Y_pros.Y_th_lin_2(F, a, xb, c))
                return {
                    "culture": culture,
                    "a": round(a, 2),
                    "xb": int(xb),
                    "c": round(c, 2),
                    "r2": round(r2, 2),
                }

            elif function_name == "exp":
                ym, k, _ = Y_pros.fit_Y_exp(culture, region)
                r2 = 0 if empty_data else r2_score(Yr, Y_pros.Y_th_exp_cap(F, ym, k))
                return {
                    "culture": culture,
                    "Ymax (kgN/ha)": int(ym),
                    "k (kgN/ha)": int(k),
                    "r2": round(r2, 2),
                }

            elif function_name == "ratio":
                ym, _ = Y_pros.fit_Y(culture, region)
                r2 = 0 if empty_data else r2_score(Yr, Y_pros.Y_th(F, ym))
                return {
                    "culture": culture,
                    "Ymax (kgN/ha)": ym,
                    "r2": round(r2, 2),
                }

            else:
                raise ValueError(
                    f"Unknown model : {function_name}. Available models are 'linear', 'linear2', 'exp', 'ratio'."
                )

        all_cultures = cultures + legumineuses + prairies
        results = []
        run_fit = partial(fit_and_store, region=region)

        with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            futures = {executor.submit(run_fit, culture): culture for culture in all_cultures}

            with tqdm(
                total=len(all_cultures), desc=f"Fitting models ({region})", position=1, leave=False, dynamic_ncols=True
            ) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    pbar.refresh()

        df_temp = pd.DataFrame(results).set_index("culture")

        # Ajouter (ou mettre à jour) les colonnes dans df_insert
        for col in df_temp.columns:
            df_insert[col] = df_temp[col]
        return df_insert

    def pre_generate_scenario_excel(self, function_name="Linear"):
        model_sheets = pd.read_excel(os.path.join(self.data_path, "scenario.xlsx"), sheet_name=None)
        for region in tqdm(
            regions, total=len(regions), desc="Régions", position=0
        ):  # tqdm(regions[23:], total=len(regions[23:]), desc="Regions", position=0):
            sheets = {}
            sheet_corres = {
                "doc": "doc",
                "main scenario": "main",
                "Surface changes": "area",
                "technical scenario": "technical",
            }
            for sheet_name, df in model_sheets.items():
                sheets[sheet_corres[sheet_name]] = df
            sheets["area"] = self.generate_crop_tab(region, function_name)

            target_dir = os.path.join(self.scenario_path, "pre_gen", function_name)
            os.makedirs(target_dir, exist_ok=True)  # Crée les dossiers si besoin
            file_path = os.path.join(target_dir, f"{region}.xlsx")
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

    def generate_scenario_excel(self, year, region, name, function_name="Linear"):
        self.region = region
        self.year = year
        self.last_data_year = annees_disponibles[-1]
        try:
            self.data = self.dataloader.pre_process_df(self.last_data_year, region)
        except:
            self.data = self.dataloader.pre_process_df(self.last_data_year, "France")
            print(f"No region named {region} in the data")

        model_sheets = pd.read_excel(
            os.path.join(self.data_path, "scenario_region", function_name, self.region + ".xlsx"), sheet_name=None
        )

        sheets = {}
        # sheet_corres = {
        #     "doc": "doc",
        #     "main scenario": "main",
        #     "Surface changes": "area",
        #     "technical scenario": "technical",
        # }

        sheet_corres = {
            "doc": "doc",
            "main": "main",
            "area": "area",
            "technical": "technical",
        }

        for sheet_name, df in model_sheets.items():
            sheets[sheet_corres[sheet_name]] = df
        sheets["doc"][None] = None
        sheets["doc"].iloc[14, 1] = name
        sheets["doc"].iloc[15, 1] = region
        sheets["doc"].iloc[16, 1] = year
        # Du à une mauvaise construction des fiches scenar...
        if function_name == "Exponential":
            sheets["doc"].loc[len(sheets["doc"])] = ["Production function", function_name]
        else:
            sheets["doc"].iloc[17, 1] = function_name

        def format_dep(dep_series):
            return " + ".join(dep_series.astype(str).unique())

        regions_dict = {
            "Ile de France": [
                "Paris",
                "Seine-et-Marne",
                "Yvelines",
                "Essonne",
                "Hauts-de-Seine",
                "Seine-Saint-Denis",
                "Val-de-Marne",
                "Val-d'Oise",
            ],
            "Eure": ["Eure"],
            "Eure-et-Loir": ["Eure-et-Loir"],
            "Picardie": ["Aisne", "Oise", "Somme"],
            "Calvados-Orne": ["Calvados", "Orne"],
            "Seine Maritime": ["Seine-Maritime"],
            "Manche": ["Manche"],
            "Nord Pas de Calais": ["Nord", "Pas-de-Calais"],
            "Champ-Ard-Yonne": ["Ardennes", "Aube", "Marne", "Yonne"],
            "Bourgogne": ["Côte-d'Or", "Haute-Marne"],
            "Grande Lorraine": ["Meurthe-et-Moselle", "Meuse", "Moselle", "Vosges", "Haute-Saône"],
            "Alsace": ["Bas-Rhin", "Haut-Rhin"],
            "Bretagne": ["Côtes-d'Armor", "Finistère", "Ille-et-Vilaine", "Morbihan"],
            "Vendée-Charente": ["Vendée", "Charente", "Charente-Maritime"],
            "Loire aval": ["Loire-Atlantique", "Maine-et-Loire", "Deux-Sèvres", "Mayenne", "Sarthe"],
            "Loire Centrale": ["Loir-et-Cher", "Indre-et-Loire", "Cher", "Loiret", "Indre", "Vienne"],
            "Loire Amont": [
                "Saône-et-Loire",
                "Nièvre",
                "Loire",
                "Haute-Loire",
                "Allier",
                "Puy-de-Dôme",
                "Creuse",
                "Haute-Vienne",
            ],
            "Grand Jura": ["Doubs", "Jura", "Territoire-de-Belfort"],
            "Savoie": ["Savoie", "Haute-Savoie"],
            "Ain-Rhône": ["Ain", "Rhône"],
            "Alpes": ["Alpes-de-Haute-Provence", "Hautes-Alpes"],
            "Isère-Drôme Ardèche": ["Isère", "Drôme", "Ardèche"],
            "Aveyron-Lozère": ["Aveyron", "Lozère"],
            "Garonne": ["Haute-Garonne", "Lot-et-Garonne", "Tarn-et-Garonne", "Gers", "Aude", "Tarn", "Ariège"],
            "Gironde": ["Gironde"],
            "Pyrénées occid": ["Pyrénées-Atlantiques", "Hautes-Pyrénées"],
            "Landes": ["Landes"],
            "Dor-Lot": ["Dordogne", "Lot"],
            "Cantal-Corrèze": ["Cantal", "Corrèze"],
            "Grand Marseille": ["Bouches-du-Rhône", "Vaucluse"],
            "Côte d'Azur": ["Alpes-Maritimes", "Var"],
            "Gard-Hérault": ["Gard", "Hérault"],
            "Pyrénées Orient": ["Pyrénées-Orientales"],
        }

        # Create a mapping dictionary {Department: Region}
        department_to_region = {
            department: region for region, departments in regions_dict.items() for department in departments
        }

        proj_pop = pd.read_excel(
            os.path.join(self.data_path, "projections_pop.xlsx"), sheet_name="Population_DEP", skiprows=5
        )

        proj_pop["Region"] = proj_pop["LIBDEP"].map(department_to_region)
        proj_pop = proj_pop.dropna(subset=["Region"])

        grouped_proj_pop = (
            proj_pop.groupby("Region")
            .agg(
                {
                    "DEP": format_dep,  # Format DEP column as "DEP1 + DEP2 + ..."
                    **{
                        col: "sum" for col in proj_pop.select_dtypes(include="number").columns
                    },  # Sum all numerical columns
                }
            )
            .reset_index()
        )

        if region == "France":
            proj = 0
            for r in regions:
                proj += grouped_proj_pop.loc[grouped_proj_pop["Region"] == r, f"POP_{year}"].item()
        else:
            proj = grouped_proj_pop.loc[grouped_proj_pop["Region"] == region, f"POP_{year}"].item()
        sheets["main"].loc[sheets["main"]["Variable"] == "Population", "Business as usual"] = (
            proj / 1000
        )  # to be in thousands, not in millions !

        if self.data is not None:
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Total per capita protein ingestion", "Business as usual"
            ] = self.historic_trend(region, 8)[-1]
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Vegetal per capita protein ingestion", "Business as usual"
            ] = self.historic_trend(region, 9)[-1]
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Edible animal per capita protein ingestion (excl fish)",
                "Business as usual",
            ] = self.historic_trend(region, 10)[-1]
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Edible animal per capita protein ingestion (excl fish)",
                "Business as usual",
            ] = self.historic_trend(region, 10)[-1]
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Synth N fertilizer application to cropland",
                "Business as usual",
            ] = self.historic_trend(region, 27)[-1]
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Synth N fertilizer application to grassland",
                "Business as usual",
            ] = self.historic_trend(region, 29)[-1]

            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "N recycling rate of human excretion in urban area",
                "Business as usual",
            ] = self.historic_trend(region, 49)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "N recycling rate of human excretion in rural area",
                "Business as usual",
            ] = self.historic_trend(region, 50)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "NH3 volatilization coefficient for synthetic nitrogen",
                "Business as usual",
            ] = self.historic_trend(region, 31)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Indirect N2O volatilization coefficient for synthetic nitrogen",
                "Business as usual",
            ] = self.historic_trend(region, 32)[-1]

            excel_dict = {
                "bovines": 1250,
                "ovines": 1264,
                "caprines": 1278,
                "porcines": 1292,
                "poultry": 1306,
                "equines": 1320,
            }
            # Excretion managment
            for t in betail:
                if t == "equine":
                    t = "equines"
                sheets["technical"].loc[
                    sheets["technical"]["Variable"] == f"{t.capitalize()} % excretion on grassland",
                    "Business as usual",
                ] = self.historic_trend(region, excel_dict[t])[-1]
                sheets["technical"].loc[
                    sheets["technical"]["Variable"] == f"{t.capitalize()} % excretion in the barn as litter manure",
                    "Business as usual",
                ] = self.historic_trend(region, excel_dict[t] + 2)[-1]
                sheets["technical"].loc[
                    sheets["technical"]["Variable"] == f"{t.capitalize()} % excretion in the barn as other manure",
                    "Business as usual",
                ] = self.historic_trend(region, excel_dict[t] + 3)[-1]
                sheets["technical"].loc[
                    sheets["technical"]["Variable"] == f"{t.capitalize()} % excretion in the barn as slurry",
                    "Business as usual",
                ] = self.historic_trend(region, excel_dict[t] + 4)[-1]

            # LU prop
            LU_prop = self.livestock_LU(self.dataloader, self.region)[annees_disponibles[-1]]
            LU_prop_tot = sum(LU_prop.values())
            LU_prop = {key: value / LU_prop_tot for key, value in LU_prop.items()}
            for t in betail:
                if t == "equine":
                    t = "equines"
                sheets["technical"].loc[
                    sheets["technical"]["Variable"] == f"{t.capitalize()} LU",
                    "Business as usual",
                ] = LU_prop[t]

            # Historical trend
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Feed nitrogen net import",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.historic_trend(self.region, 1009), self.year, seuil_bas=None)[1][-1]

            for type in betail:
                if type == "equine":
                    type = "equines"
                sheets["technical"].loc[
                    sheets["technical"]["Variable"] == f"kgN excreted by {type} LU",
                    "Business as usual",
                ] = self.extrapolate_recent_trend(self.LU_excretion(self.dataloader, self.region, type), self.year)[1][
                    -1
                ]

            for type in betail:
                if type == "equine":
                    type = "equines"
                sheets["technical"].loc[
                    sheets["technical"]["Variable"] == f"{type.capitalize()} productivity",
                    "Business as usual",
                ] = self.LU_prod(self.dataloader, self.region, f"{type} productivity")[-1]
                if type in ["bovines", "ovines", "caprines", "poultry"]:
                    sheets["technical"].loc[
                        sheets["technical"]["Variable"] == f"{type.capitalize()} dairy productivity",
                        "Business as usual",
                    ] = self.LU_prod(self.dataloader, self.region, f"{type} dairy productivity")[-1]

            tot_LU = []
            LU_prop_hist = self.livestock_LU(self.dataloader, self.region)
            for yr in annees_disponibles:
                tot_LU.append(sum(LU_prop_hist[yr].values()))

            sheets["main"].loc[
                sheets["main"]["Variable"] == "Total LU",
                "Business as usual",
            ] = self.extrapolate_recent_trend(tot_LU, self.year, seuil_bas=0)[1][-1]

            sheets["main"].loc[
                sheets["main"]["Variable"] == "Arable area",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.historic_trend(self.region, 23), self.year, seuil_bas=0)[1][-1]

            sheets["main"].loc[
                sheets["main"]["Variable"] == "Permanent grassland area",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.historic_trend(self.region, 22), self.year, seuil_bas=0)[1][-1]

            # Import (need GRAFS)
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Net import of vegetal pdcts",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.get_import_net(self.region), self.year, seuil_bas=None)[1][-1]

            # Proportion urbaine
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Urban population",
                "Business as usual",
            ] = self.logistic_urb_pop(self.region, self.year)

            # surface prop tab
            # sheets["area"] = self.generate_crop_tab(self.region)

        with pd.ExcelWriter(os.path.join(self.scenario_path, name + ".xlsx"), engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 👉 Récupérer le workbook ouvert via le writer pour ajouter une case pour vérifier la somme des proportions.
            wb = writer.book

            # ✅ Insérer une formule après l'écriture dans la feuille "area"
            sheet = wb["area"]

            # ✅ Trouver la dernière ligne (max_row donne la dernière ligne non vide)
            last_row = sheet.max_row + 2

            # ✅ Insérer le texte et la formule Excel directement
            sheet[f"A{last_row}"] = "Proportion area sum correct ?"
            sheet[f"B{last_row}"] = f'=IF(SUM(B2:B{last_row - 3})=100, "✅ OK", "❌ Erreur")'

    def generate_base_scenar(self, prod_func):
        # Create scenar for all regions with only area filled
        for region in tqdm(regions, desc="Calcul des Ymax et k", unit="region"):
            try:
                self.data = self.dataloader.pre_process_df(self.last_data_year, region)
            except:
                self.data = self.dataloader.pre_process_df(self.last_data_year, "France")
                print(f"No region named {region} in the data")

            model_sheets = pd.read_excel(os.path.join(self.data_path, "scenario.xlsx"), sheet_name=None)

            sheets = {}
            sheet_corres = {
                "doc": "doc",
                "main scenario": "main",
                "Surface changes": "area",
                "technical scenario": "technical",
            }
            for sheet_name, df in model_sheets.items():
                sheets[sheet_corres[sheet_name]] = df
            sheets["area"] = self.generate_crop_tab(region, prod_func)

            with pd.ExcelWriter(
                os.path.join(self.data_path, "scenario_region", region + ".xlsx"), engine="openpyxl"
            ) as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)


class Y:
    def __init__(self, dataloader=None):
        if dataloader == None:
            self.dataloader = DataLoader()
        else:
            self.dataloader = dataloader
        self.int_yr = [int(i) for i in annees_disponibles]

    def get_Y(self, culture, region, plot=False):
        F = []
        Y = []
        if region == "Savoie":
            years = annees_disponibles[1:]
        else:
            years = annees_disponibles
        for yr in years:
            model = NitrogenFlowModel(data=self.dataloader, year=yr, region=region)
            f = model.Ftot(culture)
            y = model.Y(culture)
            if (
                isinstance(f, (float, np.float64))
                and isinstance(y, (float, np.float64))
                and not np.isnan(f)
                and not np.isnan(y)
                and f != 0
                and y != 0
                and f < 350  # On supprime les valeur délirantes (pb données) de ferti et Y
                and y < 350
                # and y < f*0.9 # Condition NUE<90%
            ):
                F.append(f)
                Y.append(y)
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(F, Y, "o-", color="tab:blue", markersize=6, label="Historic Data")  # Points et ligne
            plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
            plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
            # plt.title("Relation entre Fertilisation et Rendement", fontsize=14, fontweight='bold')
            plt.grid(True, linestyle="--", alpha=0.4)  # Grille discrète
            plt.legend()
            plt.show()
        return np.array(F), np.array(Y)

    @staticmethod
    def Y_th(f, y_max):
        if f + y_max == 0:
            return 0
        return f * y_max / (f + y_max)

    @staticmethod
    def Y_th_exp(f, y_max, k):
        return y_max * (1 - np.exp(-f / k))

    @staticmethod
    def Y_th_exp_cap(f, y_max, k):
        return np.minimum(y_max * (1 - np.exp(-f / k)), 0.99 * f)

    @staticmethod
    def Y_th_lin(f, a, b):
        return np.minimum(a * f, b)

    @staticmethod
    def Y_th_lin_2(f, a, xb, c):
        return np.minimum(a * f, c * (f - xb) + xb * a)

    @staticmethod
    def Y_th_poly(f, a, b):
        return a * f**2 + b * f

    @staticmethod
    def Y_th_poly_cap(f, a, b):
        return np.minimum(0.99 * f, a * f**2 + b * f)

    @staticmethod
    def Y_th_sigmoid(f, Ymax, k, x0):
        return Ymax / (1 + np.exp(-k * (f - x0)))

    @staticmethod
    def Y_th_sigmoid_cap(f, Ymax, k, x0):
        return np.minimum(0.99 * f, Ymax / (1 + np.exp(-k * (f - x0))))

    @staticmethod
    def Y_th_lin_2_smooth(f, a, xb, c, s):
        transition = 1 / (1 + np.exp(-s * (f - xb)))

        return a * f * (1 - transition) + (c * (f - xb) + xb * a) * transition

    def fit_Y(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, None
        Y_max_init = max(Y)
        try:
            popt, _ = curve_fit(self.Y_th, F, Y, p0=[Y_max_init], bounds=(0, np.inf))
            Y_max_opt = popt[0]  # 📌 Paramètre ajusté Y_max
        except RuntimeError:
            print("⚠️ Ajustement impossible pour", culture, region)
            Y_max_opt = None  # Retourne None en cas d'échec

        Y_th_fitted = self.Y_th(F, Y_max_opt) if Y_max_opt is not None else None
        return Y_max_opt, Y_th_fitted

    def fit_Y_exp(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, None
        Y_max_init = max(Y)

        try:
            popt, _ = curve_fit(
                self.Y_th_exp, F, Y, p0=[Y_max_init, min(F)], bounds=([min(Y), min(F) * 0.5], [max(Y) * 2, max(F) * 2])
            )
            Y_max_opt = popt[0]  # 📌 Paramètre ajusté Y_max
            k = popt[1]
        except RuntimeError:
            print("⚠️ Ajustement impossible pour", culture, region)
            Y_max_opt = 0  # Retourne None en cas d'échec
            k = 0

        Y_th_fitted = self.Y_th_exp(F, Y_max_opt, k) if Y_max_opt is not None else None
        return Y_max_opt, k, Y_th_fitted

    def fit_Y_exp_2(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, None

        Y_max_init = max(Y)
        k_init = min(F)

        def objective(params):
            y_max, k = params
            return np.sum((self.Y_th_exp(F, y_max, k) - Y) ** 2)

        # Contraintes : k > y_max
        constraint = {"type": "ineq", "fun": lambda params: params[1] - params[0]}  # k - y_max > 0

        # Bornes : y_max > 0, k > 0
        bounds = [(0, max(Y) * 2), (0, max(F) * 2)]

        result = minimize(
            objective,
            x0=[Y_max_init, k_init],
            bounds=bounds,
            constraints=[constraint],
        )

        if result.success:
            y_max_opt, k_opt = result.x
            y_th_fit = self.Y_th_exp(F, y_max_opt, k_opt)
            return y_max_opt, k_opt, y_th_fit
        else:
            print("⚠️ Ajustement impossible pour", culture, region)
            return 0, 0, None

    def fit_Y_lin(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, None
        try:
            popt, _ = curve_fit(self.Y_th_lin, F, Y, p0=[0.75, max(Y)], bounds=([0, 0], [1, max(Y) * 2]))
            a = popt[0]  # 📌 Paramètre ajusté Y_max
            b = popt[1]
        except RuntimeError:
            print("⚠️ Ajustement impossible pour", culture, region)
            a = 0  # Retourne None en cas d'échec
            b = 0

        Y_th_fitted = self.Y_th_lin(F, a, b) if a is not None else None
        return a, b, Y_th_fitted

    def fit_Y_lin_2(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, None
        try:
            midpoint = (min(F) + max(F)) / 2
            p01 = [max(Y) / max(F), midpoint, 0.1]
            popt, _ = curve_fit(self.Y_th_lin_2, F, Y, p0=p01, bounds=([0.1, 0, 0.01], [1, 1000, 1]))
            a = popt[0]  # 📌 Paramètre ajusté Y_max
            xb = popt[1]
            c = popt[2]
        except RuntimeError:
            print("⚠️ Ajustement impossible pour", culture, region)
            a = 0  # Retourne None en cas d'échec
            xb = 0
            c = 0

        Y_th_fitted = self.Y_th_lin_2(F, a, xb, c) if a is not None else None
        return a, xb, c, Y_th_fitted

    def fit_Y_lin_2_scipy(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, 0, None

        def objective(params):
            a, xb, c = params
            Y_pred = self.Y_th_lin_2(F, a, xb, c)
            return np.sum((Y - Y_pred) ** 2)

        # ➕ Bornes pour a, xb, c
        bounds = [(0.1, 1.0), (0, max(F)), (0.0, 1.0)]

        # 🔹 Point initial (à ajuster selon le domaine)
        midpoint = (min(F) + max(F)) / 2
        p0 = [max(Y) / max(F), midpoint, 0.1]

        result = minimize(objective, p0, bounds=bounds, method="SLSQP")

        if result.success:
            a_opt, xb_opt, c_opt = result.x
            Y_fit = self.Y_th_lin_2(F, a_opt, xb_opt, c_opt)
            return a_opt, xb_opt, c_opt, Y_fit
        else:
            print(f"⚠️ Ajustement impossible pour {culture}, {region}")
            return 0, 0, 0, None

    def fit_Y_poly(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, None

        try:
            popt, _ = curve_fit(self.Y_th_poly, F, Y, p0=[0.75, 0.9, 0], bounds=([-20, -20, -20], [20, 20, 20]))
            a = popt[0]  # 📌 Paramètre ajusté Y_max
            b = popt[1]
            c = popt[2]
        except RuntimeError:
            print("⚠️ Ajustement impossible pour", culture, region)
            a = 0  # Retourne None en cas d'échec
            b = 0
            c = 0

        Y_th_fitted = self.Y_th_poly(F, a, b, c) if a is not None else None
        return a, b, c, Y_th_fitted

    def fit_Y_poly_constrained(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, 0, None

        # Initialisation
        a_init = -0.002
        b_init = 0.9
        # c_init = 0

        max_F = max(F)

        def objective(params):
            a, b = params
            return np.sum((self.Y_th_poly(F, a, b) - Y) ** 2)

        # ➕ 1ère contrainte : sommet après max(F) → -b / (2a) > max(F)
        # ⇔ -b - 2a * max(F) > 0
        # constraint1 = {"type": "ineq", "fun": lambda p: -p[1] - 2 * p[0] * max_F}

        # ➕ 2ème contrainte : pente à max(F) > 0 → 2a * max(F) + b > 0
        constraint2 = {"type": "ineq", "fun": lambda p: 2 * p[0] * max_F + p[1]}

        bounds = [(-10, 0), (-1000, 2)]  # a < 0 pour avoir une parabole concave

        result = minimize(
            objective,
            x0=[a_init, b_init],
            bounds=bounds,
            method="trust-constr",
            constraints=[constraint2],
        )

        if result.success:
            a_opt, b_opt = result.x
            y_fit = self.Y_th_poly(F, a_opt, b_opt)
            return a_opt, b_opt, y_fit
        else:
            print(f"⚠️ Ajustement impossible pour {culture}, {region}")
            return 0, 0, None

    def fit_Y_sigmoid(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, 0, None

        # Fonction à minimiser (erreur quadratique)
        def objective(params):
            Ymax, k, x0 = params
            Y_pred = self.Y_th_sigmoid(F, Ymax, k, x0)
            return np.sum((Y - Y_pred) ** 2)

        # Bornes réalistes pour les paramètres
        bounds = [(0.1, 3 * max(Y)), (1e-10, 1), (0, 2 * max(F))]  # Ymax, k, x0

        # Initialisation
        p0 = [max(Y), 0.01, np.median(F)]

        result = minimize(objective, p0, bounds=bounds)

        if result.success:
            Ymax, k_opt, x0_opt = result.x
            Y_fit = self.Y_th_sigmoid(F, Ymax, k_opt, x0_opt)
            return Ymax, k_opt, x0_opt, Y_fit
        else:
            print(f"⚠️ Ajustement impossible pour {culture}, {region}")
            return 0, 0, 0, None

    def fit_Y_lin_2_smooth(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            return 0, 0, 0, 0, None

        def objective(params):
            a, xb, c, s = params
            Y_pred = self.Y_th_lin_2_smooth(F, a, xb, c, s)
            return np.sum((Y - Y_pred) ** 2)

        # ➕ Bornes pour a, xb, c, s
        bounds = [(0.1, 1.0), (0, 2 * max(F)), (0.0, 1.0), (0, 1e6)]

        # 🔹 Point initial (à ajuster selon le domaine)
        # midpoint = (min(F) + max(F)) / 2
        # p0 = [max(Y) / max(F), midpoint, 0.1, 1]
        a, xb, c, _ = self.fit_Y_lin_2_scipy(culture, region)
        p0 = [a, xb, c, 0.01]
        print(p0)

        result = minimize(objective, p0, bounds=bounds, method="Nelder-Mead")

        if result.success:
            a_opt, xb_opt, c_opt, s_opt = result.x
            Y_fit = self.Y_th_lin_2_smooth(F, a_opt, xb_opt, c_opt, s_opt)
            return a_opt, xb_opt, c_opt, s_opt, Y_fit
        else:
            print(f"⚠️ Ajustement impossible pour {culture}, {region}")
            return 0, 0, 0, 0, None

    def plot_Y(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            print(f"no {culture} found in {region}")
            return None
        Y_max, _ = self.fit_Y(culture, region)
        F_th = np.linspace(0, 1.05 * max(F), 100)
        Y_th = self.Y_th(F_th, Y_max)
        r2 = np.round(r2_score(Y, self.Y_th(F, Y_max)), 2)
        plt.figure(figsize=(8, 6))
        plt.plot(F, Y, "o", color="tab:blue", markersize=8, label=f"Historic Data, r2 = {r2}")  # Points et ligne
        plt.plot(Y, Y, "--")
        plt.plot(F_th, Y_th, label=f"Theoric curve, Y_max = {int(Y_max)}", color="orange", linewidth=4)
        plt.xlim(0, max(F_th) * 1.1)  # Départ de l'axe X à 0
        plt.ylim(0, max(Y) * 1.1)  # Départ de l'axe Y à 0
        plt.gca().set_aspect("equal")  # Échelle identique en ajustant les limites

        plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
        plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
        # plt.title("Relation entre Fertilisation et Rendement", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.4)  # Grille discrète
        plt.legend()
        plt.show()

    def plot_Y_lin(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            print(f"no {culture} found in {region}")
            return None
        a, b, _ = self.fit_Y_lin(culture, region)
        # a, b = 0.75, 160
        F_th = np.linspace(0, 1.05 * max(F), 100)
        Y_th = self.Y_th_lin(F_th, a, b)
        r2 = np.round(r2_score(Y, self.Y_th_lin(F, a, b)), 2)
        plt.figure(figsize=(8, 6))
        plt.plot(F, Y, "o", color="tab:blue", markersize=8, label=f"Historic Data, r2 = {r2}")  # Points et ligne
        plt.plot(Y, Y, "--")
        plt.plot(F_th, Y_th, label=f"Theoric curve, a = {np.round(a, 2)}, b = {int(b)}", color="orange", linewidth=4)
        plt.xlim(0, max(F_th) * 1.1)  # Départ de l'axe X à 0
        plt.ylim(0, max(Y) * 1.1)  # Départ de l'axe Y à 0
        plt.gca().set_aspect("equal")  # Échelle identique en ajustant les limites

        plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
        plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
        # plt.title("Relation entre Fertilisation et Rendement", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.4)  # Grille discrète
        plt.legend()
        plt.show()

    def plot_Y_lin_2(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            print(f"no {culture} found in {region}")
            return None
        a, xb, c, _ = self.fit_Y_lin_2_scipy(culture, region)
        # a, b = 0.75, 160
        F_th = np.linspace(0, 1.05 * max(F), 100)
        Y_th = self.Y_th_lin_2(F_th, a, xb, c)
        r2 = np.round(r2_score(Y, self.Y_th_lin_2(F, a, xb, c)), 2)
        plt.figure(figsize=(8, 6))
        plt.plot(F, Y, "o", color="tab:blue", markersize=8, label=f"Historic Data, r2 = {r2}")  # Points et ligne
        plt.plot(Y, Y, "--")
        plt.plot(
            F_th,
            Y_th,
            label=f"Theoric curve, a = {np.round(a, 2)}, b = {int(xb)}, c = {np.round(c, 2)}",
            color="orange",
            linewidth=4,
        )
        plt.xlim(0, max(F_th) * 1.1)  # Départ de l'axe X à 0
        plt.ylim(0, max(Y) * 1.1)  # Départ de l'axe Y à 0
        plt.gca().set_aspect("equal")  # Échelle identique en ajustant les limites

        plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
        plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
        # plt.title("Relation entre Fertilisation et Rendement", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.4)  # Grille discrète
        plt.legend()
        plt.show()

    def plot_Y_poly(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            print(f"no {culture} found in {region}")
            return None
        a, b, _ = self.fit_Y_poly_constrained(culture, region)
        # a, b, c = -0.002, 0.90226, 0
        F_th = np.linspace(0, 1.05 * max(F), 100)
        Y_th = self.Y_th_poly_cap(F_th, a, b)
        r2 = np.round(r2_score(Y, self.Y_th_poly_cap(F, a, b)), 2)
        plt.figure(figsize=(8, 6))
        plt.plot(F, Y, "o", color="tab:blue", markersize=8, label=f"Historic Data, r2 = {r2}")  # Points et ligne
        plt.plot(Y, Y, "--")
        plt.plot(
            F_th,
            Y_th,
            label=f"Theoric curve, a = {np.round(a, 2)}, b = {np.round(b, 2)}",
            color="orange",
            linewidth=4,
        )
        plt.xlim(0, max(F_th) * 1.1)  # Départ de l'axe X à 0
        plt.ylim(0, max(Y) * 1.1)  # Départ de l'axe Y à 0
        plt.gca().set_aspect("equal")  # Échelle identique en ajustant les limites

        plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
        plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
        # plt.title("Relation entre Fertilisation et Rendement", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.4)  # Grille discrète
        plt.legend()
        plt.show()

    def plot_Y_exp(self, culture, region):
        F, Y = self.get_Y(culture, region)
        # F_filtered, Y_filtered = zip(
        #     *[(f, y) for f, y in zip(F, Y) if y <= 0.9 * f]
        # )  # Delete points for which NUE = Y/F > 0.9
        # F = np.array(F_filtered)
        # Y = np.array(Y_filtered)
        if len(Y) == 0:
            print(f"no {culture} found in {region}")
            return None
        Y_max, k, _ = self.fit_Y_exp_2(culture, region)
        F_th = np.linspace(0, 1.05 * max(F), 100)
        Y_th = self.Y_th_exp_cap(F_th, Y_max, k)
        r2 = np.round(r2_score(Y, self.Y_th_exp_cap(F, Y_max, k)), 2)
        plt.figure(figsize=(8, 6))
        plt.plot(F, Y, "o", color="tab:blue", markersize=8, label=f"Historic Data, r2 = {r2}")  # Points et ligne
        plt.plot(Y, Y, "--")
        plt.plot(F_th, Y_th, label=f"Theoric curve, Y_max = {int(Y_max)}, k = {int(k)}", color="orange", linewidth=4)
        plt.xlim(0, max(F_th))  # Départ de l'axe X à 0
        plt.ylim(0, max(Y))  # Départ de l'axe Y à 0
        plt.gca().set_aspect("equal")  # Échelle identique en ajustant les limites

        plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
        plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
        # plt.title("Relation entre Fertilisation et Rendement", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.4)  # Grille discrète
        plt.legend()
        plt.show()

    def plot_Y_sigmoid(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            print(f"no {culture} found in {region}")
            return None

        # Fit sigmoïde
        Ymax, k_opt, x0_opt, Y_th_fit = self.fit_Y_sigmoid(culture, region)
        F_th = np.linspace(0, 1.05 * max(F), 200)
        Y_th = self.Y_th_sigmoid_cap(F_th, Ymax, k_opt, x0_opt)

        r2 = np.round(r2_score(Y, self.Y_th_sigmoid_cap(F, Ymax, k_opt, x0_opt)), 2)

        plt.figure(figsize=(8, 6))
        plt.plot(F, Y, "o", color="tab:blue", markersize=8, label=f"Historic Data, r² = {r2}")
        plt.plot(Y, Y, "--", color="gray", label="y = x")
        plt.plot(
            F_th,
            Y_th,
            label=f"Sigmoid fit (Ymax={int(Ymax)}, x₀={int(x0_opt)}, k={k_opt:.2e})",
            color="orange",
            linewidth=3,
        )

        plt.xlim(0, max(F_th) * 1.1)
        plt.ylim(0, max(Y) * 1.1)
        plt.gca().set_aspect("equal")

        plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
        plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.show()

    def plot_Y_lin_2_smooth(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            print(f"no {culture} found in {region}")
            return None

        # Fit sigmoïde
        a, xb, c, s, Y_th_fit = self.fit_Y_lin_2_smooth(culture, region)
        F_th = np.linspace(0, 1.05 * max(F), 200)
        Y_th = self.Y_th_lin_2_smooth(F_th, a, xb, c, s)

        r2 = np.round(r2_score(Y, self.Y_th_lin_2_smooth(F, a, xb, c, s)), 2)

        plt.figure(figsize=(8, 6))
        plt.plot(F, Y, "o", color="tab:blue", markersize=8, label=f"Historic Data, r² = {r2}")
        plt.plot(Y, Y, "--", color="gray", label="y = x")
        plt.plot(
            F_th,
            Y_th,
            label=f"a={np.round(a, 2)}, xb={int(xb)}, s={s:.2e}, c={np.round(c, 2)}",
            color="orange",
            linewidth=3,
        )

        plt.xlim(0, max(F_th) * 1.1)
        plt.ylim(0, max(Y) * 1.1)
        plt.gca().set_aspect("equal")

        plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
        plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.show()

    def compare_r2(self, region):
        """
        Compare les coefficients de détermination (R²) pour chaque culture dans une région donnée
        entre les deux fonctions d'ajustement : self.Y_th et self.Y_th_exp.

        Affiche une heatmap des scores R² et ajoute une barre de progression en console.
        """
        r2_values = {
            "Ratio": {},
            "Exponential": {},
            "Linear 1 slope": {},
            "Linear 2 slopes": {},
            "Polynomial": {},
            "Sigmoid": {},
        }

        # Barre de progression avec tqdm
        for culture in tqdm(cultures + prairies, desc="Calcul des R²", unit="culture"):
            F, Y = self.get_Y(culture, region)
            if len(Y) == 0 or len(F) == 0:
                pass
                # r2_values["Ratio"][culture] = np.nan
                # r2_values["Exponential"][culture] = np.nan
                # r2_values["Linear 1 slope"][culture] = np.nan
                # r2_values["Linear 2 slopes"][culture] = np.nan
                # r2_values["Polynomial"][culture] = np.nan
            else:
                # Ajustement avec la première fonction (Y_th)
                Y_max_th, _ = self.fit_Y(culture, region)
                if Y_max_th is not None:
                    r2_values["Ratio"][culture] = Y_max_th
                    # r2_values["Ratio"][culture] = np.round(r2_score(Y, self.Y_th(F, Y_max_th)), 2)
                else:
                    r2_values["Ratio"][culture] = 0
                # Ajustement avec la seconde fonction (Y_th_exp)
                Y_max_exp, k, _ = self.fit_Y_exp(culture, region)
                if Y_max_exp is not None:
                    r2_values["Exponential"][culture] = np.round(Y_max_exp)
                    # r2_values["Exponential"][culture] = np.round(r2_score(Y, self.Y_th_exp(F, Y_max_exp, k)), 2)
                else:
                    r2_values["Exponential"][culture] = 0

                # Ajustement avec la troisieme fonction (Y_th_lin)
                a, b, _ = self.fit_Y_lin(culture, region)
                if Y_max_exp is not None:
                    r2_values["Linear 1 slope"][culture] = np.round(b)
                    # r2_values["Linear 1 slope"][culture] = np.round(r2_score(Y, self.Y_th_lin(F, a, b)), 2)
                else:
                    r2_values["Linear 1 slope"][culture] = 0

                # Ajustement avec la quatrième fonction (Y_th_lin_2)
                # a, xb, c, _ = self.fit_Y_lin_2(culture, region)
                # if a is not None:
                #     r2_values["Linear 2 slopes"][culture] = np.round(r2_score(Y, self.Y_th_lin_2(F, a, xb, c)), 2)
                # else:
                #     r2_values["Linear 2 slopes"][culture] = 0
                r2_values["Linear 2 slopes"][culture] = np.nan

                # Ajustement avec la cinquième fonction (Y_th_poly)
                a, b, _ = self.fit_Y_poly_constrained(culture, region)
                if a is not None:
                    r2_values["Polynomial"][culture] = np.round(-b / (2 * a))
                    # r2_values["Polynomial"][culture] = np.round(r2_score(Y, self.Y_th_poly(F, a, b)), 2)
                else:
                    r2_values["Polynomial"][culture] = 0

                # Ajustement avec la cinquième fonction (Y_th_poly)
                Ymax, k, x0, _ = self.fit_Y_sigmoid(culture, region)
                if Ymax is not None:
                    r2_values["Sigmoid"][culture] = np.round(Ymax)
                    # r2_values["Sigmoid"][culture] = np.round(r2_score(Y, self.Y_th_sigmoid_cap(F, Ymax, k, x0)), 2)
                else:
                    r2_values["Sigmoid"][culture] = 0

        # Création d'un DataFrame pour la heatmap
        r2_df = np.array(
            [
                list(r2_values["Ratio"].values()),
                list(r2_values["Exponential"].values()),
                list(r2_values["Linear 1 slope"].values()),
                list(r2_values["Linear 2 slopes"].values()),
                list(r2_values["Polynomial"].values()),
                list(r2_values["Sigmoid"].values()),
            ]
        )

        # Création de la heatmap
        plt.figure(figsize=(10, len(cultures) // 3))
        ax = sns.heatmap(
            r2_df.T,
            annot=True,
            cmap="plasma",  # "coolwarm",
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Ymax"},
            # cbar_kws={"label": "R² Score"},
            xticklabels=["Ratio", "Exponential", "Linear 1 slope", "Linear 2 slopes", "Polynomial", "Sigmoid"],
            yticklabels=list(r2_values["Ratio"].keys()),
            vmax=600,
            vmin=0,
        )

        plt.title(f"Comparaison des Ymax (kgN/ha) pour {region}")  # R² Ymax
        plt.xlabel("Modèle")
        plt.ylabel("Culture")

        # Affichage de la heatmap
        plt.show()

        return r2_df, r2_values


## % Prospective classes


class CultureData_prospect:
    def __init__(self, main, area, data_path, categories_mapping, func_prod):
        self.main = main
        self.area = area
        self.data_path = data_path
        self.categories_mapping = categories_mapping
        self.func_prod = func_prod
        self.df_cultures = self.create_culture_dataframe()

    def create_culture_dataframe(self):
        crops_index = self.area["Crops"][:-2]

        # Extraire les données de surface
        arable_area = self.main.loc[self.main["Variable"] == "Arable area", "Business as usual"].item()
        grassland_area = self.main.loc[self.main["Variable"] == "Permanent grassland area", "Business as usual"].item()
        area = self.area["Area proportion (%)"][:-3] * arable_area / 100
        area[35] = grassland_area
        area.index = crops_index

        # Extraire les taux de surface avec épendage et la teneur en azote des cultures
        epend = pd.read_excel(
            os.path.join(self.data_path, "GRAFS_data.xlsx"),
            sheet_name="crops",
        )
        epend = epend.drop("Note", axis=1)
        epend = epend.set_index("Culture")
        epend["Area (ha)"] = area

        # Ajouter les paramètres des fonctions de production
        if self.func_prod == "Linear":
            a = self.area["a"][:-2]
            a.index = crops_index

            b = self.area["b"][:-2]
            b.index = crops_index

            epend["a"] = a
            epend["b"] = b

        if self.func_prod == "Ratio":
            Ymax = self.area["Ymax (kgN/ha)"][:-2]
            Ymax.index = crops_index

            epend["Ymax (kgN/ha)"] = Ymax

        return epend


class ElevageData_prospect:
    def __init__(self, main, technical, data_path):
        self.main = main
        self.technical = technical
        self.df_elevage = self.create_elevage_dataframe(main, technical, data_path)

    def create_elevage_dataframe(self, main, technical, data_path):
        types = ["Bovines", "Ovines", "Caprines", "Equines", "Poultry", "Porcines"]
        LU_tot = main.loc[main["Variable"] == "Total LU", "Business as usual"].item()

        LU = {}
        prod = {}
        dairy = {}
        excr = {}
        manure = {}
        o_liter = {}
        slurry = {}
        grass = {}
        for t in types:
            LU[t] = technical.loc[technical["Variable"] == f"{t} LU", "Business as usual"].item() * LU_tot
            prod[t] = technical.loc[technical["Variable"] == f"{t} productivity", "Business as usual"].item()
            excr[t] = technical.loc[
                technical["Variable"] == f"kgN excreted by {t.lower()} LU", "Business as usual"
            ].item()
            manure[t] = technical.loc[
                technical["Variable"] == f"{t} % excretion in the barn as litter manure", "Business as usual"
            ].item()
            o_liter[t] = technical.loc[
                technical["Variable"] == f"{t} % excretion in the barn as other manure", "Business as usual"
            ].item()
            slurry[t] = technical.loc[
                technical["Variable"] == f"{t} % excretion in the barn as slurry", "Business as usual"
            ].item()
            grass[t] = technical.loc[
                technical["Variable"] == f"{t} % excretion on grassland", "Business as usual"
            ].item()
            if t in ["Bovines", "Ovines", "Poultry", "Caprines"]:
                dairy[t] = technical.loc[technical["Variable"] == f"{t} dairy productivity", "Business as usual"].item()
            else:
                dairy[t] = 0

        gas_em = pd.read_excel(os.path.join(data_path, "GRAFS_data.xlsx"), sheet_name="Volatilisation").set_index(
            "Elevage"
        )

        combined_data = {
            "LU": LU,
            "Productivity (kgcarcass/LU)": prod,
            "Dairy Productivity (kg/LU)": dairy,
            "Excretion per LU (kgN/LU)": excr,
            "% excreted on grassland": grass,
            "% excreted indoors as manure": manure,
            "% excreted indoors as other manure": o_liter,
            "% excreted indoors as slurry": slurry,
        }

        combined_df = pd.DataFrame(combined_data)
        combined_df.index = combined_df.index.str.lower()
        combined_df.rename(index={"equines": "equine"}, inplace=True)

        combined_df = combined_df.join(gas_em, how="left")

        combined_df = combined_df.fillna(0)
        combined_df["% excreted indoors"] = 100 - combined_df["% excreted on grassland"]
        combined_df["Production (ktcarcass)"] = combined_df["LU"] * combined_df["Productivity (kgcarcass/LU)"] * 1e-6
        combined_df["Dairy Production (kt)"] = combined_df["LU"] * combined_df["Dairy Productivity (kg/LU)"] * 1e-6

        combined_df["Edible Nitrogen (ktN)"] = (
            combined_df["Production (ktcarcass)"] * combined_df["% edible"]
            + combined_df["Dairy Production (kt)"] * combined_df["%N dairy"]
        )
        combined_df["Non Edible Nitrogen (ktN)"] = combined_df["Production (ktcarcass)"] * combined_df["% non edible"]

        combined_df["Excreted nitrogen (ktN)"] = combined_df["Excretion per LU (kgN/LU)"] * combined_df["LU"] * 1e-6

        combined_df["Ingestion (ktN)"] = (
            combined_df["Excreted nitrogen (ktN)"]
            + combined_df["Edible Nitrogen (ktN)"]
            + combined_df["Non Edible Nitrogen (ktN)"]
        )

        return combined_df


class NitrogenFlowModel_prospect:
    def __init__(self, scenar_path):
        self.scenar_path = scenar_path
        self.categories_mapping = categories_mapping
        self.labels = labels
        self.cultures = cultures
        self.legumineuses = legumineuses
        self.prairies = prairies
        self.betail = betail
        self.Pop = Pop
        self.ext = ext
        file_path = os.path.dirname(__file__)
        self.data_path = os.path.join(file_path, "data")

        self.scenar_sheets = pd.read_excel(os.path.join(self.scenar_path), sheet_name=None)
        self.doc = pd.DataFrame(self.scenar_sheets["doc"])
        self.main = pd.DataFrame(self.scenar_sheets["main"])
        self.area = pd.DataFrame(self.scenar_sheets["area"])
        self.technical = pd.DataFrame(self.scenar_sheets["technical"])
        self.prod_func = self.doc.loc[
            self.doc["excel sheet for scenario writing"] == "Production function", self.doc.columns.tolist()[1]
        ].item()
        self.culture_data = CultureData_prospect(
            self.main, self.area, self.data_path, categories_mapping, self.prod_func
        )
        self.elevage_data = ElevageData_prospect(self.main, self.technical, self.data_path)
        self.flux_generator = FluxGenerator(labels)

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
        main = self.main
        technical = self.technical
        area = self.area
        doc = self.doc
        year = doc.loc[doc["excel sheet for scenario writing"] == "Year", doc.columns.tolist()[1]].item()
        region = doc.loc[doc["excel sheet for scenario writing"] == "Region name", doc.columns.tolist()[1]].item()
        self.year = year
        self.region = region
        flux_generator = self.flux_generator
        if self.prod_func == "Linear":
            ym = "a"
        if self.prod_func in ["Ratio", "Exponential"]:
            ym = "Ymax (kgN/ha)"

        # Gestion du cas particulier pour 'Straw'
        cereales = ["Wheat", "Rye", "Barley", "Oat", "Grain maize", "Other cereals"]
        somme_surface_cereales = df_cultures["Area (ha)"][cereales].sum()
        df_cultures.loc["Straw", "Area (ha)"] = (
            somme_surface_cereales
            * 0.1  # On attribue 10% des surfaces de céréales à la paille. A changer avec les coproduits TODO
        )
        for cereal in cereales:
            df_cultures.loc[cereal, "Area (ha)"] -= (
                df_cultures.loc["Straw", "Area (ha)"] * df_cultures.loc[cereal, "Area (ha)"] / somme_surface_cereales
            )

        # Flux depuis 'other sectors' vers les cibles sélectionnées
        if self.prod_func in ["Ratio", "Exponential"]:
            # Pour éviter des valeur délirante, on interprète ymax comme un niveau de fertilisation caractéristique (Y(Ymax) = Ymax/2)
            target = (
                df_cultures["Seed input (kt seeds/kt Ymax)"] * df_cultures[ym] / 2 * df_cultures["Area (ha)"] * 1e-6
            ).to_dict()
        else:
            target = (
                df_cultures["Seed input (kt seeds/kt Ymax)"] * df_cultures[ym] * df_cultures["Area (ha)"] * 1e-6
            ).to_dict()
        source = {"other sectors": 1}
        flux_generator.generate_flux(source, target)

        # Fixation symbiotique
        target_fixation = (
            df_cultures["N fixation coef (kgN/kgN)"] * df_cultures[ym] * df_cultures["Area (ha)"] * 1e-6
        ).to_dict()
        source_fixation = {"atmospheric N2": 1}
        flux_generator.generate_flux(source_fixation, target_fixation)
        df_cultures["Symbiotic fixation (ktN)"] = df_cultures.index.map(target_fixation).fillna(0)

        ## Épandage de boue sur les champs
        FE_N_N02_em = 0.002
        FE_N_NH3_em = 0.118
        FE_N_N2_em = 0.425
        pop = main.loc[main["Variable"] == "Population", "Business as usual"].item()
        prop_urb = main.loc[main["Variable"] == "Urban population", "Business as usual"].item() / 100
        N_cons_cap = main.loc[main["Variable"] == "Total per capita protein ingestion", "Business as usual"].item()
        N_cap_vege = main.loc[main["Variable"] == "Vegetal per capita protein ingestion", "Business as usual"].item()
        N_cap_viande = main.loc[
            main["Variable"] == "Edible animal per capita protein ingestion (excl fish)", "Business as usual"
        ].item()
        N_boue = pop * N_cons_cap
        N_vege = pop * N_cap_vege
        N_viande = pop * N_cap_viande
        # Et calcul rapide sur les ingestions de produits de la pêche
        N_fish = N_boue - N_vege - N_viande
        source = {"fishery products": N_fish}
        target = {"urban": prop_urb, "rural": 1 - prop_urb}
        flux_generator.generate_flux(source, target)

        # Revenons aux boues
        prop_recy_urb = (
            technical.loc[
                technical["Variable"] == "N recycling rate of human excretion in urban area", "Business as usual"
            ].item()
            / 100
        )
        prop_recy_rur = (
            technical.loc[
                technical["Variable"] == "N recycling rate of human excretion in rural area", "Business as usual"
            ].item()
            / 100
        )

        Norm = sum(df_cultures["Area (ha)"] * df_cultures["Spreading Rate (%)"])
        # Création du dictionnaire target
        target_ependage = {
            culture: row["Area (ha)"] * row["Spreading Rate (%)"] / Norm for culture, row in df_cultures.iterrows()
        }

        source_boue = {
            "urban": prop_urb * N_boue * prop_recy_urb,
            "rural": (1 - prop_urb) * prop_recy_rur * N_boue,
        }

        flux_generator.generate_flux(source_boue, target_ependage)

        # Le reste est perdu dans l'environnement
        # Ajouter les fuites de N2O
        source = {
            "urban": N_boue * prop_urb * FE_N_N02_em,
            "rural": N_boue * (1 - prop_urb) * FE_N_N02_em,
        }
        target = {"N2O emission": 1}
        flux_generator.generate_flux(source, target)

        # Ajouter les fuites de NH3
        source = {
            "urban": N_boue * prop_urb * FE_N_NH3_em,
            "rural": N_boue * (1 - prop_urb) * FE_N_NH3_em,
        }
        target = {"NH3 volatilization": 1}
        flux_generator.generate_flux(source, target)

        # Ajouter les fuites de N2
        source = {
            "urban": N_boue * prop_urb * FE_N_N2_em,
            "rural": N_boue * (1 - prop_urb) * FE_N_N2_em,
        }
        target = {"atmospheric N2": 1}
        flux_generator.generate_flux(source, target)

        # Le reste est perdu dans l'hydroshere
        target = {"hydro-system": 1}
        source = {
            "urban": N_boue * prop_urb * (1 - prop_recy_urb - FE_N_N02_em - FE_N_NH3_em - FE_N_N2_em),
            "rural": N_boue * (1 - prop_urb) * (1 - prop_recy_rur - FE_N_NH3_em - FE_N_N02_em - FE_N_N2_em),
        }
        # Remplir la matrice d'adjacence
        flux_generator.generate_flux(source, target)

        # Azote excrété sur les prairies
        # Calculer les poids pour chaque cible
        # Calcul de la surface totale pour les prairies
        total_surface = (
            df_cultures.loc["Alfalfa and clover", "Area (ha)"]
            + df_cultures.loc["Non-legume temporary meadow", "Area (ha)"]
            + df_cultures.loc["Natural meadow ", "Area (ha)"]
        )

        # Création du dictionnaire target
        target = {
            "Alfalfa and clover": df_cultures.loc["Alfalfa and clover", "Area (ha)"] / total_surface,
            "Non-legume temporary meadow": df_cultures.loc["Non-legume temporary meadow", "Area (ha)"] / total_surface,
            "Natural meadow ": df_cultures.loc["Natural meadow ", "Area (ha)"] / total_surface,
        }
        source = (
            df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted on grassland"]
            / 100
            * (1 - df_elevage["N-NH3 EM. outdoor"] - df_elevage["N-N2O EM. outdoor"] - df_elevage["N-N2 EM. outdoor"])
        ).to_dict()

        flux_generator.generate_flux(source, target)

        # Le reste est émit dans l'atmosphere
        # N2
        target = {"atmospheric N2": 1}
        source = (
            df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted on grassland"]
            / 100
            * df_elevage["N-N2 EM. outdoor"]
        ).to_dict()

        flux_generator.generate_flux(source, target)

        # NH3
        # 1 % est volatilisée de manière indirecte sous forme de N2O
        target = {"NH3 volatilization": 0.99}
        source = (
            df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted on grassland"]
            / 100
            * df_elevage["N-NH3 EM. outdoor"]
        ).to_dict()

        flux_generator.generate_flux(source, target)

        volat_N2O = (
            0.01
            * df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted on grassland"]
            / 100
            * df_elevage["N-NH3 EM. outdoor"]
        )
        # N2O
        target = {"N2O emission": 1}
        source = (
            volat_N2O
            + df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted on grassland"]
            / 100
            * df_elevage["N-N2O EM. outdoor"]
        ).to_dict()

        flux_generator.generate_flux(source, target)

        ## Epandage sur champs

        source = (
            df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted indoors"]
            / 100
            * (
                df_elevage["% excreted indoors as slurry"]
                / 100
                * (
                    1
                    - df_elevage["N-NH3 EM. manure indoor"]
                    - df_elevage["N-N2O EM. manure indoor"]
                    - df_elevage["N-N2 EM. manure indoor"]
                )
                + df_elevage["% excreted indoors as manure"]
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
            df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted indoors"]
            / 100
            * (
                df_elevage["% excreted indoors as slurry"] / 100 * df_elevage["N-N2 EM. slurry indoor"]
                + df_elevage["% excreted indoors as manure"] / 100 * df_elevage["N-N2 EM. manure indoor"]
                # + other manure emission ?? TODO
            )
        ).to_dict()

        flux_generator.generate_flux(source, target)

        # NH3
        # 1 % est volatilisée de manière indirecte sous forme de N2O
        target = {"NH3 volatilization": 0.99}
        source = (
            df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted indoors"]
            / 100
            * (
                df_elevage["% excreted indoors as slurry"] / 100 * df_elevage["N-NH3 EM. slurry indoor"]
                + df_elevage["% excreted indoors as manure"] / 100 * df_elevage["N-NH3 EM. manure indoor"]
            )
        ).to_dict()

        flux_generator.generate_flux(source, target)

        volat_N2O = (
            0.01
            * df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted indoors"]
            / 100
            * (
                df_elevage["% excreted indoors as slurry"] / 100 * df_elevage["N-NH3 EM. slurry indoor"]
                + df_elevage["% excreted indoors as manure"] / 100 * df_elevage["N-NH3 EM. manure indoor"]
            )
        )
        # N2O
        target = {"N2O emission": 1}
        source = (
            volat_N2O
            + df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted indoors"]
            / 100
            * (
                df_elevage["% excreted indoors as slurry"] / 100 * df_elevage["N-N2O EM. slurry indoor"]
                + df_elevage["% excreted indoors as manure"] / 100 * df_elevage["N-N2O EM. manure indoor"]
            )
        ).to_dict()

        flux_generator.generate_flux(source, target)

        # Dépôt atmosphérique : proportionel aux emmission de gaz azoté. A voir après l'élevage !
        target = (
            df_cultures["Area (ha)"] / df_cultures["Area (ha)"].sum()
        ).to_dict()  # Dépôt proportionnel aux surface
        source = {
            "N2O emission": 0.01 * adjacency_matrix[:, label_to_index["N2O emission"]].sum(),
            "NH3 volatilization": 0.1 * adjacency_matrix[:, label_to_index["NH3 volatilization"]].sum(),
        }
        # TODO réfléchir à comment intégrer les retombées de la volatilisation des engrais synthétiques
        adjacency_matrix[:, label_to_index["N2O emission"]] *= 0.99
        adjacency_matrix[:, label_to_index["NH3 volatilization"]] *= 0.9
        flux_generator.generate_flux(source, target)

        df_cultures = df_cultures.fillna(0)

        # Calcul de l'azote épendu par hectare
        def calculer_azote_ependu(culture):
            sources = self.betail + self.Pop + ["atmospheric N2", "other sectors", "NH3 volatilization", "N2O emission"]
            adj_matrix_df = pd.DataFrame(adjacency_matrix, index=self.labels, columns=self.labels)
            return adj_matrix_df.loc[sources, culture].sum()

        df_cultures["Total Non Synthetic Fertilizer Use (ktN)"] = df_cultures.index.map(calculer_azote_ependu)
        df_cultures["Surface Non Synthetic Fertilizer Use (kgN/ha)"] = df_cultures.apply(
            lambda row: row["Total Non Synthetic Fertilizer Use (ktN)"] / row["Area (ha)"] * 10**6
            if row["Area (ha)"] > 0 and row["Total Non Synthetic Fertilizer Use (ktN)"] > 0
            else 0,
            axis=1,
        )

        df_cultures["Yield (kgN/ha)"] = 0.0
        df_cultures["Nitrogen Production (ktN)"] = 0.0

        # Les légumineuses ne reçoivent pas d'azote synthétique. On peut déjà calculer leur rendement
        for leg in df_cultures.loc[df_cultures["Category"] == "leguminous"].index.tolist() + ["Alfalfa and clover"]:
            if self.prod_func == "Linear":
                Yield = Y.Y_th_lin(
                    df_cultures.loc[df_cultures.index == leg, "Surface Non Synthetic Fertilizer Use (kgN/ha)"].item(),
                    df_cultures.loc[df_cultures.index == leg, "a"].item(),
                    df_cultures.loc[df_cultures.index == leg, "b"].item(),
                )
            if self.prod_func == "Ratio":
                Yield = Y.Y_th(
                    df_cultures.loc[df_cultures.index == leg, "Surface Non Synthetic Fertilizer Use (kgN/ha)"].item(),
                    df_cultures.loc[df_cultures.index == leg, "Ymax (kgN/ha)"].item(),
                )
            if self.prod_func == "Exponential":
                Yield = Y.Y_th_exp_cap(
                    df_cultures.loc[df_cultures.index == leg, "Surface Non Synthetic Fertilizer Use (kgN/ha)"].item(),
                    df_cultures.loc[df_cultures.index == leg, "Ymax (kgN/ha)"].item(),
                    df_cultures.loc[df_cultures.index == leg, "k (kgN/ha)"].item(),
                )
            df_cultures.loc[df_cultures.index == leg, "Yield (kgN/ha)"] = Yield

        df_cultures["Nitrogen Production (ktN)"] = df_cultures["Yield (kgN/ha)"] * df_cultures["Area (ha)"] / 1e6
        # Mécanisme d'héritage de l'azote en surplus des légumineuses
        df_cultures["Leguminous Nitrogen Surplus (ktN)"] = 0.0
        leg_with_alfalfa = self.legumineuses + ["Alfalfa and clover"]

        surplus = (
            df_cultures.loc[leg_with_alfalfa, "Total Non Synthetic Fertilizer Use (ktN)"].astype(float).values
            - df_cultures.loc[leg_with_alfalfa, "Nitrogen Production (ktN)"].astype(float).values
        )

        df_cultures.loc[leg_with_alfalfa, "Leguminous Nitrogen Surplus (ktN)"] = surplus

        # Distribution du surplus aux céréales
        total_surplus_azote = df_cultures["Leguminous Nitrogen Surplus (ktN)"].sum()
        total_surface_cereales = df_cultures.loc[
            (
                (df_cultures["Category"] == "cereals (excluding rice)")
                | (df_cultures.index.isin(["Straw", "Forage maize"]))
            ),
            "Area (ha)",
        ].sum()
        df_cultures["Leguminous heritage (ktN)"] = 0.0
        mask_cereales = (df_cultures["Category"] == "cereals (excluding rice)") | (
            df_cultures.index.isin(["Straw", "Forage maize"])
        )

        # Extraire les surfaces concernées
        surface_vals = df_cultures.loc[mask_cereales, "Area (ha)"].astype(float).values

        # Calcul du surplus hérité à répartir
        heritage_vals = surface_vals / total_surface_cereales * total_surplus_azote

        # Affecter proprement sans mismatch
        df_cultures.loc[mask_cereales, "Leguminous heritage (ktN)"] = heritage_vals

        # Mise à jour de la fertilisation non synthétique

        mask = df_cultures["Area (ha)"] != 0

        # Calcul en numpy pour éviter le mismatch d'index
        heritage = df_cultures.loc[mask, "Leguminous heritage (ktN)"].astype(float).values
        area = df_cultures.loc[mask, "Area (ha)"].astype(float).values
        delta = heritage / area * 1e6

        # Mise à jour avec des valeurs bien typées et bien alignées
        df_cultures.loc[mask, "Surface Non Synthetic Fertilizer Use (kgN/ha)"] += delta
        df_cultures["Total Non Synthetic Fertilizer Use (ktN)"] += df_cultures["Leguminous heritage (ktN)"]

        # Génération des flux pour l'héritage des légumineuses
        source_leg = (
            df_cultures.loc[df_cultures["Leguminous Nitrogen Surplus (ktN)"] > 0]["Leguminous Nitrogen Surplus (ktN)"]
            / df_cultures["Leguminous Nitrogen Surplus (ktN)"].sum()
        ).to_dict()
        target_leg = df_cultures["Leguminous heritage (ktN)"].to_dict()
        flux_generator.generate_flux(source_leg, target_leg)

        net_import = main.loc[main["Variable"] == "Net import of vegetal pdcts", "Business as usual"].item()
        Import = max(0, net_import)

        N_synth_crop = (
            main.loc[main["Variable"] == "Synth N fertilizer application to cropland", "Business as usual"].item()
            * main.loc[main["Variable"] == "Arable area", "Business as usual"].item()
            / 1e6
        )

        N_synth_grass = (
            main.loc[main["Variable"] == "Synth N fertilizer application to grassland", "Business as usual"].item()
            * main.loc[main["Variable"] == "Permanent grassland area", "Business as usual"].item()
            / 1e6
        )

        w_diet = technical.loc[technical["Variable"] == "Weight diet", "Business as usual"].item()
        w_Nsyn = technical.loc[technical["Variable"] == "Weight synthetic fertilizer use", "Business as usual"].item()
        w_imp = technical.loc[technical["Variable"] == "Weight import/export balance", "Business as usual"].item()

        df_elevage_comp = df_elevage.copy()
        df_cons_vege = df_elevage.loc[df_elevage["Ingestion (ktN)"] > 10**-8, "Ingestion (ktN)"]

        # On ajoute l'ingestion humaine
        # Une ligne urban, une ligne rural. Cela simplifiera la distinction de regime si un jour c'est pertinent

        df_cons_vege.loc["urban"] = N_vege * prop_urb
        df_cons_vege.loc["rural"] = N_vege * (1 - prop_urb)

        if len(df_cons_vege) > 0:
            CROPS = list(df_cultures.index)  # par exemple
            CONSUMERS = list(df_cons_vege.index)

            # --------------------------------------------------------------------------
            # 1) Collect data from your DataFrames (example placeholders)
            # --------------------------------------------------------------------------
            area = {c: df_cultures.at[c, "Area (ha)"] for c in CROPS}
            if self.prod_func == "Ratio":
                Ymax = {c: df_cultures.at[c, "Ymax (kgN/ha)"] for c in CROPS}
            # kparam = {c: df_cultures.at[c, "k (kgN/ha)"] for c in CROPS}
            if self.prod_func == "Linear":
                a = {c: df_cultures.at[c, "a"] for c in CROPS}
                b = {c: df_cultures.at[c, "b"] for c in CROPS}
            if self.prod_func == "Exponential":
                Ymax = {c: df_cultures.at[c, "Ymax (kgN/ha)"] for c in CROPS}
                k = {c: df_cultures.at[c, "k (kgN/ha)"] for c in CROPS}
            nonSynthFert = {c: df_cultures.at[c, "Surface Non Synthetic Fertilizer Use (kgN/ha)"] for c in CROPS}
            ingestion = {k: df_cons_vege.at[k] for k in CONSUMERS}

            # Filter out any (c,k) pairs not in the diet. We'll build an "allowed" pair list:
            allowed_ck = []
            for k in CONSUMERS:
                # Gather all crops that appear in any sub-list of regimes[k]
                authorized_crops = set()
                for p_ideal, c_list in regimes[k].items():
                    authorized_crops.update(c_list)
                # We'll keep only these (c, k) pairs
                for c in CROPS:
                    if c in authorized_crops:
                        allowed_ck.append((c, k))

            # --------------------------------------------------------------------------
            # 2) Create indexing for decision variables
            # --------------------------------------------------------------------------

            idx_synth = {}
            offset = 0
            for c in CROPS:
                idx_synth[c] = offset
                offset += 1

            idx_alloc = {}
            for c, k in allowed_ck:
                idx_alloc[(c, k)] = offset
                offset += 1

            idx_import = {}
            for c, k in allowed_ck:
                idx_import[(c, k)] = offset
                offset += 1

            n_vars = offset  # total number of decision variables

            w_spread_fixed = 1
            fixed_availability_proportion = {}  # Dict: {(k, tuple(group_crops), c): proportion}
            epsilon = 1e-9

            for k_ in CONSUMERS:
                for group_crops in regimes[k_].values():
                    if not isinstance(group_crops, (list, tuple, set)):
                        continue

                    # Use Area (ha) as the basis for fixed availability weight
                    group_total_area = sum(area.get(c2, 0) for c2 in group_crops)

                    if group_total_area < epsilon:
                        continue  # Skip if group has no area

                    group_key = tuple(sorted(group_crops))  # Use a consistent key for the group

                    for c in group_crops:
                        crop_area = area.get(c, 0)
                        proportion = crop_area / group_total_area
                        fixed_availability_proportion[(k_, group_key, c)] = proportion

            # --------------------------------------------------------------------------
            # 3) Define the objective function
            # --------------------------------------------------------------------------
            def Y_th_lin(f, a, b):
                return np.minimum(a * f, b)

            def Y_th_ratio(f, ymax):
                if f + ymax == 0:
                    return 0
                return f * ymax / (f + ymax)

            def objective(x):
                """
                Return the scalar value of the objective:
                w_diet * diet_deviation
                + w_Nsyn * fertilizer_deviation
                + w_imp * import_export_deviation
                """

                # 3.a) diet_deviation
                total_dev = 0.0
                for k_ in CONSUMERS:
                    # 1) Somme totale allouée (local + imports) pour le consommateur k_
                    denom_k = 0.0
                    for c_ in CROPS:
                        # Allocation locale
                        if (c_, k_) in idx_alloc:
                            denom_k += x[idx_alloc[(c_, k_)]]
                        # Allocation via import
                        if (c_, k_) in idx_import:
                            denom_k += x[idx_import[(c_, k_)]]

                    # 2) Pour chaque proportion idéale, calculez la somme des cultures du groupe
                    for p_ideal, c_list in regimes[k_].items():
                        group_alloc = 0.0
                        for c_ in c_list:
                            # Ajout part locale
                            if (c_, k_) in idx_alloc:
                                group_alloc += x[idx_alloc[(c_, k_)]]
                            # Ajout part importée
                            if (c_, k_) in idx_import:
                                group_alloc += x[idx_import[(c_, k_)]]

                        if denom_k < 1e-6:
                            proportion_real = 0.0
                        else:
                            proportion_real = group_alloc / denom_k

                        # Ecart diététique
                        total_dev += (proportion_real - p_ideal) ** 2

                # 3.b) fertilizer_deviation
                # sum of synthetic fertilizers in ktN
                total_synth = 0.0
                for c in CROPS:
                    # x[idx_synth[c]] is in kgN/ha
                    fert_per_ha = x[idx_synth[c]] + nonSynthFert[c]
                    total_synth += x[idx_synth[c]] * area[c]  # only the synthetic part
                # Convert from kgN to ktN
                total_synth_kt = total_synth / 1e6
                # desired total = (N_synth_crop + N_synth_grass)
                if N_synth_crop + N_synth_grass < 1:
                    scale = 1
                else:
                    scale = N_synth_crop + N_synth_grass
                fert_dev = np.maximum(0, (total_synth_kt - (N_synth_crop + N_synth_grass)) / scale) ** 2
                # fert_dev = ((total_synth_kt - (N_synth_crop + N_synth_grass)) / scale) ** 2
                # 3.c) import_export_deviation
                # sum import
                sum_imp = 0.0
                for c, k in allowed_ck:
                    sum_imp += x[idx_import[(c, k)]]

                # Calcul de la production "non allouée" => export
                export_total = 0.0
                for c in CROPS:
                    # Production locale (ktN) ; on suppose qu'elle est déjà correcte/à jour :
                    if self.prod_func == "Ratio":
                        production_c = (
                            Y_th_ratio(x[idx_synth[c]] + nonSynthFert[c], Ymax[c])
                            * df_cultures.at[c, "Area (ha)"]
                            / 1e6
                        )
                    if self.prod_func == "Linear":
                        production_c = (
                            Y_th_lin(x[idx_synth[c]] + nonSynthFert[c], a[c], b[c])
                            * df_cultures.at[c, "Area (ha)"]
                            / 1e6
                        )
                    if self.prod_func == "Exponential":
                        production_c = (
                            Y.Y_th_exp_cap(x[idx_synth[c]] + nonSynthFert[c], Ymax[c], k[c])
                            * df_cultures.at[c, "Area (ha)"]
                            / 1e6
                        )

                    # Somme des allocations locales sur c
                    allocated_c = 0.0
                    # idx_alloc[(c,k)] = indice pour allocate(c,k)
                    for k in CONSUMERS:
                        if (c, k) in idx_alloc:
                            allocated_c += x[idx_alloc[(c, k)]]

                    # leftover = production - allocated
                    # S'il est > 0 => export, s'il est < 0 => on a sur-alloué (besoin d'import net)
                    leftover_c = production_c - allocated_c
                    export_total += leftover_c

                # Net import du modèle
                net_import_model = sum_imp - export_total

                if abs(net_import) < 1:
                    imp_dev = (net_import_model - net_import) ** 2
                else:
                    imp_dev = ((net_import_model - net_import) / (net_import + 1e-6)) ** 2

                # --- NEW: Allocation Spread Penalty based on Fixed Proportions ---
                spread_penalty_fixed = 0.0
                epsilon = 1e-9

                for k_ in CONSUMERS:
                    for group_crops in regimes[k_].values():  # Iterate over the crop lists defining groups
                        if not isinstance(group_crops, (list, tuple, set)):
                            continue

                        group_key = tuple(sorted(group_crops))  # Match key used in pre-calculation

                        # --- Calculate group total allocation for consumer k_ ---
                        group_alloc_kG = 0.0
                        for c2 in group_crops:
                            if (c2, k_) in idx_alloc:
                                group_alloc_kG += x[idx_alloc[(c2, k_)]]
                            if (c2, k_) in idx_import:
                                group_alloc_kG += x[idx_import[(c2, k_)]]

                        # If total allocation to group is negligible, skip penalty
                        if group_alloc_kG < epsilon:
                            continue

                        # --- Calculate penalty for each crop in the group ---
                        for c in group_crops:
                            # Get the pre-calculated fixed proportion
                            target_proportion = fixed_availability_proportion.get((k_, group_key, c), 0)
                            if target_proportion < epsilon:  # Skip if this crop had no area in pre-calc
                                continue

                            # Calculate target allocation based on fixed proportion
                            target_alloc_c = target_proportion * group_alloc_kG

                            # Calculate this specific crop's actual allocation for consumer k_
                            alloc_ck = 0.0
                            if (c, k_) in idx_alloc:
                                alloc_ck += x[idx_alloc[(c, k_)]]
                            if (c, k_) in idx_import:
                                alloc_ck += x[idx_import[(c, k_)]]

                            # Penalize squared deviation from the target allocation
                            deviation = alloc_ck - target_alloc_c
                            spread_penalty_fixed += deviation**2
                            # Note: This penalizes both over- and under-allocation relative to the fixed proportion.
                            # If you ONLY want to penalize under-allocation (concentration), use:
                            # if deviation < 0: # i.e., alloc_ck < target_alloc_c
                            #     spread_penalty_fixed += deviation**2

                return w_diet * total_dev + w_Nsyn * fert_dev + w_imp * imp_dev + w_spread_fixed * spread_penalty_fixed

            def objective_gradient(x):
                """
                Computes the gradient of the objective function.
                """
                area = df_cultures["Area (ha)"].to_dict()  # Faster access than .loc inside loops
                grad = np.zeros(n_vars, dtype=float)
                epsilon = 1e-12  # For safe division

                # --- Part 1: Import Deviation Gradient ---
                # dDi/dx[j] = 1 if x[j] is an import variable, 0 otherwise
                for c, k in allowed_ck:  # Assuming allowed_ck contains all keys for idx_import
                    if (c, k) in idx_import:  # Check if the import variable exists for this combo
                        import_idx = idx_import[(c, k)]
                        grad[import_idx] += w_imp * 1.0

                # --- Part 2: Fertilizer Deviation Gradient ---
                total_synth_kg = 0.0
                synth_indices = []  # Store indices of synth vars for later gradient update
                synth_crop_map = {}  # Map index back to crop
                for c in CROPS:
                    if c in idx_synth:  # Make sure the crop has a synth variable
                        synth_idx = idx_synth[c]
                        total_synth_kg += x[synth_idx] * area.get(c, 0)  # Use .get for safety
                        synth_indices.append(synth_idx)
                        synth_crop_map[synth_idx] = c

                total_synth_kt = total_synth_kg / 1e6
                Target = N_synth_crop + N_synth_grass
                Scale = max(1.0, Target)  # Ensure scale is at least 1 and float
                Z = (total_synth_kt - Target) / Scale

                if Z > 0:
                    # Base derivative factor d(Df)/d(total_synth_kt)
                    base_fert_deriv = (2.0 * Z) / Scale
                    for synth_idx in synth_indices:
                        c_syn = synth_crop_map[synth_idx]
                        # d(total_synth_kt)/dx[j] = area[c_syn] / 1e6
                        deriv_contrib = base_fert_deriv * (area.get(c_syn, 0) / 1e6)
                        grad[synth_idx] += w_Nsyn * deriv_contrib

                # --- Part 3: Diet Deviation Gradient ---
                for k_ in CONSUMERS:
                    # Calculate denom_k (total allocation for consumer k) *once*
                    denom_k = 0.0
                    # Store indices and type relevant to this consumer for quick lookup
                    alloc_indices_k = {}  # Map index to crop
                    import_indices_k = {}  # Map index to crop
                    for c_ in CROPS:
                        alloc_key = (c_, k_)
                        if alloc_key in idx_alloc:
                            idx = idx_alloc[alloc_key]
                            denom_k += x[idx]
                            alloc_indices_k[idx] = c_
                        if alloc_key in idx_import:
                            idx = idx_import[alloc_key]
                            denom_k += x[idx]
                            import_indices_k[idx] = c_

                    if denom_k < epsilon:  # If total allocation is near zero, derivative is zero
                        continue  # Skip this consumer

                    # Calculate contributions for each group in the regime
                    for p_ideal, c_list in regimes[k_].items():
                        # Calculate group_alloc *once* for this group
                        group_alloc = 0.0
                        group_alloc_indices = set()  # Indices contributing to this group_alloc
                        group_import_indices = set()

                        for c_in_list in c_list:
                            alloc_key = (c_in_list, k_)
                            if alloc_key in idx_alloc:
                                idx = idx_alloc[alloc_key]
                                group_alloc += x[idx]
                                group_alloc_indices.add(idx)
                            if alloc_key in idx_import:
                                idx = idx_import[alloc_key]
                                group_alloc += x[idx]
                                group_import_indices.add(idx)

                        # Calculate proportion_real and the base deviation factor
                        prop_real = group_alloc / denom_k
                        base_diet_deriv = 2.0 * (prop_real - p_ideal)

                        # Calculate d(prop_real)/d(x_j) contributions using the quotient rule components
                        # Derivative Factor = base_diet_deriv * (1 / denom_k**2)
                        # This avoids repeated division inside the loops below
                        quotient_deriv_factor = base_diet_deriv / (denom_k * denom_k)  # denom_k**2 can be large

                        # Term 1: + denom_k * d(group_alloc)/dx[j]
                        # Term 2: - group_alloc * d(denom_k)/dx[j]

                        # Apply derivative contributions to relevant grad entries

                        # d(group_alloc)/dx[j] is 1 ONLY for alloc/import vars in this c_list
                        # Contribution: quotient_deriv_factor * denom_k * (+1)
                        for group_idx in group_alloc_indices.union(group_import_indices):
                            grad[group_idx] += w_diet * quotient_deriv_factor * denom_k  # * (+1 is implicit)

                        # d(denom_k)/dx[j] is 1 for ALL alloc/import vars for this consumer k
                        # Contribution: quotient_deriv_factor * (-group_alloc) * (+1)
                        neg_group_alloc_term = -group_alloc  # Precompute
                        for k_alloc_idx in alloc_indices_k:  # Indices of alloc vars for consumer k
                            grad[k_alloc_idx] += (
                                w_diet * quotient_deriv_factor * neg_group_alloc_term
                            )  # * (+1 implicit)
                        for k_import_idx in import_indices_k:  # Indices of import vars for consumer k
                            grad[k_import_idx] += (
                                w_diet * quotient_deriv_factor * neg_group_alloc_term
                            )  # * (+1 implicit)

                return grad

            def compute_objective_terms(x):
                """
                Return the scalar value of the objective:
                w_diet * diet_deviation
                + w_Nsyn * fertilizer_deviation
                + w_imp * import_export_deviation
                """

                # 3.a) diet_deviation
                total_dev = 0.0
                for k_ in CONSUMERS:
                    # 1) Somme totale allouée (local + imports) pour le consommateur k_
                    denom_k = 0.0
                    for c_ in CROPS:
                        # Allocation locale
                        if (c_, k_) in idx_alloc:
                            denom_k += x[idx_alloc[(c_, k_)]]
                        # Allocation via import
                        if (c_, k_) in idx_import:
                            denom_k += x[idx_import[(c_, k_)]]

                    # 2) Pour chaque proportion idéale, calculez la somme des cultures du groupe
                    for p_ideal, c_list in regimes[k_].items():
                        group_alloc = 0.0
                        for c_ in c_list:
                            # Ajout part locale
                            if (c_, k_) in idx_alloc:
                                group_alloc += x[idx_alloc[(c_, k_)]]
                            # Ajout part importée
                            if (c_, k_) in idx_import:
                                group_alloc += x[idx_import[(c_, k_)]]

                        if denom_k < 1e-6:
                            proportion_real = 0.0
                        else:
                            proportion_real = group_alloc / denom_k

                        # Ecart diététique
                        total_dev += (proportion_real - p_ideal) ** 2

                # 3.b) fertilizer_deviation
                # sum of synthetic fertilizers in ktN
                total_synth = 0.0
                for c in CROPS:
                    # x[idx_synth[c]] is in kgN/ha
                    fert_per_ha = x[idx_synth[c]] + nonSynthFert[c]
                    total_synth += x[idx_synth[c]] * area[c]  # only the synthetic part
                # Convert from kgN to ktN
                total_synth_kt = total_synth / 1e6
                # desired total = (N_synth_crop + N_synth_grass)
                if N_synth_crop + N_synth_grass < 1:
                    scale = 1
                else:
                    scale = N_synth_crop + N_synth_grass
                fert_dev = np.maximum(0, (total_synth_kt - (N_synth_crop + N_synth_grass)) / scale) ** 2
                # fert_dev = ((total_synth_kt - (N_synth_crop + N_synth_grass)) / scale) ** 2
                # 3.c) import_export_deviation
                # sum import
                sum_imp = 0.0
                for c, k in allowed_ck:
                    sum_imp += x[idx_import[(c, k)]]

                # Calcul de la production "non allouée" => export
                export_total = 0.0
                for c in CROPS:
                    # Production locale (ktN) ; on suppose qu'elle est déjà correcte/à jour :
                    if self.prod_func == "Ratio":
                        production_c = (
                            Y_th_ratio(x[idx_synth[c]] + nonSynthFert[c], Ymax[c])
                            * df_cultures.at[c, "Area (ha)"]
                            / 1e6
                        )
                    if self.prod_func == "Linear":
                        production_c = (
                            Y_th_lin(x[idx_synth[c]] + nonSynthFert[c], a[c], b[c])
                            * df_cultures.at[c, "Area (ha)"]
                            / 1e6
                        )
                    if self.prod_func == "Exponential":
                        production_c = (
                            Y.Y_th_exp_cap(x[idx_synth[c]] + nonSynthFert[c], Ymax[c], k[c])
                            * df_cultures.at[c, "Area (ha)"]
                            / 1e6
                        )

                    # Somme des allocations locales sur c
                    allocated_c = 0.0
                    # idx_alloc[(c,k)] = indice pour allocate(c,k)
                    for k in CONSUMERS:
                        if (c, k) in idx_alloc:
                            allocated_c += x[idx_alloc[(c, k)]]

                    # leftover = production - allocated
                    # S'il est > 0 => export, s'il est < 0 => on a sur-alloué (besoin d'import net)
                    leftover_c = production_c - allocated_c
                    export_total += leftover_c

                # Net import du modèle
                net_import_model = sum_imp - export_total

                if abs(net_import) < 1:
                    imp_dev = (net_import_model - net_import) ** 2
                else:
                    imp_dev = ((net_import_model - net_import) / (net_import + 1e-6)) ** 2

                return total_dev, fert_dev, imp_dev, net_import, net_import_model, sum_imp, export_total

            def my_callback(xk):
                # xk is the current solution vector at iteration k
                # Evaluate any terms you want here, e.g.:
                f_val = objective(xk)
                # Possibly store separate sub-terms:
                diet_dev_term, fertilizer_term, imp_term, net_imp, net_import_model, sum_imp, export_total = (
                    compute_objective_terms(xk)
                )
                # Append them to some global or nonlocal list
                iteration_log.append(
                    {
                        # "x": xk.copy(),
                        "objective": f_val,
                        "diet_dev": diet_dev_term,
                        "fert_dev": fertilizer_term,
                        "import term": imp_term,
                        "net import target": net_imp,
                        "net import model": net_import_model,
                        "Import model": sum_imp,
                        "Export model": export_total,
                    }
                )

            # --------------------------------------------------------------------------
            # 4) Define constraints as a list of dicts (SLSQP form)
            # --------------------------------------------------------------------------
            # We'll have:
            #   (i) production_balance(c) = sum_{k} allocate(c,k) + import_export(c) - prod_c == 0
            #   (ii) consumption_rule(k) = sum_{c} allocate(c,k) - ingestion[k] == 0
            #
            # The “(iii) if not authorized => allocate=0” we already excluded from x.
            # --------------------------------------------------------------------------

            def build_min_fraction_constraints_reformulated(BETA=0.05):
                """
                Returns constraint dict for SciPy using a reformulated inequality
                to avoid direct division within the constraint function.

                Constraint Logic: We want the allocation of a crop 'c' within a group 'G'
                for consumer 'k' (alloc_ck) to be at least a certain fraction (BETA)
                of what its proportional production (prod_c / sum_prod_g) would suggest
                applied to the total allocation for that group (group_alloc_kG).

                Original Form: alloc_ck - BETA * (prod_c / sum_prod_g) * group_alloc_kG >= 0
                Reformulated : alloc_ck * sum_prod_g - BETA * prod_c * group_alloc_kG >= 0
                            (for sum_prod_g > 0)
                """

                # --- Helper function to calculate production ---
                # (Avoid recalculating this repeatedly inside the constraint function if possible,
                # but here we need it dependent on x_synth which changes)
                def calculate_production(x_synth_val, c):
                    # Make sure idx_synth[c] exists if you use it here, or adjust logic
                    fert_tot = x_synth_val + nonSynthFert[c]  # Assuming x_ maps directly for synth
                    if self.prod_func == "Ratio":
                        y_ha = Y_th_ratio(fert_tot, df_cultures.loc[c, "Ymax (kgN/ha)"])
                    if self.prod_func == "Linear":
                        y_ha = Y_th_lin(fert_tot, df_cultures.loc[c, "a"], df_cultures.loc[c, "b"])
                    if self.prod_func == "Exponential":
                        y_ha = Y.Y_th_exp_cap(
                            fert_tot, df_cultures.loc[c, "Ymax (kgN/ha)"], df_cultures.loc[c, "k (kgN/ha)"]
                        )
                    # Production in ktN (consistent units assumed)
                    return (y_ha * df_cultures.loc[c, "Area (ha)"]) / 1e6

                # We still need to determine the list of constraints to evaluate consistently
                constraint_tuples = []
                for k_ in CONSUMERS:
                    # regimes[k_] has format {proportion: [crop_list]}
                    # We need the groups (crop_lists) themselves. Using values() is fine.
                    for group_crops in regimes[k_].values():
                        # Ensure group_crops is actually a list/iterable of crop names
                        if not isinstance(group_crops, (list, tuple, set)):
                            # Handle potential errors in regimes structure if needed
                            # print(f"Warning: Expected list of crops for consumer {k_}, got {group_crops}")
                            continue
                        for c in group_crops:
                            # Only add constraints for crops that *can* be allocated (locally or imported)
                            # This avoids creating constraints for crops that can never meet the condition.
                            can_be_allocated = (c, k_) in idx_alloc or (c, k_) in idx_import
                            if can_be_allocated:
                                # Store (consumer, list_of_crops_in_group, specific_crop)
                                constraint_tuples.append(
                                    (k_, tuple(group_crops), c)
                                )  # Use tuple for hashability if needed

                num_constraints = len(constraint_tuples)
                # print(f"Building {num_constraints} min_fraction constraints.")

                # --- The actual constraint function passed to SciPy ---
                def min_fraction_fn(x_):
                    constraints_array = np.zeros(num_constraints, dtype=float)
                    calculated_productions = {}  # Cache production calc within one function call

                    for i, (k_, group_crops, c) in enumerate(constraint_tuples):
                        # 1) Calculate production values for the group
                        sum_prod_g = 0.0
                        prod_values = {}
                        for c2 in group_crops:
                            if c2 not in calculated_productions:
                                # Ensure c2 is a valid crop index/name
                                if c2 in idx_synth:
                                    prod_c2_val = calculate_production(x_[idx_synth[c2]], c2)
                                    calculated_productions[c2] = prod_c2_val
                                else:
                                    # Handle case where crop might be in regimes but not synthesizable (maybe fixed production?)
                                    # For now, assume 0 production if no synth variable
                                    prod_c2_val = 0
                                    calculated_productions[c2] = prod_c2_val

                            prod_c2_val = calculated_productions[c2]
                            prod_values[c2] = prod_c2_val
                            sum_prod_g += prod_c2_val

                        # The specific production for the crop 'c' we're constraining
                        # Use the value already computed and stored in prod_values
                        prod_c_val = prod_values.get(
                            c, 0.0
                        )  # Default to 0 if c wasn't calculable (shouldn't happen based on loop structure)

                        # 2) Calculate total allocation to the group for this consumer
                        group_alloc_kG = 0.0
                        for c2 in group_crops:
                            if (c2, k_) in idx_alloc:
                                group_alloc_kG += x_[idx_alloc[(c2, k_)]]
                            if (c2, k_) in idx_import:
                                group_alloc_kG += x_[idx_import[(c2, k_)]]

                        # 3) Calculate this specific crop's allocation
                        alloc_ck = 0.0
                        if (c, k_) in idx_alloc:
                            alloc_ck += x_[idx_alloc[(c, k_)]]
                        if (c, k_) in idx_import:
                            alloc_ck += x_[idx_import[(c, k_)]]

                        # 4) Apply the reformulated constraint
                        # If total production in the group is negligible, the concept
                        # of proportional allocation breaks down. Make constraint trivially true (>=0).
                        # Also handle the case where there's no allocation to the group.
                        if sum_prod_g < 1e-9 or group_alloc_kG < 1e-9:
                            # LHS must be >= 0. Setting to 0 or a small positive value is safe.
                            constraints_array[i] = 1.0  # Or 0.0, ensures >= 0 is met
                        else:
                            # Reformulated: alloc_ck * sum_prod_g - BETA * prod_c * group_alloc_kG >= 0
                            lhs = alloc_ck * sum_prod_g - BETA * prod_c_val * group_alloc_kG
                            constraints_array[i] = lhs

                    return constraints_array

                return {"type": "ineq", "fun": min_fraction_fn}

            # We will define them in a wrapper that references a global or closure “x_current”
            # so that SLSQP can evaluate.
            # One standard pattern is to define a single function that returns an array of
            # constraints. However, SLSQP (via scipy) also supports a list of constraint dicts.

            def build_constraints():
                cons = []
                # Production >= allocations (ineq)
                for c in CROPS:
                    cons.append({"type": "ineq", "fun": lambda x_, c=c: production_balance_expr(x_, c)})

                # Ingestion = local + import (eq)
                for k_ in CONSUMERS:
                    cons.append({"type": "eq", "fun": lambda x_, k_=k_: consumption_rule_expr(x_, k_)})
                return cons

            def production_balance_expr(x_, c):
                sum_local = 0.0
                for k_ in CONSUMERS:
                    if (c, k_) in idx_alloc:
                        sum_local += x_[idx_alloc[(c, k_)]]

                # production c
                fert_tot = x_[idx_synth[c]] + nonSynthFert[c]
                if self.prod_func == "Ratio":
                    y = Y_th_ratio(fert_tot, Ymax[c])
                if self.prod_func == "Linear":
                    y = Y_th_lin(fert_tot, a[c], b[c])
                if self.prod_func == "Exponential":
                    y = Y.Y_th_exp_cap(fert_tot, Ymax[c], k[c])
                prod_c = (y * area[c]) / 1e6

                return prod_c - sum_local  # doit être >= 0

            def consumption_rule_expr(x_, k_):
                sum_local = 0.0
                sum_import = 0.0
                for c_ in CROPS:
                    if (c_, k_) in idx_alloc:
                        sum_local += x_[idx_alloc[(c_, k_)]]
                    if (c_, k_) in idx_import:  # si vous stockez les imports sous (culture, cons)
                        sum_import += x_[idx_import[(c_, k_)]]

                return sum_local + sum_import - ingestion[k_]

            # --------------------------------------------------------------------------
            # 5) Build bounds
            # --------------------------------------------------------------------------
            #  - synth_fert[c] >= 0
            #  - allocate[c,k] >= 0
            #  - import_export[c] unbounded
            # We must create a bounds array for each variable in x
            # --------------------------------------------------------------------------

            bounds = [None] * n_vars
            for c in CROPS:
                # For synth_fert[c]
                i = idx_synth[c]
                if df_cultures.loc[c, "Category"] == "leguminous":
                    bounds[i] = (0.0, 0.0)  # null
                else:
                    bounds[i] = (0.0, None)  # nonnegative

            for c, k_ in allowed_ck:
                i = idx_alloc[(c, k_)]
                bounds[i] = (0.0, None)  # nonnegative

            for c, k_ in allowed_ck:
                i = idx_import[(c, k_)]
                bounds[i] = (0.0, None)  # nonnegative

            # --------------------------------------------------------------------------
            # 6) Initial guess
            # --------------------------------------------------------------------------
            # You can choose a naive guess, e.g. 0 for everything
            # or some heuristic.
            # --------------------------------------------------------------------------
            x0 = np.array([0.01 for i in range(n_vars)])  # np.zeros(n_vars, dtype=float)
            if self.prod_func in ["Ratio", "Exponential"]:
                x0[: len(df_cultures)] = df_cultures["Ymax (kgN/ha)"].values
            if self.prod_func == "Linear":
                x0[: len(df_cultures)] = df_cultures["b"].values

            # --------------------------------------------------------------------------
            # 7) Call the optimizer
            # --------------------------------------------------------------------------
            cons = build_constraints()
            # min_frac_cons = build_min_fraction_constraints_reformulated()
            # cons.append(min_frac_cons)

            iteration_log = []

            # Stage 1: quick approximate solve
            # res1 = minimize(
            #     fun=objective,
            #     x0=x0,
            #     method="SLSQP",
            #     bounds=bounds,
            #     constraints=cons,
            #     callback=my_callback,
            #     options={"maxiter": 1000, "ftol": 1e-5, "disp": True},
            # )

            # Stage 2: refine from the stage 1 solution with a tighter tolerance
            res = minimize(
                fun=objective,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                callback=my_callback,
                options={"maxiter": 1000, "ftol": 1e-5, "disp": True, "eps": 1e-6},
            )

            self.log = iteration_log

            x_opt = res.x

            # On crée les nouvelles colonnes à zéro par défaut
            df_cultures["Surface Synthetic Fertilizer Use (kgN/ha)"] = 0.0
            df_cultures["Synthetic Fertilizer Use (ktN)"] = 0.0

            # Parcours de toutes les cultures
            for c in df_cultures.index:
                if df_cultures.at[c, "Category"] != "leguminous":
                    # 1) Récupérer la fertilisation synthétique du solveur, en kgN/ha
                    #    -> synth_fert_sol[c] = variable du solveur
                    if c in idx_synth and x_opt[idx_synth[c]] > 1e-1:
                        fert_synth = x_opt[idx_synth[c]]
                    else:
                        fert_synth = 0.0

                    # On met à jour la colonne "Surface Synthetic Fertilizer Use (kgN/ha)"
                    fert_synth_with_volat = (
                        fert_synth * 1 / (0.99 - 0.01 * 0.01)
                    )  # Pour prendre en compte l'azote synthétique volatilizé et non consommé par la plante
                    df_cultures.at[c, "Surface Synthetic Fertilizer Use (kgN/ha)"] = fert_synth_with_volat

                    # 2) Calculer la fertilisation synthétique totale en ktN
                    area_ha = df_cultures.at[c, "Area (ha)"]
                    total_synth_ktN = (fert_synth_with_volat * area_ha) / 1e6  # (kgN/ha * ha) / 1e6 = ktN
                    df_cultures.at[c, "Synthetic Fertilizer Use (ktN)"] = total_synth_ktN

                    # 3) Mettre à jour rendement et production
                    # On calcule la fertilisation totale (synthétique + éventuel non-synthétique)
                    non_synth_fert = df_cultures.at[c, "Surface Non Synthetic Fertilizer Use (kgN/ha)"]
                    fert_tot = fert_synth + non_synth_fert

                    if self.prod_func == "Ratio":
                        y_max = df_cultures.at[c, "Ymax (kgN/ha)"]

                        new_yield = Y_th_ratio(fert_tot, y_max)
                    if self.prod_func == "Linear":
                        a = df_cultures.at[c, "a"]
                        b = df_cultures.at[c, "b"]

                    if self.prod_func == "Eponential":
                        y_max = df_cultures.at[c, "Ymax (kgN/ha)"]
                        k = df_cultures.at[c, "k (kgN/ha)"]

                        new_yield = Y_th_lin(fert_tot, a, b)
                    df_cultures.at[c, "Yield (kgN/ha)"] = new_yield
                    # Production en ktN
                    nitro_prod_ktN = (new_yield * area_ha) / 1e6
                    df_cultures.at[c, "Nitrogen Production (ktN)"] = nitro_prod_ktN

                else:
                    # Pour les légumineuses, on ne touche pas Yield ni Nitrogen Production
                    pass

            ## Azote synthétique volatilisé par les terres
            # Est ce qu'il n'y a que l'azote synthétique qui est volatilisé ?
            coef_volat_NH3 = (
                technical.loc[
                    technical["Variable"] == "NH3 volatilization coefficient for synthetic nitrogen",
                    "Business as usual",
                ].item()
                / 100
            )
            coef_volat_N2O = 0.01

            # 1 % des emissions de NH3 du aux fert. synth sont volatilisées sous forme de N2O
            df_cultures["Volatilized Nitrogen N-NH3 (ktN)"] = (
                df_cultures["Synthetic Fertilizer Use (ktN)"] * 0.99 * coef_volat_NH3
            )
            df_cultures["Volatilized Nitrogen N-N2O (ktN)"] = df_cultures["Synthetic Fertilizer Use (ktN)"] * (
                coef_volat_N2O + 0.01 * coef_volat_NH3
            )
            df_cultures["Synthetic Fertilizer Use (ktN)"] = df_cultures["Synthetic Fertilizer Use (ktN)"] * (
                1 - coef_volat_NH3 - coef_volat_N2O
            )

            source = {"Haber-Bosch": 1}
            target = df_cultures["Synthetic Fertilizer Use (ktN)"].to_dict()

            flux_generator.generate_flux(source, target)

            source = df_cultures["Volatilized Nitrogen N-NH3 (ktN)"].to_dict()
            target = {"NH3 volatilization": 1}

            flux_generator.generate_flux(source, target)

            source = df_cultures["Volatilized Nitrogen N-N2O (ktN)"].to_dict()
            target = {"N2O emission": 1}

            flux_generator.generate_flux(source, target)

            # A cela on ajoute les emissions indirectes de N2O lors de la fabrication des engrais
            # epend_tot_synt = (
            #     df_cultures["Volatilized Nitrogen N-NH3 (ktN)"]
            #     + df_cultures["Volatilized Nitrogen N-N2O (ktN)"]
            #     + df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"]
            # ).sum()
            epend_tot_synt = df_cultures["Synthetic Fertilizer Use (ktN)"].sum()
            coef_emis_N_N2O = (
                technical.loc[
                    technical["Variable"] == "Indirect N2O volatilization coefficient for synthetic nitrogen",
                    "Business as usual",
                ].item()
                / 100
            )
            target = {"N2O emission": 1}
            source = {"Haber-Bosch": epend_tot_synt * coef_emis_N_N2O}

            flux_generator.generate_flux(source, target)

            # Azote issu de la partie non comestible des carcasses
            source_non_comestible = df_elevage["Non Edible Nitrogen (ktN)"].to_dict()
            target_other_sectors = {"other sectors": 1}
            flux_generator.generate_flux(source_non_comestible, target_other_sectors)

            allocations = []

            for (c, k), idx in idx_alloc.items():
                val = x_opt[idx]
                if val > 1e-6:
                    # Décider du "Type" (local feed vs local food)
                    # Par exemple, si 'k' fait partie d'un set df_elevage.index => feed
                    if k in df_elevage.index:
                        type_ = "Local culture feed"
                    else:
                        type_ = "Local culture food"

                    allocations.append(
                        {
                            "Culture": c,
                            "Consumer": k,
                            "Allocated Nitrogen": val,
                            "Type": type_,
                        }
                    )

            # Idem pour les imports (I, E).
            # Import feed:
            for (c, k), idx in idx_import.items():
                val = x_opt[idx]
                if k in df_elevage.index:
                    label = "Feed"
                else:
                    label = "Food"
                if val > 1e-6:
                    allocations.append(
                        {
                            "Culture": c,
                            "Consumer": k,
                            "Allocated Nitrogen": val,
                            "Type": f"Imported {label}",
                        }
                    )

            allocations_df = pd.DataFrame(allocations)
            self.allocations_df = allocations_df
            self.allocation_vege = allocations_df

            df_cultures["Yield (qtl/ha)"] = (
                df_cultures["Yield (kgN/ha)"]
                / (df_cultures["Nitrogen Content (%)"] / 100)
                / 100  # Conversion de kg en qtl (100kg)
            )

            df_cultures["Nitrogen for Feed (ktN)"] = 0.0
            df_cultures["Nitrogen for Food (ktN)"] = 0.0

            group_sums = allocations_df.groupby(["Culture", "Type"])["Allocated Nitrogen"].sum()
            table_sums = group_sums.unstack(fill_value=0.0)
            for c in table_sums.index:
                if "Local culture feed" in table_sums.columns:
                    df_cultures.at[c, "Nitrogen for Feed (ktN)"] = table_sums.at[c, "Local culture feed"]
                if "Local culture food" in table_sums.columns:
                    df_cultures.at[c, "Nitrogen for Food (ktN)"] = table_sums.at[c, "Local culture food"]

            df_cultures["Available Nitrogen After Feed and Food (ktN)"] = (
                df_cultures["Nitrogen Production (ktN)"]
                - df_cultures["Nitrogen for Feed (ktN)"]
                - df_cultures["Nitrogen for Food (ktN)"]
            )

            # Mise à jour de df_elevage

            df_elevage["Consummed Nitrogen from local feed (ktN)"] = 0.0
            df_elevage["Consummed Nitrogen from imported feed (ktN)"] = 0.0

            group_sums = allocations_df.groupby(["Consumer", "Type"])["Allocated Nitrogen"].sum()
            table_sums = group_sums.unstack(fill_value=0.0)

            for k in df_elevage.index:
                if "Local culture feed" in table_sums.columns:
                    df_elevage.at[k, "Consummed Nitrogen from local feed (ktN)"] = table_sums.at[
                        k, "Local culture feed"
                    ]
                if "Imported Feed" in table_sums.columns:
                    df_elevage.at[k, "Consummed Nitrogen from imported feed (ktN)"] = table_sums.at[k, "Imported Feed"]

            deviations_list = []

            for k_ in CONSUMERS:
                # Somme totale (local + import) pour le conso k_
                denom_k = 0.0
                for c_ in CROPS:
                    if (c_, k_) in idx_alloc:
                        denom_k += x_opt[idx_alloc[(c_, k_)]]
                    if (c_, k_) in idx_import:
                        denom_k += x_opt[idx_import[(c_, k_)]]

                for p_ideal, c_list in regimes[k_].items():
                    group_alloc = 0.0
                    for c_ in c_list:
                        if (c_, k_) in idx_alloc:
                            group_alloc += x_opt[idx_alloc[(c_, k_)]]
                        if (c_, k_) in idx_import:
                            group_alloc += x_opt[idx_import[(c_, k_)]]

                    if denom_k < 1e-6:
                        proportion_real = 0.0
                    else:
                        proportion_real = group_alloc / denom_k

                    deviation = proportion_real - p_ideal

                    deviations_list.append(
                        {
                            "Consumer": k_,
                            "Cultures": ", ".join(c_list),
                            "Expected Proportion (%)": p_ideal * 100,
                            "Allocated Proportion (%)": proportion_real * 100,
                            "Deviation (%)": deviation * 100,
                        }
                    )

            df_deviation = pd.DataFrame(deviations_list)
            self.deviations_df = df_deviation

            # Génération des flux pour les cultures locales
            allocations_locales = allocations_df[
                allocations_df["Type"].isin(["Local culture food", "Local culture feed"])
            ]

            for cons in df_cons_vege.index:
                target = {cons: 1}
                source = (
                    allocations_locales[allocations_locales["Consumer"] == cons]
                    .set_index("Culture")["Allocated Nitrogen"]
                    .to_dict()
                )
                if source:
                    flux_generator.generate_flux(source, target)

            # Génération des flux pour les importations
            allocations_imports = allocations_df[
                allocations_df["Type"].isin(["Imported Feed", "Imported Food", "Excess feed imports"])
            ]

            for cons in df_cons_vege.index:
                target = {cons: 1}
                cons_vege_imports = allocations_imports[allocations_imports["Consumer"] == cons]

                # Initialisation d'un dictionnaire pour collecter les flux par catégorie
                flux = {}

                for _, row in cons_vege_imports.iterrows():
                    culture = row["Culture"]
                    azote_alloue = row["Allocated Nitrogen"]

                    # Récupération de la catégorie de la culture
                    categorie = df_cultures.loc[culture, "Category"]

                    # Construction du label source pour l'importation
                    if cons in ["urban", "rural"]:
                        label_source = f"{categorie} food trade"
                    else:
                        label_source = f"{categorie} feed trade"

                    # Accumuler les flux par catégorie
                    if label_source in flux:
                        flux[label_source] += azote_alloue
                    else:
                        flux[label_source] = azote_alloue

                # Génération des flux pour l'élevage
                if sum(flux.values()) > 0:
                    flux_generator.generate_flux(flux, target)

            # On redonne à df_elevage sa forme d'origine et à import_feed_net sa vraie valeur
            # Utiliser `infer_objects(copy=False)` pour éviter l'avertissement sur le downcasting
            df_elevage = df_elevage.combine_first(df_elevage_comp)

            # Remplir les valeurs manquantes avec 0
            df_elevage = df_elevage.fillna(0)

            # Inférer les types pour éviter le warning sur les colonnes object
            df_elevage = df_elevage.infer_objects(copy=False)

            # feed_export = import_feed - import_feed_net

            # flux_exported = {}
            # if feed_export > 10**-6:  # On a importé plus que les imports net, la diff est l'export de feed
            #     feed_export = min(
            #         feed_export,
            #         df_cultures["Available Nitrogen After Feed and Food (ktN)"].sum(),
            #     )  # Patch pour gérer les cas où on a une surexportation (cf Bretagne 2010)
            #     # On distingue les exports de feed prioritaires (prairies et fourrages) au reste
            #     # On distingue le cas où il y a assez dans les exports prioritaires pour couvrir
            #     # les export de feed au cas où il faut en plus exporter les autres cultures (mais d'abord les exports prio)
            #     if (
            #         feed_export
            #         > df_cultures.loc[
            #             df_cultures["Category"].isin(["forages", "temporary meadows"]),
            #             "Available Nitrogen After Feed and Food (ktN)",
            #         ].sum()
            #     ):
            #         feed_export_prio = df_cultures.loc[
            #             df_cultures["Category"].isin(["forages", "temporary meadows"]),
            #             "Available Nitrogen After Feed and Food (ktN)",
            #         ].sum()
            #         feed_export_other = feed_export - feed_export_prio
            #     else:
            #         feed_export_prio = feed_export
            #         feed_export_other = 0
            #     # Répartition de l'azote exporté inutilisé par catégorie
            #     # On fait un premier tour sur les cultures prioritaires
            #     for culture in df_cultures.loc[df_cultures["Category"].isin(["forages", "temporary meadows"])].index:
            #         categorie = df_cultures.loc[df_cultures.index == culture, "Category"].item()
            #         # On exporte pas en feed des catégories dédiées aux humains
            #         if categorie not in ["rice", "fruits and vegetables", "roots"]:
            #             # Calculer la quantité exportée par catégorie proportionnellement aux catégories présentes dans df_cultures
            #             culture_nitrogen_available = df_cultures.loc[df_cultures.index == culture][
            #                 "Available Nitrogen After Feed and Food (ktN)"
            #             ].item()

            #             if culture_nitrogen_available > 0:
            #                 flux_exported[culture] = feed_export_prio * (
            #                     culture_nitrogen_available
            #                     / df_cultures["Available Nitrogen After Feed and Food (ktN)"].sum()
            #                 )

            #     # On écoule le reste des export de feed (si il y en a) sur les autres cultures
            #     if feed_export_other > 10**-6:
            #         for culture in df_cultures.loc[
            #             ~df_cultures["Category"].isin(["forages", "temporary meadows"])
            #         ].index:
            #             categorie = df_cultures.loc[df_cultures.index == culture, "Category"].item()
            #             # On exporte pas en feed des catégories dédiées aux humains
            #             if categorie not in ["rice", "fruits and vegetables", "roots"]:
            #                 # Calculer la quantité exportée par catégorie proportionnellement aux catégories présentes dans df_cultures
            #                 culture_nitrogen_available = df_cultures.loc[df_cultures.index == culture][
            #                     "Available Nitrogen After Feed and Food (ktN)"
            #                 ].item()

            #                 if culture_nitrogen_available > 0:
            #                     flux_exported[culture] = feed_export_prio * (
            #                         culture_nitrogen_available
            #                         / df_cultures["Available Nitrogen After Feed and Food (ktN)"].sum()
            #                     )

            #     # Générer des flux les exportations vers leur catégorie d'origine
            #     for label_source, azote_exported in flux_exported.items():
            #         if azote_exported > 0:
            #             categorie = df_cultures.loc[df_cultures.index == label_source, "Category"].item()
            #             label_target = f"{categorie} feed trade"
            #             target = {label_target: 1}
            #             source = {label_source: azote_exported}
            #             flux_generator.generate_flux(source, target)

        # # Mise à jour du DataFrame avec les quantités exportées
        # df_cultures["Nitrogen Exported For Feed (ktN)"] = df_cultures.index.map(flux_exported).fillna(
        #     0
        # )  # df_cultures.index.map(source).fillna(0)

        # df_cultures["Available Nitrogen After Feed, Export Feed and Food (ktN)"] = (
        #     df_cultures["Available Nitrogen After Feed and Food (ktN)"]
        #     - df_cultures["Nitrogen Exported For Feed (ktN)"]
        # ).apply(lambda x: 0 if abs(x) < 1e-6 else x)

        # import/export food
        # Le surplus est food exporté (ou stocké mais cela ne nous regarde pas)
        for idx, row in df_cultures.iterrows():
            culture = row.name
            categorie = df_cultures.loc[df_cultures.index == culture, "Category"].item()
            if categorie not in ["temporary meadows", "forages", "natural meadows "]:
                source = {
                    culture: df_cultures.loc[
                        df_cultures.index == culture,
                        "Available Nitrogen After Feed and Food (ktN)",
                    ].item()
                }
                target = {f"{categorie} food trade": 1}
            elif (
                culture != "Natural meadow "
            ):  # TODO Que faire des production de feed qui ne sont ni consommées ni exportées ? Pour l'instant on les exporte....
                # NOOOON Il faut les laisser retourner en terre si c'est une prairie naturelle (recommandation de JLN)
                source = {
                    culture: df_cultures.loc[
                        df_cultures.index == culture,
                        "Available Nitrogen After Feed and Food (ktN)",
                    ].item()
                }
                target = {f"{categorie} feed trade": 1}
            else:
                source = {
                    culture: df_cultures.loc[
                        df_cultures.index == culture,
                        "Available Nitrogen After Feed and Food (ktN)",
                    ].item()
                }
                target = {"soil stock": 1}
            flux_generator.generate_flux(source, target)

        # Que faire d'eventuel surplus de prairies ou forage ?
        # Sur recommandation de JLN c'est retourné vers le sol pour les prairies permanentes et exporté pour tout le reste

        ## Usage de l'azote animal pour nourir la population, on pourrait améliorer en distinguant viande, oeufs et lait

        viande_cap = main.loc[
            main["Variable"] == "Edible animal per capita protein ingestion (excl fish)", "Business as usual"
        ].item()
        cons_viande = viande_cap * pop

        # Reflechir a considerer un regime alimentaire carne (national) apres 1960
        if cons_viande < df_elevage["Edible Nitrogen (ktN)"].sum():  # Il y a assez de viande locale
            target = {
                "urban": prop_urb * cons_viande,
                "rural": (1 - prop_urb) * cons_viande,
            }
            source = (df_elevage["Edible Nitrogen (ktN)"] / df_elevage["Edible Nitrogen (ktN)"].sum()).to_dict()
            df_elevage["Net animal nitrogen exports (ktN)"] = df_elevage[
                "Edible Nitrogen (ktN)"
            ] - df_elevage.index.map(source) * sum(target.values())
            flux_generator.generate_flux(source, target)

        else:
            # On commence par consommer tout l'azote disponible
            target = {"urban": prop_urb, "rural": (1 - prop_urb)}
            source = df_elevage["Edible Nitrogen (ktN)"].to_dict()
            flux_generator.generate_flux(source, target)

            cons_viande_import = cons_viande - df_elevage["Edible Nitrogen (ktN)"].sum()
            commerce_path = "FAOSTAT_data_fr_viande_import.csv"
            commerce = pd.read_csv(os.path.join(self.data_path, commerce_path))
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

            target = {
                "urban": prop_urb * cons_viande_import,
                "rural": (1 - prop_urb) * cons_viande_import,
            }
            source = {
                "animal trade": 1
            }  # commerce["Ratio"].to_dict() On peut distinguer les différents types d'azote importé
            flux_generator.generate_flux(source, target)
            # Et on reporte ce qu'il manque dans la colonne "Azote animal exporté net"
            df_elevage["Net animal nitrogen exports (ktN)"] = -commerce["Ratio"] * (cons_viande_import)

        if cons_viande < df_elevage["Edible Nitrogen (ktN)"].sum():
            source = df_elevage["Net animal nitrogen exports (ktN)"].to_dict()
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
                if label != "Natural meadow ":  # 70% de l'excès fini dans les ecosystèmes aquatiques
                    source = {label: -imbalance}
                    # Ajouter soil stock parmis les surplus de fertilisation.
                    target = {
                        "other losses": 0.2925,
                        "hydro-system": 0.7,
                        "N2O emission": 0.0075,
                    }
                else:
                    if (
                        imbalance * 10**6 / df_cultures.loc[df_cultures.index == "Natural meadow ", "Area (ha)"].item()
                        > 100
                    ):  # Si c'est une prairie, l'azote est lessivé seulement au dela de 100 kgN/ha
                        source = {
                            label: -imbalance
                            - 100 * df_cultures.loc[df_cultures.index == "Natural meadow ", "Area (ha)"].item() / 10**6
                        }
                        target = {
                            "other losses": 0.2925,
                            "hydro-system": 0.7,
                            "N2O emission": 0.0075,
                        }
                        flux_generator.generate_flux(source, target)
                        source = {
                            label: 100
                            * df_cultures.loc[df_cultures.index == "Natural meadow ", "Area (ha)"].item()
                            / 10**6
                        }
                        target = {label: 1}
                    else:  # Autrement, l'azote reste dans le sol (cas particulier, est ce que cela a du sens, quid des autres cultures ?)
                        source = {label: -imbalance}
                        target = {"soil stock": 1}
                flux_generator.generate_flux(source, target)
            # Si imbalance == 0, aucun ajustement nécessaire

        # Calcul de imbalance dans df_cultures
        df_cultures["Balance (ktN)"] = (
            df_cultures["Synthetic Fertilizer Use (ktN)"]
            + df_cultures["Total Non Synthetic Fertilizer Use (ktN)"]
            - df_cultures["Nitrogen Production (ktN)"]
            - df_cultures["Volatilized Nitrogen N-NH3 (ktN)"]
            - df_cultures["Volatilized Nitrogen N-N2O (ktN)"]  # Pas de volat sous forme de N2 ?
        )

        # On équilibre Haber-Bosch avec atmospheric N2 pour le faire entrer dans le système
        target = {"Haber-Bosch": adjacency_matrix[:, label_to_index["Haber-Bosch"]].sum()}
        source = {"atmospheric N2": 1}
        flux_generator.generate_flux(source, target)

        # On ajoute une ligne total à df_cultures et df_elevage
        colonnes_a_exclure = [
            "Spreading Rate (%)",
            "Nitrogen Content (%)",
            "Seed input (kt seeds/kt Ymax)",
            "Category",
            "N fixation coef (kgN/kgN)",
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
        df_cultures = pd.concat([df_cultures, total.to_frame().T])

        colonnes_a_exclure = [
            "% edible",
            "% excreted indoors",
            "% excreted indoors as manure",
            "% excreted indoors as slurry",
            "% excreted on grassland",
            "% non edible",
            "%N dairy",
            "N-N2 EM. manure indoor",
            "N-N2 EM. outdoor",
            "N-N2 EM. slurry indoor",
            "N-N2O EM. manure indoor",
            "N-N2O EM. outdoor",
            "N-N2O EM. slurry indoor",
            "N-NH3 EM. manure indoor",
            "N-NH3 EM. outdoor",
            "N-NH3 EM. slurry indoor",
            "Conversion factor (%)",
        ]
        colonnes_a_sommer = df_elevage.columns.difference(colonnes_a_exclure)
        total = df_elevage[colonnes_a_sommer].sum()
        total.name = "Total"
        df_elevage = pd.concat([df_elevage, total.to_frame().T])

        self.df_cultures = df_cultures
        self.df_elevage = df_elevage
        self.adjacency_matrix = adjacency_matrix

    def get_df_culture(self):
        return self.df_cultures

    def get_df_elevage(self):
        return self.df_elevage

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
        _ = self.get_core_matrix()
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

    def imported_nitrogen(self):
        return self.allocation_vege.loc[
            self.allocation_vege["Type"].isin(["Imported Food", "Imported Feed", "Excess feed imports"]),
            "Allocated Nitrogen",
        ].sum()

    def net_imported_plant(self):
        return (
            self.importations_df["Imported Nitrogen (ktN)"].sum()
            - self.df_cultures["Available Nitrogen After Feed and Food (ktN)"].sum()
        )

    def net_imported_animal(self):
        return self.df_elevage["Net animal nitrogen exports (ktN)"].sum()

    def total_plant_production(self):
        return self.df_cultures["Nitrogen Production (ktN)"].sum()

    def cereals_production(self):
        return self.df_cultures.loc[
            self.df_cultures["Category"].isin(["cereals (excluding rice)", "rice"]), "Nitrogen Production (ktN)"
        ].sum()

    def leguminous_production(self):
        return self.df_cultures.loc[
            self.df_cultures["Category"].isin(["leguminous"]), "Nitrogen Production (ktN)"
        ].sum()

    def oleaginous_production(self):
        return self.df_cultures.loc[
            self.df_cultures["Category"].isin(["oleaginous"]), "Nitrogen Production (ktN)"
        ].sum()

    def grassland_and_forages_production(self):
        return self.df_cultures.loc[
            self.df_cultures["Category"].isin(["temporary meadows", "natural meadows forages"]),
            "Nitrogen Production (ktN)",
        ].sum()

    def roots_production(self):
        return self.df_cultures.loc[self.df_cultures["Category"].isin(["roots"]), "Nitrogen Production (ktN)"].sum()

    def fruits_and_vegetable_production(self):
        return self.df_cultures.loc[
            self.df_cultures["Category"].isin(["fruits and vegetables"]), "Nitrogen Production (ktN)"
        ].sum()

    def cereals_production_r(self):
        return (
            self.df_cultures.loc[
                self.df_cultures["Category"].isin(["cereals (excluding rice)", "rice"]), "Nitrogen Production (ktN)"
            ].sum()
            * 100
            / self.total_plant_production()
        )

    def leguminous_production_r(self):
        return (
            self.df_cultures.loc[self.df_cultures["Category"].isin(["leguminous"]), "Nitrogen Production (ktN)"].sum()
            * 100
            / self.total_plant_production()
        )

    def oleaginous_production_r(self):
        return (
            self.df_cultures.loc[self.df_cultures["Category"].isin(["oleaginous"]), "Nitrogen Production (ktN)"].sum()
            * 100
            / self.total_plant_production()
        )

    def grassland_and_forages_production_r(self):
        return (
            self.df_cultures.loc[
                self.df_cultures["Category"].isin(["temporary meadows", "natural meadows ", "forages"]),
                "Nitrogen Production (ktN)",
            ].sum()
            * 100
            / self.total_plant_production()
        )

    def roots_production_r(self):
        return (
            self.df_cultures.loc[self.df_cultures["Category"].isin(["roots"]), "Nitrogen Production (ktN)"].sum()
            * 100
            / self.total_plant_production()
        )

    def fruits_and_vegetable_production_r(self):
        return (
            self.df_cultures.loc[
                self.df_cultures["Category"].isin(["fruits and vegetables"]), "Nitrogen Production (ktN)"
            ].sum()
            * 100
            / self.total_plant_production()
        )

    def animal_production(self):
        return self.df_elevage["Edible Nitrogen (ktN)"].sum()

    def emissions(self):
        return pd.Series(
            {
                "N2O emission": self.adjacency_matrix[:, label_to_index["N2O emission"]].sum()
                * (14 * 2 + 16)
                / (14 * 2),
                "atmospheric N2": self.adjacency_matrix[:, label_to_index["atmospheric N2"]].sum(),
                "NH3 volatilization": self.adjacency_matrix[:, label_to_index["NH3 volatilization"]].sum() * 17 / 14,
            },
            name="Emission",
        ).to_frame()["Emission"]

    def surfaces(self):
        return self.df_cultures["Area (ha)"]

    def N_eff(self):
        return gr.GraphAnalyzer.calculate_Neff(self.adjacency_matrix)

    def C_eff(self):
        return gr.GraphAnalyzer.calculate_Ceff(self.adjacency_matrix)

    def F_eff(self):
        return gr.GraphAnalyzer.calculate_Feff(self.adjacency_matrix)

    def R_eff(self):
        return gr.GraphAnalyzer.calculate_Reff(self.adjacency_matrix)
