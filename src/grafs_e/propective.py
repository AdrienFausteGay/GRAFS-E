import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.metrics import r2_score
from tqdm import tqdm

import grafs_e.graphes_objet as gr
from grafs_e.donnees import *
from grafs_e.N_class import DataLoader, FluxGenerator, NitrogenFlowModel


class scenario:
    def __init__(self, scenario_path, dataloader=None):
        self.data_path = os.path.join(os.path.dirname(__file__), "data")
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

    def cap_excretion(self, region, type):
        L = []
        livestock = {
            "bovines": [(1150, 1164), (1196, 1210)],
            "caprines": [(1166, 1168), (1212, 1214)],
            "ovines": [(1170, 1172), (1216, 1218)],
            "porcines": [(1174, 1178), (1220, 1224)],
            "poultry": [(1180, 1190), (1226, 1236)],
            "equines": [(1192, 1193), (1238, 1239)],
        }
        for i in annees_disponibles:
            df = self.dataloader.pre_process_df(i, region)
            heads = df.loc[df["index_excel"].between(livestock[type][0][0], livestock[type][0][1]), region]
            excr_cap = df.loc[df["index_excel"].between(livestock[type][1][0], livestock[type][1][1]), region]
            if heads.sum() == 0:
                L.append(0)
            else:
                L.append(np.dot(heads, excr_cap) / heads.sum())
        return L

    @staticmethod
    def livestock_LU(dataloader, region):
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
        return LU

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

    def generate_crop_tab(self, region):
        df = self.dataloader.pre_process_df(annees_disponibles[-1], region)

        cultures_df = df.loc[df["index_excel"].isin(range(259, 294)), ("nom", region)]
        cultures_df[region] = cultures_df[region] * 100 / cultures_df[region].sum()
        cultures_df.loc[cultures_df["nom"] == "Natural meadow ", region] = None

        df_insert = pd.DataFrame(cultures_df.values, columns=["Cultures", "Area proportion (%)"])
        df_insert["Enforce Area"] = False

        Y_pros = Y(self.dataloader)

        # Y_max = []
        # kl = []
        # for culture in cultures + legumineuses + prairies:
        #     ym, k, _ = Y_pros.fit_Y_exp(culture, self.region)
        #     Y_max.append(int(ym))
        #     kl.append(k)

        def fit_and_store(culture):
            ym, k, _ = Y_pros.fit_Y_exp(culture, region)
            return int(ym), k

        all_cultures = cultures + legumineuses + prairies

        # Lancer le traitement en parallèle
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(executor.map(fit_and_store, all_cultures), total=len(all_cultures), desc="Fitting models")
            )

        Y_max, kl = zip(*results)
        df_insert["Y_max"] = Y_max
        df_insert["k"] = kl
        df_insert["k"] = df_insert["k"].map("{:.2e}".format)
        return df_insert

    def generate_scenario_excel(self, year, region, name):
        self.region = region
        self.year = year
        self.last_data_year = annees_disponibles[-1]
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
        sheets["doc"][None] = None
        sheets["doc"].iloc[14, 1] = name
        sheets["doc"].iloc[15, 1] = region
        sheets["doc"].iloc[16, 1] = year

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

        proj = grouped_proj_pop.loc[grouped_proj_pop["Region"] == region, f"POP_{year}"].item()
        sheets["main"].loc[sheets["main"]["Variable"] == "Population", "Business as usual"] = proj

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
            # sheets["technical"].loc[
            #     sheets["technical"]["Variable"] == "Synth N fertilizer application to cropland",
            #     "Business as usual",
            # ] = self.historic_trend(region, 27)[-1]
            # sheets["technical"].loc[
            #     sheets["technical"]["Variable"] == "Synth N fertilizer application to grassland",
            #     "Business as usual",
            # ] = self.historic_trend(region, 29)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "N recycling rate of human excretion in urban area",
                "Business as usual",
            ] = self.historic_trend(region, 49)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "N recycling rate of human excretion in rural area",
                "Business as usual",
            ] = self.historic_trend(region, 50)[-1]

            # Bovines
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Bovines % excretion on grassland",
                "Business as usual",
            ] = self.historic_trend(region, 1250)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Bovines % excretion in the barn",
                "Business as usual",
            ] = self.historic_trend(region, 1251)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Bovines % excretion in the barn as litter manure",
                "Business as usual",
            ] = self.historic_trend(region, 1252)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Bovines % excretion in the barn as other manure",
                "Business as usual",
            ] = self.historic_trend(region, 1253)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Bovines % excretion in the barn as slurry",
                "Business as usual",
            ] = self.historic_trend(region, 1254)[-1]

            # ovines
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Ovines % excretion on grassland",
                "Business as usual",
            ] = self.historic_trend(region, 1264)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Ovines % excretion in the barn",
                "Business as usual",
            ] = self.historic_trend(region, 1265)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Ovines % excretion in the barn as litter manure",
                "Business as usual",
            ] = self.historic_trend(region, 1266)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Ovines % excretion in the barn as other manure",
                "Business as usual",
            ] = self.historic_trend(region, 1267)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Ovines % excretion in the barn as slurry",
                "Business as usual",
            ] = self.historic_trend(region, 1268)[-1]

            # caprines
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Caprines % excretion on grassland",
                "Business as usual",
            ] = self.historic_trend(region, 1278)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Caprines % excretion in the barn",
                "Business as usual",
            ] = self.historic_trend(region, 1279)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Caprines % excretion in the barn as litter manure",
                "Business as usual",
            ] = self.historic_trend(region, 1280)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Caprines % excretion in the barn as other manure",
                "Business as usual",
            ] = self.historic_trend(region, 1281)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Caprines % excretion in the barn as slurry",
                "Business as usual",
            ] = self.historic_trend(region, 1282)[-1]

            # porcines
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Porcines % excretion on grassland",
                "Business as usual",
            ] = self.historic_trend(region, 1292)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Porcines % excretion in the barn",
                "Business as usual",
            ] = self.historic_trend(region, 1293)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Porcines % excretion in the barn as litter manure",
                "Business as usual",
            ] = self.historic_trend(region, 1294)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Porcines % excretion in the barn as other manure",
                "Business as usual",
            ] = self.historic_trend(region, 1295)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Porcines % excretion in the barn as slurry",
                "Business as usual",
            ] = self.historic_trend(region, 1296)[-1]

            # poultry
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Poultry % excretion on grassland",
                "Business as usual",
            ] = self.historic_trend(region, 1306)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Poultry % excretion in the barn",
                "Business as usual",
            ] = self.historic_trend(region, 1307)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Poultry % excretion in the barn as litter manure",
                "Business as usual",
            ] = self.historic_trend(region, 1308)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Poultry % excretion in the barn as other manure",
                "Business as usual",
            ] = self.historic_trend(region, 1309)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Poultry % excretion in the barn as slurry",
                "Business as usual",
            ] = self.historic_trend(region, 1310)[-1]

            # equines
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Equines % excretion on grassland",
                "Business as usual",
            ] = self.historic_trend(region, 1320)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Equines % excretion in the barn",
                "Business as usual",
            ] = self.historic_trend(region, 1321)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Equines % excretion in the barn as litter manure",
                "Business as usual",
            ] = self.historic_trend(region, 1322)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Equines % excretion in the barn as other manure",
                "Business as usual",
            ] = self.historic_trend(region, 1323)[-1]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Equines % excretion in the barn as slurry",
                "Business as usual",
            ] = self.historic_trend(region, 1244)[-1]

            # LU prop
            LU_prop = self.livestock_LU(self.dataloader, self.region)[annees_disponibles[-1]]
            LU_prop_tot = sum(LU_prop.values())
            LU_prop = {key: value / LU_prop_tot for key, value in LU_prop.items()}
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Bovines LU",
                "Business as usual",
            ] = LU_prop["bovines"]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Ovines LU",
                "Business as usual",
            ] = LU_prop["ovines"]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Caprines LU",
                "Business as usual",
            ] = LU_prop["caprines"]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Porcines LU",
                "Business as usual",
            ] = LU_prop["porcines"]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Poultry LU",
                "Business as usual",
            ] = LU_prop["poultry"]
            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "Equines LU",
                "Business as usual",
            ] = LU_prop["equines"]

            # Historical trend
            sheets["main"].loc[
                sheets["main"]["Variable"] == "Feed nitrogen net import",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.historic_trend(self.region, 1009), self.year, seuil_bas=None)[1][-1]

            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "kgN excreted by bovines head",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.cap_excretion(self.region, "bovines"), self.year)[1][-1]

            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "kgN excreted by ovines head",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.cap_excretion(self.region, "ovines"), self.year)[1][-1]

            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "kgN excreted by caprines head",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.cap_excretion(self.region, "caprines"), self.year)[1][-1]

            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "kgN excreted by porcines head",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.cap_excretion(self.region, "porcines"), self.year)[1][-1]

            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "kgN excreted by poultry head",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.cap_excretion(self.region, "poultry"), self.year)[1][-1]

            sheets["technical"].loc[
                sheets["technical"]["Variable"] == "kgN excreted by equines head",
                "Business as usual",
            ] = self.extrapolate_recent_trend(self.cap_excretion(self.region, "equines"), self.year)[1][-1]

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
            sheets["area"] = self.generate_crop_tab(self.region)

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
        for yr in annees_disponibles:
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
        return f * y_max / (f + y_max)

    @staticmethod
    def Y_th_exp(f, y_max, k):
        return y_max * (1 - np.exp(-f / k))

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
                self.Y_th_exp, F, Y, p0=[Y_max_init, min(F)], bounds=([min(Y), max(F)], [max(Y) * 2, 1])
            )
            Y_max_opt = popt[0]  # 📌 Paramètre ajusté Y_max
            k = popt[1]
        except RuntimeError:
            print("⚠️ Ajustement impossible pour", culture, region)
            Y_max_opt = None  # Retourne None en cas d'échec
            k = None

        Y_th_fitted = self.Y_th_exp(F, Y_max_opt, k) if Y_max_opt is not None else None
        return Y_max_opt, k, Y_th_fitted

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

    def plot_Y_exp(self, culture, region):
        F, Y = self.get_Y(culture, region)
        if len(Y) == 0:
            print(f"no {culture} found in {region}")
            return None
        Y_max, k, _ = self.fit_Y_exp(culture, region)
        F_th = np.linspace(0, 1.05 * max(F), 100)
        Y_th = self.Y_th_exp(F_th, Y_max, k)
        r2 = np.round(r2_score(Y, self.Y_th_exp(F, Y_max, k)), 2)
        plt.figure(figsize=(8, 6))
        plt.plot(F, Y, "o", color="tab:blue", markersize=8, label=f"Historic Data, r2 = {r2}")  # Points et ligne
        plt.plot(F_th, Y_th, label=f"Theoric curve, Y_max = {int(Y_max)}", color="orange", linewidth=4)
        plt.xlim(0, max(F_th))  # Départ de l'axe X à 0
        plt.ylim(0, max(Y))  # Départ de l'axe Y à 0
        plt.gca().set_aspect("equal")  # Échelle identique en ajustant les limites

        plt.xlabel("Fertilization (kgN/ha/yr)", fontsize=12)
        plt.ylabel("Yield (kgN/ha/yr)", fontsize=12)
        # plt.title("Relation entre Fertilisation et Rendement", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.4)  # Grille discrète
        plt.legend()
        plt.show()

    def compare_r2(self, region):
        """
        Compare les coefficients de détermination (R²) pour chaque culture dans une région donnée
        entre les deux fonctions d'ajustement : self.Y_th et self.Y_th_exp.

        Affiche une heatmap des scores R² et ajoute une barre de progression en console.
        """
        r2_values = {"Y_th": {}, "Y_th_exp": {}}

        # Barre de progression avec tqdm
        for culture in tqdm(cultures + prairies, desc="Calcul des R²", unit="culture"):
            F, Y = self.get_Y(culture, region)
            if len(Y) == 0 or len(F) == 0:
                r2_values["Y_th"][culture] = np.nan
                r2_values["Y_th_exp"][culture] = np.nan
            else:
                # Ajustement avec la première fonction (Y_th)
                Y_max_th, _ = self.fit_Y(culture, region)
                if Y_max_th is not None:
                    r2_values["Y_th"][culture] = Y_max_th  # np.round(r2_score(Y, self.Y_th(F, Y_max_th)), 2)
                else:
                    r2_values["Y_th"][culture] = 0
                # Ajustement avec la seconde fonction (Y_th_exp)
                Y_max_exp, k, _ = self.fit_Y_exp(culture, region)
                if Y_max_exp is not None:
                    r2_values["Y_th_exp"][culture] = np.round(
                        Y_max_exp
                    )  # np.round(r2_score(Y, self.Y_th_exp(F, Y_max_exp, k)), 2)
                else:
                    r2_values["Y_th_exp"][culture] = 0

        # Création d'un DataFrame pour la heatmap
        r2_df = np.array([list(r2_values["Y_th"].values()), list(r2_values["Y_th_exp"].values())])

        # Création de la heatmap
        plt.figure(figsize=(10, len(cultures) // 3))
        ax = sns.heatmap(
            r2_df.T,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Ymax"},  # {"label": "R² Score"},
            xticklabels=["Y_th", "Y_th_exp"],
            yticklabels=list(r2_values["Y_th"].keys()),
        )

        plt.title(f"Comparaison des Ymax pour {region}")  # R²
        plt.xlabel("Modèle")
        plt.ylabel("Culture")

        # Affichage de la heatmap
        plt.show()


## % Prospective classes


class CultureData_prospect:
    def __init__(self, main, area, data_path, categories_mapping):
        self.main = main
        self.area = area
        self.data_path = data_path
        self.categories_mapping = categories_mapping
        self.df_cultures = self.create_culture_dataframe()

    def create_culture_dataframe(self):
        crops_index = self.area["Crops"][:-2]

        # Extraire les données de surface
        arable_area = self.main.loc[self.main["Variable"] == "Arable area", "Business as usual"].item()
        grassland_area = self.main.loc[self.main["Variable"] == "Permanent grassland area", "Business as usual"].item()
        area = self.area["Area proportion (%)"][:-3] * arable_area / 100
        area[35] = grassland_area

        # Extraire les taux de surface avec épendage et la teneur en azote des cultures
        epend = pd.read_excel(
            os.path.join(self.data_path, "GRAFS_data.xlsx"),
            usecols=[0, 1],
            sheet_name="Surf N org",
        )
        epend_org = epend.set_index("Culture").to_dict()["Surface recevant N organique maîtrisable"]
        epend_N_content = epend.set_index("Culture").to_dict()["Nitrogen Content (%)"]
        # Créer un DataFrame combiné
        combined_data = {
            "Area (ha)": area,
            "Nitrogen Content (%)": epend_N_content,
            "Spreading Rate (%)": epend_org,
        }

        combined_df = pd.DataFrame(combined_data, index=crops_index)

        # Ajouter la colonne 'catégories' en mappant les cultures sur leurs catégories
        combined_df["Category"] = combined_df.index.map(self.categories_mapping)

        return combined_df


class ElevageData_prospect:
    def __init__(self, main, technical):
        self.main = main
        self.technical = technical
        self.df_elevage = self.create_elevage_dataframe(main, technical)

    def create_elevage_dataframe(self):
        # def add_data(nom, ligne, delta, keys):
        #     # Extraire les données supplémentaires
        #     additional_data = df.loc[[ligne + i * delta - 2 for i in range(6)], ["nom", region]]
        #     additional_dict = dict(zip(keys, additional_data[region].values))
        #     # Ajouter les nouvelles données au DataFrame existant dans l'ordre
        #     for key, value in additional_dict.items():
        #         if value <= 10**-5:
        #             value = 0
        #         combined_df.loc[key, nom] = value

        # Production animale, attention, contrairement au reste, ici on est en kton carcasse
        production_data = df[(df["index_excel"] >= 1017) & (df["index_excel"] <= 1022)][["nom", region]]
        production_dict = production_data.set_index("nom")[region].to_dict()

        gas_em = pd.read_excel(os.path.join(data_path, "GRAFS_data.xlsx"), sheet_name="Volatilisation").set_index(
            "Elevage"
        )

        combined_data = {"Production": production_dict}
        combined_df = pd.DataFrame(combined_data)

        combined_df = combined_df.join(gas_em, how="left")

        add_data("% edible", 1092, 12, ["bovines", "ovines", "porcines", "poultry", "equine"])
        combined_df.loc["caprines", "% edible"] = combined_df["% edible"]["ovines"]
        combined_df["% edible"] = combined_df["% edible"] / 100
        add_data(
            "% non edible",
            1093,
            12,
            ["bovines", "ovines", "porcines", "poultry", "equine"],
        )
        combined_df.loc["caprines", "% non edible"] = combined_df["% non edible"]["ovines"]
        combined_df["% non edible"] = combined_df["% non edible"] / 100

        combined_df = combined_df.fillna(0)
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

        self.culture_data = CultureData_prospect(self.main, self.area, self.data_path, categories_mapping)
        self.elevage_data = ElevageData_prospect(self.main, self.technical)
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
        year = self.year
        region = self.region
        data_loader = self.data_loader
        flux_generator = self.flux_generator
        data = data_loader.pre_process_df(year, region)

        # Calcul de l'azote disponible pour les cultures
        df_cultures["Nitrogen Production (ktN)"] = (
            df_cultures["Crop Production (ktonDFW)"] * df_cultures["Nitrogen Content (%)"] / 100
        )

        # Gestion du cas particulier pour 'Straw'
        cereales = ["Wheat", "Rye", "Barley", "Oat", "Grain maize", "Other cereals"]
        somme_azote_produit_cereales = df_cultures["Nitrogen Production (ktN)"][cereales].sum()
        somme_surface_cereales = df_cultures["Area (ha)"][cereales].sum()
        df_cultures.loc["Straw", "Area (ha)"] = (
            somme_surface_cereales
            * df_cultures.loc["Straw", "Nitrogen Production (ktN)"]
            / somme_azote_produit_cereales
        )
        for cereal in cereales:
            df_cultures.loc[cereal, "Area (ha)"] -= (
                df_cultures.loc["Straw", "Area (ha)"] * df_cultures.loc[cereal, "Area (ha)"] / somme_surface_cereales
            )
        df_cultures.loc["Straw", "Yield (qtl/ha)"] = (
            df_cultures["Crop Production (ktonDFW)"]["Straw"] / df_cultures["Area (ha)"]["Straw"] * 1000
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
            df_cultures.index.isin(["Natural meadow ", "Non-legume temporary meadow", "Alfalfa and clover"]),
            "Area (ha)",
        ].to_dict()
        source_atmosphere = {"Atmospheric deposition": coef_surf / 1e6}
        flux_generator.generate_flux(source_atmosphere, target_prairies)

        # Dépôt sur les terres arables
        Surf_reel = data.loc[data["index_excel"] == 23, region].item()
        Surf = df_cultures.loc[
            ~df_cultures.index.isin(["Natural meadow ", "Non-legume temporary meadow", "Alfalfa and clover"]),
            "Area (ha)",
        ].sum()
        target_arable = (
            df_cultures.loc[
                ~df_cultures.index.isin(
                    [
                        "Natural meadow ",
                        "Non-legume temporary meadow",
                        "Alfalfa and clover",
                    ]
                ),
                "Area (ha)",
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

                vege_prods = df_cultures.at[culture, "Crop Production (ktonDFW)"]
                teneur_en_azote = df_cultures.at[culture, "Nitrogen Content (%)"]
                target_fixation[culture] = vege_prods * teneur_en_azote * coefficient / 100

        source_fixation = {"atmospheric N2": 1}
        flux_generator.generate_flux(source_fixation, target_fixation)
        df_cultures["Symbiotic fixation (ktN)"] = df_cultures.index.map(target_fixation).fillna(0)

        ## Épandage de boue sur les champs
        # Fonction pour calculer la redistribution de boues autour de Paris
        def compute_N_supp():
            if self.region not in ["Eure", "Ile de France", "Eure-et-Loir", "Picardie"]:
                return 0, 0
            if self.region == "Ile de France":
                factor = -0.5
            elif self.region in ["Eure", "Eure-et-Loir"]:
                factor = 0.15
            elif self.region == "Picardie":
                factor = 0.2
            data_IDF = self.data_loader.pre_process_df(year, "Ile de France")
            pop_IDF = data_IDF[data_IDF["index_excel"] == 5]["Ile de France"].item()
            prop_urb_IDF = data_IDF[data_IDF["nom"] == "Urban population"]["Ile de France"].item() / 100
            N_cons_cap_IDF = data_IDF[data_IDF["index_excel"] == 8]["Ile de France"].item()
            N_prop_recy_urb_IDF = (
                data_IDF[data_IDF["nom"] == "N recycling rate of human excretion in urban area"]["Ile de France"].item()
                / 100
            )
            N_prop_recy_rur_IDF = (
                data_IDF[data_IDF["nom"] == "N recycling rate of human excretion in rural area"]["Ile de France"].item()
                / 100
            )
            return (
                factor * pop_IDF * N_cons_cap_IDF * prop_urb_IDF * N_prop_recy_urb_IDF,
                factor * pop_IDF * N_cons_cap_IDF * (1 - prop_urb_IDF) * N_prop_recy_rur_IDF,
            )

        FE_N_N02_em = 0.002
        FE_N_NH3_em = 0.118
        FE_N_N2_em = 0.425
        pop = data[data["index_excel"] == 5][region].item()
        prop_urb = data[data["nom"] == "Urban population"][region].item() / 100
        N_cons_cap = data[data["index_excel"] == 8][region].item()
        N_cap_vege = data[data["index_excel"] == 9][region].item()
        N_cap_viande = data[data["index_excel"] == 10][region].item()
        N_boue = pop * N_cons_cap
        N_vege = pop * N_cap_vege
        N_viande = pop * N_cap_viande
        # Et calcul rapide sur les ingestions de produits de la pêche
        N_fish = N_boue - N_vege - N_viande
        source = {"fishery products": N_fish}
        target = {"urban": prop_urb, "rural": 1 - prop_urb}
        flux_generator.generate_flux(source, target)

        # Revenons aux boues
        # data[data["nom"] == "Total per capita protein ingestion"][region].item() * pop Formule fausse dans PVAR
        # data[data["nom"] == "N Sludges to cropland"][region].item()
        prop_recy_urb = data[data["nom"] == "N recycling rate of human excretion in urban area"][region].item() / 100
        prop_recy_rur = data[data["nom"] == "N recycling rate of human excretion in rural area"][region].item() / 100

        Norm = sum(df_cultures["Area (ha)"] * df_cultures["Spreading Rate (%)"])
        # Création du dictionnaire target
        target_ependage = {
            culture: row["Area (ha)"] * row["Spreading Rate (%)"] / Norm for culture, row in df_cultures.iterrows()
        }

        N_supp_urb, N_supp_rur = compute_N_supp()

        source_boue = {
            "urban": prop_urb * N_boue * prop_recy_urb + N_supp_urb,
            "rural": (1 - prop_urb) * prop_recy_rur * N_boue + N_supp_rur,
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

        # Azote excrété sur prairies
        # Production d'azote comestible

        df_elevage["Edible Nitrogen (ktN)"] = df_elevage["Production"] * df_elevage["% edible"]
        df_elevage.loc["poultry", "Edible Nitrogen (ktN)"] += (
            data[data["index_excel"] == 1023][region].item() * data[data["index_excel"] == 1067][region].item() / 100
        )  # ajout des oeufs
        df_elevage.loc["bovines", "Edible Nitrogen (ktN)"] += (
            data[data["index_excel"] == 1024][region].item() * data[data["index_excel"] == 1068][region].item() / 100
        )  # ajout du lait de vache

        # Plus délicat pour les ovins/caprins car la production de lait est mélangée
        tete_ovins_femelle = data[data["index_excel"] == 1171][region].item()
        tete_caprins_femelle = data[data["index_excel"] == 1167][region].item()
        production_par_tete_caprins = 1000  # kg/tete vu sur internet
        production_par_tete_ovins = 300  # kg/tete vu sur internet
        df_elevage.loc["ovines", "Edible Nitrogen (ktN)"] += (
            0
            if (production_par_tete_ovins * tete_ovins_femelle + production_par_tete_caprins * tete_caprins_femelle)
            == 0
            else data[data["index_excel"] == 1025][region].item()
            * data[data["index_excel"] == 1069][region].item()
            / 100
            * production_par_tete_ovins
            * tete_ovins_femelle
            / (production_par_tete_ovins * tete_ovins_femelle + production_par_tete_caprins * tete_caprins_femelle)
        )  # ajout du lait de brebis
        df_elevage.loc["caprines", "Edible Nitrogen (ktN)"] += (
            0
            if (production_par_tete_ovins * tete_ovins_femelle + production_par_tete_caprins * tete_caprins_femelle)
            == 0
            else data[data["index_excel"] == 1025][region].item()
            * data[data["index_excel"] == 1069][region].item()
            / 100
            * production_par_tete_caprins
            * tete_caprins_femelle
            / (production_par_tete_ovins * tete_ovins_femelle + production_par_tete_caprins * tete_caprins_femelle)
        )  # ajout du lait de brebis

        df_elevage["Non Edible Nitrogen (ktN)"] = df_elevage["Production"] * df_elevage["% non edible"]

        index = [1241 + j for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)]
        selected_data.loc[:, "nom"] = selected_data["nom"].apply(lambda x: x.split()[0])
        selected_data = selected_data.groupby("nom").agg({region: "sum", "index_excel": "first"}).reset_index()

        df_elevage["Excreted nitrogen (ktN)"] = selected_data.set_index("nom")[region]
        df_elevage["Ingestion (ktN)"] = (
            df_elevage["Excreted nitrogen (ktN)"]
            + df_elevage["Edible Nitrogen (ktN)"]
            + df_elevage["Non Edible Nitrogen (ktN)"]
        )

        index = [1250 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excreted on grassland"] = selected_data

        index = [1251 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excreted indoors"] = selected_data

        index = [1252 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excreted indoors as slurry"] = selected_data

        # On ajouter la catégorie other manure dans la catégorie liter manure
        index = [1253 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excreted indoors as slurry"] += selected_data

        index = [1254 + j * 14 for j in range(6)]
        selected_data = data[data["index_excel"].isin(index)][region]
        selected_data.index = df_elevage.index

        df_elevage["% excreted indoors as manure"] = selected_data

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
                df_elevage["% excreted indoors as slurry"] / 100 * df_elevage["N-N2 EM. manure indoor"]
                + df_elevage["% excreted indoors as manure"] / 100 * df_elevage["N-N2 EM. slurry indoor"]
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
                df_elevage["% excreted indoors as slurry"] / 100 * df_elevage["N-NH3 EM. manure indoor"]
                + df_elevage["% excreted indoors as manure"] / 100 * df_elevage["N-NH3 EM. slurry indoor"]
            )
        ).to_dict()

        flux_generator.generate_flux(source, target)

        volat_N2O = (
            0.01
            * df_elevage["Excreted nitrogen (ktN)"]
            * df_elevage["% excreted indoors"]
            / 100
            * (
                df_elevage["% excreted indoors as slurry"] / 100 * df_elevage["N-NH3 EM. manure indoor"]
                + df_elevage["% excreted indoors as manure"] / 100 * df_elevage["N-NH3 EM. slurry indoor"]
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
                df_elevage["% excreted indoors as slurry"] / 100 * df_elevage["N-N2O EM. manure indoor"]
                + df_elevage["% excreted indoors as manure"] / 100 * df_elevage["N-N2O EM. slurry indoor"]
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
        df_cultures["Fertilization Need (ktN/qtl)"] = df_cultures.index.map(besoin_azote)
        df_cultures["Surface Fertilization Need (ktN/ha)"] = (
            df_cultures["Fertilization Need (ktN/qtl)"] * df_cultures["Yield (qtl/ha)"]
        )

        # Fixer manuellement les besoins pour certaines cultures
        df_cultures.loc["Sugar beet", "Surface Fertilization Need (ktN/ha)"] = 220
        df_cultures.loc["Potatoes", "Surface Fertilization Need (ktN/ha)"] = 220
        df_cultures.loc["Other roots", "Surface Fertilization Need (ktN/ha)"] = 220
        df_cultures.loc["Dry vegetables", "Surface Fertilization Need (ktN/ha)"] = 50
        df_cultures.loc["Dry fruits", "Surface Fertilization Need (ktN/ha)"] = 100
        df_cultures.loc["Squash and melons", "Surface Fertilization Need (ktN/ha)"] = 180
        df_cultures.loc["Cabbage", "Surface Fertilization Need (ktN/ha)"] = 300
        df_cultures.loc["Leaves vegetables", "Surface Fertilization Need (ktN/ha)"] = 150
        df_cultures.loc["Fruits", "Surface Fertilization Need (ktN/ha)"] = 100
        df_cultures.loc["Olives", "Surface Fertilization Need (ktN/ha)"] = 80
        df_cultures.loc["Citrus", "Surface Fertilization Need (ktN/ha)"] = 130
        df_cultures.loc["Hemp", "Surface Fertilization Need (ktN/ha)"] = 120
        df_cultures.loc["Flax", "Surface Fertilization Need (ktN/ha)"] = 60
        df_cultures.loc["Non-legume temporary meadow", "Surface Fertilization Need (ktN/ha)"] = 150
        df_cultures.loc["Natural meadow ", "Surface Fertilization Need (ktN/ha)"] = 150
        df_cultures.loc["Rice", "Surface Fertilization Need (ktN/ha)"] = 125
        df_cultures.loc["Forage cabbages", "Surface Fertilization Need (ktN/ha)"] = 300

        df_cultures = df_cultures.fillna(0)

        # Calcul de l'azote épendu par hectare
        def calculer_azote_ependu(culture):
            sources = self.betail + self.Pop + ["atmospheric N2", "Atmospheric deposition", "other sectors"]
            adj_matrix_df = pd.DataFrame(adjacency_matrix, index=self.labels, columns=self.labels)
            return adj_matrix_df.loc[sources, culture].sum()

        df_cultures["Total Non Synthetic Fertilizer Use (ktN)"] = df_cultures.index.map(calculer_azote_ependu)
        df_cultures["Surface Non Synthetic Fertilizer Use (kgN/ha)"] = df_cultures.apply(
            lambda row: row["Total Non Synthetic Fertilizer Use (ktN)"] / row["Area (ha)"] * 10**6
            if row["Area (ha)"] > 0 and row["Total Non Synthetic Fertilizer Use (ktN)"] > 0
            else 0,
            axis=1,
        )

        # Mécanisme d'héritage de l'azote en surplus des légumineuses
        df_cultures["Leguminous Nitrogen Surplus (ktN)"] = 0.0
        df_cultures.loc[self.legumineuses, "Leguminous Nitrogen Surplus (ktN)"] = (
            df_cultures.loc[self.legumineuses, "Total Non Synthetic Fertilizer Use (ktN)"]
            - df_cultures.loc[self.legumineuses, "Nitrogen Production (ktN)"]
        )

        # Distribution du surplus aux céréales
        total_surplus_azote = df_cultures.loc[self.legumineuses, "Leguminous Nitrogen Surplus (ktN)"].sum()
        total_surface_cereales = df_cultures.loc[
            (
                (df_cultures["Category"] == "cereals (excluding rice)")
                | (df_cultures.index.isin(["Straw", "Forage maize"]))
            ),
            "Area (ha)",
        ].sum()
        df_cultures["Leguminous heritage (ktN)"] = 0.0
        df_cultures.loc[
            (
                (df_cultures["Category"] == "cereals (excluding rice)")
                | (df_cultures.index.isin(["Straw", "Forage maize"]))
            ),
            "Leguminous heritage (ktN)",
        ] = (
            df_cultures.loc[
                (
                    (df_cultures["Category"] == "cereals (excluding rice)")
                    | (df_cultures.index.isin(["Straw", "Forage maize"]))
                ),
                "Area (ha)",
            ]
            / total_surface_cereales
            * total_surplus_azote
        )

        # Génération des flux pour l'héritage des légumineuses
        source_leg = (
            df_cultures.loc[df_cultures["Leguminous Nitrogen Surplus (ktN)"] > 0]["Leguminous Nitrogen Surplus (ktN)"]
            / df_cultures["Leguminous Nitrogen Surplus (ktN)"].sum()
        ).to_dict()
        target_leg = df_cultures["Leguminous heritage (ktN)"].to_dict()
        flux_generator.generate_flux(source_leg, target_leg)

        # Calcul de l'azote à épendre
        df_cultures["Raw Surface Synthetic Fertilizer Use (ktN/ha)"] = df_cultures.apply(
            lambda row: row["Surface Fertilization Need (ktN/ha)"]
            - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"]
            - (row["Leguminous heritage (ktN)"] / row["Area (ha)"] * 1e6)
            if row["Area (ha)"] > 0
            else row["Surface Fertilization Need (ktN/ha)"] - row["Surface Non Synthetic Fertilizer Use (kgN/ha)"],
            axis=1,
        )
        df_cultures["Raw Surface Synthetic Fertilizer Use (ktN/ha)"] = df_cultures[
            "Raw Surface Synthetic Fertilizer Use (ktN/ha)"
        ].apply(lambda x: max(x, 0))
        df_cultures["Raw Total Synthetic Fertilizer Use (ktN)"] = (
            df_cultures["Raw Surface Synthetic Fertilizer Use (ktN/ha)"] * df_cultures["Area (ha)"] / 1e6
        )

        # Calcul de la quantité moyenne (kgN) d'azote synthétique épendu par hectare
        # Séparer les données en prairies et champs
        df_prairies = df_cultures[df_cultures.index.isin(prairies)].copy()
        df_champs = df_cultures[df_cultures.index.isin(cultures)].copy()

        moyenne_ponderee_prairies = (
            df_prairies["Raw Surface Synthetic Fertilizer Use (ktN/ha)"] * df_prairies["Area (ha)"]
        ).sum()  # / df_prairies['Surface'].sum()
        moyenne_ponderee_champs = (
            df_champs["Raw Surface Synthetic Fertilizer Use (ktN/ha)"] * df_champs["Area (ha)"]
        ).sum()  # / df_champs['Surface'].sum()

        moyenne_reel_champs = (
            data[data["index_excel"] == 27][region].item() * data[data["index_excel"] == 23][region].item()
        )
        moyenne_reel_prairies = (
            data[data["index_excel"] == 29][region].item() * data[data["index_excel"] == 22][region].item() / 10**6
        )

        df_prairies.loc[:, "Adjusted Total Synthetic Fertilizer Use (ktN)"] = moyenne_reel_prairies
        df_champs.loc[:, "Adjusted Total Synthetic Fertilizer Use (ktN)"] = (
            df_champs["Raw Total Synthetic Fertilizer Use (ktN)"] * moyenne_reel_champs / moyenne_ponderee_champs
        )

        # Mise à jour de df_cultures

        df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"] = (
            df_champs["Adjusted Total Synthetic Fertilizer Use (ktN)"]
            .combine_first(df_prairies["Adjusted Total Synthetic Fertilizer Use (ktN)"])
            .reindex(df_cultures.index, fill_value=0)  # Remplissage des clés manquantes (les légumineuses) avec 0
        )

        ## Azote synthétique volatilisé par les terres
        # Est ce qu'il n'y a que l'azote synthétique qui est volatilisé ?
        coef_volat_NH3 = data[data["index_excel"] == 31][region].item() / 100
        coef_volat_N2O = 0.01

        # 1 % des emissions de NH3 du aux fert. synth sont volatilisées sous forme de N2O
        df_cultures["Volatilized Nitrogen N-NH3 (ktN)"] = (
            df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"] * 0.99 * coef_volat_NH3
        )
        df_cultures["Volatilized Nitrogen N-N20 (ktN)"] = df_cultures[
            "Adjusted Total Synthetic Fertilizer Use (ktN)"
        ] * (coef_volat_N2O + 0.01 * coef_volat_NH3)
        df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"] = df_cultures[
            "Adjusted Total Synthetic Fertilizer Use (ktN)"
        ] * (1 - coef_volat_NH3 - coef_volat_N2O)
        # La quantité d'azote réellement épendue est donc un peu plus faible car une partie est volatilisée

        source = {"Haber-Bosch": 1}
        target = df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"].to_dict()

        flux_generator.generate_flux(source, target)

        source = df_cultures["Volatilized Nitrogen N-NH3 (ktN)"].to_dict()
        target = {"NH3 volatilization": 1}

        flux_generator.generate_flux(source, target)

        source = df_cultures["Volatilized Nitrogen N-N20 (ktN)"].to_dict()
        target = {"N2O emission": 1}

        flux_generator.generate_flux(source, target)

        # A cela on ajoute les emissions indirectes de N2O lors de la fabrication des engrais
        # epend_tot_synt = (
        #     df_cultures["Volatilized Nitrogen N-NH3 (ktN)"]
        #     + df_cultures["Volatilized Nitrogen N-N20 (ktN)"]
        #     + df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"]
        # ).sum()
        epend_tot_synt = df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"].sum()
        coef_emis_N_N2O = data[data["index_excel"] == 32][region].item() / 100
        target = {"N2O emission": 1}
        source = {"Haber-Bosch": epend_tot_synt * coef_emis_N_N2O}

        flux_generator.generate_flux(source, target)

        # Azote issu de la partie non comestible des carcasses
        source_non_comestible = df_elevage["Non Edible Nitrogen (ktN)"].to_dict()
        target_other_sectors = {"other sectors": 1}
        flux_generator.generate_flux(source_non_comestible, target_other_sectors)

        # On va chercher les éventuelles corrections apportées par JLN (=0 si export, donc pas vraiment net...)
        import_feed = self.data_loader.get_import_feed(year, region)
        # Et la valeur net
        import_feed_net = data[data["index_excel"] == 1009][region].item()

        df_elevage_comp = df_elevage.copy()
        df_cons_vege = df_elevage.loc[df_elevage["Ingestion (ktN)"] > 10**-8, "Ingestion (ktN)"]

        # On ajoute l'ingestion humaine
        # Une ligne urban, une ligne rural. Cela simplifiera la distinction de regime si un jour c'est pertinent

        df_cons_vege.loc["urban"] = N_vege * prop_urb
        df_cons_vege.loc["rural"] = N_vege * (1 - prop_urb)

        # On distingue les imports feed et food #TODO dans l'optim
        import_food_net = df_cons_vege.sum() - df_cultures["Nitrogen Production (ktN)"].sum() - import_feed_net
        import_food = max(0, import_food_net)

        supp_export = 0
        if import_feed_net > df_elevage["Ingestion (ktN)"].sum():
            supp_export = import_feed_net - df_elevage["Ingestion (ktN)"].sum()  # On augmentera d'autant les exports
            import_feed_net = df_elevage["Ingestion (ktN)"].sum()

        if len(df_cons_vege) > 0:
            # Dictionnaire enregistrant toutes les cultures présentes dans le régime d'un élevage
            all_cultures_regime = {}
            for cons in df_cons_vege.index:
                cultures_name = set()
                for cultures_liste in regimes[cons].values():
                    cultures_name.update(cultures_liste)
                all_cultures_regime[cons] = cultures_name

            # Initialisation du problème
            prob = LpProblem("Allocation_Azote_Animaux", LpMinimize)

            # Variables de décision pour les allocations
            x_vars = LpVariable.dicts(
                "x",
                [(culture, cons) for culture in df_cultures.index for cons in df_cons_vege.index],
                lowBound=0,
                cat="Continuous",
            )

            # Variable de depassement des importations feed
            E_vars_feed = LpVariable.dicts(
                "E",
                [(cons, culture) for cons in df_cons_vege.index[:-2] for culture in all_cultures_regime[cons]],
                lowBound=0,
                cat="Continuous",
            )

            # Variables de déviation des régimes alimentaires
            delta_vars = LpVariable.dicts(
                "delta",
                [(cons, proportion) for cons in df_cons_vege.index for proportion in regimes[cons].keys()],
                lowBound=0,
                cat=LpContinuous,
            )

            # Variables de pénalité pour la concentration des allocations
            # penalite_vars = LpVariable.dicts(
            #     "penalite",
            #     [(culture, cons) for culture in df_cultures.index for cons in df_cons_vege.index],
            #     lowBound=0,
            #     cat=LpContinuous,
            # )

            # Variables de pénalité pour la distribution au sein des catégories
            penalite_culture_vars = LpVariable.dicts(
                "penalite_culture",
                [
                    (cons, proportion, culture)
                    for cons in df_cons_vege.index
                    for proportion in regimes[cons].keys()
                    for culture in regimes[cons][proportion]
                ],
                lowBound=0,
                cat=LpContinuous,
            )

            # Variables d'importation pour chaque élevage et catégorie
            I_vars_feed = LpVariable.dicts(
                "I",
                [(cons, culture) for cons in df_cons_vege.index[:-2] for culture in all_cultures_regime[cons]],
                lowBound=0,
                cat="Continuous",
            )

            # Variables d'importation pour chaque population et catégorie
            I_vars_food = LpVariable.dicts(
                "I",
                [(cons, culture) for cons in df_cons_vege.index[-2:] for culture in all_cultures_regime[cons]],
                lowBound=0,
                cat="Continuous",
            )

            # # Variables pour capturer les importations associées aux déviations négatives
            # delta_import_vars = LpVariable.dicts(
            #     "delta_import",
            #     [(cons, proportion) for cons in df_cons_vege.index for proportion in regimes[cons].keys()],
            #     lowBound=0,
            #     cat=LpContinuous,
            # )

            # Pondération pour le terme de pénalité
            poids_penalite_deviation = 10

            # poids_penalite = 0  # Ajustez ce poids selon vos préférences

            # Poids pour équilibrer la distribution des cultures dans les categories
            poids_penalite_culture = 0.5  # À ajuster selon vos préférences

            # Définir un poids élevé pour pénaliser les importations
            if int(year) > 1960:
                poids_exces_import = 1.0
                poids_import_food = 1.0
            else:
                poids_exces_import = 1000.0  # Ajustez ce poids selon vos préférences
                poids_import_food = 1000.0

            # poids_delta_import = (
            #     0.5  # Poids supplémentaire pour orienter les importations pour minimiser les fortes déviations
            # )

            prob += (
                poids_penalite_deviation
                * lpSum(
                    delta_vars[(cons, proportion)] for cons in df_cons_vege.index for proportion in regimes[cons].keys()
                )
                # + poids_delta_import
                # * lpSum(
                #     delta_import_vars[(cons, proportion)]
                #     for cons in df_cons_vege.index
                #     for proportion in regimes[cons].keys()
                # )
                # + poids_penalite
                # * lpSum(penalite_vars[(culture, cons)] for culture in df_cultures.index for cons in df_cons_vege.index)
                + poids_penalite_culture
                * lpSum(
                    penalite_culture_vars[(cons, proportion, culture)]
                    for cons in df_cons_vege.index
                    for proportion in regimes[cons].keys()
                    for culture in regimes[cons][proportion]
                )
                + poids_exces_import
                * lpSum(
                    E_vars_feed[(cons, culture)]
                    for cons in df_cons_vege.index[:-2]
                    for culture in all_cultures_regime[cons]
                )
                + poids_import_food
                * lpSum(
                    I_vars_food[(cons, culture)]
                    for cons in df_cons_vege.index[-2:]
                    for culture in all_cultures_regime[cons]
                ),
                "Minimiser_Deviations_Penalties_Et_Excès_Importation",
            )

            # Les besoins en feed sont complétés par la prod locale, l'importation de feed (donnees GRAFS) et un eventuel import excedentaire
            for cons in df_cons_vege.index[:-2]:
                besoin = df_cons_vege.loc[cons]
                prob += (
                    lpSum(x_vars[(culture, cons)] for culture in df_cultures.index)
                    + lpSum(I_vars_feed[(cons, culture)] for culture in all_cultures_regime[cons])
                    + lpSum(E_vars_feed[(cons, culture)] for culture in all_cultures_regime[cons])
                    == besoin,
                    f"Besoin_{cons}",
                )

            # Les besoins en food sont complétés par la production locale et les imports de food
            for cons in df_cons_vege.index[-2:]:
                besoin = df_cons_vege.loc[cons]
                prob += (
                    lpSum(x_vars[(culture, cons)] for culture in df_cultures.index)
                    + lpSum(I_vars_food[(cons, culture)] for culture in all_cultures_regime[cons])
                    == besoin,
                    f"Besoin_{cons}",
                )

            # Cette contrainte assure que la somme de l'azote alloué de chaque culture aux différents types de consommateurs ne dépasse pas l'azote disponible pour cette culture.
            for culture in df_cultures.index:
                azote_disponible = df_cultures.loc[culture, "Nitrogen Production (ktN)"]
                prob += (
                    lpSum(x_vars[(culture, cons)] for cons in df_cons_vege.index) <= azote_disponible,
                    f"Disponibilite_{culture}",
                )

            # interdiction de consommation locale et d'import pour des cultures qui ne sont pas dans le feed regime
            for cons in df_cons_vege.index[:-2]:
                cultures_autorisees = set()
                for cultures_liste in regimes[cons].values():
                    cultures_autorisees.update(cultures_liste)
                for culture in df_cultures.index:
                    if culture not in cultures_autorisees:
                        prob += (
                            x_vars[(culture, cons)] == 0,
                            f"Culture_Non_Autorisee_{culture}_{cons}",
                        )
                        # Vérifier si la variable I_vars existe avant d'ajouter la contrainte
                        if (cons, culture) in I_vars_feed:
                            prob += (
                                I_vars_feed[(cons, culture)] == 0,
                                f"Import_Non_Autorise_{cons}_{culture}",
                            )
                        if (cons, culture) in E_vars_feed:
                            prob += (
                                E_vars_feed[(cons, culture)] == 0,
                                f"Import_excedentaire_Non_Autorise_{cons}_{culture}",
                            )

            # même chose pour food
            for cons in df_cons_vege.index[-2:]:
                cultures_autorisees = set()
                for cultures_liste in regimes[cons].values():
                    cultures_autorisees.update(cultures_liste)
                for culture in df_cultures.index:
                    if culture not in cultures_autorisees:
                        prob += (
                            x_vars[(culture, cons)] == 0,
                            f"Culture_Non_Autorisee_{culture}_{cons}",
                        )
                        # Vérifier si la variable I_vars existe avant d'ajouter la contrainte
                        if (cons, culture) in I_vars_food:
                            prob += (
                                I_vars_food[(cons, culture)] == 0,
                                f"Import_Non_Autorise_{cons}_{culture}",
                            )

            # Ces contraintes calculent les déviations entre les proportions effectives des catégories consommées par chaque élevage et les proportions initiales du régime alimentaire.
            for cons in df_cons_vege.index[:-2]:
                besoin = df_cons_vege.loc[cons]
                for proportion_initiale, cultures_liste in regimes[cons].items():
                    # Azote total des cultures dans la liste
                    azote_cultures = (
                        lpSum(x_vars[(culture, cons)] for culture in cultures_liste if culture in df_cultures.index)
                        + lpSum(I_vars_feed[(cons, culture)] for culture in cultures_liste)
                        + lpSum(E_vars_feed[(cons, culture)] for culture in cultures_liste)
                    )
                    proportion_effective = azote_cultures / besoin
                    # Déviation par rapport à la proportion initiale
                    delta_var = delta_vars[(cons, proportion_initiale)]
                    prob += (
                        proportion_effective - proportion_initiale <= delta_var,
                        f"Deviation_Plus_{cons}_{proportion_initiale}",
                    )
                    prob += (
                        proportion_initiale - proportion_effective <= delta_var,
                        f"Deviation_Moins_{cons}_{proportion_initiale}",
                    )

            # Pareil pour food
            for cons in df_cons_vege.index[-2:]:
                besoin = df_cons_vege.loc[cons]
                for proportion_initiale, cultures_liste in regimes[cons].items():
                    # Azote total des cultures dans la liste
                    azote_cultures = lpSum(
                        x_vars[(culture, cons)] for culture in cultures_liste if culture in df_cultures.index
                    ) + lpSum(I_vars_food[(cons, culture)] for culture in cultures_liste)
                    proportion_effective = azote_cultures / besoin
                    # Déviation par rapport à la proportion initiale
                    delta_var = delta_vars[(cons, proportion_initiale)]
                    prob += (
                        proportion_effective - proportion_initiale <= delta_var,
                        f"Deviation_Plus_{cons}_{proportion_initiale}",
                    )
                    prob += (
                        proportion_initiale - proportion_effective <= delta_var,
                        f"Deviation_Moins_{cons}_{proportion_initiale}",
                    )

            # Les importations normales de feed sont égales aux données de GRAFS
            prob += (
                lpSum(
                    I_vars_feed[(cons, culture)]
                    for cons in df_cons_vege.index[:-2]
                    for culture in all_cultures_regime[cons]
                )
                == import_feed,
                "Limite_Imports_Normaux",
            )

            # # Calcul de l'allocation cible (par exemple, allocation uniforme)
            # for culture in df_cultures.index:
            #     azote_disponible_culture = df_cultures.loc[culture, "Azote disponible"]
            #     allocation_cible = azote_disponible_culture / len(df_cons_vege.index)  # Allocation uniforme
            #     for cons in df_cons_vege.index:
            #         allocation_reelle = x_vars[(culture, cons)]
            #         # Pénalité est la valeur absolue de la différence entre l'allocation réelle et l'allocation cible
            #         prob += (
            #             allocation_reelle - allocation_cible <= penalite_vars[(culture, cons)],
            #             f"Penalite_Plus_{culture}_{cons}",
            #         )
            #         prob += (
            #             allocation_cible - allocation_reelle <= penalite_vars[(culture, cons)],
            #             f"Penalite_Moins_{culture}_{cons}",
            #         )

            # Pénaliser si on nourrit les animaux avec une seule culture dans chaque groupe de proportions
            for cons in df_cons_vege.index[:-2]:
                besoin = df_cons_vege.loc[cons]
                for proportion, cultures_liste in regimes[cons].items():
                    # Allocation totale pour ce groupe de cultures
                    allocation_groupe = (
                        lpSum(x_vars[(culture, cons)] for culture in cultures_liste if culture in df_cultures.index)
                        + lpSum(I_vars_feed[(cons, culture)] for culture in cultures_liste)
                        + lpSum(E_vars_feed[(cons, culture)] for culture in cultures_liste)
                    )
                    # Azote total disponible pour ce groupe de cultures
                    azote_total_groupe = df_cultures.loc[
                        df_cultures.index.isin(cultures_liste),
                        "Nitrogen Production (ktN)",
                    ].sum()
                    if azote_total_groupe > 0:
                        for culture in cultures_liste:
                            if culture in df_cultures.index:
                                azote_disponible_culture = df_cultures.loc[culture, "Nitrogen Production (ktN)"]
                                # Calcul de l'allocation cible proportionnelle à la disponibilité
                                allocation_cible_culture = (
                                    azote_disponible_culture / azote_total_groupe
                                ) * allocation_groupe
                                # Allocation réelle
                                allocation_reelle_culture = x_vars[(culture, cons)]
                                # Pénalités pour la déviation
                                prob += (
                                    allocation_reelle_culture - allocation_cible_culture
                                    <= penalite_culture_vars[(cons, proportion, culture)],
                                    f"Penalite_Culture_Plus_{cons}_{proportion}_{culture}",
                                )
                                prob += (
                                    allocation_cible_culture - allocation_reelle_culture
                                    <= penalite_culture_vars[(cons, proportion, culture)],
                                    f"Penalite_Culture_Moins_{cons}_{proportion}_{culture}",
                                )
                    else:
                        pass

            # Pareil pour les humains
            for cons in df_cons_vege.index[-2:]:
                besoin = df_cons_vege.loc[cons]
                for proportion, cultures_liste in regimes[cons].items():
                    # Allocation totale pour ce groupe de cultures
                    allocation_groupe = lpSum(
                        x_vars[(culture, cons)] for culture in cultures_liste if culture in df_cultures.index
                    ) + lpSum(I_vars_food[(cons, culture)] for culture in cultures_liste)
                    # Azote total disponible pour ce groupe de cultures
                    azote_total_groupe = df_cultures.loc[
                        df_cultures.index.isin(cultures_liste),
                        "Nitrogen Production (ktN)",
                    ].sum()
                    if azote_total_groupe > 0:
                        for culture in cultures_liste:
                            if culture in df_cultures.index:
                                azote_disponible_culture = df_cultures.loc[culture, "Nitrogen Production (ktN)"]
                                # Calcul de l'allocation cible proportionnelle à la disponibilité
                                allocation_cible_culture = (
                                    azote_disponible_culture / azote_total_groupe
                                ) * allocation_groupe
                                # Allocation réelle
                                allocation_reelle_culture = x_vars[(culture, cons)]
                                # Pénalités pour la déviation
                                prob += (
                                    allocation_reelle_culture - allocation_cible_culture
                                    <= penalite_culture_vars[(cons, proportion, culture)],
                                    f"Penalite_Culture_Plus_{cons}_{proportion}_{culture}",
                                )
                                prob += (
                                    allocation_cible_culture - allocation_reelle_culture
                                    <= penalite_culture_vars[(cons, proportion, culture)],
                                    f"Penalite_Culture_Moins_{cons}_{proportion}_{culture}",
                                )
                    else:
                        pass

            # Contrainte pour importer le feed là où les déviations sont les plus importantes
            for cons in df_cons_vege.index[:-2]:
                for proportion, cultures_liste in regimes[cons].items():
                    # Total des importations pour cette proportion
                    azote_importe = lpSum(
                        I_vars_feed[(cons, culture)] + E_vars_feed[(cons, culture)]
                        for culture in cultures_liste
                        if culture in df_cultures.index
                    )
                    # # Lier aux variables de déviation
                    # prob += (
                    #     delta_import_vars[(cons, proportion)]
                    #     >= azote_importe - delta_vars[(cons, proportion)] * df_cons_vege.loc[cons],
                    #     f"Delta_Import_Lien_{cons}_{proportion}",
                    # )

            # Pareil pour les humains
            for cons in df_cons_vege.index[-2:]:
                for proportion, cultures_liste in regimes[cons].items():
                    # Total des importations pour cette proportion
                    azote_importe = lpSum(
                        I_vars_food[(cons, culture)] for culture in cultures_liste if culture in df_cultures.index
                    )
                    # # Lier aux variables de déviation
                    # prob += (
                    #     delta_import_vars[(cons, proportion)]
                    #     >= azote_importe - delta_vars[(cons, proportion)] * df_cons_vege.loc[cons],
                    #     f"Delta_Import_Lien_{cons}_{proportion}",
                    # )

            # Résolution du problème
            prob.solve()

            allocations = []
            for var in prob.variables():
                if var.name.startswith("x") and var.varValue > 0:
                    # Nom de la variable : x_(culture, cons)
                    chaine = str(var)
                    matches = re.findall(r"'([^']*)'", chaine)
                    parts = [match.replace("_", " ").strip() for match in matches]
                    culture = parts[0]
                    # Gestion du tiret dans le nom
                    if culture == "Non legume temporary meadow":
                        culture = "Non-legume temporary meadow"
                    if culture == "Natural meadow":
                        culture = "Natural meadow "
                    cons = parts[1]
                    if any(index in var.name for index in df_elevage.index):
                        Type = "Local culture feed"
                    else:
                        Type = "Local culture food"
                    allocations.append(
                        {
                            "Culture": culture,
                            "Consumer": cons,
                            "Allocated Nitrogen": var.varValue,
                            "Type": Type,
                        }
                    )
                elif var.name.startswith("I") and var.varValue > 0:
                    # Nom de la variable : I_(cons, culture)
                    chaine = str(var)
                    matches = re.findall(r"'([^']*)'", chaine)
                    parts = [match.replace("_", " ").strip() for match in matches]
                    cons = parts[0]
                    culture = parts[1]
                    if culture == "Non legume temporary meadow":
                        culture = "Non-legume temporary meadow"
                    if culture == "Natural meadow":
                        culture = "Natural meadow "
                    if any(index in var.name for index in df_elevage.index):
                        Type = "Imported Feed"
                    else:
                        Type = "Imported Food"
                    allocations.append(
                        {
                            "Culture": culture,
                            "Consumer": cons,
                            "Allocated Nitrogen": var.varValue,
                            "Type": Type,
                        }
                    )

                elif var.name.startswith("E") and var.varValue > 0:
                    # Nom de la variable : E_(cons, culture)
                    chaine = str(var)
                    matches = re.findall(r"'([^']*)'", chaine)
                    parts = [match.replace("_", " ").strip() for match in matches]
                    cons = parts[0]
                    culture = parts[1]
                    if culture == "Non legume temporary meadow":
                        culture = "Non-legume temporary meadow"
                    if culture == "Natural meadow":
                        culture = "Natural meadow "
                    allocations.append(
                        {
                            "Culture": culture,
                            "Consumer": cons,
                            "Allocated Nitrogen": var.varValue,
                            "Type": "Excess feed imports",
                        }
                    )

            allocations_df = pd.DataFrame(allocations)

            # Filtrer les lignes en supprimant celles dont 'Allocated Nitrogen' est très proche de zéro
            allocations_df = allocations_df[allocations_df["Allocated Nitrogen"].abs() >= 1e-6]

            self.allocation_vege = allocations_df

            # Extraction des déviations avec le signe
            deviations = []
            for cons in df_cons_vege.index[:-2]:
                for proportion in regimes[cons].keys():
                    proportion_rounded = round(proportion, 5)
                    delta_var_key = (cons, proportion_rounded)
                    deviation = delta_vars[delta_var_key].varValue
                    if deviation != 0:
                        # Récupérer la liste des cultures associées à cette proportion
                        cultures_liste = regimes[cons][proportion]
                        cultures_str = ", ".join(cultures_liste)

                        # Calcul de l'allocation totale (local et importée)
                        azote_cultures_feed = (
                            sum(
                                x_vars[(culture, cons)].varValue
                                for culture in cultures_liste
                                if (culture, cons) in x_vars
                            )
                            + sum(
                                I_vars_feed[(cons, culture)].varValue
                                for culture in cultures_liste
                                if (cons, culture) in I_vars_feed
                            )
                            + sum(
                                E_vars_feed[(cons, culture)].varValue
                                for culture in cultures_liste
                                if (cons, culture) in E_vars_feed
                            )
                        )
                        besoin_total = df_cons_vege.loc[cons]

                        # Calcul de la proportion effective
                        proportion_effective = azote_cultures_feed / besoin_total if besoin_total > 0 else 0

                        # Déterminer le signe
                        signe = 1 if proportion_effective > proportion else -1

                        deviations.append(
                            {
                                "Consumer": cons,
                                "Expected Proportion (%)": proportion_rounded * 100,
                                "Deviation (%)": signe * round(deviation, 4) * 100,  # Convertir en pourcentage
                                "Porportion Allocated (%)": proportion_rounded * 100
                                + signe * round(deviation, 4) * 100,
                                "Cultures": cultures_str,
                            }
                        )
            for cons in df_cons_vege.index[-2:]:
                for proportion in regimes[cons].keys():
                    proportion_rounded = round(proportion, 5)
                    delta_var_key = (cons, proportion_rounded)
                    deviation = delta_vars[delta_var_key].varValue
                    if deviation != 0:
                        # Récupérer la liste des cultures associées à cette proportion
                        cultures_liste = regimes[cons][proportion]
                        cultures_str = ", ".join(cultures_liste)

                        # Calcul de l'allocation totale (local et importée)
                        azote_cultures_food = sum(
                            x_vars[(culture, cons)].varValue for culture in cultures_liste if (culture, cons) in x_vars
                        ) + sum(
                            I_vars_food[(cons, culture)].varValue
                            for culture in cultures_liste
                            if (cons, culture) in I_vars_food
                        )
                        besoin_total = df_cons_vege.loc[cons]

                        # Calcul de la proportion effective
                        proportion_effective = azote_cultures_food / besoin_total if besoin_total > 0 else 0

                        # Déterminer le signe
                        signe = 1 if proportion_effective > proportion else -1

                        deviations.append(
                            {
                                "Consumer": cons,
                                "Expected Proportion (%)": proportion_rounded * 100,
                                "Deviation (%)": signe * round(deviation, 4) * 100,  # Convertir en pourcentage
                                "Porportion Allocated (%)": proportion_rounded * 100
                                + signe * round(deviation, 4) * 100,
                                "Cultures": cultures_str,
                            }
                        )
            self.deviations_df = pd.DataFrame(deviations)

            # Extraction des importations normales
            importations = []
            for cons in df_cons_vege.index[:-2]:
                for culture in all_cultures_regime[cons]:
                    if (cons, culture) in I_vars_feed:
                        import_value = I_vars_feed[(cons, culture)].varValue
                        if import_value > 0:
                            importations.append(
                                {
                                    "Consumer": cons,
                                    "Culture": culture,
                                    "Type": "Normal feed",
                                    "Imported Nitrogen (ktN)": import_value,
                                }
                            )
            for cons in df_cons_vege.index[-2:]:
                for culture in all_cultures_regime[cons]:
                    if (cons, culture) in I_vars_food:
                        import_value = I_vars_food[(cons, culture)].varValue
                        if import_value > 0:
                            importations.append(
                                {
                                    "Consumer": cons,
                                    "Culture": culture,
                                    "Type": "Normal food",
                                    "Imported Nitrogen (ktN)": import_value,
                                }
                            )

            # Extraction des imports excédentaires
            for cons in df_cons_vege.index[:-2]:
                for culture in all_cultures_regime[cons]:
                    if (cons, culture) in E_vars_feed:
                        excess_value = E_vars_feed[(cons, culture)].varValue
                        if excess_value > 0:
                            importations.append(
                                {
                                    "Consumer": cons,
                                    "Culture": culture,
                                    "Type": "Excédentaire feed",
                                    "Imported Nitrogen (ktN)": excess_value,
                                }
                            )

            # Convertir en DataFrame
            self.importations_df = pd.DataFrame(importations)

            # Calcul de la quantité d'azote importé non utilisée
            azote_importe_alloue = allocations_df[
                allocations_df["Type"].isin(["Imported Feed", "Imported Food", "Excess feed imports"])
            ]["Allocated Nitrogen"].sum()

            # Mise à jour de df_cultures
            for idx, row in df_cultures.iterrows():
                culture = row.name
                azote_alloue = allocations_df[
                    (allocations_df["Culture"] == culture)
                    & (allocations_df["Type"].isin(["Local culture food", "Local culture feed"]))
                ]["Allocated Nitrogen"].sum()
                azote_alloue_feed = allocations_df[
                    (allocations_df["Culture"] == culture) & (allocations_df["Type"] == "Local culture feed")
                ]["Allocated Nitrogen"].sum()
                azote_alloue_food = allocations_df[
                    (allocations_df["Culture"] == culture) & (allocations_df["Type"] == "Local culture food")
                ]["Allocated Nitrogen"].sum()
                df_cultures.loc[idx, "Available Nitrogen After Feed and Food (ktN)"] = (
                    row["Nitrogen Production (ktN)"] - azote_alloue
                )
                df_cultures.loc[idx, "Nitrogen For Feed (ktN)"] = azote_alloue_feed
                df_cultures.loc[idx, "Nitrogen For Food (ktN)"] = azote_alloue_food
            # Correction des valeurs proches de zéro
            df_cultures["Available Nitrogen After Feed and Food (ktN)"] = df_cultures[
                "Available Nitrogen After Feed and Food (ktN)"
            ].apply(lambda x: 0 if abs(x) < 1e-6 else x)
            df_cultures["Nitrogen For Feed (ktN)"] = df_cultures["Nitrogen For Feed (ktN)"].apply(
                lambda x: 0 if abs(x) < 1e-6 else x
            )
            df_cultures["Nitrogen For Food (ktN)"] = df_cultures["Nitrogen For Food (ktN)"].apply(
                lambda x: 0 if abs(x) < 1e-6 else x
            )

            # Mise à jour de df_elevage
            # Calcul de l'azote total alloué à chaque élevage
            azote_alloue_elevage = (
                allocations_df.groupby(["Consumer", "Type"])["Allocated Nitrogen"].sum().unstack(fill_value=0)
            )

            # Sélectionner uniquement les élevages (présents dans la liste `betail`)
            azote_alloue_elevage = azote_alloue_elevage.loc[
                azote_alloue_elevage.index.get_level_values("Consumer").isin(betail)
            ]

            # Ajouter les colonnes d'azote alloué dans df_elevage
            df_elevage.loc[:, "Consummed nitrogen from local feed (ktN)"] = df_elevage.index.map(
                azote_alloue_elevage.get("Local culture feed", pd.Series(0, index=df_elevage.index))
            )
            df_elevage.loc[:, "Consummed Nitrogen from imported feed (ktN)"] = df_elevage.index.map(
                lambda elevage: azote_alloue_elevage.get("Imported Feed", pd.Series(0, index=df_elevage.index)).get(
                    elevage, 0
                )
                + azote_alloue_elevage.get("Excess feed imports", pd.Series(0, index=df_elevage.index)).get(elevage, 0)
            )
            # df_elevage['Azote alloué importations'] = df_elevage.index.map(azote_alloue_elevage['Importation'])

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

            feed_export = import_feed - import_feed_net
            flux_exported = {}
            if feed_export > 10**-6:  # On a importé plus que les imports net, la diff est l'export de feed
                feed_export = min(
                    feed_export,
                    df_cultures["Available Nitrogen After Feed and Food (ktN)"].sum(),
                )  # Patch pour gérer les cas où on a une surexportation (cf Bretagne 2010)
                # On distingue les exports de feed prioritaires (prairies et fourrages) au reste
                # On distingue le cas où il y a assez dans les exports prioritaires pour couvrir
                # les export de feed au cas où il faut en plus exporter les autres cultures (mais d'abord les exports prio)
                if (
                    feed_export
                    > df_cultures.loc[
                        df_cultures["Category"].isin(["forages", "grasslands"]),
                        "Available Nitrogen After Feed and Food (ktN)",
                    ].sum()
                ):
                    feed_export_prio = df_cultures.loc[
                        df_cultures["Category"].isin(["forages", "grasslands"]),
                        "Available Nitrogen After Feed and Food (ktN)",
                    ].sum()
                    feed_export_other = feed_export - feed_export_prio
                else:
                    feed_export_prio = feed_export
                    feed_export_other = 0
                # Répartition de l'azote exporté inutilisé par catégorie
                # On fait un premier tour sur les cultures prioritaires
                for culture in df_cultures.loc[df_cultures["Category"].isin(["forages", "grasslands"])].index:
                    categorie = df_cultures.loc[df_cultures.index == culture, "Category"].item()
                    # On exporte pas en feed des catégories dédiées aux humains
                    if categorie not in ["rice", "fruits and vegetables", "roots"]:
                        # Calculer la quantité exportée par catégorie proportionnellement aux catégories présentes dans df_cultures
                        culture_nitrogen_available = df_cultures.loc[df_cultures.index == culture][
                            "Available Nitrogen After Feed and Food (ktN)"
                        ].item()

                        if culture_nitrogen_available > 0:
                            flux_exported[culture] = feed_export_prio * (
                                culture_nitrogen_available
                                / df_cultures["Available Nitrogen After Feed and Food (ktN)"].sum()
                            )

                # On écoule le reste des export de feed (si il y en a) sur les autres cultures
                if feed_export_other > 10**-6:
                    for culture in df_cultures.loc[~df_cultures["Category"].isin(["forages", "grasslands"])].index:
                        categorie = df_cultures.loc[df_cultures.index == culture, "Category"].item()
                        # On exporte pas en feed des catégories dédiées aux humains
                        if categorie not in ["rice", "fruits and vegetables", "roots"]:
                            # Calculer la quantité exportée par catégorie proportionnellement aux catégories présentes dans df_cultures
                            culture_nitrogen_available = df_cultures.loc[df_cultures.index == culture][
                                "Available Nitrogen After Feed and Food (ktN)"
                            ].item()

                            if culture_nitrogen_available > 0:
                                flux_exported[culture] = feed_export_prio * (
                                    culture_nitrogen_available
                                    / df_cultures["Available Nitrogen After Feed and Food (ktN)"].sum()
                                )

                # Générer des flux les exportations vers leur catégorie d'origine
                for label_source, azote_exported in flux_exported.items():
                    if azote_exported > 0:
                        categorie = df_cultures.loc[df_cultures.index == label_source, "Category"].item()
                        label_target = f"{categorie} feed trade"
                        target = {label_target: 1}
                        source = {label_source: azote_exported}
                        flux_generator.generate_flux(source, target)

        # Mise à jour du DataFrame avec les quantités exportées
        df_cultures["Nitrogen Exported For Feed (ktN)"] = df_cultures.index.map(flux_exported).fillna(
            0
        )  # df_cultures.index.map(source).fillna(0)

        df_cultures["Available Nitrogen After Feed, Export Feed and Food (ktN)"] = (
            df_cultures["Available Nitrogen After Feed and Food (ktN)"]
            - df_cultures["Nitrogen Exported For Feed (ktN)"]
        ).apply(lambda x: 0 if abs(x) < 1e-6 else x)

        # import/export food
        # Le surplus est food exporté (ou stocké mais cela ne nous regarde pas)
        for idx, row in df_cultures.iterrows():
            culture = row.name
            categorie = df_cultures.loc[df_cultures.index == culture, "Category"].item()
            if categorie not in ["grasslands", "forages"]:
                source = {
                    culture: df_cultures.loc[
                        df_cultures.index == culture,
                        "Available Nitrogen After Feed, Export Feed and Food (ktN)",
                    ].item()
                }
                target = {f"{categorie} food trade": 1}
                flux_generator.generate_flux(source, target)

        # Que faire d'eventuel surplus de prairies ou forage ? Pour l'instant on les ignores... Ou alors vers soil stock ?

        ## Usage de l'azote animal pour nourir la population, on pourrait améliorer en distinguant viande, oeufs et lait

        viande_cap = data[data["index_excel"] == 10][region].item()
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
            commerce = pd.read_csv(os.path.join(self.data_loader.data_path, commerce_path))
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
                            "N20 emission": 0.0075,
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
            df_cultures["Adjusted Total Synthetic Fertilizer Use (ktN)"]
            + df_cultures["Total Non Synthetic Fertilizer Use (ktN)"]
            + df_cultures["Leguminous heritage (ktN)"]
            - df_cultures["Leguminous Nitrogen Surplus (ktN)"]
            - df_cultures["Nitrogen Production (ktN)"]
            - df_cultures["Volatilized Nitrogen N-NH3 (ktN)"]
            - df_cultures["Volatilized Nitrogen N-N20 (ktN)"]  # Pas de volat sous forme de N2 ?
        )

        # On équilibre Haber-Bosch avec atmospheric N2 pour le faire entrer dans le système
        target = {"Haber-Bosch": adjacency_matrix[:, label_to_index["Haber-Bosch"]].sum()}
        source = {"atmospheric N2": 1}
        flux_generator.generate_flux(source, target)

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
            self.df_cultures["Category"].isin(["grasslands", "forages"]), "Nitrogen Production (ktN)"
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
                self.df_cultures["Category"].isin(["grasslands", "forages"]), "Nitrogen Production (ktN)"
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
