import os

import numpy as np
import pandas as pd

from grafs_e.donnees import *
from grafs_e.N_class import DataLoader


class scenario:
    def __init__(self, scenario_path, dataloader=None):
        self.data_path = os.path.join(os.path.dirname(__file__), "data")
        if dataloader is None:
            self.dataloader = DataLoader()
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

        with pd.ExcelWriter(os.path.join(self.scenario_path, name + ".xlsx"), engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
