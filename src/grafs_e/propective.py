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
            L.append(np.dot(heads, excr_cap) / heads.sum())
        return L

    def livestock_prop(self, region):
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
            df = self.dataloader.pre_process_df(i, region)
            LU[i] = {}
            for type in livestock.keys():
                heads = df.loc[df["index_excel"].between(livestock[type][0][0], livestock[type][0][1]), region]
                lu_coef = lu_list[livestock[type][1][0] : livestock[type][1][1]]
                LU[i][type] = np.dot(heads, lu_coef)
        return LU

    def generate_scenario_excel(self, year, region, name):
        self.region = region
        self.year = year
        self.last_data_year = annees_disponibles[-1]
        try:
            self.data = self.dataloader.pre_process_df(self.last_data_year, region)
        except:
            self.data = None
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

        with pd.ExcelWriter(os.path.join(self.scenario_path, name + ".xlsx"), engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
