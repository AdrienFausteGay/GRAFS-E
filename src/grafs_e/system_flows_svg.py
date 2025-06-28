# %%
import re
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from lxml import etree

from grafs_e.donnees import *
from grafs_e.N_class import DataLoader
from grafs_e.typology import compute_mean_matrice, get_matrices, merge_nodes, plot_dendrogram

# %%
# etree.register_namespace('', "http://www.w3.org/2000/svg")
run = False
if run == True:
    data = DataLoader()
    matrices, norm_matrices = get_matrices(data)
    df_plot = plot_dendrogram(norm_matrices)
    mean_matrices_by_cluster, global_mean_matrices = compute_mean_matrice(norm_matrices, df_plot)
# %%
if run == True:
    mean_matrices_by_cluster_merged = {}
    for cluster in mean_matrices_by_cluster.keys():
        mean_matrices_by_cluster_merged[cluster], new_labels, _ = merge_nodes(
            mean_matrices_by_cluster[cluster],
            labels,
            merges={
                "cereals (excluding rice)": [
                    "Wheat",
                    "Rye",
                    "Barley",
                    "Oat",
                    "Grain maize",
                    "Rice",
                    "Other cereals",
                ],
                "fruits and vegetables": [
                    "Dry vegetables",
                    "Dry fruits",
                    "Squash and melons",
                    "Cabbage",
                    "Leaves vegetables",
                    "Fruits",
                    "Olives",
                    "Citrus",
                ],
                "leguminous": [
                    "Horse beans and faba beans",
                    "Peas",
                    "Other protein crops",
                    "Green peas",
                    "Dry beans",
                    "Green beans",
                    "Soybean",
                ],
                "oleaginous": ["Rapeseed", "Sunflower", "Other oil crops", "Flax", "Hemp"],
                "forages": [
                    "Forage maize",
                    "Forage cabbages",
                    "Straw",
                ],
                "temporary meadows": ["Non-legume temporary meadow", "Alfalfa and clover"],
                "natural meadows ": ["Natural meadow "],
                "trade": [
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
                ],
                "ruminants": ["bovines", "ovines", "caprines", "equine"],
                "monogastrics": ["porcines", "poultry"],
                "population": ["urban", "rural"],
                "Environment": [
                    "NH3 volatilization",
                    "N2O emission",
                    "hydro-system",
                    "other losses",
                ],
                "roots": ["Sugar beet", "Potatoes", "Other roots"],
            },
        )

# %% lire les id


def list_svg_paths(svg_path):
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_path, parser)
    root = tree.getroot()
    ns = root.nsmap.get(None)

    paths = root.xpath("//svg:path", namespaces={"svg": ns})
    print(f"Nombre total de paths : {len(paths)}")
    for path in paths:
        path_id = path.get("id", None)
        print(f"id: {path_id}")


# Exemple d'usage
# list_svg_paths("C:/Users/faustega/Documents/These/Typologie/test schéma/system_flows.svg")
# %% Dictionnaire des correspondances flux et path (svg)
new_labels = [
    "Environment",
    "Haber-Bosch",
    "atmospheric N2",
    "cereals (excluding rice)",
    "fishery products",
    "forages",
    "fruits and vegetables",
    "green waste",
    "leguminous",
    "methaniser",
    "monogastrics",
    "natural meadows ",
    "oleaginous",
    "other sectors",
    "population",
    "roots",
    "ruminants",
    "soil stock",
    "temporary meadows",
    "trade",
]
boxes_labels = {new_labels[i]: int(i) for i in range(len(new_labels))}

mapping_svg_fluxes_hist = {
    "path12-2": [(16, 14), (16, 19), (16, 13)],
    "path15": [(11, 17), (11, 0)],
    "path17": [(19, 10)],
    "path23": [(16, 18)],
    "path26": [(11, 17)],
    "path27": [(11, 0)],
    "path29": [(10, 19), (10, 14), (10, 13)],
    "path31": [(16, 11)],
    "path32": [(11, 16)],
    "path34": [(16, 0), (16, 2), (16, 3), (16, 5), (16, 6), (16, 9), (16, 12), (16, 15)],
    "path35": [
        (10, 0),
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 9),
        (10, 11),
        (10, 12),
        (10, 15),
        (10, 18),
        (16, 0),
        (16, 2),
        (16, 3),
        (16, 5),
        (16, 6),
        (16, 9),
        (16, 12),
        (16, 15),
        (14, 0),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 11),
        (14, 12),
        (14, 15),
        (14, 18),
    ],
    "path36": [(10, 0), (16, 0), (14, 0)],
    "path37": [(14, 0), (14, 2), (14, 3), (14, 5), (14, 6), (14, 11), (14, 12), (14, 15), (14, 18)],
    "path39": [(3, 16), (5, 16), (8, 16), (18, 16)],
    "path40": [(3, 14), (6, 14), (8, 14), (10, 14), (12, 14), (15, 14), (16, 14)],
    "path41": [(3, 10), (8, 10), (12, 10)],
    "path42": [(8, 19), (8, 14), (8, 16), (8, 10)],
    "path43": [(5, 19), (5, 16)],
    "path45": [(3, 10), (3, 14), (3, 16), (3, 19)],
    "path46": [(12, 14), (12, 10), (12, 16), (12, 19)],
    "path47": [(15, 14), (15, 19)],
    "path48": [(6, 14), (6, 19)],
    "path49": [(3, 19), (5, 19), (6, 19), (8, 19), (12, 19), (15, 19), (18, 19)],
    "path50": [(6, 19)],
    "path52": [(3, 19), (5, 19), (8, 19), (12, 19), (15, 19), (18, 19)],
    "path53": [(15, 19)],
    "path54": [(3, 19), (5, 19), (8, 19), (12, 19), (18, 19)],
    "path55": [(12, 19)],
    "path56": [(3, 19), (5, 19), (8, 19), (18, 19)],
    "path57": [(3, 19)],
    "path58": [(5, 19), (8, 19), (18, 19)],
    "path62": [(18, 19), (18, 16)],
    "path65": [(1, 3), (1, 5), (1, 6), (1, 8), (1, 11), (1, 12), (1, 15), (1, 18)],
    "path66": [(1, 3)],
    "path68": [(1, 6)],
    "path69": [(1, 15)],
    "path70": [(1, 6), (1, 15)],
    "path71": [(1, 12)],
    "path72": [(1, 12), (1, 6), (1, 15)],
    "path74": [(0, 3), (0, 6), (0, 8), (0, 5), (0, 11), (0, 12), (0, 15), (0, 18)],
    "path75": [(0, 11), (1, 11), (2, 11), (10, 11), (14, 11), (9, 11)],
    "path76": [(0, 18), (1, 18), (2, 18), (10, 18), (14, 18), (9, 18)],
    "path77": [(1, 5), (1, 18), (1, 11)],
    "path78": [(1, 5)],
    "path79": [(2, 8), (2, 11), (2, 18)],
    "path80": [(2, 8), (0, 8)],
    "path81": [(2, 1)],
    "path83": [(2, 8)],
    "path123": [
        (14, 3),
        (14, 6),
        (14, 12),
        (14, 15),
        (10, 3),
        (10, 6),
        (10, 12),
        (10, 15),
        (16, 3),
        (16, 6),
        (16, 12),
        (16, 15),
    ],
    "path87": [(2, 11)],
    "path88": [(2, 18)],
    "path89": [(2, 11), (2, 18)],
    "path91": [(0, 11)],
    "path92": [(0, 3), (0, 6), (0, 8), (0, 5), (0, 12), (0, 15), (0, 18)],
    "path93": [(0, 18)],
    "path118": [(15, 14)],
    "path119": [(6, 14)],
    "path120": [(12, 14), (12, 16), (12, 10)],
    "path121": [(12, 10)],
    "path122": [(12, 14)],
    "path124": [(14, 5), (10, 5), (16, 5)],
    "path125": [(14, 6), (14, 12), (14, 15), (10, 6), (10, 12), (10, 15), (16, 6), (16, 12), (16, 15)],
    "path126": [(14, 3), (16, 3), (10, 3)],
    "path127": [(14, 6), (14, 15), (10, 6), (10, 15), (16, 6), (16, 15)],
    "path128": [(14, 12), (16, 12), (10, 12)],
    "path129": [(14, 15), (16, 15), (10, 15)],
    "path130": [(14, 6), (16, 6), (10, 6)],
    "path131": [(0, 3), (0, 6), (0, 8), (0, 5), (0, 12), (0, 15)],
    "path132": [(0, 8)],
    "path133": [
        (14, 3),
        (14, 6),
        (14, 12),
        (14, 15),
        (14, 5),
        (10, 3),
        (10, 6),
        (10, 12),
        (10, 15),
        (10, 5),
        (16, 3),
        (16, 6),
        (16, 12),
        (16, 15),
        (16, 5),
    ],
    "path134": [(0, 3), (0, 6), (0, 5), (0, 12), (0, 15)],
    "path135": [(0, 3), (0, 6), (0, 12), (0, 15)],
    "path136": [(0, 6), (0, 12), (0, 15)],
    "path137": [(0, 6), (0, 15)],
    "path138": [(0, 6)],
    "path139": [(0, 15)],
    "path140": [(0, 12)],
    "path141": [(0, 3)],
    "path142": [(0, 5)],
    "path143": [(0, 5), (14, 5), (10, 5), (16, 5)],
    "path144": [(0, 3), (14, 3), (10, 3), (16, 3), (18, 3), (8, 3), (17, 3)],
    "path145": [(0, 12), (14, 12), (10, 12), (16, 12)],
    "path146": [(0, 15), (14, 15), (10, 15), (16, 15)],
    "path147": [[0, 6], (14, 6), (10, 6), (16, 6)],
    "path151": [(1, 18)],
    "path152": [(1, 11)],
    "path153": [(1, 18), (1, 11)],
    "path155": [(16, 14), (16, 19), (16, 13)],
    "path156": [(3, 14), (6, 14), (8, 14), (12, 14), (15, 14)],
    "path157": [(16, 14), (10, 14)],
    "path158": [(10, 19), (16, 19)],
    "path159": [(3, 10), (3, 14), (3, 16)],
    "path160": [(3, 10), (3, 16)],
    "path166": [(3, 14)],
    "path171": [(5, 19)],
    "path172": [(12, 16), (5, 16)],
    "path173": [(12, 16), (5, 16), (3, 16)],
    "path174": [(3, 10)],
    "path175": [(3, 16)],
    "path176": [(3, 10)],
    "path177": [(8, 14)],
    "path178": [(8, 16), (8, 10), (8, 19)],
    "path179": [(8, 19), (18, 19)],
    "path180": [(8, 19)],
    "path181": [(8, 10), (8, 16)],
    "path182": [(8, 16)],
    "path183": [(8, 10)],
    "path184": [(18, 19)],
    "path185": [(18, 16)],
    "path186": [(18, 0)],
    "path187": [(18, 0), (11, 0), (3, 0), (5, 0), (6, 0), (8, 0), (12, 0), (15, 0)],
    "path188": [(19, 16), (19, 10), (19, 14)],
    "path189": [(19, 16)],
    "path190": [(19, 14)],
    "path191": [(17, 11)],
    "path192": [(17, 18), (17, 3), (17, 5), (17, 6), (17, 8), (17, 12), (17, 15)],
    "path193": [(17, 6)],
    "path194": [(17, 3), (17, 5), (17, 8), (17, 12), (17, 15)],
    "path195": [(17, 15)],
    "path196": [(17, 3), (17, 5), (17, 8), (17, 12)],
    "path197": [(17, 12)],
    "path198": [(17, 8), (17, 5), (17, 3)],
    "path199": [(17, 8)],
    "path201": [(17, 12)],
    "path6": [(4, 14)],
    "path7": [(4, 14), (19, 14)],
    "path8": [(5, 16)],
    "path9": [(5, 16)],
    "path10": [(12, 16)],
    "path11": [(12, 14), (12, 10)],
    "path12": [(10, 0), (10, 2), (10, 3), (10, 5), (10, 6), (10, 9), (10, 11), (10, 12), (10, 15), (10, 18)],
    "path13": [
        (10, 0),
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 9),
        (10, 11),
        (10, 12),
        (10, 15),
        (10, 18),
        (14, 0),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 11),
        (14, 12),
        (14, 15),
        (14, 18),
    ],
    "path14": [(14, 2), (16, 2), (10, 2)],
    "path16": [
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 11),
        (10, 12),
        (10, 15),
        (10, 18),
        (16, 2),
        (16, 3),
        (16, 5),
        (16, 6),
        (16, 12),
        (16, 15),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 11),
        (14, 12),
        (14, 15),
        (14, 18),
    ],
    "path18": [(16, 14), (16, 13), (16, 19), (10, 14), (10, 13), (10, 19)],
    "path19": [(16, 13), (16, 19), (10, 13), (10, 19)],
    "path20": [(16, 13), (10, 13)],
    "path21": [(17, 18)],
    "path22": [(17, 3), (17, 5), (17, 6), (17, 8), (17, 12), (17, 15)],
    "path24": [(18, 0), (11, 0)],
    "path25": [(6, 0)],
    "path28": [(18, 0), (11, 0), (6, 0)],
    "path30": [(15, 0)],
    "path33": [(18, 0), (11, 0), (6, 0), (15, 0)],
    "path38": [(3, 0), (12, 0)],
    "path51": [(12, 0)],
    "path59": [(3, 0)],
    "path63": [(18, 0), (11, 0), (3, 0), (6, 0), (12, 0), (15, 0)],
    "path64": [(8, 0)],
    "path67": [(18, 0), (11, 0), (3, 0), (6, 0), (8, 0), (12, 0), (15, 0)],
    "path73": [(5, 0)],
    "path82": [(0, 11), (2, 11), (10, 11), (14, 11), (9, 11)],
    "path84": [(2, 18), (0, 18), (10, 18), (14, 18), (9, 18)],
    "path148": [(18, 3), (18, 5)],
    "path161": [(18, 3), (18, 0), (18, 5)],
    "path162": [(14, 3), (16, 3), (10, 3), (0, 3)],
    "path163": [(18, 3), (18, 5)],
    "path164": [(8, 3), (8, 5)],
    "path168": [(8, 3), (8, 0), (8, 5)],
    "path169": [(8, 3), (18, 3), (17, 3)],
    "path202": [(17, 3), (17, 5)],
    "path203": [(17, 3), (17, 5), (18, 3), (18, 5)],
    "path205": [(0, 5), (14, 5), (10, 5), (16, 5)],
    "path206": [(17, 3), (17, 5)],
    "path44": [(10, 11), (10, 18), (14, 11), (14, 18)],
    "path90": [(10, 11), (14, 11), (9, 11)],
    "path94": [(10, 18), (14, 18), (9, 18)],
    "path149": [(2, 18), (10, 18), (14, 18), (9, 18)],
    "path150": [(2, 11), (10, 11), (14, 11), (9, 11)],
    "path165": [
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 12),
        (10, 15),
        (16, 2),
        (16, 3),
        (16, 5),
        (16, 6),
        (16, 12),
        (16, 15),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 12),
        (14, 15),
    ],
    "path1": [(17, 3), (18, 3), (18, 5), (8, 3), (8, 5), (17, 5)],
    "path2": [(18, 5), (8, 5), (17, 5)],
    "path3": [(18, 5), (8, 5), (17, 5), (0, 5), (14, 5), (10, 5), (16, 5)],
    "path5": [(1, 0)],
}

mapping_svg_fluxes_pros = {
    "path12-2": [(16, 14), (16, 19), (16, 13)],
    "path15": [(11, 17), (11, 0)],
    "path17": [(19, 10)],
    "path23": [(16, 18)],
    "path26": [(11, 17)],
    "path27": [(11, 0)],
    "path29": [(10, 19), (10, 14), (10, 13)],
    "path31": [(16, 11)],
    "path32": [(11, 16)],
    "path34": [(16, 0), (16, 2), (16, 3), (16, 5), (16, 6), (16, 9), (16, 12), (16, 15)],
    "path35": [
        (10, 0),
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 9),
        (10, 11),
        (10, 12),
        (10, 15),
        (10, 18),
        (16, 0),
        (16, 2),
        (16, 3),
        (16, 5),
        (16, 6),
        (16, 9),
        (16, 12),
        (16, 15),
        (14, 0),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 11),
        (14, 12),
        (14, 15),
        (14, 18),
    ],
    "path36": [(10, 0), (16, 0), (14, 0)],
    "path37": [(14, 0), (14, 2), (14, 3), (14, 5), (14, 6), (14, 11), (14, 12), (14, 15), (14, 18)],
    "path39": [(3, 16), (5, 16), (8, 16), (18, 16)],
    "path40": [(3, 14), (6, 14), (8, 14), (10, 14), (12, 14), (15, 14), (16, 14)],
    "path41": [(3, 10), (8, 10), (12, 10)],
    "path42": [(8, 19), (8, 14), (8, 16), (8, 10)],
    "path43": [(5, 19), (5, 16)],
    "path45": [(3, 10), (3, 14), (3, 16), (3, 19)],
    "path46": [(12, 14), (12, 10), (12, 16), (12, 19)],
    "path47": [(15, 14), (15, 19)],
    "path48": [(6, 14), (6, 19)],
    "path49": [(3, 19), (5, 19), (6, 19), (8, 19), (12, 19), (15, 19), (18, 19)],
    "path50": [(6, 19)],
    "path52": [(3, 19), (5, 19), (8, 19), (12, 19), (15, 19), (18, 19)],
    "path53": [(15, 19)],
    "path54": [(3, 19), (5, 19), (8, 19), (12, 19), (18, 19)],
    "path55": [(12, 19)],
    "path56": [(3, 19), (5, 19), (8, 19), (18, 19)],
    "path57": [(3, 19)],
    "path58": [(5, 19), (8, 19), (18, 19)],
    "path62": [(18, 19), (18, 16)],
    "path65": [(1, 3), (1, 5), (1, 6), (1, 8), (1, 11), (1, 12), (1, 15), (1, 18)],
    "path66": [(1, 3)],
    "path68": [(1, 6)],
    "path69": [(1, 15)],
    "path70": [(1, 6), (1, 15)],
    "path71": [(1, 12)],
    "path72": [(1, 12), (1, 6), (1, 15)],
    "path74": [(0, 3), (0, 6), (0, 8), (0, 5), (0, 11), (0, 12), (0, 15), (0, 18)],
    "path75": [(0, 11), (1, 11), (2, 11), (10, 11), (14, 11), (9, 11)],
    "path76": [(0, 18), (1, 18), (2, 18), (10, 18), (14, 18), (9, 18)],
    "path77": [(1, 5), (1, 18), (1, 11)],
    "path78": [(1, 5)],
    "path79": [(2, 8), (2, 11), (2, 18)],
    "path80": [(2, 8), (0, 8)],
    "path81": [(2, 1)],
    "path83": [(2, 8)],
    "path123": [
        (14, 3),
        (14, 6),
        (14, 12),
        (14, 15),
        (10, 3),
        (10, 6),
        (10, 12),
        (10, 15),
        (16, 3),
        (16, 6),
        (16, 12),
        (16, 15),
    ],
    "path87": [(2, 11)],
    "path88": [(2, 18)],
    "path89": [(2, 11), (2, 18)],
    "path91": [(0, 11)],
    "path92": [(0, 3), (0, 6), (0, 8), (0, 5), (0, 12), (0, 15), (0, 18)],
    "path93": [(0, 18)],
    "path118": [(15, 14)],
    "path119": [(6, 14)],
    "path120": [(12, 14), (12, 16), (12, 10)],
    "path121": [(12, 10)],
    "path122": [(12, 14)],
    "path124": [(14, 5), (10, 5), (16, 5)],
    "path125": [(14, 6), (14, 12), (14, 15), (10, 6), (10, 12), (10, 15), (16, 6), (16, 12), (16, 15)],
    "path126": [(14, 3), (16, 3), (10, 3)],
    "path127": [(14, 6), (14, 15), (10, 6), (10, 15), (16, 6), (16, 15)],
    "path128": [(14, 12), (16, 12), (10, 12)],
    "path129": [(14, 15), (16, 15), (10, 15)],
    "path130": [(14, 6), (16, 6), (10, 6)],
    "path131": [(0, 3), (0, 6), (0, 8), (0, 5), (0, 12), (0, 15)],
    "path132": [(0, 8)],
    "path133": [
        (14, 3),
        (14, 6),
        (14, 12),
        (14, 15),
        (14, 5),
        (10, 3),
        (10, 6),
        (10, 12),
        (10, 15),
        (10, 5),
        (16, 3),
        (16, 6),
        (16, 12),
        (16, 15),
        (16, 5),
    ],
    "path134": [(0, 3), (0, 6), (0, 5), (0, 12), (0, 15)],
    "path135": [(0, 3), (0, 6), (0, 12), (0, 15)],
    "path136": [(0, 6), (0, 12), (0, 15)],
    "path137": [(0, 6), (0, 15)],
    "path138": [(0, 6)],
    "path139": [(0, 15)],
    "path140": [(0, 12)],
    "path141": [(0, 3)],
    "path142": [(0, 5)],
    "path143": [(0, 5), (14, 5), (10, 5), (16, 5)],
    "path144": [(0, 3), (14, 3), (10, 3), (16, 3), (18, 3), (8, 3), (17, 3)],
    "path145": [(0, 12), (14, 12), (10, 12), (16, 12)],
    "path146": [(0, 15), (14, 15), (10, 15), (16, 15)],
    "path147": [[0, 6], (14, 6), (10, 6), (16, 6)],
    "path151": [(1, 18)],
    "path152": [(1, 11)],
    "path153": [(1, 18), (1, 11)],
    "path155": [(16, 14), (16, 19), (16, 13)],
    "path156": [(3, 14), (6, 14), (8, 14), (12, 14), (15, 14)],
    "path157": [(16, 14), (10, 14)],
    "path158": [(10, 19), (16, 19)],
    "path159": [(3, 10), (3, 14), (3, 16)],
    "path160": [(3, 10), (3, 16)],
    "path166": [(3, 14)],
    "path171": [(5, 19)],
    "path172": [(12, 16), (5, 16)],
    "path173": [(12, 16), (5, 16), (3, 16)],
    "path174": [(3, 10)],
    "path175": [(3, 16)],
    "path176": [(3, 10)],
    "path177": [(8, 14)],
    "path178": [(8, 16), (8, 10), (8, 19)],
    "path179": [(8, 19), (18, 19)],
    "path180": [(8, 19)],
    "path181": [(8, 10), (8, 16)],
    "path182": [(8, 16)],
    "path183": [(8, 10)],
    "path184": [(18, 19)],
    "path185": [(18, 16)],
    "path186": [(18, 0)],
    "path187": [(18, 0), (11, 0), (3, 0), (5, 0), (6, 0), (8, 0), (12, 0), (15, 0)],
    "path188": [(19, 16), (19, 10), (19, 14)],
    "path189": [(19, 16)],
    "path190": [(19, 14)],
    "path191": [(17, 11)],
    "path192": [(17, 18), (17, 3), (17, 5), (17, 6), (17, 8), (17, 12), (17, 15)],
    "path193": [(17, 6)],
    "path194": [(17, 3), (17, 5), (17, 8), (17, 12), (17, 15)],
    "path195": [(17, 15)],
    "path196": [(17, 3), (17, 5), (17, 8), (17, 12)],
    "path197": [(17, 12)],
    "path198": [(17, 8), (17, 5), (17, 3)],
    "path199": [(17, 8)],
    "path201": [(17, 12)],
    "path6": [(4, 14)],
    "path7": [(4, 14), (19, 14)],
    "path8": [(5, 16)],
    "path9": [(5, 16)],
    "path10": [(12, 16)],
    "path11": [(12, 14), (12, 10)],
    "path12": [(10, 0), (10, 2), (10, 3), (10, 5), (10, 6), (10, 9), (10, 11), (10, 12), (10, 15), (10, 18)],
    "path13": [
        (10, 0),
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 9),
        (10, 11),
        (10, 12),
        (10, 15),
        (10, 18),
        (14, 0),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 11),
        (14, 12),
        (14, 15),
        (14, 18),
    ],
    "path14": [(14, 2), (16, 2), (10, 2)],
    "path16": [
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 11),
        (10, 12),
        (10, 15),
        (10, 18),
        (16, 2),
        (16, 3),
        (16, 5),
        (16, 6),
        (16, 12),
        (16, 15),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 11),
        (14, 12),
        (14, 15),
        (14, 18),
    ],
    "path18": [(16, 14), (16, 13), (16, 19), (10, 14), (10, 13), (10, 19)],
    "path19": [(16, 13), (16, 19), (10, 13), (10, 19)],
    "path20": [(16, 13), (10, 13)],
    "path21": [(17, 18)],
    "path22": [(17, 3), (17, 5), (17, 6), (17, 8), (17, 12), (17, 15)],
    "path24": [(18, 0), (11, 0)],
    "path25": [(6, 0)],
    "path28": [(18, 0), (11, 0), (6, 0)],
    "path30": [(15, 0)],
    "path33": [(18, 0), (11, 0), (6, 0), (15, 0)],
    "path38": [(3, 0), (12, 0)],
    "path51": [(12, 0)],
    "path59": [(3, 0)],
    "path63": [(18, 0), (11, 0), (3, 0), (6, 0), (12, 0), (15, 0)],
    "path64": [(8, 0)],
    "path67": [(18, 0), (11, 0), (3, 0), (6, 0), (8, 0), (12, 0), (15, 0)],
    "path73": [(5, 0)],
    "path82": [(0, 11), (2, 11), (10, 11), (14, 11), (9, 11)],
    "path84": [(2, 18), (0, 18), (10, 18), (14, 18), (9, 18)],
    "path148": [(18, 3), (18, 5)],
    "path161": [(18, 3), (18, 0), (18, 5)],
    "path162": [(14, 3), (16, 3), (10, 3), (0, 3)],
    "path163": [(18, 3), (18, 5)],
    "path164": [(8, 3), (8, 5)],
    "path168": [(8, 3), (8, 0), (8, 5)],
    "path169": [(8, 3), (18, 3), (17, 3)],
    "path202": [(17, 3), (17, 5)],
    "path203": [(17, 3), (17, 5), (18, 3), (18, 5)],
    "path205": [(0, 5), (14, 5), (10, 5), (16, 5)],
    "path206": [(17, 3), (17, 5)],
    "path44": [(10, 11), (10, 18), (14, 11), (14, 18)],
    "path90": [(10, 11), (14, 11), (9, 11)],
    "path94": [(10, 18), (14, 18), (9, 18)],
    "path149": [(2, 18), (10, 18), (14, 18), (9, 18)],
    "path150": [(2, 11), (10, 11), (14, 11), (9, 11)],
    "path165": [
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 12),
        (10, 15),
        (16, 2),
        (16, 3),
        (16, 5),
        (16, 6),
        (16, 12),
        (16, 15),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 12),
        (14, 15),
    ],
    "path1": [(17, 3), (18, 3), (18, 5), (8, 3), (8, 5), (17, 5)],
    "path2": [(18, 5), (8, 5), (17, 5)],
    "path3": [(18, 5), (8, 5), (17, 5), (0, 5), (14, 5), (10, 5), (16, 5)],
    "path5": [(1, 0)],
    "path95": [(5, 9)],
    "path96": [(7, 9)],
    "path97": [
        (14, 3),
        (14, 6),
        (14, 12),
        (14, 15),
        (14, 5),
        (10, 3),
        (10, 6),
        (10, 12),
        (10, 15),
        (10, 5),
        (16, 3),
        (16, 6),
        (16, 12),
        (16, 15),
        (16, 5),
        (9, 3),
        (9, 5),
        (9, 6),
        (9, 12),
        (9, 15),
    ],
    "path99": [
        (9, 3),
        (9, 5),
        (9, 6),
        (9, 12),
        (9, 15),
    ],
    "path154": [(5, 9), (10, 9), (16, 9)],
    "path167": [(10, 9), (16, 9)],
    "path200": [
        (10, 0),
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 6),
        (10, 11),
        (10, 12),
        (10, 15),
        (10, 18),
        (16, 0),
        (16, 2),
        (16, 3),
        (16, 5),
        (16, 6),
        (16, 12),
        (16, 15),
        (14, 0),
        (14, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 11),
        (14, 12),
        (14, 15),
        (14, 18),
    ],
    "path204": [(9, 3), (9, 5), (9, 6), (9, 11), (9, 12), (9, 15), (9, 18)],
    "path207": [(9, 11), (9, 18)],
    "path208": [(10, 11), (10, 18), (14, 11), (14, 18), (9, 11), (9, 18)],
}

# %%


def update_svg_fluxes(
    svg_path,
    output_path,
    flux_matrix,
    mapping_svg_fluxes,
    scale=100,
    inkscape_exe="C:/Program Files/Inkscape/bin/inkscape.exe",
    save=False,
):
    """
    Met à jour les épaisseurs des flux dans le SVG selon la matrice et le mapping.

    Args:
        svg_path (str): chemin vers le SVG original.
        output_path (str): chemin pour enregistrer le SVG modifié.
        flux_matrix (np.array): matrice carrée des flux.
        labels (list): liste des noms des nœuds, alignée avec flux_matrix.
        mapping_svg_fluxes (dict): dict {id_svg_flux: [(i,j), ...]} où (i,j) sont indices dans flux_matrix.
        scale (float): facteur multiplicateur pour l'épaisseur (ajuste selon besoin).

    """
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(svg_path, parser)
    root = tree.getroot()
    ns = root.nsmap.get(None)

    for id_svg, ij_list in mapping_svg_fluxes.items():
        total_flux = sum(flux_matrix[i, j] for i, j in ij_list)
        if total_flux > 0:
            width = total_flux * scale  # Épaisseur minimale à 0.5 pour visibilité
        else:
            width = 0
        xpath = f"//svg:path[@id='{id_svg}']"
        elems = root.xpath(xpath, namespaces={"svg": ns})
        if not elems:
            print(f"⚠️ Élément non trouvé dans SVG : {id_svg}")
            continue

        path = elems[0]
        style = path.attrib.get("style", "")
        if "stroke-width" in style:
            new_style = re.sub(r"stroke-width:[^;]+", f"stroke-width:{width}", style)
        else:
            new_style = style.rstrip(";") + f";stroke-width:{width}"
        path.attrib["style"] = new_style
        # path = elems[0]
        # style = path.attrib.get("style", "")
        # # Mise à jour du stroke-width dans l'attribut style
        # new_style = re.sub(r"stroke-width:[^;]+", f"stroke-width:{width}", style)
        # path.attrib["style"] = new_style

    # if output_path.lower().endswith((".jpg", ".jpeg")):
    #     # 2. ⬇️  Écrit un SVG temporaire (même dossier que output)
    #     tmp_svg = Path(output_path).with_suffix(".tmp.svg")
    #     tree.write(tmp_svg, xml_declaration=True, encoding="UTF-8", pretty_print=False)

    #     # 3. ⬇️  Construit la commande Inkscape CLI
    #     cmd = [
    #         inkscape_exe,
    #         str(tmp_svg),
    #         f"--export-filename={output_path}",
    #         "--export-background=#ffffff",
    #         "--export-area-drawing",
    #     ]
    #     export_width = 2000
    #     export_height = None
    #     if export_width:
    #         cmd.append(f"--export-width={export_width}")
    #     if export_height:
    #         cmd.append(f"--export-height={export_height}")

    #     # 4. ⬇️  Exécute Inkscape
    #     subprocess.run(cmd, check=True)
    #     print(f"✅  Image générée via Inkscape : {output_path}")

    #     # 5. ⬇️  Si JPG, convertir le PNG temporaire
    #     if output_path.lower().endswith((".jpg", ".jpeg")):
    #         # Inkscape vient d’exporter un JPG directement; s’il a exporté un PNG,
    #         # décocher ce bloc et adapter: convertir PNG -> JPG via Pillow.
    #         pass

    #     # 6. ⬇️  Nettoyage
    #     tmp_svg.unlink(missing_ok=True)
    if save == True:
        tree.write(output_path, pretty_print=False, xml_declaration=True, encoding="UTF-8")
    else:
        return etree.tostring(
            tree,
            encoding="utf-8",
            xml_declaration=True,  # on retire <?xml…> pour l’embed
            pretty_print=False,
        ).decode()


# base = Path(__file__).parent.parent  # par exemple, un dossier au-dessus de app.py
# # 2) Construire le chemin vers le SVG
# svg_template_path = base / "grafs_e" / "data" / "system_flows.svg"

# for cluster in mean_matrices_by_cluster_merged:
#     update_svg_fluxes(
#         svg_template_path,
#         f"C:/Users/faustega/Documents/These/Typologie/system_flows_clusters/{cluster}.svg",
#         mean_matrices_by_cluster_merged[cluster],
#         mapping_svg_fluxes,
#         save = True
#     )


def build_flow_svg(
    svg_template_path: str,
    flux_matrix: np.ndarray,
    mapping_svg_fluxes: dict[str, list[tuple[int, int]]],
    scale_max_px: float = 25.0,
) -> str:
    """
    Charge le template SVG et met à jour les `stroke-width`
    en normalisant l’épaisseur maximale à `scale_max_px` pixels.
    Retourne le SVG en **chaîne de caractères**.
    """
    parser = etree.XMLParser(remove_blank_text=False)
    # try:
    #     tree = etree.parse(svg_template_path, parser)
    # except Exception as e:
    #     print("❌ Erreur durant le parsing du SVG :")
    #     print(f"   Chemin utilisé  : {svg_template_path}")
    #     print(f"   Exception levée : {repr(e)}")
    #     raise
    tree = etree.parse(svg_template_path, parser)
    root = tree.getroot()
    # # Force le SVG à être redimensionnable selon son viewBox
    # if "width" in root.attrib:
    #     del root.attrib["width"]
    # if "height" in root.attrib:
    #     del root.attrib["height"]
    # # On s’assure que le viewBox sera pris en compte à 100 %
    # root.attrib["preserveAspectRatio"] = "xMinYMin meet"
    ns = {"svg": root.nsmap.get(None)}

    # Valeur max pour la normalisation
    vmax = max(
        (flux_matrix[i, j] for pairs in mapping_svg_fluxes.values() for i, j in pairs if flux_matrix[i, j] > 0),
        default=1e-9,
    )

    for path_id, ij_list in mapping_svg_fluxes.items():
        total = sum(float(flux_matrix[i, j]) for i, j in ij_list)
        width = 0 if total == 0 else total / vmax * scale_max_px

        nodes = root.xpath(f"//svg:path[@id='{path_id}']", namespaces=ns)
        if not nodes:
            continue
        path = nodes[0]
        style = path.attrib.get("style", "")
        if "stroke-width" in style:
            style = re.sub(r"stroke-width:[^;]+", f"stroke-width:{width}", style)
        else:
            style = style.rstrip(";") + f";stroke-width:{width}"
        path.attrib["style"] = style

    return etree.tostring(
        tree,
        encoding="utf-8",
        xml_declaration=False,  # on retire <?xml…> pour l’embed
        pretty_print=False,
    ).decode()


def build_flow_svg_full(
    svg_template_path: str,
    flux_matrix: np.ndarray,
    mapping_svg_fluxes: dict[str, list[tuple[int, int]]],
    scale_max_px: float = 25.0,
) -> str:
    """
    1) Met à jour les `stroke-width` des chemins SVG selon flux_matrix (normalisé).
    2) Calcule la bounding‐box de toutes les formes (paths + rects).
    3) Réécrit le <svg> pour supprimer width/height, ajouter viewBox et overflow="visible".
    4) Retourne le SVG complet (chaîne UTF‐8), prêt à être injecté tel quel.
    """

    # --- 1) parser le SVG ---
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(svg_template_path, parser)
    root = tree.getroot()
    ns = {"svg": root.nsmap.get(None)}

    # --- 2) Traiter les stroke‐width comme avant ---
    # 2.1 Calculer vmax pour la normalisation
    all_flux = [flux_matrix[i, j] for pairs in mapping_svg_fluxes.values() for i, j in pairs if flux_matrix[i, j] > 0]
    vmax = max(all_flux) if all_flux else 1e-9

    # 2.2 Mettre à jour chaque <path> ciblé
    for path_id, ij_list in mapping_svg_fluxes.items():
        total = sum(float(flux_matrix[i, j]) for i, j in ij_list)
        width = 0 if total == 0 else (total / vmax) * scale_max_px

        nodes = root.xpath(f"//svg:path[@id='{path_id}']", namespaces=ns)
        if not nodes:
            continue
        path = nodes[0]
        style = path.attrib.get("style", "")
        if "stroke-width" in style:
            style = re.sub(r"stroke-width:[^;]+", f"stroke-width:{width}", style)
        else:
            style = style.rstrip(";") + f";stroke-width:{width}"
        path.attrib["style"] = style

    # --- 3) Calculer la bounding‐box de tout le contenu ---
    # On cherche minx, miny, maxx, maxy parmi:
    #   • Tous les ‘M x,y’, ‘L x,y’, ‘C …’ d’un <path> (on prend TOUTES les coordonnées absolues)
    #   • Tous les coins des <rect> (x,y) + (x+width, y+height)
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    # 3.1 Pour chaque <path> : extraire tous les nombres de ‘d=’
    for elem in root.xpath("//svg:path", namespaces=ns):
        d = elem.attrib.get("d", "")
        # trouver toutes les valeurs numériques (x1, y1, x2, y2, …)
        nums = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", d)
        # on convertit en float et on prend paires (0,1), (2,3), …
        coords = [float(x) for x in nums]
        xs = coords[0::2]
        ys = coords[1::2]
        if xs and ys:
            minx = min(minx, min(xs))
            maxx = max(maxx, max(xs))
            miny = min(miny, min(ys))
            maxy = max(maxy, max(ys))

    # 3.2 Pour chaque <rect> : coins (x, y) et (x+width, y+height)
    for elem in root.xpath("//svg:rect", namespaces=ns):
        x = float(elem.attrib.get("x", 0))
        y = float(elem.attrib.get("y", 0))
        w = float(elem.attrib.get("width", 0))
        h = float(elem.attrib.get("height", 0))
        minx = min(minx, x)
        maxx = max(maxx, x + w)
        miny = min(miny, y)
        maxy = max(maxy, y + h)

    # Si on n’a rien trouvé (aucun path ni rect), on reste sur le viewBox existant (ou on quitte)
    if minx == float("inf"):
        # Rien à calculer, on renvoie le SVG original modifié
        return etree.tostring(tree, encoding="utf-8", pretty_print=False).decode()

    # --- 4) Réécrire le <svg> pour englober ces bornes ---
    # 4.1 Supprimer width/height (pour que le viewBox gouverne l’affichage)
    for attr in ("width", "height"):
        if attr in root.attrib:
            del root.attrib[attr]

    # 4.2 Ajouter le viewBox = "minx miny width_box height_box"
    vb_x = minx
    vb_y = miny
    vb_width = maxx - minx
    vb_height = maxy - miny
    root.attrib["viewBox"] = f"{vb_x} {vb_y} {vb_width} {vb_height}"

    # 4.3 S’assurer que rien n’est masqué : overflow="visible"
    root.attrib["overflow"] = "visible"

    tree.write(Path(svg_template_path).parent / "test.svg", pretty_print=False, xml_declaration=True, encoding="UTF-8")

    # --- 5) Renvoyer le SVG final (sans <?xml …>) ---
    return etree.tostring(tree, encoding="utf-8", xml_declaration=False, pretty_print=False).decode()


# ──────────────────────────────────────────────
# 2)  Fonction Streamlit qui affiche le SVG + légende
# ──────────────────────────────────────────────
# def streamlit_sankey_systemic_flows_svg(
#     flux_matrix: np.ndarray,
#     mapping_svg_fluxes: dict,
#     svg_template_path: str,
#     n_legend_steps: int = 4,
#     scale_max_px: float = 25.0,
# ):
#     # 2.1  produire le SVG final (string)
#     svg_str = build_flow_svg(svg_template_path, flux_matrix, mapping_svg_fluxes, scale_max_px)

#     # 2.2  injection directe dans la page
#     st.markdown(f'<div style="text-align:center">{svg_str}</div>', unsafe_allow_html=True)

#     # 2.3  légende dynamique
#     vmax = max(
#         (flux_matrix[i, j] for pairs in mapping_svg_fluxes.values() for i, j in pairs if flux_matrix[i, j] > 0),
#         default=1e-9,
#     )
#     step_vals = np.linspace(0, vmax, n_legend_steps + 1)[1:]

#     legend_svg = [
#         '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="{}">'.format(
#             int((scale_max_px + 6) * len(step_vals))
#         )
#     ]
#     y_cursor = 0
#     for v in step_vals:
#         w = int(v / vmax * scale_max_px)
#         legend_svg.append(
#             f'<path d="M 10 {y_cursor + w / 2 + 3} L 150 {y_cursor + w / 2 + 3}" stroke="black" stroke-width="{w}" />'
#         )
#         legend_svg.append(
#             f'<text x="80" y="{y_cursor + w + 6}" font-size="9" text-anchor="middle">{v:.2f} kt N / an</text>'
#         )
#         y_cursor += w + 10
#     legend_svg.append("</svg>")

#     st.markdown("<br/><b>Légende – Intensité des flux</b>", unsafe_allow_html=True)
#     st.markdown("".join(legend_svg), unsafe_allow_html=True)


def streamlit_sankey_systemic_flows_svg(model, mapping_svg_fluxes, svg_template_path):
    # 1. on génère le SVG string (avec build_flow_svg déjà modifié)
    # svg_str = build_flow_svg_full(svg_template_path, flux_matrix, mapping_svg_fluxes, scale_max_px=25.0)
    merged_matrix, _, _ = merge_nodes(
        model.get_transition_matrix(),
        labels_init,
        merges={
            "cereals (excluding rice)": [
                "Wheat",
                "Rye",
                "Barley",
                "Oat",
                "Grain maize",
                "Rice",
                "Other cereals",
            ],
            "fruits and vegetables": [
                "Dry vegetables",
                "Dry fruits",
                "Squash and melons",
                "Cabbage",
                "Leaves vegetables",
                "Fruits",
                "Olives",
                "Citrus",
            ],
            "leguminous": [
                "Horse beans and faba beans",
                "Peas",
                "Other protein crops",
                "Green peas",
                "Dry beans",
                "Green beans",
                "Soybean",
            ],
            "oleaginous": ["Rapeseed", "Sunflower", "Other oil crops", "Flax", "Hemp"],
            "forages": [
                "Forage maize",
                "Forage cabbages",
                "Straw",
            ],
            "temporary meadows": ["Non-legume temporary meadow", "Alfalfa and clover"],
            "natural meadows ": ["Natural meadow "],
            "trade": [
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
            ],
            "ruminants": ["bovines", "ovines", "caprines", "equine"],
            "monogastrics": ["porcines", "poultry"],
            "population": ["urban", "rural"],
            "Environment": [
                "NH3 volatilization",
                "N2O emission",
                "hydro-system",
                "other losses",
            ],
            "roots": ["Sugar beet", "Potatoes", "Other roots"],
        },
    )
    normed_matrix = merged_matrix / merged_matrix.sum()
    svg_str = update_svg_fluxes(
        svg_template_path, Path(svg_template_path).parent / "test.svg", normed_matrix, mapping_svg_fluxes, save=False
    )
    # 2. on l’affiche dans un composant HTML dédié, en précisant largeur et hauteur ou en laissant défiler
    components.html(
        f'<div style="transform: scale(0.3); transform-origin: 0 0;">{svg_str}</div>',
        # svg_str,
        width=5000,  # ou None pour full-width
        height=600,  # adapter si besoin
        scrolling=False,
    )


# %%
