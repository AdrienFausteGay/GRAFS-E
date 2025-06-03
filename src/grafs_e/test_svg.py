# %%
import re

from lxml import etree

from grafs_e.donnees import *
from grafs_e.N_class import DataLoader
from grafs_e.typology import compute_mean_matrice, get_matrices, merge_nodes, plot_dendrogram

# %%
# etree.register_namespace('', "http://www.w3.org/2000/svg")

data = DataLoader()
matrices, norm_matrices = get_matrices(data)
df_plot = plot_dendrogram(norm_matrices)
mean_matrices_by_cluster, global_mean_matrices = compute_mean_matrice(norm_matrices, df_plot)
# %%

mean_matrices_by_cluster_merged = {}
for cluster in mean_matrices_by_cluster.keys():
    mean_matrices_by_cluster_merged[cluster], _, _ = merge_nodes(
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


# %% Premier test
def update_svg_flows(svg_path, output_path, stroke_by_id):
    """
    Met à jour l'épaisseur des traits (stroke-width) pour des éléments SVG donnés par leur ID.
    Le marqueur (flèche) s'adapte automatiquement si markerUnits="strokeWidth".
    """
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_path, parser)
    root = tree.getroot()
    ns = root.nsmap.get(None)

    for elem_id, new_width in stroke_by_id.items():
        xpath = f"//svg:path[@id='{elem_id}']"
        elems = root.xpath(xpath, namespaces={"svg": ns})
        if not elems:
            print(f"⚠️ Élément non trouvé : {elem_id}")
            continue
        path = elems[0]
        style = path.attrib.get("style", "")
        # Mise à jour du stroke-width dans l'attribut style
        new_style = re.sub(r"stroke-width:[^;]+", f"stroke-width:{new_width}", style)
        path.attrib["style"] = new_style

    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")


# update_svg_flows(
#     svg_path="C:/Users/faustega/Documents/These/Typologie/test schéma/plain_svg.svg",
#     output_path="C:/Users/faustega/Documents/These/Typologie/test schéma/test_updated.svg",
#     stroke_by_id={
#         "flux_cereals_trade": 3.0  # ou n’importe quelle valeur
#     },
# )

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
list_svg_paths("C:/Users/faustega/Documents/These/Typologie/test schéma/system_flows.svg")
# %% Dictionnaire des correspondances flux et path (svg)
new_labels = [
    "Environment",
    "Haber-Bosch",
    "atmospheric N2",
    "cereals (excluding rice)",
    "fishery products",
    "forages",
    "fruits and vegetables",
    "leguminous",
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

mapping_svg_fluxes = {
    "path12-2": [(14, 12), (14, 17)],
    "path15": [(9, 15), (9, 0)],
    "path17": [(17, 8)],
    "path23": [(14, 16)],
    "path26": [(9, 15)],
    "path27": [(9, 0)],
    "path29": [(8, 17), (8, 12)],
    "path31": [(14, 9)],
    "path32": [(9, 14)],
    "path33": [(8, 0), (8, 3), (8, 5), (8, 6), (8, 10), (8, 13)],
    "path34": [(14, 0), (14, 3), (14, 5), (14, 6), (14, 10), (14, 13)],
    "path35": [
        (8, 0),
        (8, 3),
        (8, 5),
        (8, 6),
        (8, 10),
        (8, 13),
        (14, 0),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 10),
        (14, 13),
    ],
    "path36": [(8, 0), (14, 0)],
    "path37": [(12, 0), (12, 3), (12, 5), (12, 6), (12, 10), (12, 13)],
    "path39": [(3, 14), (5, 14), (7, 14), (16, 14)],
}

# %%


def update_svg_fluxes(svg_path, output_path, flux_matrix, labels, mapping_svg_fluxes, scale=100):
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
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_path, parser)
    root = tree.getroot()
    ns = root.nsmap.get(None)

    for id_svg, ij_list in mapping_svg_fluxes.items():
        total_flux = sum(flux_matrix[i, j] for i, j in ij_list)
        width = max(total_flux * scale, 0.5)  # Épaisseur minimale à 0.5 pour visibilité

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

    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")


# %%
