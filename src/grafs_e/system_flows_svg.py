# %%
import re
import subprocess
from pathlib import Path

import cairosvg
import numpy as np
import streamlit as st
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
    "path12-2": [(14, 12), (14, 17), (14, 11)],
    "path15": [(9, 15), (9, 0)],
    "path17": [(17, 8)],
    "path23": [(14, 16)],
    "path26": [(9, 15)],
    "path27": [(9, 0)],
    "path29": [(8, 17), (8, 12), (8, 11)],
    "path31": [(14, 9)],
    "path32": [(9, 14)],
    "path34": [(14, 0), (14, 3), (14, 5), (14, 6), (14, 10), (14, 13)],
    "path35": [
        (8, 0),
        (8, 3),
        (8, 5),
        (8, 6),
        (8, 10),
        (8, 13),
        (8, 2),
        (14, 0),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 10),
        (14, 13),
        (14, 2),
        (12, 0),
        (12, 3),
        (12, 5),
        (12, 6),
        (12, 10),
        (12, 13),
        (12, 2),
    ],
    "path36": [(8, 0), (14, 0)],
    "path37": [(12, 0), (12, 3), (12, 5), (12, 6), (12, 10), (12, 13), (12, 2)],
    "path39": [(3, 14), (5, 14), (7, 14), (16, 14)],
    "path40": [(3, 12), (6, 12), (7, 12), (8, 12), (10, 12), (13, 12), (14, 12)],
    "path41": [(3, 8), (7, 8), (10, 8)],
    "path42": [(7, 17), (7, 12), (7, 14), (7, 8)],
    "path43": [(5, 17), (5, 14)],
    "path45": [(3, 8), (3, 12), (3, 14), (3, 17)],
    "path46": [(10, 12), (10, 8), (10, 14), (10, 17)],
    "path47": [(13, 12), (13, 17)],
    "path48": [(6, 12), (6, 17)],
    "path49": [(3, 17), (5, 17), (6, 17), (7, 17), (10, 17), (13, 17), (16, 17)],
    "path50": [(6, 17)],
    "path52": [(3, 17), (5, 17), (7, 17), (10, 17), (13, 17), (16, 17)],
    "path53": [(13, 17)],
    "path54": [(3, 17), (5, 17), (7, 17), (10, 17), (16, 17)],
    "path55": [(10, 17)],
    "path56": [(3, 17), (5, 17), (7, 17), (16, 17)],
    "path57": [(3, 17)],
    "path58": [(5, 17), (7, 17), (16, 17)],
    "path62": [(16, 17), (16, 14)],
    "path65": [(1, 3), (1, 5), (1, 6), (1, 7), (1, 9), (1, 10), (1, 13), (1, 16)],
    "path66": [(1, 3)],
    "path68": [(1, 6)],
    "path69": [(1, 13)],
    "path70": [(1, 6), (1, 13)],
    "path71": [(1, 10)],
    "path72": [(1, 10), (1, 6), (1, 13)],
    "path74": [(0, 3), (0, 6), (0, 7), (0, 5), (0, 9), (0, 10), (0, 13), (0, 16)],
    "path75": [(0, 9), (1, 9), (2, 9)],
    "path76": [(0, 16), (1, 16), (2, 16)],
    "path77": [(1, 5), (1, 16), (1, 9)],
    "path78": [(1, 5)],
    "path79": [(2, 7), (2, 9), (2, 16)],
    "path80": [(2, 7), (0, 7)],
    "path81": [(0, 1)],
    "path83": [(2, 7)],
    "path123": [
        (12, 3),
        (12, 6),
        (12, 10),
        (12, 13),
        (8, 3),
        (8, 6),
        (8, 10),
        (8, 13),
        (14, 3),
        (14, 6),
        (14, 10),
        (14, 13),
    ],
    "path87": [(2, 9)],
    "path88": [(2, 16)],
    "path89": [(2, 9), (2, 16)],
    "path91": [(0, 9)],
    "path92": [(0, 3), (0, 6), (0, 7), (0, 5), (0, 10), (0, 13), (0, 16)],
    "path93": [(0, 16)],
    "path118": [(13, 12)],
    "path119": [(6, 12)],
    "path120": [(10, 12), (10, 14), (10, 8)],
    "path121": [(10, 8)],
    "path122": [(10, 12)],
    "path124": [(12, 5), (8, 5), (14, 5)],
    "path125": [
        (12, 6),
        (12, 10),
        (12, 13),
        (8, 6),
        (8, 10),
        (8, 13),
        (14, 6),
        (14, 10),
        (14, 13),
    ],
    "path126": [(12, 3), (14, 3), (8, 3)],
    "path127": [
        (12, 6),
        (12, 13),
        (8, 6),
        (8, 13),
        (14, 6),
        (14, 13),
    ],
    "path128": [(12, 10), (14, 10), (8, 10)],
    "path129": [(12, 13), (14, 13), (8, 13)],
    "path130": [(12, 6), (14, 6), (8, 6)],
    "path131": [(0, 3), (0, 6), (0, 7), (0, 5), (0, 10), (0, 13)],
    "path132": [(0, 7)],
    "path133": [
        (12, 3),
        (12, 6),
        (12, 10),
        (12, 13),
        (12, 5),
        (8, 3),
        (8, 6),
        (8, 10),
        (8, 13),
        (8, 5),
        (14, 3),
        (14, 6),
        (14, 10),
        (14, 13),
        (14, 5),
    ],
    "path134": [(0, 3), (0, 6), (0, 5), (0, 10), (0, 13)],
    "path135": [(0, 3), (0, 6), (0, 10), (0, 13)],
    "path136": [(0, 6), (0, 10), (0, 13)],
    "path137": [(0, 6), (0, 13)],
    "path138": [(0, 6)],
    "path139": [(0, 13)],
    "path140": [(0, 10)],
    "path141": [(0, 3)],
    "path142": [(0, 5)],
    "path143": [(0, 5), (12, 5), (8, 5), (14, 5)],
    "path144": [(0, 3), (12, 3), (8, 3), (14, 3), (16, 3), (7, 3), (15, 3)],
    "path145": [(0, 10), (12, 10), (8, 10), (14, 10)],
    "path146": [(0, 13), (12, 13), (8, 13), (14, 13)],
    "path147": [[0, 6], (12, 6), (8, 6), (14, 6)],
    "path151": [(1, 16)],
    "path152": [(1, 9)],
    "path153": [(1, 16), (1, 9)],
    "path155": [(14, 12), (14, 17), (14, 11)],
    "path156": [(3, 12), (6, 12), (7, 12), (10, 12), (13, 12)],
    "path157": [(14, 12), (8, 12)],
    "path158": [(8, 12), (14, 12)],
    "path159": [(3, 8), (3, 12), (3, 14)],
    "path160": [(3, 8), (3, 14)],
    "path166": [(3, 12)],
    "path171": [(5, 17)],
    "path172": [(10, 14), (5, 14)],
    "path173": [(10, 14), (5, 14), (3, 14)],
    "path174": [(3, 8)],
    "path175": [(3, 14)],
    "path176": [(3, 8)],
    "path177": [(7, 12)],
    "path178": [(7, 14), (7, 8), (7, 17)],
    "path179": [(7, 17), (16, 17)],
    "path180": [(7, 17)],
    "path181": [(7, 8), (7, 14)],
    "path182": [(7, 14)],
    "path183": [(7, 8)],
    "path184": [(16, 17)],
    "path185": [(16, 14)],
    "path186": [(16, 0)],
    "path187": [(16, 0), (9, 0), (3, 0), (5, 0), (6, 0), (7, 0), (10, 0), (13, 0)],
    "path188": [(17, 14), (17, 8), (17, 12)],
    "path189": [(17, 14)],
    "path190": [(17, 12)],
    "path191": [(15, 9)],
    "path192": [(15, 16), (15, 3), (15, 5), (15, 6), (15, 7), (15, 10), (15, 13)],
    "path193": [(15, 6)],
    "path194": [(15, 3), (15, 5), (15, 7), (15, 10), (15, 13)],
    "path195": [(15, 13)],
    "path196": [(15, 3), (15, 5), (15, 7), (15, 10)],
    "path197": [(15, 10)],
    "path198": [(15, 7), (15, 5), (15, 3)],
    "path199": [(15, 7)],
    "path201": [(15, 10)],
    "path6": [(4, 12)],
    "path7": [(4, 12), (17, 12)],
    "path8": [(5, 14)],
    "path9": [(5, 14)],
    "path10": [(10, 14)],
    "path11": [(10, 12), (10, 8)],
    "path12": [(8, 0), (8, 3), (8, 5), (8, 6), (8, 10), (8, 13), (8, 2)],
    "path13": [
        (8, 0),
        (8, 3),
        (8, 5),
        (8, 6),
        (8, 10),
        (8, 13),
        (8, 2),
        (12, 0),
        (12, 3),
        (12, 5),
        (12, 6),
        (12, 10),
        (12, 13),
        (12, 2),
    ],
    "path14": [(12, 2), (14, 2), (8, 2)],
    "path16": [
        (8, 3),
        (8, 5),
        (8, 6),
        (8, 10),
        (8, 13),
        (8, 2),
        (14, 3),
        (14, 5),
        (14, 6),
        (14, 10),
        (14, 13),
        (14, 2),
        (12, 3),
        (12, 5),
        (12, 6),
        (12, 10),
        (12, 13),
        (12, 2),
    ],
    "path18": [(14, 12), (14, 11), (14, 17), (8, 12), (8, 11), (8, 17)],
    "path19": [(14, 11), (14, 17), (8, 11), (8, 17)],
    "path20": [(14, 11), (8, 11)],
    "path21": [(15, 16)],
    "path22": [(15, 3), (15, 5), (15, 6), (15, 7), (15, 10), (15, 13)],
    "path24": [(16, 0), (9, 0)],
    "path25": [(6, 0)],
    "path28": [(16, 0), (9, 0), (6, 0)],
    "path30": [(13, 0)],
    "path33": [(16, 0), (9, 0), (6, 0), (13, 0)],
    "path38": [(3, 0), (10, 0)],
    "path51": [(10, 0)],
    "path59": [(3, 0)],
    "path63": [(16, 0), (9, 0), (3, 0), (6, 0), (10, 0), (13, 0)],
    "path64": [(7, 0)],
    "path67": [(16, 0), (9, 0), (3, 0), (6, 0), (7, 0), (10, 0), (13, 0)],
    "path73": [(5, 0)],
    "path82": [(2, 9), (0, 9)],
    "path84": [(2, 16), (0, 16)],
    "path148": [(16, 3)],
    "path161": [(16, 3), (16, 0)],
    "path162": [(12, 3), (14, 3), (8, 3), (0, 3)],
    "path163": [(16, 3)],
    "path164": [(7, 3)],
    "path168": [(7, 3), (7, 0)],
    "path169": [(7, 3), (16, 3), (15, 3)],
    "path202": [(15, 3)],
    "path203": [(15, 3), (16, 3)],
    "path204": [(15, 5)],
    "path205": [(0, 5), (12, 5), (8, 5), (14, 5), (15, 5)],
    "path206": [(15, 3), (15, 5)],
}

# %%


def update_svg_fluxes(
    svg_path,
    output_path,
    flux_matrix,
    labels,
    mapping_svg_fluxes,
    scale=100,
    inkscape_exe="C:/Program Files/Inkscape/bin/inkscape.exe",
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

    if output_path.lower().endswith((".jpg", ".jpeg")):
        # 2. ⬇️  Écrit un SVG temporaire (même dossier que output)
        tmp_svg = Path(output_path).with_suffix(".tmp.svg")
        tree.write(tmp_svg, xml_declaration=True, encoding="UTF-8", pretty_print=False)

        # 3. ⬇️  Construit la commande Inkscape CLI
        cmd = [
            inkscape_exe,
            str(tmp_svg),
            f"--export-filename={output_path}",
            "--export-background=#ffffff",
            "--export-area-drawing",
        ]
        export_width = 2000
        export_height = None
        if export_width:
            cmd.append(f"--export-width={export_width}")
        if export_height:
            cmd.append(f"--export-height={export_height}")

        # 4. ⬇️  Exécute Inkscape
        subprocess.run(cmd, check=True)
        print(f"✅  Image générée via Inkscape : {output_path}")

        # 5. ⬇️  Si JPG, convertir le PNG temporaire
        if output_path.lower().endswith((".jpg", ".jpeg")):
            # Inkscape vient d’exporter un JPG directement; s’il a exporté un PNG,
            # décocher ce bloc et adapter: convertir PNG -> JPG via Pillow.
            pass

        # 6. ⬇️  Nettoyage
        tmp_svg.unlink(missing_ok=True)
    else:
        tree.write(output_path, pretty_print=False, xml_declaration=True, encoding="UTF-8")


def streamlit_sankey_systemic_flows(
    svg_template_path: str,
    flux_matrix: np.ndarray,
    mapping_svg_fluxes: dict[str, list[tuple[int, int]]],
    scale_px: float = 100.0,
    legend_steps: int = 4,
):
    """
    Affiche le diagramme (SVG → PNG) + légende dans Streamlit.

    • flux_matrix est normalisée (somme = 1) avant mise à l'échelle.
    • Légende : barres d'épaisseur correspondant aux paliers de flux.

    Parameters
    ----------
    svg_template_path : chemin du SVG « vide » (flux fins).
    flux_matrix       : np.ndarray carrée des flux (kt N / yr).
    mapping_svg_fluxes: {id_svg: [(i,j), …]}  liste des paires sommées.
    scale_px          : facteur px / (kt N / yr).
    legend_steps      : nombre de paliers de légende.
    """

    # ───── 1. normalisation ─────
    total_flux = flux_matrix.sum()
    if total_flux == 0:
        st.warning("Matrice vide ; aucun flux à tracer.")
        return
    mat_norm = flux_matrix / total_flux  # somme = 1

    # ───── 2. mise à jour du SVG en mémoire ─────
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(svg_template_path, parser)
    root = tree.getroot()
    ns = {"svg": root.nsmap.get(None)}

    max_width_px = 0  # pour savoir jusqu'où va le scale, utile à la légende
    for path_id, ij_list in mapping_svg_fluxes.items():
        flux_value = sum(float(mat_norm[i, j]) for i, j in ij_list)  # normalisé
        width_px = flux_value * scale_px
        max_width_px = max(max_width_px, width_px)

        nodes = root.xpath(f"//svg:path[@id='{path_id}']", namespaces=ns)
        if not nodes:
            st.write(f"⚠️ id {path_id} absent du SVG.")
            continue

        style = nodes[0].attrib.get("style", "")
        if "stroke-width" in style:
            style = re.sub(r"stroke-width:[^;]+", f"stroke-width:{width_px}", style)
        else:
            style = style.rstrip(";") + f";stroke-width:{width_px}"
        nodes[0].attrib["style"] = style

    # ───── 3. SVG → PNG (fond blanc) ─────
    svg_bytes = etree.tostring(tree, xml_declaration=True, encoding="utf-8", pretty_print=False)
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, background_color="white", unsafe=True, output_width=2000)

    # ───── 4. Streamlit : affiche le diagramme ─────
    st.image(png_bytes, caption="Diagramme systémique (flux normalisés)")

    # ───── 5. Légende (barres horizontales) ─────
    st.markdown("#### Légende (kt N / yr → épaisseur en pixels)")
    legend_vals = np.linspace(0, total_flux / legend_steps, legend_steps + 1)[1:]
    for val in legend_vals:
        width_px = val / total_flux * scale_px
        bar_svg = f"""
        <svg width="200" height="{width_px + 4}">
            <line x1="0" y1="{width_px / 2 + 2}" x2="150" y2="{width_px / 2 + 2}"
                  stroke="black" stroke-width="{width_px}" />
            <text x="160" y="{width_px / 2 + 6}" font-size="12">{val:.2e}</text>
        </svg>
        """
        st.markdown(bar_svg, unsafe_allow_html=True)
