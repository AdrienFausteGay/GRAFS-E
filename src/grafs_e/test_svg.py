import re

from lxml import etree

# etree.register_namespace('', "http://www.w3.org/2000/svg")


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


update_svg_flows(
    svg_path="C:/Users/faustega/Documents/These/Typologie/test schéma/plain_svg.svg",
    output_path="C:/Users/faustega/Documents/These/Typologie/test schéma/test_updated.svg",
    stroke_by_id={
        "flux_cereals_trade": 3.0  # ou n’importe quelle valeur
    },
)
