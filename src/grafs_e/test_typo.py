from graphviz import Digraph


def scale_width(v, vmin, vmax, wmin=0.5, wmax=10):
    # w = wmin + (v-vmin)/(vmax-vmin)*(wmax-wmin)
    return wmin + (v - vmin) / max(vmax - vmin, 1e-6) * (wmax - wmin)


nodes = {
    "A": {"pos": "100,200!", "ports": ["e", "s"]},
    "B": {"pos": "300,100!", "ports": ["w"]},
}
F = {("A", "B"): 20, ("A", "A_s"): 5}

dot = Digraph(engine="neato")  # neato/fdp pour positionnement libre
dot.attr("node", shape="box", width="1", height="0.5", fixedsize="true")

# 1) déclarer les nœuds avec position
for name, cfg in nodes.items():
    dot.node(name, pos=cfg["pos"])

# 2) déclarer les arêtes en précisant port et épaisseur
vmin, vmax = min(F.values()), max(F.values())
for (src, tgt), val in F.items():
    pen = str(scale_width(val, vmin, vmax))
    if "_" in src:
        n, port = src.split("_")
        src_spec = f"{n}:{port}"
    else:
        src_spec = src
    if "_" in tgt:
        n, port = tgt.split("_")
        tgt_spec = f"{n}:{port}"
    else:
        tgt_spec = tgt

    dot.edge(src_spec, tgt_spec, penwidth=pen, arrowsize="0.8")

dot.render("flux_diagram", format="png", cleanup=True)
