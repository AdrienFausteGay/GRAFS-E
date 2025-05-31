import plotly.graph_objects as go

fig = go.Figure()

# --- 1. Boîtes ----------------------------------
nodes = {
    "A": {"x": 0.1, "y": 0.5, "w": 0.1, "h": 0.2},
    "B": {"x": 0.5, "y": 0.6, "w": 0.1, "h": 0.2},
    "C": {"x": 0.8, "y": 0.2, "w": 0.1, "h": 0.2},
}

for name, n in nodes.items():
    fig.add_shape(
        type="rect",
        x0=n["x"],
        x1=n["x"] + n["w"],
        y0=n["y"],
        y1=n["y"] + n["h"],
        line=dict(color="black"),
        fillcolor="lightblue",
        layer="above",
    )
    fig.add_trace(
        go.Scatter(x=[n["x"] + n["w"] / 2], y=[n["y"] + n["h"] / 2], text=[name], mode="text", showlegend=False)
    )


# --- 2. Fonction de lien --------------------------------
def make_link_perfect(x0, y0, dir0, x1, y1, dir1, width, color, arrow_frac=0.07):
    """
    Trace un flux à épaisseur constante, avec :
    - entrée/sortie perpendiculaires,
    - un biseau discret à la fin pour indiquer le sens.
    """
    OFFSET = 0.1  # pour tangence
    ARROW_LEN = arrow_frac

    # direction → vecteurs de tangente
    VEC = {"right": (1, 0), "left": (-1, 0), "top": (0, 1), "bottom": (0, -1)}
    dx0, dy0 = VEC[dir0]
    dx1, dy1 = VEC[dir1]

    # point de contrôle pour Bézier (sortie)
    cx0 = x0 + dx0 * OFFSET
    cy0 = y0 + dy0 * OFFSET

    # point de fin du flux principal (juste avant flèche)
    xf = x1 - dx1 * ARROW_LEN
    yf = y1 - dy1 * ARROW_LEN

    # point de contrôle à l'arrivée
    cx1 = xf + dx1 * OFFSET
    cy1 = yf + dy1 * OFFSET

    # décalage pour l’épaisseur (demi-largeur)
    wx, wy = width / 2, width / 2
    if dir0 in ["left", "right"]:
        # horizontal
        p1 = (x0, y0 + wy)
        p2 = (xf, yf + wy)
        p3 = (x1, y1)  # pointe
        p4 = (xf, yf - wy)
        p5 = (x0, y0 - wy)
    else:
        # vertical
        p1 = (x0 + wx, y0)
        p2 = (xf + wx, yf)
        p3 = (x1, y1)
        p4 = (xf - wx, yf)
        p5 = (x0 - wx, y0)

    # construire le chemin SVG (flux + biseau)
    path = f"""
        M {p1[0]},{p1[1]}
        C {cx0},{cy0} {cx1},{cy1} {p2[0]},{p2[1]}
        L {p3[0]},{p3[1]}
        L {p4[0]},{p4[1]}
        C {cx1},{cy1} {cx0},{cy0} {p5[0]},{p5[1]}
        Z
    """
    fig.add_shape(type="path", path=path, fillcolor=color, line=dict(width=0), layer="below")


# --- 3. Flux à tracer -------------------------------------
F = {("A", "B"): 0.05, ("B", "C"): 0.08}

for (src, tgt), val in F.items():
    # sortie à droite de src
    x0 = nodes[src]["x"] + nodes[src]["w"]
    y0 = nodes[src]["y"] + nodes[src]["h"] / 2
    dir0 = "right"

    # entrée à gauche de tgt
    x1 = nodes[tgt]["x"]
    y1 = nodes[tgt]["y"] + nodes[tgt]["h"] / 2
    dir1 = "left"

    make_link_perfect(x0, y0, dir0, x1, y1, dir1, width=val, color="royalblue")

# --- 4. Layout -------------------------------------
fig.update_layout(
    width=800,
    height=500,
    xaxis=dict(range=[0, 1], visible=False),
    yaxis=dict(range=[0, 1], visible=False),
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="white",
)
fig.show()
