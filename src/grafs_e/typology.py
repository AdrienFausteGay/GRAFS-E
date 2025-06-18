# %%

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, normalize

from grafs_e.donnees import *
from grafs_e.N_class import DataLoader, NitrogenFlowModel

# %%
run = True
if run:
    data = DataLoader()

# %%


def get_matrices(data):
    matrices = {}
    exclusions = [
        ("Savoie", "1852"),
        ("Alsace", "1885"),
        ("Alsace", "1906"),
        ("Alpes", "1852"),
        ("Pyrénées Orient", "1989"),
    ]
    for reg in regions:  # « regions » est ta liste de 33 régions
        for year in annees_disponibles:
            if (reg, year) not in exclusions:
                model = NitrogenFlowModel(data, year, reg)
                T = model.get_transition_matrix()  # (n_nodes × n_nodes)
                matrices[reg + "_" + year] = T.astype(float)

    # Normalisation

    # Global M/T
    norm_matrices = {}
    for reg, M in matrices.items():
        s = M.sum()
        norm_matrices[reg] = M / s if s else M
    return matrices, norm_matrices


# %%
if run:
    matrices, norm_matrices = get_matrices(data)
# %%
# Leontieff-like normalization: divide each column by its total
# norm_matrices = {}
# for reg, M in matrices.items():
#     col_sums = M.sum(axis=0)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         norm_M = np.divide(M, col_sums, where=col_sums != 0)  # avoid division by 0
#         norm_M[np.isnan(norm_M)] = 0  # replace NaNs from division by zero
#     norm_matrices[reg] = norm_M

# %%


def plot_dendrogram(norm_matrices, seuil=0.22):
    # vectorization
    # X = np.vstack([M.flatten() for M in norm_matrices.values()])  # shape = (33, n*n)
    # regions_order = list(norm_matrices.keys())
    X = np.vstack([M.flatten() for M in norm_matrices.values()])
    X_normalized_l2 = normalize(X, norm="l2", axis=1)
    regions_order = list(norm_matrices.keys())

    X_scaled = StandardScaler().fit_transform(X)
    Z = linkage(X, method="complete")  # ou "average", "complete", …
    Z2 = linkage(normalize(X, norm="l2"), method="ward")
    Z3 = linkage(MaxAbsScaler().fit_transform(X), method="ward")
    D = pdist(X_normalized_l2, metric="cosine")
    Zc = linkage(D, method="average")
    plt.figure(figsize=(80, 4), dpi=500)
    dendrogram(Zc, labels=regions_order, leaf_rotation=90, color_threshold=seuil)
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    # Si tu veux garder, disons, 4 clusters après inspection visuelle :
    # clt = AgglomerativeClustering(n_clusters=4, metric="euclidean", linkage="ward")
    # labels = clt.fit_predict(X)
    # df_clusters = pd.DataFrame({"Region": regions_order, "Cluster": labels}).sort_values("Cluster")

    labels_cluster = fcluster(Zc, t=seuil, criterion="distance")
    df_plot = pd.DataFrame(
        {"Region_year": regions_order, "Cluster": labels_cluster, "ClusterName": labels_cluster.astype(str)}
    ).sort_values("Cluster")
    return df_plot


if run:
    df_plot = plot_dendrogram(norm_matrices)

# %% Affichage de la frise


# ────────────────────────────────────────────────────────────────
#  utilitaire : ordonnée automatique « par colonne »
# ────────────────────────────────────────────────────────────────
def _compute_column_y(nodes: pd.DataFrame) -> pd.Series:
    for yr, sub in nodes.groupby("Year"):
        ordered = sub.sort_values(["ClusterName", "Node"])
        k = len(ordered)
        # on ne touche pas aux bords : marge haute & basse = 5 %
        ys = np.linspace(0.05, 0.15, k + 2)[1:-1]
        nodes.loc[ordered.index, "_y"] = ys
    return nodes["_y"]


# ────────────────────────────────────────────────────────────────
#  fonction principale
# ────────────────────────────────────────────────────────────────
def sankey_clusters_aggregate(df_plot: pd.DataFrame, highlight: list[str] | None = None) -> go.Figure:
    """
    Frise Sankey : un lien = nombre de régions qui passent
    d’un cluster à un autre entre deux années successives.

    Parameters
    ----------
    df_plot : DataFrame
        colonnes requises : 'Region_year' (« Nom_YYYY »), 'ClusterName'.
    highlight : list[str] | None
        Liste de régions à surligner ; chaque région reçoit sa propre
        couleur.  Si `None` (défaut) : colorisation par cluster d’arrivée.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # ── 1. Préparation des nœuds ────────────────────────────────────────
    df = df_plot.copy()
    df[["Region", "Year"]] = df["Region_year"].str.rsplit("_", n=1, expand=True)
    df["Year"] = df["Year"].astype(int)
    df["Node"] = df["Year"].astype(str) + " – " + df["ClusterName"]

    years = sorted(df["Year"].unique())
    y_min, y_max, span = years[0], years[-1], (years[-1] - years[0] or 1)

    nodes = (
        df[["Node", "Year", "ClusterName"]]
        .drop_duplicates()
        .sort_values(["Year", "ClusterName"])
        .reset_index(drop=True)
    )
    nodes["x"] = 0.05 + 0.90 * (nodes["Year"] - y_min) / span
    nodes["y"] = _compute_column_y(nodes)

    node_id = {n: i for i, n in nodes["Node"].items()}

    # ── 2. Transitions + liste exacte des régions ───────────────────────
    transitions = []
    for reg, g in df.groupby("Region"):
        g = g.sort_values("Year")
        for (_, r0), (_, r1) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            transitions.append(
                dict(
                    src=node_id[r0["Node"]],
                    dst=node_id[r1["Node"]],
                    src_clu=r0["ClusterName"],
                    dst_clu=r1["ClusterName"],
                    region=reg,
                )
            )

    links = (
        pd.DataFrame(transitions)
        .groupby(["src", "dst", "src_clu", "dst_clu"])
        .agg(regions=("region", lambda r: ", ".join(sorted(r))))
        .reset_index()
    )
    links["value"] = links["regions"].str.count(",") + 1  # nb de régions

    # ── 3. Couleurs ────────────────────────────────────────────────────
    if highlight:
        # palette distincte pour les régions mises en avant
        col_pal = dict(zip(highlight, sns.color_palette("husl", len(highlight)).as_hex()))

        def _link_color(row):
            for hreg in highlight:
                if hreg in row["regions"].split(", "):
                    return col_pal[hreg]  # couleur de la 1re région highlight trouvée
            return "#DDDDDD"  # gris clair pour le reste

        links["color"] = links.apply(_link_color, axis=1)

        # nœuds : on garde la palette par cluster, mais on peut opacifier
        clusters = nodes["ClusterName"].unique()
        node_palette = dict(zip(clusters, sns.color_palette("husl", len(clusters)).as_hex()))
        nodes["n_color"] = nodes["ClusterName"].map(node_palette)

    else:
        # colorisation « classique » par cluster d’arrivée
        clusters = nodes["ClusterName"].unique()
        node_palette = dict(zip(clusters, sns.color_palette("husl", len(clusters)).as_hex()))
        nodes["n_color"] = nodes["ClusterName"].map(node_palette)
        links["color"] = links["dst_clu"].map(node_palette)

    # ── 4. Figure Sankey ────────────────────────────────────────────────
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            domain=dict(x=[0, 1], y=[0.06, 1]),
            node=dict(
                x=nodes["x"].to_numpy(),
                y=nodes["y"].to_numpy(),
                # label=nodes["ClusterName"],
                color=nodes["n_color"],
                pad=6,
                thickness=4,
                line=dict(width=0.5, color="black"),
            ),
            link=dict(
                source=links["src"],
                target=links["dst"],
                value=links["value"],
                color=links["color"],
                customdata=np.stack([links["regions"], links["src_clu"], links["dst_clu"]], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[1]}</b> → <b>%{customdata[2]}</b><br>Régions : %{customdata[0]}<extra></extra>"
                ),
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        # title="Trajectoires agrégées des régions entre clusters",
        width=1900,
        height=1200,
        margin=dict(l=10, r=10, t=20, b=20),
        font_size=12,
    )

    # Annotations verticales des années
    for yr in years:
        fig.add_annotation(
            x=0.05 + 0.90 * (yr - y_min) / span,
            y=-0.0,
            xref="paper",
            yref="paper",
            text=str(yr),
            textangle=-90,
            showarrow=False,
            font=dict(size=13),
        )

    return fig


# %%
if run:
    # ──────────────────────────────────────────────────────────────────────────
    # 0)  Données : df_numeric    (index = Region_Year)      ───────────
    #     + colonne "Cluster"            (int)
    #     + mapping numéro  → nom  :  cluster_names
    # ──────────────────────────────────────────────────────────────────────────
    # cluster_names = summary["name"]  # Series : index = n° , value = libellé
    # cluster_names = [str(i) for i in range(max(df_plot["Cluster"]))]
    # df_plot = df_numeric[["Cluster"]].copy()
    # df_plot["ClusterName"] = df_plot["Cluster"].map(cluster_names)

    df_plot = df_plot.reset_index()
    # Séparation de l’index  "Region_Year"  → deux colonnes
    df_plot[["Region", "Year"]] = (
        df_plot["Region_year"].str.rsplit("_", n=1, expand=True)  # ← ici expand est bien nommé
        # .rename({0: "Region", 1: "Year"})
    )
    df_plot["Year"] = df_plot["Year"].astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # 1)  Génération des nœuds « année-cluster » et des liens régionaux
    # ──────────────────────────────────────────────────────────────────────────
    # (ex. nœud  "1970 – Cluster A")
    df_plot["Node"] = df_plot["Year"].astype(str) + " – " + df_plot["ClusterName"]

    # table des nœuds uniques
    nodes_df = df_plot[["Node", "Year", "ClusterName"]].drop_duplicates().reset_index(drop=True)
    node_index = {n: i for i, n in nodes_df["Node"].items()}  # nom → id entier

    # table des liens : on relie année‐t  →  année-t+1   pour chaque région
    links = []
    for reg, grp in df_plot.sort_values("Year").groupby("Region"):
        for (_, row0), (_, row1) in zip(grp.iloc[:-1].iterrows(), grp.iloc[1:].iterrows()):
            links.append(
                {
                    "source": node_index[row0["Node"]],
                    "target": node_index[row1["Node"]],
                    "value": 1,  # un passage = 1
                    "region": reg,
                    "src_clu": row0["ClusterName"],
                    "dst_clu": row1["ClusterName"],
                }
            )
    links_df = pd.DataFrame(links)

    # ──────────────────────────────────────────────────────────────────────────
    # 2)  Couleurs : une couleur par cluster (husl → rgba)
    # ──────────────────────────────────────────────────────────────────────────
    clusters_unique = nodes_df["ClusterName"].unique()
    palette = dict(
        zip(
            clusters_unique,
            sns.color_palette("husl", len(clusters_unique)).as_hex(),  # hex OK pour Plotly
        )
    )
    # couleur du lien = couleur du cluster d’arrivée
    links_df["color"] = links_df["dst_clu"].map(palette)

    # ──────────────────────────────────────────────────────────────────────────
    # 3)  Sankey Plotly
    # ──────────────────────────────────────────────────────────────────────────
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=3,
                thickness=12,
                line=dict(width=0.5, color="black"),
                label=nodes_df["ClusterName"],
                color=nodes_df["ClusterName"].map(palette),
                customdata=np.stack([nodes_df["Year"]], axis=-1),
                hovertemplate=("<b>Year :</b> %{customdata[0]}<br>"),
            ),
            link=dict(
                source=links_df["source"],
                target=links_df["target"],
                value=links_df["value"],
                color=links_df["color"],
                customdata=np.stack([links_df["region"], links_df["src_clu"], links_df["dst_clu"]], axis=-1),
                hovertemplate=(
                    "<b>Région :</b> %{customdata[0]}<br>"
                    + "<b>%{customdata[1]}</b> ➜ <b>%{customdata[2]}</b><extra></extra>"
                ),
            ),
            arrangement="snap",
        )
    )

    fig.update_layout(
        title="Transitions des régions entre clusters (1852-2014)",
        font_size=12,
        height=180,
        width=1900,
        template="plotly_white",
    )

    for i in range(len(annees_disponibles)):
        fig.add_annotation(
            x=i * 1.05 / len(annees_disponibles),
            y=-0.05,  # ou y=-0.05 pour dessous
            xref="paper",
            yref="paper",  # repère du canevas complet
            text=str(annees_disponibles[i]),
            showarrow=False,
            font=dict(size=14),
        )

    fig.show()

# %% test


def sankey_highlight_region_ordered(df_plot, highlight=None):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import seaborn as sns

    # df_plot = df_plot.reset_index()
    df_plot[["Region", "Year"]] = df_plot["Region_year"].str.rsplit("_", n=1, expand=True)
    df_plot["Year"] = df_plot["Year"].astype(int)
    df_plot["Node"] = df_plot["Year"].astype(str) + " – " + df_plot["ClusterName"]

    # Construction de nodes_df avec tri par année puis région
    nodes_df = (
        df_plot[["Node", "Year", "ClusterName", "Region"]]
        .drop_duplicates()
        .sort_values(["Year", "Region"])
        .reset_index(drop=True)
    )
    node_index = {n: i for i, n in nodes_df["Node"].items()}

    # Génération des liens avec tri cohérent
    df_plot = df_plot.sort_values(["Region", "Year"])
    links = []
    for reg, grp in df_plot.groupby("Region"):
        grp = grp.sort_values("Year")
        for (_, row0), (_, row1) in zip(grp.iloc[:-1].iterrows(), grp.iloc[1:].iterrows()):
            links.append(
                {
                    "source": node_index[row0["Node"]],
                    "target": node_index[row1["Node"]],
                    "value": 1,
                    "region": reg,
                    "src_clu": row0["ClusterName"],
                    "dst_clu": row1["ClusterName"],
                }
            )
    links_df = pd.DataFrame(links)

    # Palette pour clusters
    clusters_unique = nodes_df["ClusterName"].unique()
    cluster_palette = dict(zip(clusters_unique, sns.color_palette("husl", len(clusters_unique)).as_hex()))
    node_colors = nodes_df["ClusterName"].map(cluster_palette)

    # Gestion des couleurs des liens
    if highlight is None:
        links_df["color"] = links_df["dst_clu"].map(cluster_palette)
    else:
        regions_unique = links_df["region"].unique()
        region_palette = dict(zip(regions_unique, sns.color_palette("tab10", len(regions_unique)).as_hex()))
        links_df["color"] = links_df["region"].map(lambda r: region_palette[r] if r in highlight else "lightgray")

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=3,
                thickness=12,
                line=dict(width=0.5, color="black"),
                label=nodes_df["ClusterName"],
                color=node_colors,
                customdata=np.stack([nodes_df["Year"]], axis=-1),
                hovertemplate=("<b>Year :</b> %{customdata[0]}<br>"),
            ),
            link=dict(
                source=links_df["source"],
                target=links_df["target"],
                value=links_df["value"],
                color=links_df["color"],
                customdata=np.stack([links_df["region"], links_df["src_clu"], links_df["dst_clu"]], axis=-1),
                hovertemplate=(
                    "<b>Région :</b> %{customdata[0]}<br>"
                    + "<b>%{customdata[1]}</b> ➜ <b>%{customdata[2]}</b><extra></extra>"
                ),
            ),
            arrangement="snap",
        )
    )

    fig.update_layout(
        title="Transitions des régions entre clusters",
        font_size=12,
        height=1000,
        width=1800,
        template="plotly_white",
    )

    for i in range(len(annees_disponibles)):
        fig.add_annotation(
            x=i * 1.05 / len(annees_disponibles),
            y=-0.05,  # ou y=-0.05 pour dessous
            xref="paper",
            yref="paper",  # repère du canevas complet
            text=str(annees_disponibles[i]),
            showarrow=False,
            font=dict(size=14),
        )

    return fig


# %% Test sous forme de fonction


def sankey_highlight_region(df_plot, highlight=None):
    # df_plot = df_plot.reset_index()
    df_plot[["Region", "Year"]] = df_plot["Region_year"].str.rsplit("_", n=1, expand=True)
    df_plot["Year"] = df_plot["Year"].astype(int)
    df_plot["Node"] = df_plot["Year"].astype(str) + " – " + df_plot["ClusterName"]

    # nodes_df = df_plot[["Node", "Year", "ClusterName"]].drop_duplicates().reset_index(drop=True)
    df_plot["Region"] = df_plot["Region_year"].str.rsplit("_", n=1).str[0]
    nodes_df = (
        df_plot[["Node", "Year", "ClusterName", "Region"]]
        .drop_duplicates()
        .sort_values(["Year", "Region"])
        .reset_index(drop=True)
    )
    node_index = {n: i for i, n in nodes_df["Node"].items()}

    links = []
    for reg, grp in df_plot.sort_values("Year").groupby("Region"):
        for (_, row0), (_, row1) in zip(grp.iloc[:-1].iterrows(), grp.iloc[1:].iterrows()):
            links.append(
                {
                    "source": node_index[row0["Node"]],
                    "target": node_index[row1["Node"]],
                    "value": 1,
                    "region": reg,
                    "src_clu": row0["ClusterName"],
                    "dst_clu": row1["ClusterName"],
                }
            )
    links_df = pd.DataFrame(links)

    # Palette pour clusters (noeuds)
    clusters_unique = nodes_df["ClusterName"].unique()
    cluster_palette = dict(zip(clusters_unique, sns.color_palette("husl", len(clusters_unique)).as_hex()))

    # Palette pour régions à surligner (liens)
    region_palette = {}
    if highlight is not None:
        colors = sns.color_palette("husl", len(highlight)).as_hex()
        region_palette = dict(zip(highlight, colors))

    def get_link_color(row):
        if highlight is None:
            return cluster_palette.get(row["dst_clu"], "gray")
        return region_palette.get(row["region"], "#DDDDDD")  # gris clair si pas dans highlight

    links_df["color"] = links_df.apply(get_link_color, axis=1)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=3,
                thickness=12,
                line=dict(width=0.5, color="black"),
                label=nodes_df["ClusterName"],
                color=nodes_df["ClusterName"].map(cluster_palette),
                customdata=np.stack([nodes_df["Year"]], axis=-1),
                hovertemplate=("<b>Year :</b> %{customdata[0]}<br>"),
            ),
            link=dict(
                source=links_df["source"],
                target=links_df["target"],
                value=links_df["value"],
                color=links_df["color"],
                customdata=np.stack([links_df["region"], links_df["src_clu"], links_df["dst_clu"]], axis=-1),
                hovertemplate="<b>Région :</b> %{customdata[0]}<br><b>%{customdata[1]}</b> ➜ <b>%{customdata[2]}</b><extra></extra>",
            ),
            arrangement="snap",
        )
    )

    fig.update_layout(
        title="Transitions des régions entre clusters (1852-2014)",
        font_size=12,
        height=1080,
        width=1900,
        template="plotly_white",
    )

    for i in range(len(annees_disponibles)):
        fig.add_annotation(
            x=i * 1.05 / len(annees_disponibles),
            y=-0.05,  # ou y=-0.05 pour dessous
            xref="paper",
            yref="paper",  # repère du canevas complet
            text=str(annees_disponibles[i]),
            showarrow=False,
            font=dict(size=14),
        )

    return fig


# %%
def compute_mean_matrice(norm_matrices, df_plot):
    # On utilise norm_matrices car ce sont les matrices qui ont été la base du X pour le clustering.
    all_normed_matrices_values = list(norm_matrices.values())

    if all_normed_matrices_values:
        stacked_all_matrices = np.array(all_normed_matrices_values)
        global_mean_matrix = np.mean(stacked_all_matrices, axis=0)
        print("\n--- Matrice Moyenne Globale ---")
        print(global_mean_matrix)
        print(f"Forme de la matrice moyenne globale: {global_mean_matrix.shape}")
    else:
        print("\nAucune matrice disponible pour calculer la moyenne globale.")
        global_mean_matrix = None  # Assurer que la variable est définie

    # --- Calcul des Matrices Moyennes par Cluster ---
    mean_matrices_by_cluster = {}
    # mean_matrices_by_cluster_notnormed = {}
    for cluster_id in df_plot["Cluster"].unique():
        regions_in_cluster = df_plot[df_plot["Cluster"] == cluster_id]["Region_year"].tolist()
        matrices_in_current_cluster = []
        matrices_in_current_cluster_notnormed = []
        for region_year_key in regions_in_cluster:
            if region_year_key in norm_matrices:  # Utilisez norm_matrices pour la cohérence
                matrices_in_current_cluster.append(norm_matrices[region_year_key])
                # matrices_in_current_cluster_notnormed.append(matrices[region_year_key])
            else:
                print(f"Attention : La clé '{region_year_key}' n'a pas été trouvée dans 'norm_matrices'.")

        if matrices_in_current_cluster:
            stacked_matrices = np.array(matrices_in_current_cluster)
            mean_matrix = np.mean(stacked_matrices, axis=0)
            mean_matrices_by_cluster[f"Cluster_{cluster_id}"] = mean_matrix

            # stacked_matrices_notnormed = np.array(matrices_in_current_cluster_notnormed)
            # mean_matrix = np.mean(stacked_matrices, axis=0)
            # mean_matrices_by_cluster[f"Cluster_{cluster_id}"] = mean_matrix
        else:
            print(f"Le cluster {cluster_id} ne contient aucune matrice valide.")

    print(f"\nCalculées les moyennes pour {len(mean_matrices_by_cluster)} clusters.")
    return mean_matrices_by_cluster, global_mean_matrix


if run:
    mean_matrices_by_cluster, global_mean_matrix = compute_mean_matrice(norm_matrices, df_plot)
# %%
if run:
    if global_mean_matrix is not None and mean_matrices_by_cluster:
        n_clusters = len(mean_matrices_by_cluster)
        fig_cols = 3  # Nombre de colonnes pour l'affichage
        fig_rows = int(np.ceil(n_clusters / fig_cols))  # Calcul du nombre de lignes

        plt.figure(figsize=(fig_cols * 6, fig_rows * 5))  # Ajuster la taille de la figure

        for i, (cluster_name, cluster_mean_matrix) in enumerate(mean_matrices_by_cluster.items()):
            difference_matrix = cluster_mean_matrix - global_mean_matrix

            plt.subplot(fig_rows, fig_cols, i + 1)
            sns.heatmap(
                difference_matrix, cmap="coolwarm", center=0, annot=False, fmt=".2f", cbar=True, square=True
            )  # cmap="coolwarm" pour différences, center=0 pour le point neutre
            plt.title(f"Différence: {cluster_name} - Moyenne Globale")
            plt.xlabel("Colonne de la Matrice")
            plt.ylabel("Ligne de la Matrice")

        plt.tight_layout()
        plt.show()

        # --- Dendrogramme pour référence (votre code existant) ---
        plt.figure(figsize=(80, 4), dpi=500)
        dendrogram(Zc, labels=regions_order, leaf_rotation=90, color_threshold=seuil)
        plt.ylabel("Dissimilarity")
        plt.title(f"Dendrogramme de Clusterisation (Seuil: {seuil})")
        plt.axhline(y=seuil, color="r", linestyle="--", label=f"Seuil de coupure ({seuil})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        print("\nImpossible d'afficher les heatmaps : matrices moyennes non disponibles.")
# %%
if run:
    # Définir les labels des nœuds (lignes/colonnes de la matrice de transition)
    # Ces labels sont supposés être constants pour toutes vos matrices de transition.
    # Si votre NitrogenFlowModel a un attribut self.labels, utilisez-le.
    # Sinon, utilisez une liste générique basée sur n_nodes.
    if "model" in locals() and hasattr(model, "labels"):
        node_labels = model.labels
    else:
        # Fallback si model.labels n'est pas défini ou n'existe pas
        # Assurez-vous que n_nodes est correctement défini plus haut dans votre code
        n_nodes_example = matrices[list(matrices.keys())[0]].shape[0] if matrices else 10
        node_labels = [f"N_Pool_{i + 1}" for i in range(n_nodes_example)]


def plot_matrix_heatmap(matrix_to_plot, title, node_labels_list):
    """
    Génère une heatmap Plotly pour une matrice de valeurs absolues (avec log10).
    Adapté de votre fonction plot_heatmap_interactive.
    """
    x_labels = list(range(1, len(node_labels_list) + 1))
    y_labels = list(range(1, len(node_labels_list) + 1))

    # Assurez-vous que la matrice a la bonne dimension si elle a été tronquée
    adjacency_subset = matrix_to_plot[: len(node_labels_list), : len(node_labels_list)]

    # Gestion min/max et transformation log10
    # Éviter les zéros ou négatifs pour log10
    cmin = max(1e-4, np.min(adjacency_subset[adjacency_subset > 0]))
    cmax = np.max(adjacency_subset)
    if cmax == 0:
        cmax = 1  # Éviter log(0) si toutes les valeurs sont 0
    log_matrix = np.where(adjacency_subset > 0, np.log10(adjacency_subset), np.nan)

    # Construire un tableau 2D de chaînes pour le survol
    strings_matrix = []
    for row_i, y_val in enumerate(y_labels):
        row_texts = []
        for col_i, x_val in enumerate(x_labels):
            real_val = adjacency_subset[row_i, col_i]
            real_val_str = f"{real_val:.2e}" if not np.isnan(real_val) else "0"
            tooltip_str = (
                f"Source : {node_labels_list[y_val - 1]}<br>"
                f"Target : {node_labels_list[x_val - 1]}<br>"
                f"Value  : {real_val_str} ktN/yr"
            )
            row_texts.append(tooltip_str)
        strings_matrix.append(row_texts)

    trace = go.Heatmap(
        z=log_matrix,
        x=x_labels,
        y=y_labels,
        colorscale="Plasma_r",
        zmin=np.log10(cmin),
        zmax=np.log10(cmax),
        text=strings_matrix,
        hoverinfo="text",
        colorbar=dict(
            title="ktN/year (log scale)",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            thickness=15,
            len=1,
        ),
    )

    fig = go.Figure(data=[trace])

    # Discrétisation manuelle des ticks sur la colorbar
    tickvals = np.linspace(np.floor(np.log10(cmin)), np.ceil(np.log10(cmax)), num=7)
    # Filtrer les ticktext pour ne pas avoir de valeurs trop petites ou trop grandes si cmin/cmax sont extrêmes
    valid_ticktext_values = [v for v in [10**x for x in range(-4, 3, 1)] if cmin <= v <= cmax]
    ticktext = [f"{v:.1e}" for v in valid_ticktext_values]
    tickvals_filtered = [np.log10(v) for v in valid_ticktext_values]  # Convert back to log scale for tickvals

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
            tickvals=tickvals_filtered,
            ticktext=ticktext,
        )
    )

    fig.update_layout(
        width=1000,
        height=1000,
        margin=dict(t=50, b=100, l=0, r=220),
        title_text=title,
        title_x=0.5,  # Titre centré
    )
    fig.update_layout(yaxis_scaleanchor="x")

    fig.update_xaxes(
        title="Target",
        side="top",
        tickangle=90,
        tickmode="array",
        tickfont=dict(size=10),
        tickvals=x_labels,
        ticktext=[str(x) for x in x_labels],
    )
    fig.update_yaxes(
        title="Source",
        autorange="reversed",
        tickmode="array",
        tickfont=dict(size=10),
        tickvals=y_labels,
        ticktext=[str(y) for y in y_labels],
    )

    legend_text = "<br>".join(f"{i + 1} : {lbl}" for i, lbl in enumerate(node_labels_list))
    fig.add_annotation(
        x=1.25,
        y=0.45,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=legend_text,
        align="left",
        valign="middle",
        font=dict(size=11),
        bordercolor="rgba(0,0,0,0)",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_difference_heatmap(matrix_to_plot, title, node_labels_list):
    """
    Génère une heatmap Plotly pour une matrice de différences (échelle linéaire, centrée sur 0).
    """
    x_labels = list(range(1, len(node_labels_list) + 1))
    y_labels = list(range(1, len(node_labels_list) + 1))

    # Construire un tableau 2D de chaînes pour le survol
    strings_matrix = []
    for row_i, y_val in enumerate(y_labels):
        row_texts = []
        for col_i, x_val in enumerate(x_labels):
            real_val = matrix_to_plot[row_i, col_i]
            real_val_str = f"{real_val:.2e}"
            tooltip_str = (
                f"Source : {node_labels_list[y_val - 1]}<br>"
                f"Target : {node_labels_list[x_val - 1]}<br>"
                f"Différence : {real_val_str} ktN/yr"
            )
            row_texts.append(tooltip_str)
        strings_matrix.append(row_texts)

    # Déterminer la plage symétrique pour la colorbar
    max_abs_val = np.max(np.abs(matrix_to_plot))

    trace = go.Heatmap(
        z=matrix_to_plot,
        x=x_labels,
        y=y_labels,
        colorscale="RdBu",  # Rouge-Bleu est bon pour les différences
        zmin=-max_abs_val,
        zmax=max_abs_val,
        text=strings_matrix,
        hoverinfo="text",
        colorbar=dict(
            title="Différence (ktN/year)",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            thickness=15,
            len=1,
        ),
    )

    fig = go.Figure(data=[trace])

    fig.update_layout(width=1000, height=1000, margin=dict(t=50, b=100, l=0, r=220), title_text=title, title_x=0.5)
    fig.update_layout(yaxis_scaleanchor="x")

    fig.update_xaxes(
        title="Target",
        side="top",
        tickangle=90,
        tickmode="array",
        tickfont=dict(size=10),
        tickvals=x_labels,
        ticktext=[str(x) for x in x_labels],
    )
    fig.update_yaxes(
        title="Source",
        autorange="reversed",
        tickmode="array",
        tickfont=dict(size=10),
        tickvals=y_labels,
        ticktext=[str(y) for y in y_labels],
    )

    legend_text = "<br>".join(f"{i + 1} : {lbl}" for i, lbl in enumerate(node_labels_list))
    fig.add_annotation(
        x=1.25,
        y=0.45,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=legend_text,
        align="left",
        valign="middle",
        font=dict(size=11),
        bordercolor="rgba(0,0,0,0)",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --- 5. Fonction pour afficher les heatmaps directement ---


def display_cluster_heatmap(cluster_id, plot_type="mean"):
    """
    Affiche la heatmap d'une matrice moyenne de cluster ou de sa différence avec la moyenne globale.

    Args:
        cluster_id (int or str): L'ID numérique du cluster (ex: 1, 2, 3) ou la chaîne "global" pour la moyenne globale.
        plot_type (str): 'mean' pour afficher la matrice moyenne du cluster,
                         'diff' pour afficher la différence avec la moyenne globale.
    """
    if cluster_id == "global":
        if global_mean_matrix is not None:
            title = "Matrice Moyenne Globale de Tous les Systèmes"
            fig = plot_matrix_heatmap(global_mean_matrix, title, node_labels)
            fig.show()
        else:
            print("La matrice moyenne globale n'est pas disponible.")
    else:
        cluster_name = f"Cluster_{cluster_id}"
        if cluster_name in mean_matrices_by_cluster:
            if plot_type == "mean":
                title = f"Matrice Moyenne pour {cluster_name}"
                fig = plot_matrix_heatmap(mean_matrices_by_cluster[cluster_name], title, node_labels)
                fig.show()
            elif plot_type == "diff":
                if global_mean_matrix is not None:
                    diff_matrix = mean_matrices_by_cluster[cluster_name] - global_mean_matrix
                    title = f"Différence : {cluster_name} - Moyenne Globale"
                    fig = plot_difference_heatmap(diff_matrix, title, node_labels)
                    fig.show()
                else:
                    print("La matrice moyenne globale n'est pas disponible pour le calcul de différence.")
            else:
                print("Type de plot invalide. Utilisez 'mean' ou 'diff'.")
        else:
            print(f"Le cluster {cluster_name} n'a pas de matrice moyenne.")


# --- Exemple d'utilisation (vous pouvez appeler cette fonction dans de nouvelles cellules) ---
# print("\nExemples d'utilisation de la fonction display_cluster_heatmap :")
# # Afficher la matrice moyenne du Cluster 1
# display_cluster_heatmap(1, 'mean')

# # Afficher la différence du Cluster 2 avec la moyenne globale
# display_cluster_heatmap(2, 'diff')

# # Afficher la matrice moyenne globale
# display_cluster_heatmap("global", 'mean')
# %%


def merge_nodes(adjacency_matrix, labels, merges):
    """
    Fusionne des groupes de nœuds (labels) dans la matrice d'adjacence.

    :param adjacency_matrix: np.array carré de taille (n, n) avec flux de i vers j
    :param labels: liste des labels d'origine, de longueur n
    :param merges: dict indiquant les fusions à faire. Exemple :
                   {
                       "population": ["urban", "rural"],
                       "livestock":  ["bovines", "ovines", "equine", "poultry", "porcines", "caprines"],
                       "industry":   ["haber-bosch", "other sectors"]
                   }
    :return:
      - new_matrix: la matrice d'adjacence après fusion
      - new_labels: la liste des labels après fusion
      - old_to_new: dict qui mappe index d'origine -> nouvel index
    """
    n = len(labels)

    # 1) Construire un mapping "label d'origine" -> "label fusionné"
    #    s'il est mentionné dans merges; sinon, il reste tel quel
    merged_label_map = {}
    for group_label, group_list in merges.items():
        for lbl in group_list:
            merged_label_map[lbl] = group_label

    def get_merged_label(lbl):
        # Si le label apparaît dans merges, on retourne le label fusionné
        # Sinon on le laisse tel quel
        return merged_label_map[lbl] if lbl in merged_label_map else lbl

    # 2) Construire la liste de tous les "nouveaux" labels
    #    On peut d'abord faire un set, puis trier pour stabilité
    new_label_set = set()
    for lbl in labels:
        new_label_set.add(get_merged_label(lbl))
    new_labels = sorted(list(new_label_set))

    # 3) Créer un mapping "new_label" -> "nouvel index"
    new_label_to_index = {lbl: i for i, lbl in enumerate(new_labels)}

    # 4) Construire la nouvelle matrice
    #    On fait une somme des flux entre les groupes
    new_n = len(new_labels)
    new_matrix = np.zeros((new_n, new_n))

    # 5) Construire un dict old_to_new : index d'origine -> index fusionné
    old_to_new = {}

    for old_i in range(n):
        old_label = labels[old_i]
        merged_i_label = get_merged_label(old_label)
        i_new = new_label_to_index[merged_i_label]
        old_to_new[old_i] = i_new

    # 6) Parcourir la matrice d'origine pour agréger les flux
    for i in range(n):
        for j in range(n):
            flow = adjacency_matrix[i, j]
            if flow != 0:
                i_new = old_to_new[i]
                j_new = old_to_new[j]
                new_matrix[i_new, j_new] += flow

    return new_matrix, new_labels, old_to_new


def compute_node_positions(node_labels_to_position, label_to_x, y_spacing=0.05):
    """
    Calcule les positions x et y pour les nœuds d'un diagramme de Sankey,
    en utilisant un mapping de labels vers des coordonnées x souhaitées.
    Les positions y sont calculées pour espacer les nœuds dans chaque colonne.

    :param node_labels_to_position: La liste des labels des nœuds pour lesquels calculer les positions.
                                    Cette liste doit être l'ordre final des labels dans le Sankey.
    :param label_to_x: Dictionnaire mappant les labels (normalisés en minuscules/sans espaces)
                       à leurs positions x (0.0 à 1.0).
    :param y_spacing: L'espacement minimum entre les nœuds sur l'axe y dans une même colonne.
    :return: Une paire de listes (final_x, final_y) des coordonnées x et y des nœuds.
    """
    x_pos_by_label = {}
    col_to_labels_sorted = defaultdict(list)

    # 1. Déterminer la position X pour chaque label affiché et regrouper par colonne
    # Utilisation d'un dictionnaire intermédiaire pour conserver l'ordre de `node_labels_to_position`
    # tout en remplissant `col_to_labels_sorted` avec des listes ordonnées.
    for label in node_labels_to_position:
        label_key = label.lower().strip()
        x = label_to_x.get(label_key, 0.5)  # Default to 0.5 if not found
        x_pos_by_label[label] = x
        col_to_labels_sorted[x].append(label)

    # 2. Calculer les positions Y pour chaque nœud dans chaque colonne
    final_x = []
    final_y = []

    # Créer un dictionnaire pour stocker les positions y calculées pour chaque label
    # afin de les récupérer dans l'ordre de `node_labels_to_position`
    calculated_y_positions = {}

    for x_coord in sorted(col_to_labels_sorted.keys()):
        labels_in_current_col = col_to_labels_sorted[x_coord]
        num_labels_in_col = len(labels_in_current_col)

        # Calculer l'espace total nécessaire pour cette colonne
        total_y_height_needed = num_labels_in_col * y_spacing

        # Calculer le point de départ pour centrer les nœuds verticalement
        # Nous voulons que les nœuds s'étendent sur tout l'axe Y (0 à 1)
        # Assurez-vous que l'espacement ne pousse pas les nœuds hors de la zone visible [0, 1]
        if num_labels_in_col > 1:
            # Distribue les nœuds uniformément sur l'axe Y (0 à 1)
            # Ajout d'un petit décalage pour ne pas commencer à 0 ou finir à 1 exactement
            y_increment = (1.0 - 2 * y_spacing) / (num_labels_in_col - 1)
            y_start = y_spacing
        else:
            y_increment = 0
            y_start = 0.5  # Center single node in column

        for i, label in enumerate(labels_in_current_col):
            calculated_y_positions[label] = y_start + i * y_increment

    # 3. Remplir final_x et final_y dans l'ordre des `node_labels_to_position`
    for label in node_labels_to_position:
        final_x.append(x_pos_by_label[label])
        final_y.append(calculated_y_positions.get(label, 0.5))  # Fallback to 0.5 if somehow not calculated

    return final_x, final_y


label_to_x = {
    "atmospheric n2": 0.0,
    "haber-bosch": 0.25,
    "leguminous": 0.25,
    "cereals (excluding rice)": 0.5,
    "oleaginous": 0.5,
    "fruits and vegetables": 0.5,
    "roots": 0.5,
    "forages": 0.5,
    "fishery products": 0.5,
    "temporary meadows": 0.5,
    "natural meadows ": 0.5,
    "monogastrics": 0.75,
    "ruminants": 0.75,
    "population": 0.9,
    "trade": 0.75,
    "soil stock": 0.75,
    "other sectors": 0.75,
    "environment": 1.0,
}


def sankey_systemic_flows(
    adjacency_matrix,
    labels=labels,
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
            "atmospheric N2",
        ],
        "roots": ["Sugar beet", "Potatoes", "Other roots"],
    },
    label_to_x=label_to_x,
):
    """
    Crée un diagramme de Sankey systémique montrant tous les flux à partir d'une matrice d'adjacence et de labels.
    Les nœuds sont fusionnés selon les règles de `merges`, et les flux inférieurs à `THRESHOLD` sont éliminés.
    :param adjacency_matrix: La matrice d'adjacence des flux.
    :param labels: La liste des labels correspondant aux lignes/colonnes de la matrice.
    :param merges: Dictionnaire définissant les fusions de nœuds.
    :param THRESHOLD: Seuil en dessous duquel les flux sont supprimés (par défaut : 1e-1).
    :return: Une figure Plotly du diagramme de Sankey.
    """
    # 1) Fusion des nœuds
    new_matrix, new_labels, old_to_new = merge_nodes(adjacency_matrix, labels, merges)
    n_new = len(new_labels)
    THRESHOLD = 0.01
    # 2) Définir les couleurs des nœuds fusionnés
    color_dict = {
        "cereals (excluding rice)": "gold",
        "fruits and vegetables": "lightgreen",
        "leguminous": "darkgreen",
        "oleaginous": "lightgreen",
        "meadow and forage": "green",
        "trade": "gray",
        "monogastrics": "lightblue",
        "ruminants": "lightblue",
        "population": "darkblue",
        "losses": "crimson",
        "roots": "orange",
        "forages": "limegreen",
        "Environment": "crimson",
        "temporary meadows": "seagreen",
        "natural meadows ": "darkgreen",
        "soil stock": "sienna",
        "Haber-Bosch": "purple",
        "atmospheric N2": "seagreen",
    }
    # node_color et index_to_label ne sont pas définis ici.
    # Si des couleurs spécifiques pour les labels d'origine sont nécessaires,
    # assurez-vous qu'elles sont passées ou définies dans le scope.
    # Pour l'instant, je les ignore ou utilise une couleur par défaut.

    default_color = "gray"

    def get_color_for_label(lbl):
        return color_dict.get(lbl, default_color)

    new_node_colors = [get_color_for_label(lbl) for lbl in new_labels]

    # 3) Collecter tous les flux de la matrice fusionnée
    sources_raw = []
    targets_raw = []
    values = []
    link_colors = []
    link_hover_texts = []

    def format_scientific(value):
        return f"{value:.2e} ktN/yr"

    for s_idx in range(n_new):
        for t_idx in range(n_new):
            flow = new_matrix[s_idx, t_idx]
            if flow > THRESHOLD:  # Seuil pour éliminer les petits flux
                sources_raw.append(s_idx)
                targets_raw.append(t_idx)
                values.append(flow)
                link_colors.append(new_node_colors[s_idx])  # Couleur des liens selon la source
                link_hover_texts.append(
                    f"Source: {new_labels[s_idx]}<br>Target: {new_labels[t_idx]}<br>Value: {format_scientific(flow)}"
                )

    # 4) Calcul du throughflow pour chaque nœud (flux entrants + sortants)
    throughflows = np.sum(new_matrix, axis=0) + np.sum(new_matrix, axis=1)

    # 5) Filtrage des nœuds avec throughflow < THRESHOLD
    kept_nodes = [i for i in range(n_new) if throughflows[i] >= THRESHOLD]

    # Filtrer les flux qui impliquent des nœuds supprimés
    final_links = [
        idx for idx in range(len(sources_raw)) if sources_raw[idx] in kept_nodes and targets_raw[idx] in kept_nodes
    ]
    sources_raw = [sources_raw[i] for i in final_links]
    targets_raw = [targets_raw[i] for i in final_links]
    values = [values[i] for i in final_links]
    link_colors = [link_colors[i] for i in final_links]
    link_hover_texts = [link_hover_texts[i] for i in final_links]

    # 6) Re-mappage des indices pour le Sankey
    unique_final_nodes = []
    for idx in sources_raw + targets_raw:
        if idx not in unique_final_nodes:
            unique_final_nodes.append(idx)
    node_map = {old_i: new_i for new_i, old_i in enumerate(unique_final_nodes)}
    sankey_sources = [node_map[s] for s in sources_raw]
    sankey_targets = [node_map[t] for t in targets_raw]

    # 7) Création des labels et couleurs finaux pour les nœuds
    node_labels = [new_labels[idx] for idx in unique_final_nodes]
    node_final_colors = [new_node_colors[idx] for idx in unique_final_nodes]
    node_hover_data = [
        f"Node: {new_labels[idx]}<br>Throughflow: {format_scientific(throughflows[idx])}" for idx in unique_final_nodes
    ]

    # 8) Calcul des positions X/Y si un mapping est fourni
    if label_to_x:
        sankey_x, sankey_y = compute_node_positions(
            [new_labels[i] for i in unique_final_nodes],
            label_to_x={k.lower().strip(): v for k, v in label_to_x.items()},  # normalisation
        )
    else:
        sankey_x = sankey_y = None

    # 8) Création du Sankey final
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_final_colors,
                customdata=node_hover_data,
                hovertemplate="%{customdata}<extra></extra>",
                **({"x": sankey_x, "y": sankey_y} if sankey_x and sankey_y else {}),  # 👈 ajout conditionnel
            ),
            link=dict(
                source=sankey_sources,
                target=sankey_targets,
                value=values,
                color=link_colors,
                customdata=link_hover_texts,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            arrangement="snap",
        )
    )
    fig.update_layout(
        template="plotly_dark",  # Thème sombre de Plotly
        font_color="white",  # Couleur générale du texte (titres, axes, etc.)
        width=2000,
        height=750,
    )

    return fig, new_labels, new_matrix


# %%
