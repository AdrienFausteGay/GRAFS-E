# %%

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
data = DataLoader()

# %%
matrices = {}

for reg in regions:  # « regions » est ta liste de 33 régions
    for year in annees_disponibles:
        model = NitrogenFlowModel(data, year, reg)
        T = model.get_transition_matrix()  # (n_nodes × n_nodes)
        matrices[reg + "_" + year] = T.astype(float)
# %%
# %% Normalisation
# Global M/T
norm_matrices = {}
for reg, M in matrices.items():
    s = M.sum()
    norm_matrices[reg] = M / s if s else M

# Leontieff-like normalization: divide each column by its total
# norm_matrices = {}
# for reg, M in matrices.items():
#     col_sums = M.sum(axis=0)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         norm_M = np.divide(M, col_sums, where=col_sums != 0)  # avoid division by 0
#         norm_M[np.isnan(norm_M)] = 0  # replace NaNs from division by zero
#     norm_matrices[reg] = norm_M

# %% vectorization
# X = np.vstack([M.flatten() for M in norm_matrices.values()])  # shape = (33, n*n)
# regions_order = list(norm_matrices.keys())

X = np.vstack([M.flatten() for M in norm_matrices.values()])
regions_order = list(norm_matrices.keys())

X_scaled = StandardScaler().fit_transform(X)

# %% dictionnaire de modèles
models = {}

for reg in regions:  # « regions » est ta liste de 33 régions
    for year in annees_disponibles:
        model = NitrogenFlowModel(data, year, reg)
        models[reg + "_" + year] = model

# %% Clusterisation par indicateur
rows = []

for reg in regions:
    for year in annees_disponibles:
        # m = NitrogenFlowModel(data, "2014", reg)
        m = models[reg + "_" + year]

        # --- quelques helpers ------------
        def _safe_sum(x):
            return np.nansum(list(x.values())) if isinstance(x, dict) else x

        # Y moyen sur 4 cultures repères
        cultures_ref = ["Wheat", "Barley", "Rapeseed", "Forage maize"]
        y_vals = [m.Y(c) for c in cultures_ref if m.Y(c) > 0 and m.Y(c) < 200]
        Y_mean = np.mean(y_vals)

        surf = m.surfaces()
        rows.append(
            {
                "Region_year": reg + "_" + year,
                "Yield_mean": Y_mean,
                "Tot_fert": m.tot_fert(),
                "ImpN_ratio": m.imported_nitrogen() / max(m.total_plant_production(), 1e-6),
                "Net_footp": m.net_footprint(),
                "Emissions": m.emissions(),
                "NUE": m.NUE_system(),
                "Grass_share": surf.get("Natural meadow ", 0) / max(m.surfaces_tot(), 1e-6),
                "Imp_anim%": m.net_imported_animal() / max(m.animal_production(), 1e-6),
            }
        )

df = pd.DataFrame(rows).set_index("Region_year")


# %%
def expand_dict_column(df, col, prefix):
    # transforme chaque dict en Series puis concatène
    expanded = df[col].apply(lambda d: pd.Series(d)).add_prefix(f"{prefix}_")
    return df.drop(columns=[col]).join(expanded)


df = expand_dict_column(df, "Emissions", "E")
df = expand_dict_column(df, "Tot_fert", "F")
# %%
df_numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0)

df_numeric = df_numeric[
    ["Yield_mean", "Net_footp", "Grass_share", "E_NH3 volatilization", "F_atmospheric N2", "F_Haber-Bosch"]
]
# %%
X = StandardScaler().fit_transform(df_numeric)  # toutes colonnes désormais numériques
# %% Clusterisation
Z = linkage(X, method="complete")  # ou "average", "complete", …
Z2 = linkage(normalize(X, norm="l2"), method="ward")
Z3 = linkage(MaxAbsScaler().fit_transform(X), method="ward")
D = pdist(X, metric="cosine")
Zc = linkage(D, method="average")
plt.figure(figsize=(80, 4), dpi=500)
dendrogram(Zc, labels=regions_order, leaf_rotation=90, color_threshold=0.27)
plt.ylabel("Dissimilarity")
plt.tight_layout()
plt.show()

# Si tu veux garder, disons, 4 clusters après inspection visuelle :
# clt = AgglomerativeClustering(n_clusters=4, metric="euclidean", linkage="ward")
# labels = clt.fit_predict(X)
# df_clusters = pd.DataFrame({"Region": regions_order, "Cluster": labels}).sort_values("Cluster")

labels = fcluster(Zc, t=0.27, criterion="distance")
df_plot = pd.DataFrame(
    {"Region_year": regions_order, "Cluster": labels, "ClusterName": labels.astype(str)}
).sort_values("Cluster")
# %%

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
    )
)

fig.update_layout(
    title="Transitions des régions entre clusters (1852-2014)",
    font_size=12,
    height=1080,
    width=1900,
    template="plotly_white",
)

fig.show()
# %%
