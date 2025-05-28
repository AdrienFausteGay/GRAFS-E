# %%
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, normalize

from grafs_e.donnees import *
from grafs_e.N_class import DataLoader, NitrogenFlowModel

# %%

data = DataLoader()

# %%
matrices = {}

for reg in regions:  # « regions » est ta liste de 33 régions
    for year in annees_disponibles:
        if reg == "Savoie" and year == "1852":
            pass
        model = NitrogenFlowModel(data, year, reg)
        T = model.get_transition_matrix()  # (n_nodes × n_nodes)
        matrices[reg + "_" + year] = T.astype(float)

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
X_normalized_l2 = normalize(X, norm="l2", axis=1)
regions_order = list(norm_matrices.keys())

X_scaled = StandardScaler().fit_transform(X)
# %% Clusterisation

Z = linkage(X, method="complete")  # ou "average", "complete", …
Z2 = linkage(normalize(X, norm="l2"), method="ward")
Z3 = linkage(MaxAbsScaler().fit_transform(X), method="ward")
D = pdist(X_normalized_l2, metric="cosine")
Zc = linkage(D, method="average")
plt.figure(figsize=(80, 4), dpi=500)
dendrogram(Zc, labels=regions_order, leaf_rotation=90, color_threshold=0.24)
plt.ylabel("Dissimilarity")
plt.tight_layout()
plt.show()

# Si tu veux garder, disons, 4 clusters après inspection visuelle :
# clt = AgglomerativeClustering(n_clusters=4, metric="euclidean", linkage="ward")
# labels = clt.fit_predict(X)
# df_clusters = pd.DataFrame({"Region": regions_order, "Cluster": labels}).sort_values("Cluster")

labels = fcluster(Zc, t=0.24, criterion="distance")
df_plot = pd.DataFrame(
    {"Region_year": regions_order, "Cluster": labels, "ClusterName": labels.astype(str)}
).sort_values("Cluster")

# %% Visualisation des clusters en fonction des indicateurs

rows = []
for reg, lab in zip(regions, labels):  # `labels` vient du clustering précédent
    m = NitrogenFlowModel(data, "2014", reg)
    rows.append(
        {
            "Region": reg,
            "Cluster": lab,
            "Haber-Bosch": m.rel_fert()["Mining"],  # kt  → Mt
            "Leguminous fertilization": m.tot_fert()["Leguminous soil enrichment"],
            "NUE": m.NUE(),  # rendement global
            # "ImpN_ratio"   : m.imported_nitrogen(),        # 0-1
            # "Animal_share" : m.animal_production() /
            #  m.total_plant_production(),
            # "Self-sufficiency": m.N_self_sufficient(),
            # "Animal production": m.animal_production(),
            "Relative leguminous": m.leguminous_production_r(),
            "Net_footprint": m.net_footprint(),
        }
    )

df_indic = pd.DataFrame(rows).set_index("Region").sort_values("Cluster")

(
    sns.pairplot(
        df_indic,
        vars=[
            "Haber-Bosch",
            "Leguminous fertilization",
            "NUE",
            # "Self-sufficiency",
            "Relative leguminous",
            "Net_footprint",
        ],
        hue="Cluster",
        palette="tab10",
        diag_kind="kde",
        plot_kws=dict(s=80, edgecolor="k", linewidth=0.3),
        diag_kws=dict(common_norm=False),
    ),
)
plt.suptitle("Position des régions dans l'espace des 5 indicateurs", y=1.02)
plt.show()

# %% Recherche des indicateurs les plus pertinents pour expliquer la clusterisation

rows = []
models = {
    reg: NitrogenFlowModel(  # ➜ instance pour chaque région
        data=data,  # le chargeur de données
        year="2014",  # année choisie
        region=reg,  # la région courante
    )
    for reg in regions
}
scalar_funcs = [
    "imported_nitrogen",
    "net_imported_plant",
    "net_imported_animal",
    "total_plant_production",
    "cereals_production",
    "leguminous_production",
    "oleaginous_production",
    "grassland_and_forages_production",
    "roots_production",
    "fruits_and_vegetable_production",
    "cereals_production_r",
    "leguminous_production_r",
    "oleaginous_production_r",
    "grassland_and_forages_production_r",
    "roots_production_r",
    "fruits_and_vegetable_production_r",
    "animal_production",
    "surfaces_tot",
    "N_eff",
    "C_eff",
    "F_eff",
    "R_eff",
    "NUE",
    "NUE_system",
    "NUE_system_2",
    "N_self_sufficient",
    "primXsec",
    "net_footprint",
]

series_funcs = [
    ("emissions", "EMI_"),  #  ➜ N2O …, NH3… (= 3 colonnes préfixées EMI_)
    ("tot_fert", "FERT_"),  #  ➜ Mining, Seeds, … (= 8 colonnes FERT_)
    ("rel_fert", "FERTrel_"),  #  ➜ parts en %        (= 8 colonnes FERTrel_)
]
for reg, mdl in models.items():
    row = {"Region": reg}

    # ---- indicateurs scalaires ---------------------------------------------
    for f in scalar_funcs:
        try:
            row[f] = getattr(mdl, f)()
        except Exception as err:
            print(f"[{reg}]  ⚠️ {f} impossible : {err}")
            row[f] = pd.NA  # on mettra NaN ensuite

    # ---- indicateurs Series / dict -----------------------------------------
    for func_name, prefix in series_funcs:
        try:
            s = getattr(mdl, func_name)()  # Series ou dict
            if isinstance(s, dict):
                s = pd.Series(s)
            for k, v in s.items():
                row[f"{prefix}{k}"] = v
        except Exception as err:
            print(f"[{reg}]  ⚠️ {func_name} impossible : {err}")

    rows.append(row)

df_ind = (
    pd.DataFrame(rows)
    .set_index("Region")
    .apply(pd.to_numeric, errors="coerce")  # force en numérique, NaN sinon
    .sort_index()
)

X = df_ind.copy()
y = df_clusters["Cluster"]  # mêmes régions, même ordre

# ─── standardisation
Xz = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)

mask = pd.Series(y, index=Xz.index)  # même index que Xz
# ─── 1) ANOVA F-score
anova = {
    col: f_oneway(
        *[
            Xz.loc[mask == c, col].values  # 1-D ndarray
            for c in np.unique(y)
        ]
    ).statistic
    for col in Xz.columns
}

# ─── 2) Silhouette 1-D
silh = {col: silhouette_samples(Xz[[col]].values, y).mean() for col in Xz.columns}

# ─── 3) Random-Forest importance + permutation
rf = RandomForestClassifier(n_estimators=500, random_state=0).fit(Xz, y)
rf_imp = dict(zip(Xz.columns, rf.feature_importances_))
perm_imp = dict(
    zip(
        Xz.columns,
        permutation_importance(rf, Xz, y, n_repeats=200, random_state=0).importances_mean,
    )
)

# ─── 4) Mutual information
mi = dict(zip(Xz.columns, mutual_info_classif(Xz, y, random_state=0)))

# ─── synthèse des rangs
rank = (
    pd.DataFrame({"ANOVA": anova, "Silh": silh, "RF": rf_imp, "Perm": perm_imp, "MI": mi})
    .rank(ascending=False)
    .mean(axis=1)
    .sort_values()
)

print("Indicateurs les plus discriminants :\n", rank.head(15))

best = rank.index[:4]  # les 4 plus explicatifs
sns.pairplot(df_ind.join(y), vars=best, hue="Cluster", palette="tab10", height=2.5)
plt.suptitle("Indicateurs les plus discriminants des clusters", y=1.02)
plt.show()

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

# %% Affichage

X = StandardScaler().fit_transform(df_numeric)  # toutes colonnes désormais numériques
# D = pdist(X, metric="cosine")
# Zc = linkage(D, method="average")  # ou "ward", "complete", …

Zc = linkage(X, method="ward")

plt.figure(figsize=(100, 4))
dendrogram(Zc, labels=df_numeric.index, leaf_rotation=90, color_threshold=0.6)
plt.ylabel("Dissimilarity")
plt.show()

# %% Visualisation des liens avec indicateurs

labels = fcluster(Zc, t=12, criterion="distance")
# ou   labels  = fcluster(Z, t=4,  criterion='maxclust')

df_numeric["Cluster"] = labels

# palette de couleurs, une couleur / cluster
n_clust = df_numeric["Cluster"].nunique()
palette = sns.color_palette("tab10", n_clust)
lut = dict(zip(sorted(df_numeric["Cluster"].unique()), palette))
row_colors = df_numeric["Cluster"].map(lut)  # couleur associée à chaque région

g = sns.clustermap(
    df_numeric.drop(columns="Cluster"),
    row_linkage=Zc,
    col_cluster=False,
    cmap="vlag",
    center=0,
    standard_scale=1,
    figsize=(10, 10),
    row_colors=row_colors,  # ← la bande de couleur à gauche
)

# Ajouter la légende des couleurs
handles = [Patch(facecolor=col, label=f"Cluster {k}") for k, col in lut.items()]
g.ax_row_dendrogram.legend(
    handles=handles,
    loc="upper right",
    ncol=1,
    title="Clusters",
    bbox_to_anchor=(2, 1.3),
)
plt.show()

# %% Rapport des indicateurs par cluster

summary = (
    df_numeric.groupby("Cluster")
    .agg(["mean", "std"])  # moyenne & écart-type par variable
    .round(2)
)
summary["name"] = list(string.ascii_uppercase[: len(summary)])
# %% Visualisation des trajectoires

# ─── on remet Region / Year & Cluster dans le même DF
df_plot = df_numeric.copy()

df_plot = df_plot.reset_index()

df_plot = df_plot.rename(columns={"index": "Region_year"})

df_plot[["Region", "Year"]] = df_plot["Region_year"].str.rsplit("_", n=1, expand=True)
df_plot["Year"] = df_plot["Year"].astype(int)

# %%

# ────────────────── 3) visualisation des trajectoires ───────────────────
palette = dict(zip(regions, sns.color_palette("husl", n_colors=len(regions))))

fig, ax = plt.subplots(figsize=(12, 8))

# a) nuage de points (x=cluster, y=année)
sns.scatterplot(data=df_plot, x="Cluster", y="Year", hue="Region", palette=palette, s=90, edgecolor="k", ax=ax)

# b) segments région → région (années ordonnées)
for reg, grp in df_plot.sort_values("Year").groupby("Region"):
    ax.plot(grp["Cluster"], grp["Year"], color=palette[reg], linewidth=1, alpha=0.4)

ax.set(xlabel="Cluster assigné", ylabel="Année", title="Trajectoires régionales dans l’espace des clusters")
ax.invert_yaxis()  # année la plus ancienne en haut
ax.grid(alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Région")

plt.tight_layout()
plt.show()

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
# %%
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
    dendrogram(Zc, labels=regions_order, leaf_rotation=90, color_threshold=my_threshold)
    plt.ylabel("Dissimilarity")
    plt.title(f"Dendrogramme de Clusterisation (Seuil: {my_threshold})")
    plt.axhline(y=my_threshold, color="r", linestyle="--", label=f"Seuil de coupure ({my_threshold})")
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    print("\nImpossible d'afficher les heatmaps : matrices moyennes non disponibles.")
# %%

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
        "oleaginous": [
            "Rapeseed",
            "Sunflower",
            "Other oil crops",
        ],
        "forages": [
            "Forage maize",
            "Forage cabbages",
            "Straw",
        ],
        "temporary meadows": ["Non-legume temporary meadow", "Alfalfa and clover"],
        "natural meadows ": ["Natural meadows "],
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
        height=1500,
    )

    return fig


# %%
