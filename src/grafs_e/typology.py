# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway
from sklearn.cluster import AgglomerativeClustering
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
regions_order = list(norm_matrices.keys())

X_scaled = StandardScaler().fit_transform(X)
# %% Clusterisation

Z = linkage(X, method="complete")  # ou "average", "complete", …
Z2 = linkage(normalize(X, norm="l2"), method="ward")
Z3 = linkage(MaxAbsScaler().fit_transform(X), method="ward")
D = pdist(X, metric="cosine")
Zc = linkage(D, method="average")
plt.figure(figsize=(80, 4))
dendrogram(Zc, labels=regions_order, leaf_rotation=90, color_threshold=0.27)
plt.ylabel("Dissimilarity")
plt.tight_layout()
plt.show()

# Si tu veux garder, disons, 4 clusters après inspection visuelle :
clt = AgglomerativeClustering(n_clusters=4, metric="euclidean", linkage="ward")
labels = clt.fit_predict(X)

df_clusters = pd.DataFrame({"Region": regions_order, "Cluster": labels}).sort_values("Cluster")

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
        y_vals = [m.Y(c) for c in cultures_ref if m.Y(c) > 0]
        Y_mean = np.mean(y_vals)

        surf = m.surfaces()
        rows.append(
            {
                "Region": reg,
                "Yield_mean": Y_mean,
                "Tot_fert": m.tot_fert(),
                "ImpN_ratio": m.imported_nitrogen() / max(m.total_plant_production(), 1e-6),
                "Net_footp": m.net_footprint(),
                "Emissions": _safe_sum(m.emissions()),
                "NUE": m.NUE_system(),
                "Grass_share": surf.get("Natural meadow ", 0) / max(m.surfaces_tot(), 1e-6),
                "Imp_anim%": m.net_imported_animal() / max(m.animal_production(), 1e-6),
            }
        )

df = pd.DataFrame(rows).set_index("Region")


# %%
def expand_dict_column(df, col, prefix):
    # transforme chaque dict en Series puis concatène
    expanded = df[col].apply(lambda d: pd.Series(d)).add_prefix(f"{prefix}_")
    return df.drop(columns=[col]).join(expanded)


df = expand_dict_column(df, "Emissions", "E")
df = expand_dict_column(df, "Tot_fert", "F")
df_numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0)

# %% Affichage

X = StandardScaler().fit_transform(df_numeric)  # toutes colonnes désormais numériques
D = pdist(X, metric="cosine")
Zc = linkage(D, method="average")  # ou "ward", "complete", …

plt.figure(figsize=(10, 4))
dendrogram(Zc, labels=df_numeric.index, leaf_rotation=90, color_threshold=0.6)
plt.ylabel("Dissimilarity")
plt.show()

# %% Visualisation des liens avec indicateurs

labels = fcluster(Zc, t=0.7, criterion="distance")
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
# %%
