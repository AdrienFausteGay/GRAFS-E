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

# %% Try sankeys with matplotlib

label_to_level = {
    "atmospheric n2": 0,
    "haber-bosch": 1,
    "leguminous": 1,
    "cereals (excluding rice)": 2,
    "oleaginous": 2,
    "fruits and vegetables": 2,
    "roots": 2,  # <--- Still at level 2
    "forages": 2,
    "fishery products": 2,
    "temporary meadows": 2,
    "natural meadows ": 2,
    "monogastrics": 3,
    "ruminants": 3,  # <--- Still at level 3
    "population": 4,
    "trade": 3,
    "soil stock": 3,
    "other sectors": 3,
    "environment": 4,
}


def sankey_systemic_flows_matplotlib(
    adjacency_matrix,
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
    label_to_level=label_to_level,  # Renamed to reflect 'levels'
    THRESHOLD=0.01,
    figure_size=(16, 12),  # For publication quality, control figure size
    dpi=300,  # High DPI for publication
):
    """
    Crée un diagramme de Sankey systémique pour publication en utilisant sankeyflow.
    Les nœuds sont fusionnés, les flux sous le seuil sont éliminés,
    et les positions des nœuds sont définies par des niveaux.
    """
    # 1) Fusion des nœuds
    new_matrix, new_labels, old_to_new = merge_nodes(adjacency_matrix, labels, merges)
    n_new = len(new_labels)

    # 2) Définir les couleurs des nœuds fusionnés
    color_dict = {
        "cereals (excluding rice)": "gold",
        "fruits and vegetables": "lightgreen",
        "leguminous": "darkgreen",
        "oleaginous": "lightgreen",
        "meadow and forage": "green",  # Is this merged label present in new_labels?
        "trade": "gray",
        "monogastrics": "lightblue",
        "ruminants": "lightblue",
        "population": "darkblue",
        "losses": "crimson",  # Is this merged label present in new_labels?
        "roots": "orange",
        "forages": "limegreen",
        "Environment": "crimson",
        "temporary meadows": "seagreen",
        "natural meadows ": "darkgreen",
        "soil stock": "sienna",
        "haber-bosch": "purple",
        "atmospheric n2": "seagreen",
        "other sectors": "lightgray",
        "fishery products": "cadetblue",
    }

    default_node_color = "gray"
    # Ensure merged labels in color_dict actually exist in new_labels or are handled.
    # Otherwise, they won't apply.
    # It's better to get the actual labels after merge and then map colors.

    # 3) Collecter tous les flux de la matrice fusionnée
    # We now collect as (source_label, target_label, value) tuples for sankeyflow
    all_flows_data = []

    def format_scientific(value):
        return f"{value:.2e} ktN/yr"

    for s_idx in range(n_new):
        for t_idx in range(n_new):
            flow = new_matrix[s_idx, t_idx]
            if flow > THRESHOLD:
                source_label = new_labels[s_idx]
                target_label = new_labels[t_idx]
                all_flows_data.append((source_label, target_label, flow))

    # 4) Filter nodes based on throughflow
    throughflows = np.sum(new_matrix, axis=0) + np.sum(new_matrix, axis=1)
    # Map original merged labels to their throughflows for filtering
    label_throughflow = {new_labels[i]: throughflows[i] for i in range(n_new)}

    # Filter out flows where source or target node has insufficient throughflow
    filtered_flows_data = []
    kept_labels_set = set()  # To keep track of nodes that remain
    for s_label, t_label, flow_val in all_flows_data:
        if label_throughflow.get(s_label, 0) >= THRESHOLD and label_throughflow.get(t_label, 0) >= THRESHOLD:
            filtered_flows_data.append((s_label, t_label, flow_val))
            kept_labels_set.add(s_label)
            kept_labels_set.add(t_label)

    # Ensure label_to_level keys are normalized to match
    # new_labels (which are already normalized by merge_nodes if original labels were).
    normalized_label_to_level = {k.lower().strip(): v for k, v in label_to_level.items()}

    # Create the nodes list for sankeyflow
    nodes = defaultdict(list)  # Maps level -> list of nodes for that level
    node_metadata = {}  # To store colors and hover info

    # Sort kept_labels for consistent Y-ordering within each level
    # This is critical for controlling vertical stacking.
    # We will sort alphabetically within each level, but you could define a custom order
    # if you have specific vertical stacking preferences for publication.

    # First, group labels by their assigned level
    labels_by_level = defaultdict(list)
    for label in sorted(list(kept_labels_set)):  # Sort alphabetically for consistent ordering
        level = normalized_label_to_level.get(label.lower().strip())
        if level is not None:
            labels_by_level[level].append(label)
        else:
            print(f"Warning: Node '{label}' does not have a defined level in label_to_level. Skipping.")
            # If a node doesn't have a level, it won't be drawn.
            # You might want to assign a default level or handle it.

    # Now, build the `nodes` structure for sankeyflow, which is a list of lists.
    # Each inner list is a "level" (column), and contains tuples of (node_name, node_value)
    # The order within the inner list determines y-position.
    sankeyflow_nodes = []

    for level_idx in sorted(labels_by_level.keys()):
        current_level_nodes = []
        for label in labels_by_level[level_idx]:
            # sankeyflow needs a 'value' for the node's overall size, often the throughflow
            node_val = label_throughflow.get(label, 0)
            current_level_nodes.append((label, node_val))
            # Store color and hover info for this node
            node_metadata[label] = {
                "color": color_dict.get(label, default_node_color),
                "hover_text": f"Node: {label}<br>Throughflow: {format_scientific(node_val)}",
            }
        sankeyflow_nodes.append(current_level_nodes)

    # Create flows for sankeyflow, adding colors and potentially curvature
    sankeyflow_flows = []
    for s_label, t_label, flow_val in filtered_flows_data:
        link_color = color_dict.get(s_label, default_node_color)  # Color link by source node
        hover_text = f"Source: {s_label}<br>Target: {t_label}<br>Value: {format_scientific(flow_val)}"

        flow_opts = {
            "color": link_color,
            "data": {"hover_text": hover_text},  # Custom data for tooltips if we want to add later
        }

        # You can adjust curvature here.
        # For backward flows (level_source > level_target), positive curvature can help.
        # For forward flows, 0 (straight) or a small positive value is often good.
        s_level = normalized_label_to_level.get(s_label.lower().strip())
        t_level = normalized_label_to_level.get(t_label.lower().strip())

        if s_level is not None and t_level is not None and s_level > t_level:
            # This is a backward flow. Increase curvature to make it loop nicely.
            flow_opts["curvature"] = 0.5  # Experiment with this value (e.g., 0.2 to 1.0)
        else:
            # Forward flow or same-level flow. Keep it relatively straight or slightly curved.
            flow_opts["curvature"] = 0.1  # A small value can make it look smoother

        sankeyflow_flows.append((s_label, t_label, flow_val, flow_opts))

    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)

    # Create the Sankey diagram
    # Use ProcessGroup for defining nodes more explicitly.
    # node_color_mode can be 'default' or 'value' or a custom dict.
    # We'll apply colors after creation for more control.

    # Map from node label to a unique integer ID for SankeyFlow's internal use if needed
    # (though it largely works with string labels)

    sankey = Sankey(
        flows=sankeyflow_flows,
        nodes=sankeyflow_nodes,  # Pass the structured nodes list
        flow_color_mode="individual",  # We provide colors per flow in flow_opts
        node_opts=dict(
            label_format="{label}"  # Only show label, we'll add value to hover if needed later
        ),
        ax=ax,  # Pass the Matplotlib axes
    )

    # Draw the Sankey diagram
    sankey.draw()

    # --- Post-processing for node colors and labels (sankeyflow gives good control here) ---
    for node_name, node_patch in sankey.node_patches.items():
        if node_name in node_metadata:
            node_patch.set_facecolor(node_metadata[node_name]["color"])
            # sankeyflow automatically adds tooltips if 'data' is in flow_opts or node_opts,
            # but for publication, you might add hover text when saving as HTML (not default Matplotlib).
            # For static publication, you'd typically rely on labels or external captions.

    # Customize node labels if you want more than just the name
    for node_label, label_obj in sankey.node_labels.items():
        if node_label in node_metadata:
            # You can set font size, color, etc.
            label_obj.set_fontsize(10)
            # You can also customize position if needed
            # For publication, you might want to show throughflow next to the label
            label_obj.set_text(f"{node_label}\n({format_scientific(label_throughflow.get(node_label, 0))})")

    # Customize flow labels if needed (e.g., add value labels)
    # sankeyflow doesn't automatically put flow labels. You'd add them manually using ax.text.

    ax.set_title("Système de Flux de Nutriments", fontsize=16, color="black")
    ax.axis("off")  # Turn off standard Matplotlib axes

    # For publication, save the figure
    # You can save as PDF, SVG (vector formats are best for publication), or high-res PNG
    # plt.savefig("sankey_publication.pdf", bbox_inches='tight')
    # plt.savefig("sankey_publication.png", dpi=dpi, bbox_inches='tight')

    # To show the plot (for debugging or interactive viewing during development)
    plt.show()

    return fig  # Return the matplotlib figure object


# %%
