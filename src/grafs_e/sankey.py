import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.cm as cm
from dataclasses import dataclass


def merge_nodes(adjacency_matrix, labels, merges):
    """
    Merge groups of nodes (labels) in the adjacency matrix by combining their nitrogen fluxes.

    This function allows you to combine multiple nodes (e.g., different sectors) into one, by
    summing their nitrogen fluxes. It takes a dictionary of node groupings (`merges`), where each
    key corresponds to a merged label, and the values are the original labels to be merged.

    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        A square matrix of size (n, n) representing nitrogen fluxes, where each entry (i, j)
        indicates the nitrogen flow from node i to node j (in ktN/year).

    labels : list of str
        A list of the original labels for the nodes, with length n.

    merges : dict
        A dictionary defining which nodes to merge. The keys are the new labels for merged groups,
        and the values are lists of the labels to be merged. For example:
        {
            "population": ["urban", "rural"],
            "livestock": ["bovines", "ovines", "equine", "poultry", "porcines", "caprines"],
            "industry": ["haber-bosch", "other sectors"]
        }

    Returns:
    --------
    new_matrix : np.ndarray
        The new adjacency matrix after merging the nodes, where the fluxes between merged nodes are summed.

    new_labels : list of str
        The list of the new, merged labels.

    old_to_new : dict
        A dictionary mapping the original node indices to the new merged indices.

    Notes:
    ------
    - The function combines nitrogen fluxes between all nodes defined in `merges`, and returns a reduced matrix
      with fewer nodes, where the original nodes within each merged group are summed together.
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
                if i_new == j_new and i != j:
                    continue
                new_matrix[i_new, j_new] += flow

    return new_matrix, new_labels, old_to_new


# 2. Créer une fonction de génération de couleurs dynamiques
def get_unique_colors(new_labels, fixed_colors_map):
    labels_to_color = sorted(list(set(new_labels) - set(fixed_colors_map.keys())))

    # 2.2 Choisir une Colormap (e.g., 'tab20', 'hsv', 'nipy_spectral'). 'tab20' est bonne pour max 20 catégories.
    # Si le nombre de labels est élevé (> 20), 'hsv' ou 'nipy_spectral' est préférable.
    N = len(labels_to_color)
    colormap_name = "hsv" if N > 20 else "tab20"
    cmap = cm.get_cmap(colormap_name)

    # 2.3 Générer les couleurs RGB pour les labels manquants
    dynamic_colors = {
        label: cmap(i / N)  # Donne une couleur distincte de 0 à 1
        for i, label in enumerate(labels_to_color)
    }

    all_colors = fixed_colors_map.copy()
    all_colors.update(dynamic_colors)

    new_index_to_color = {
        i: all_colors.get(new_labels[i]) for i in range(len(new_labels))
    }

    from matplotlib.colors import to_hex

    final_index_to_color = {
        idx: to_hex(color) if isinstance(color, tuple) else color
        for idx, color in new_index_to_color.items()
    }

    return final_index_to_color


@dataclass
class SankeyView:
    """Paquetage des infos nécessaires à l'affichage + mapping vers le modèle brut."""

    matrix: np.ndarray  # matrice (fusionnée ou non)
    labels: list  # labels (fusionnés ou non)
    old_to_new: dict  # map index_origine -> index_nouveau (identité si pas de merge)
    new_to_old: dict  # map index_nouveau -> [indices_originaux]
    index_to_label: dict  # map index_nouveau -> label
    label_to_index: dict  # map label -> index_nouveau


def build_sankey_view(model, do_merge: bool, merges: dict | None = None) -> SankeyView:
    """
    Construit la matrice/labels à afficher pour le sankey, avec ou sans fusion,
    tout en conservant des mappings explicites entre indices d'origine et nouveaux indices.
    Le 'model' n'est JAMAIS modifié.
    """
    base_matrix = model.adjacency_matrix
    base_labels = list(model.labels)  # copie défensive

    if not do_merge or not merges:
        # Pas de merge : identité
        n = len(base_labels)
        old_to_new = {i: i for i in range(n)}
        new_to_old = {i: [i] for i in range(n)}
        matrix = base_matrix.copy()
        labels = base_labels
    else:
        matrix, labels, old_to_new = merge_nodes(
            base_matrix, base_labels, merges=merges
        )
        # reconstruit le reverse mapping
        new_to_old = {}
        for old_i, new_i in old_to_new.items():
            new_to_old.setdefault(new_i, []).append(old_i)

    index_to_label = {i: labels[i] for i in range(len(labels))}
    label_to_index = {labels[i]: i for i in range(len(labels))}

    return SankeyView(
        matrix=matrix,
        labels=labels,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        index_to_label=index_to_label,
        label_to_index=label_to_index,
    )


def app_sankey(
    adjacency_matrix: np.ndarray,
    labels: list[str],
    main_node: int,
    index_to_color: dict[int, str] | None = None,
    scope: int = 1,
    extra_forward_nodes: list[int] | None = None,
):
    """
    Sankey focalisé sur 'main_node'.
    - Aval (forward): liens colorés par la cible (target).
    - Amont (backward): liens colorés par la source.
    - Auto-boucles (self-flows): ajoutées exactement UNE fois (source==target).
    """
    n = adjacency_matrix.shape[0]
    assert n == len(labels) and 0 <= main_node < n
    extra_forward_nodes = extra_forward_nodes or []

    if index_to_color is None:
        index_to_color = get_unique_colors(labels, fixed_colors_map={})

    def fmt(v):
        return f"{v:.2e} ktN/yr"

    # node data
    outflows = adjacency_matrix.sum(axis=1)
    inflows = adjacency_matrix.sum(axis=0)
    node_colors = [index_to_color.get(i, "#AAAAAA") for i in range(n)]
    node_hover = [
        f"Node: {labels[i]}<br>Total throughflow: {fmt(inflows[i])}"  # <br>Out: {fmt(outflows[i])}<br>Total: {fmt(inflows[i] + outflows[i])}"
        for i in range(n)
    ]

    sources, targets, values, link_colors, link_hover = [], [], [], [], []

    # ---------- Parcours (sans self-links) ----------
    def add_forward(node, depth):
        if depth > scope:
            return
        for j in range(n):
            f = adjacency_matrix[node, j]
            if f > 0 and node != j:  # <- exclut self-link ici
                sources.append(node)
                targets.append(j)
                values.append(f)
                link_colors.append(index_to_color[j])  # aval -> target
                link_hover.append(
                    f"Source: {labels[node]}<br>Target: {labels[j]}<br>Value: {fmt(f)}"
                )
                add_forward(j, depth + 1)

    def add_backward(node, depth):
        if depth > scope:
            return
        for i in range(n):
            f = adjacency_matrix[i, node]
            if f > 0 and i != node:  # <- exclut self-link ici
                sources.append(i)
                targets.append(node)
                values.append(f)
                link_colors.append(index_to_color[i])  # amont -> source
                link_hover.append(
                    f"Source: {labels[i]}<br>Target: {labels[node]}<br>Value: {fmt(f)}"
                )
                add_backward(i, depth + 1)

    add_forward(main_node, 1)
    add_backward(main_node, 1)
    for n0 in extra_forward_nodes:
        add_forward(n0, 1)

    # ---------- Auto-boucles : ajouter UNE seule fois ----------
    # On ne le fait que pour les nœuds visibles (main + extra), pour éviter du bruit.
    for i in {main_node, *extra_forward_nodes}:
        self_flow = float(adjacency_matrix[i, i])
        if self_flow > 0:
            sources.append(i)
            targets.append(i)
            values.append(self_flow)
            # couleur: peu importe source/target, c'est le même nœud; on prend sa couleur
            link_colors.append(index_to_color[i])
            link_hover.append(
                f"Self-flow (seeds): {labels[i]} → {labels[i]}<br>Value: {fmt(self_flow)}"
            )

    # ---------- Figure ----------
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=10,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
                customdata=node_hover,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                customdata=link_hover,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            # 'snap' convient bien; pas besoin d'arrangement 'fixed' pour les self-links
            arrangement="freeform",
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), font_size=14)
    return fig


def streamlit_sankey(
    model,
    do_merge: bool,
    merges: dict | None = None,
    fixed_colors: dict[str, str] | None = None,
    scope: int = 1,
):
    import streamlit as st

    view = build_sankey_view(model, do_merge=do_merge, merges=merges)

    prod_labels = (
        set(getattr(model, "df_prod", []).index.tolist())
        if hasattr(model, "df_prod")
        else set()
    )
    excr_labels = (
        set(getattr(model, "df_excr", []).index.tolist())
        if hasattr(model, "df_excr")
        else set()
    )

    # -> On exclut seulement produits & excrétions (les autres restent sélectionnables)
    selectable_indices = [
        i
        for i, lbl in enumerate(view.labels)
        if lbl not in prod_labels and lbl not in excr_labels
    ]
    if not selectable_indices:
        st.error("Aucun compartiment sélectionnable (hors products & excretions).")
        return

    sel_index = st.selectbox(
        "Select compartment",
        options=selectable_indices,
        index=0,
        format_func=lambda i: view.labels[i],
    )
    main_node = sel_index

    fixed_colors = fixed_colors or {
        "Livestock": "lightblue",
        "Excretion": "salmon",
        "Population": "darkblue",
        "Environment": "crimson",
        "Industry": "purple",
        "Trade": "silver",
        "soil stock": "sienna",
    }
    index_to_color = get_unique_colors(view.labels, fixed_colors)

    # Extra nœuds à étendre en AVAL (produits pour crops, excrétions pour livestock)
    extra_forward_nodes = []
    orig_indices = view.new_to_old.get(main_node, [main_node])
    orig_labels = [model.labels[i] for i in orig_indices]

    # crops -> produits
    if (
        hasattr(model, "df_cultures")
        and hasattr(model, "df_prod")
        and not model.df_prod.empty
    ):
        crop_orig = [l for l in orig_labels if l in model.df_cultures.index]
        if crop_orig:
            prod_from_crop = model.df_prod.loc[
                model.df_prod["Origin compartment"].isin(crop_orig)
            ].index.tolist()
            extra_forward_nodes += [
                view.label_to_index[p]
                for p in prod_from_crop
                if p in view.label_to_index
            ]

    # livestock -> excrétions
    if (
        hasattr(model, "df_elevage")
        and hasattr(model, "df_excr")
        and not model.df_excr.empty
    ):
        lsk_orig = [l for l in orig_labels if l in model.df_elevage.index]
        if lsk_orig:
            excr_from_lsk = model.df_excr.loc[
                model.df_excr["Origin compartment"].isin(lsk_orig)
            ].index.tolist()
            extra_forward_nodes += [
                view.label_to_index[e]
                for e in excr_from_lsk
                if e in view.label_to_index
            ]

    fig = app_sankey(
        adjacency_matrix=view.matrix,
        labels=view.labels,
        main_node=main_node,
        index_to_color=index_to_color,
        scope=scope,
        extra_forward_nodes=extra_forward_nodes,  # aval -> couleur target (gérée dans app_sankey)
    )
    st.plotly_chart(fig, use_container_width=True)

    # with st.expander("Mappings (traceability)"):
    #     st.write("old_to_new:", view.old_to_new)
    #     st.write("new_to_old:", view.new_to_old)


def streamlit_sankey_fertilization(
    model,
    merges: dict | None = None,
    THRESHOLD: float = 1e-1,
    filter_nodes: bool = True,
):
    """
    Sankey des flux *entrants vers les cultures* (fertilisation) + boîte unique 'Seeds'.

    - Utilise build_sankey_view(model, do_merge=True, merges) pour obtenir la vue (fusionnée ou non).
    - Les flux de graines (self-flows) sont lus sur la diagonale d'origine et représentés
      comme des liens 'Seeds' -> cultures. La diagonale de travail est ensuite mise à 0 pour
      éviter les doubles comptes.
    - Les liens sont colorés par la *source*. Seuil THRESHOLD sur les liens et, si filter_nodes=True,
      filtrage des nœuds à faible throughflow.
    """
    import streamlit as st
    import numpy as np
    import plotly.graph_objects as go

    if model is None:
        st.error("❌ The model has not run yet. Please use the run tab.")
        return

    # 1) Construire la vue (avec ou sans merge selon 'merges')
    view = build_sankey_view(model, do_merge=True, merges=merges)

    # Matrice de travail + diagonale d'ORIGINE (avant neutralisation)
    mat = view.matrix.copy()
    diag_orig = np.diag(view.matrix).astype(float).copy()

    labels = view.labels
    n = len(labels)

    # 2) Indices de nœuds qui contiennent au moins UNE culture d'origine
    crop_index_set = set()
    if hasattr(model, "df_cultures") and not model.df_cultures.empty:
        crop_labels_orig = set(model.df_cultures.index)
        for new_i, old_list in view.new_to_old.items():
            if any(model.labels[old_i] in crop_labels_orig for old_i in old_list):
                crop_index_set.add(new_i)
    if not crop_index_set:
        st.info("Aucun nœud 'cultures' trouvé après fusion.")
        return

    # 3) Couleurs
    FIXED = {
        "Livestock": "lightblue",
        "Excretion": "salmon",
        "Population": "darkblue",
        "Environment": "crimson",
        "Industry": "purple",
        "Trade": "silver",
        "soil stock": "sienna",
        "Atmospheric deposition": "skyblue",
    }
    index_to_color = get_unique_colors(labels, FIXED)

    def fmt(v: float) -> str:
        return f"{v:.2e} ktN/yr"

    # 4) SEEDS : lire dans la diagonale d'origine, puis neutraliser la diagonale de travail
    seeds_to_crop = {}
    for tgt in sorted(crop_index_set):
        val = float(diag_orig[tgt])
        if val > 0:
            seeds_to_crop[tgt] = val
    # neutralise la diagonale pour la suite des calculs (évite double comptage)
    np.fill_diagonal(mat, 0.0)

    # 5) Liens "sources -> cultures" (hors seeds)
    S, T, V, LINK_COLORS, LINK_HOV = [], [], [], [], []
    for tgt in sorted(crop_index_set):
        for src in range(n):
            if src == tgt:
                continue  # les self-flows seront portés par 'Seeds'
            flow = float(mat[src, tgt])
            if flow <= 0 or flow < THRESHOLD:
                continue
            S.append(src)
            T.append(tgt)
            V.append(flow)
            LINK_COLORS.append(index_to_color.get(src, "#888888"))  # couleur = source
            LINK_HOV.append(
                f"Source: {labels[src]}<br>Target: {labels[tgt]}<br>Value: {fmt(flow)}"
            )

    # 6) Ajouter la boîte unique 'Seeds' -> cultures
    use_seeds = any(v >= THRESHOLD for v in seeds_to_crop.values())
    seeds_index_virtual = None
    if use_seeds:
        seeds_index_virtual = n  # index virtuel (sera remappé)
        for tgt, val in seeds_to_crop.items():
            if val < THRESHOLD:
                continue
            S.append(seeds_index_virtual)
            T.append(tgt)
            V.append(val)
            LINK_COLORS.append("#d4a017")  # goldenrod
            LINK_HOV.append(
                f"Source: Seeds<br>Target: {labels[tgt]}<br>Value: {fmt(val)}"
            )

    if not S:
        st.info("Aucun flux de fertilisation au-dessus du seuil.")
        return

    # 7) (Optionnel) Filtrage des nœuds par throughflow
    if filter_nodes:
        # taille potentiellement augmentée si 'Seeds' est actif
        size = n + (1 if use_seeds else 0)
        through = np.zeros(size, dtype=float)

        # pour les cultures: on regarde les entrées ; pour les autres: les sorties
        for i in range(n):
            if i in crop_index_set:
                through[i] = float(mat[:, i].sum())
            else:
                through[i] = float(mat[i, :].sum())
        if use_seeds:
            # throughflow du noeud Seeds = somme de ses sorties
            through[seeds_index_virtual] = float(sum(seeds_to_crop.values()))

        kept_nodes = {i for i in range(size) if through[i] >= THRESHOLD}
        kept_links_idx = [
            k for k in range(len(S)) if (S[k] in kept_nodes and T[k] in kept_nodes)
        ]
        S = [S[k] for k in kept_links_idx]
        T = [T[k] for k in kept_links_idx]
        V = [V[k] for k in kept_links_idx]
        LINK_COLORS = [LINK_COLORS[k] for k in kept_links_idx]
        LINK_HOV = [LINK_HOV[k] for k in kept_links_idx]

        if not S:
            st.info("Tous les flux ont été filtrés par le seuil/nœuds.")
            return

    # 8) Remap compact (inclut Seeds si présent)
    used_nodes = sorted(set(S) | set(T))
    remap = {old: new for new, old in enumerate(used_nodes)}
    sankey_sources = [remap[x] for x in S]
    sankey_targets = [remap[x] for x in T]

    node_labels = []
    node_colors = []
    node_hover = []
    for old in used_nodes:
        if use_seeds and old == seeds_index_virtual:
            node_labels.append("Seeds")
            node_colors.append("#d4a017")
            node_hover.append(
                f"Node: Seeds<br>Total out: {fmt(sum(seeds_to_crop.values()))}"
            )
        else:
            node_labels.append(labels[old])
            node_colors.append(index_to_color.get(old, "#AAAAAA"))
            node_hover.append(
                f"Node: {labels[old]}<br>"
                f"In: {fmt(mat[:, old].sum())}<br>"
                f"Out: {fmt(mat[old, :].sum())}"
            )

    # 9) Figure
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=16,
                thickness=18,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
                customdata=node_hover,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=sankey_sources,
                target=sankey_targets,
                value=V,
                color=LINK_COLORS,
                customdata=LINK_HOV,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            arrangement="snap",
        )
    )
    fig.update_layout(
        title="Fertilization flows → crops (with Seeds box)",
        font_size=14,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def streamlit_sankey_food_flows(
    model,
    merges: dict | None = None,
    THRESHOLD: float = 1e-1,
):
    """
    Sankey des flux food/feed avec duplication des nœuds 'trade' en (import)/(export).
    Filtrage des liens < THRESHOLD. Couleurs robustes (labels fusionnés).
    """
    import streamlit as st
    import numpy as np
    import plotly.graph_objects as go

    if model is None:
        st.error("❌ Le modèle n'est pas encore exécuté. Lancez d'abord le modèle.")
        return

    adjacency_matrix = model.adjacency_matrix
    labels_model = list(model.labels)  # labels d'origine (avant merge)

    # 1) Fusion des nœuds
    merges = merges or {}
    new_matrix, new_labels, old_to_new = merge_nodes(
        adjacency_matrix, labels_model, merges
    )
    n_new = len(new_labels)

    # Helper: ancien label -> index fusionné (ou None)
    def old_label_to_new_index(old_label: str):
        try:
            old_idx = labels_model.index(old_label)
        except ValueError:
            return None
        return old_to_new.get(old_idx)

    # 2) Sets 'source' et 'target' (fusionnés)
    src_labels = []
    if hasattr(model, "df_prod") and model.df_prod is not None:
        src_labels += list(model.df_prod.index)
    if hasattr(model, "data_loader") and hasattr(model.data_loader, "trade_labels"):
        src_labels += list(model.data_loader.trade_labels)

    tgt_labels = []
    if hasattr(model, "df_elevage") and model.df_elevage is not None:
        tgt_labels += list(model.df_elevage.index)
    if hasattr(model, "df_pop") and model.df_pop is not None:
        tgt_labels += list(model.df_pop.index)
    if hasattr(model, "data_loader") and hasattr(model.data_loader, "trade_labels"):
        tgt_labels += list(model.data_loader.trade_labels)
    if hasattr(model, "df_energy") and model.df_energy is not None:
        tgt_labels += list(model.df_energy.index)

    sources_merged = {
        idx
        for lbl in src_labels
        for idx in [old_label_to_new_index(lbl)]
        if idx is not None
    }
    targets_merged = {
        idx
        for lbl in tgt_labels
        for idx in [old_label_to_new_index(lbl)]
        if idx is not None
    }

    # 3) Nœuds 'trade' (post-merge)
    trade_merged = {
        i
        for i, lbl in enumerate(new_labels)
        if isinstance(lbl, str) and "trade" in lbl.lower()
    }

    # 4) Construire les nœuds finaux: duplication (import/export) pour trade
    all_sankey_nodes = []
    node_map_import = {}
    node_map_export = {}
    node_map_normal = {}

    def _strip_trade_suffix(label_i: str) -> str:
        # Si tes labels mergés se terminent par " trade" (6 caractères), on slicera proprement
        s = label_i.strip()
        # essaie d'enlever un suffixe "trade" éventuel
        return s[:-6].rstrip() if s.lower().endswith("trade") else s

    for i in range(n_new):
        label_i = new_labels[i]
        if i in trade_merged:
            base = _strip_trade_suffix(label_i)
            idx_import = len(all_sankey_nodes)
            all_sankey_nodes.append(f"{base} import")
            node_map_import[i] = idx_import

            idx_export = len(all_sankey_nodes)
            all_sankey_nodes.append(f"{base} export")
            node_map_export[i] = idx_export
        else:
            idx_norm = len(all_sankey_nodes)
            all_sankey_nodes.append(label_i)
            node_map_normal[i] = idx_norm

    # 5) Balayage des flux fusionnés, filtrage par sets et typage import/export
    final_sources, final_targets, final_values, final_hover_texts = [], [], [], []

    def format_scientific(value):
        return f"{value:.2e} ktN/yr"

    for s_idx in range(n_new):
        if s_idx not in sources_merged:
            continue
        for t_idx in range(n_new):
            if t_idx not in targets_merged:
                continue
            flow = float(new_matrix[s_idx, t_idx])
            if flow <= 0:
                continue

            # Typage des flux
            if s_idx in trade_merged and t_idx not in trade_merged:
                sankey_s = node_map_import[s_idx]  # import
                sankey_t = node_map_normal[t_idx]
                flow_type = "Import"
            elif s_idx not in trade_merged and t_idx in trade_merged:
                sankey_s = node_map_normal[s_idx]
                sankey_t = node_map_export[t_idx]  # export
                flow_type = "Export"
            elif s_idx not in trade_merged and t_idx not in trade_merged:
                sankey_s = node_map_normal[s_idx]
                sankey_t = node_map_normal[t_idx]
                flow_type = "Internal"
            else:
                # trade->trade : on peut choisir import->export
                sankey_s = node_map_import[s_idx]
                sankey_t = node_map_export[t_idx]
                flow_type = "Trade→Trade"

            final_sources.append(sankey_s)
            final_targets.append(sankey_t)
            final_values.append(flow)
            final_hover_texts.append(
                f"Source: {all_sankey_nodes[sankey_s]}<br>"
                f"Target: {all_sankey_nodes[sankey_t]}<br>"
                f"Type: {flow_type}<br>"
                f"Value: {format_scientific(flow)}"
            )

    # 6) Seuil sur les liens
    kept = [i for i, v in enumerate(final_values) if v >= THRESHOLD]
    final_sources = [final_sources[i] for i in kept]
    final_targets = [final_targets[i] for i in kept]
    final_values = [final_values[i] for i in kept]
    final_hover_texts = [final_hover_texts[i] for i in kept]

    if not final_sources:
        st.info("Aucun flux au-dessus du seuil.")
        return

    # 7) Throughflow & filtrage des nœuds
    nb_nodes = len(all_sankey_nodes)
    flow_in = [0.0] * nb_nodes
    flow_out = [0.0] * nb_nodes
    for s, t, v in zip(final_sources, final_targets, final_values):
        flow_out[s] += v
        flow_in[t] += v
    throughflow = [flow_in[i] + flow_out[i] for i in range(nb_nodes)]
    kept_nodes = {i for i, thr in enumerate(throughflow) if thr >= THRESHOLD}

    kept_links_idx = [
        i
        for i in range(len(final_sources))
        if (final_sources[i] in kept_nodes and final_targets[i] in kept_nodes)
    ]
    final_sources = [final_sources[i] for i in kept_links_idx]
    final_targets = [final_targets[i] for i in kept_links_idx]
    final_values = [final_values[i] for i in kept_links_idx]
    final_hover_texts = [final_hover_texts[i] for i in kept_links_idx]

    if not final_sources:
        st.info("Tous les flux ont été filtrés par le seuil/nœuds.")
        return

    # 8) Remap compact des nœuds
    used_nodes = sorted(set(final_sources) | set(final_targets))
    remap = {old: new for new, old in enumerate(used_nodes)}
    sankey_sources = [remap[s] for s in final_sources]
    sankey_targets = [remap[t] for t in final_targets]

    # 9) Labels & Couleurs
    sankey_labels = [all_sankey_nodes[i] for i in used_nodes]

    # Couleurs: calcule d'abord index->couleur sur les labels fusionnés,
    # puis dérive label->couleur pour new_labels seulement.
    fixed_color_dict = {
        "Livestock": "lightblue",
        "Population": "darkblue",
    }
    index_to_color = get_unique_colors(
        new_labels, fixed_color_dict
    )  # index -> hex (sur *new_labels*)
    label_to_color = {new_labels[i]: index_to_color[i] for i in range(len(new_labels))}

    sankey_colors = []
    for lbl in sankey_labels:
        lbl_l = (lbl or "").lower()
        if " import" in lbl_l:
            sankey_colors.append("slategray")
        elif " export" in lbl_l:
            sankey_colors.append("silver")
        elif lbl in label_to_color:
            sankey_colors.append(label_to_color[lbl])
        else:
            sankey_colors.append("gray")

    # 10) Figure
    def format_scientific(value):
        return f"{value:.2e} ktN/yr"

    node_hover = [
        f"Node: {lbl}<br>Throughflow: {format_scientific(throughflow[old])}"
        for old, lbl in zip(used_nodes, sankey_labels)
    ]

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=sankey_labels,
                color=sankey_colors,
                customdata=node_hover,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=sankey_sources,
                target=sankey_targets,
                value=final_values,
                color=[sankey_colors[s] for s in sankey_sources],  # couleur = source
                customdata=final_hover_texts,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            arrangement="snap",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        font_color="white",
        width=1200,
        height=1000,
    )
    st.plotly_chart(fig, use_container_width=False)
