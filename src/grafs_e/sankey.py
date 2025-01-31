import numpy as np
import plotly.graph_objects as go
import streamlit as st

from grafs_e.donnees import *
from grafs_e.N_class import *


def create_sankey():
    # Définir les labels des nœuds
    labels = [
        "Wheat",
        "atmosphere",
        "hydro-system",
        "other losses",
        "cereals food export",
        "cereals feed export",
        "bovines",
        "rural",
        "urban",
        "Sugar beet",
        "Natural meadow",
    ]

    labels = [
        "Haber-Bosch",
        "Rapeseed",
        "Sugarbeet",
        "Natural meadow ",
        "Wheat",
        "Barley",
        "Roots food export",
        "Bovines",
        "Cereals food export",
    ]

    color_labels = ["red", "yellow", "yellow", "darkgreen", "yellow", "yellow", "gray", "lightblue", "gray"]

    # Définir les liens entre les nœuds
    # source, target, value (proportions ou flux entre les nœuds)
    # sources = [0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6]
    # targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    # values = [4.15, 7.97, 3.42, 70.2, 8.56, 3.77, 2.32, 3.41, 1.84, 18.67, 10]

    sources = [0, 0, 0, 0, 2, 3, 4, 5, 7]
    targets = [1, 2, 3, 4, 6, 7, 8, 8, 3]
    values = [16.99, 15.26, 28, 81.47, 22.61, 13.96, 70.2, 11.28, 18.66]
    color_links = ["red", "red", "red", "red", "yellow", "darkgreen", "yellow", "yellow", "lightblue"]

    # Créer le diagramme de Sankey
    fig = go.Figure(
        go.Sankey(
            node=dict(pad=7, thickness=20, line=dict(color="black", width=0.5), label=labels, color=color_labels),
            link=dict(source=sources, target=targets, value=values, color=color_links),
        )
    )

    # Ajouter un titre
    # fig.update_layout(title_text="Main flows from Wheat in Picardy 2010", font_size=10)

    # Afficher le graphique
    fig.show()


def create_sankey_from_transition_matrix(transition_matrix, main_node, scope=2):
    """
    Crée un diagramme de Sankey à partir d'une matrice de transition.

    :param transition_matrix: Matrice de transition (numpy array) où chaque élément [i, j] est le flux de i vers j.
    :param main_node: Le nœud principal qui servira de base pour le Sankey.
    :param scope: Le nombre de niveaux à inclure dans le Sankey (profils de profondeur de la hiérarchie).
    """
    # Récupérer le nombre de nœuds
    n_nodes = transition_matrix.shape[0]

    # Créer une liste vide de labels pour les nœuds et une liste vide pour les flux
    labels = []
    sources = []
    targets = []
    values = []

    # Générer les labels à partir des nœuds
    for i in range(n_nodes):
        labels.append(index_to_label[i])  # Utiliser les indices des nœuds comme labels de base

    # Fonction récursive pour ajouter les flux dans la direction descendante
    def add_flows(node, depth, parent=None):
        if depth > scope:
            return
        # Ajouter le flux pour les nœuds voisins du nœud courant
        for target_node in range(n_nodes):
            flow = transition_matrix[node, target_node]
            if flow > 0:  # Si un flux existe
                sources.append(node)
                targets.append(target_node)
                values.append(flow)

                # Ajouter récursivement les flux pour les nœuds cibles
                add_flows(target_node, depth + 1, parent=node)

    # Ajouter les flux à partir du nœud principal
    add_flows(main_node, 1)

    # Créer le diagramme de Sankey
    fig = go.Figure(
        go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
            link=dict(source=sources, target=targets, value=values),
        )
    )

    # Ajouter un titre
    # fig.update_layout(title_text=f"Flow Diagram starting from Node {main_node}", font_size=10)

    # Afficher le graphique
    fig.show()


def create_sankey_from_transition_matrix_2(
    transition_matrix, main_node, scope=1, index_to_label=index_to_label, index_to_color=node_color
):
    """
    Crée un diagramme de Sankey montrant à la fois les flux entrants (sources) et sortants (cibles) d'un nœud principal.

    :param transition_matrix: Matrice de transition (numpy array) où chaque élément [i, j] est le flux de i vers j.
    :param main_node: Le nœud principal qui servira de base pour le Sankey.
    :param scope: Le nombre de niveaux à inclure dans le Sankey (profondeur).
    """
    # Récupérer le nombre de nœuds
    n_nodes = transition_matrix.shape[0]

    # Créer une liste vide de labels pour les nœuds et une liste vide pour les flux
    labels = []
    sources = []
    targets = []
    values = []
    node_colors = []  # Liste pour les couleurs des nœuds
    link_colors = []  # Liste pour les couleurs des flux

    # Générer les labels à partir des nœuds
    for i in range(n_nodes):
        if index_to_label[i] == "cereals (excluding rice) food nitrogen import-export":
            labels.append("cereals food export")
        elif index_to_label[i] == "cereals (excluding rice) feed nitrogen import-export":
            labels.append("cereals feed export")
        else:
            labels.append(index_to_label[i])  # Utiliser les indices des nœuds comme labels de base
        node_colors.append(index_to_color[i])  # Définir une couleur de base pour les nœuds (par exemple, lightblue)

    # Ajouter les flux sortants (cibles) à partir du nœud principal
    def add_forward_flows(node, depth):
        if depth > scope:
            return
        for target_node in range(n_nodes):
            flow = transition_matrix[node, target_node]
            if flow > 0:  # Si un flux existe
                sources.append(node)
                targets.append(target_node)
                values.append(flow)
                link_colors.append(index_to_color[target_node])  # Couleur des flux sortants
                add_forward_flows(target_node, depth + 1)

    # Ajouter les flux entrants (sources) vers le nœud principal
    def add_backward_flows(node, depth):
        if depth > scope:
            return
        for source_node in range(n_nodes):
            flow = transition_matrix[source_node, node]
            if flow > 0:  # Si un flux existe
                sources.append(source_node)
                targets.append(node)
                values.append(flow)
                link_colors.append(index_to_color[source_node])  # Couleur des flux entrants
                add_backward_flows(source_node, depth + 1)

    # Ajouter les flux sortants (cibles) et entrants (sources)
    add_forward_flows(main_node, 1)
    add_backward_flows(main_node, 1)

    # Créer le diagramme de Sankey
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=7,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,  # Application des couleurs aux nœuds
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,  # Application des couleurs aux flux
            ),
        )
    )
    # fig.write_image("sankey_high_resolution.png", width=2400, height=1600, scale=2)

    # Afficher le graphique
    fig.show()


def create_sankey_tot(transition_matrix, index_to_label, index_to_color):
    """
    Fonction toute claquée, il y a trop d'infos
    Crée un diagramme de Sankey à partir d'une matrice de transition des flux d'azote.
    Les nœuds sont organisés en trois colonnes :
    1. Nœuds 47 à 50 (première colonne)
    2. Nœuds 0 à 35 (deuxième colonne)
    3. Les autres nœuds (troisième colonne)

    :param transition_matrix: Matrice de transition des flux d'azote (61x61 numpy array)
    :param index_to_label: Dictionnaire associant les indices des nœuds à leurs labels
    :param index_to_color: Dictionnaire associant les indices des nœuds à leurs couleurs
    """
    n_nodes = transition_matrix.shape[0]

    # Créer les labels et couleurs des nœuds
    labels = [index_to_label[i] for i in range(n_nodes)]
    node_colors = [index_to_color[i] for i in range(n_nodes)]

    # Déclarer les nœuds et les flux dans l'ordre correct
    sources = []
    targets = []
    values = []
    link_colors = []

    # Première colonne : nœuds 47 à 50
    col1_nodes = list(range(47, 51))  # Nœuds 47 à 50

    # Deuxième colonne : nœuds 0 à 35
    col2_nodes = list(range(36))  # Nœuds 0 à 35

    # Troisième colonne : nœuds 51 à 60
    col3_nodes = list(range(36, 47)) + list(range(51, 63))  # Nœuds 51 à 60

    # Organiser les nœuds en 3 colonnes
    all_nodes = col1_nodes + col2_nodes + col3_nodes

    # Ajouter les flux sortants et entrants en fonction des nœuds
    for i in all_nodes:
        for j in all_nodes:
            flow = transition_matrix[i, j]
            if flow > 0:  # Si un flux existe
                sources.append(i)
                targets.append(j)
                values.append(flow)
                link_colors.append(index_to_color[j])  # Couleur des flux sortants

    # Créer le diagramme de Sankey
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=7,
                thickness=10,
                line=dict(color="black", width=0.5),
                label=None,
                color=node_colors,  # Application des couleurs aux nœuds
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,  # Application des couleurs aux flux
            ),
        )
    )

    # Afficher le graphique
    fig.show()


def create_sankey_agreg(transition_matrix):
    """
    Crée un diagramme de Sankey simplifié en fusionnant les nœuds par groupe.
    Les groupes sont fusionnés et organisés en trois colonnes :
    1. Première colonne : "Industry", "Cereals", "Oleaginous", "Roots"
    2. Deuxième colonne : "Fruits and vegetables", "Grasslands and forages", "Leguminous"
    3. Troisième colonne : "Livestock", "Population", "Losses", "Trade", "Atmosphere"

    :param transition_matrix: Matrice de transition des flux d'azote (61x61 numpy array)
    :param index_to_label: Dictionnaire associant les indices des nœuds à leurs labels
    :param index_to_color: Dictionnaire associant les indices des nœuds à leurs couleurs
    """
    # Définir les groupes de nœuds à fusionner
    groups = {
        "Industry": [49, 50],  # Place Industry en haut
        "Cereals": list(range(8)),
        "Oleaginous": list(range(8, 11)),
        "Roots": list(range(11, 14)),
        "Fruits and vegetables": list(range(14, 24)),
        "Grasslands and forages": [24, 25, 26, 35],
        "Leguminous": list(range(27, 35)),
        "Livestock": list(range(36, 42)),
        "Population": [42, 43],
        "Losses": list(range(44, 47)),
        "Atmosphere": list(range(47, 49)),
        "Import": [51, 53],
        "Export": [52, 54, 55, 56, 57, 59, 60, 61, 62],
    }

    # Créer un dictionnaire de nouveaux labels pour les nœuds fusionnés
    merged_labels = {i: label for i, label in enumerate(groups.keys())}

    # Créer les labels et couleurs des nœuds fusionnés
    labels = list(groups.keys())  # Ajouter les labels des groupes fusionnés
    node_colors = [
        "purple",
        "yellow",
        "olive",
        "orange",
        "lightyellow",
        "darkgreen",
        "lightgreen",
        "lightblue",
        "darkblue",
        "red",
        "cyan",
        "gray",
        "gray",
    ]  # Définir les couleurs de chaque groupe

    # Créer les flux agrégés
    sources = []
    targets = []
    values = []
    link_colors = []

    # Créer les flux entre les groupes de nœuds fusionnés
    for i, group_i in enumerate(groups.values()):
        for j, group_j in enumerate(groups.values()):
            flow = np.sum(transition_matrix[np.ix_(group_i, group_j)])  # Additionner les flux entre les groupes
            if flow > 0:  # Si un flux existe entre ces deux groupes
                sources.append(i)
                targets.append(j)
                values.append(flow)
                link_colors.append(node_colors[i])  # Appliquer la couleur du groupe source

    # Organiser les nœuds dans les trois colonnes
    col1_nodes = [0, 1, 2, 3]  # Nœuds "Industry", "Cereals", "Oleaginous", "Roots"
    col2_nodes = [4, 5, 6]  # Nœuds "Fruits and vegetables", "Grasslands and forages", "Leguminous"
    col3_nodes = [7, 8, 9, 10]  # Nœuds "Livestock", "Population", "Losses", "Trade", "Atmosphere"

    # Créer le diagramme de Sankey
    fig = go.Figure(
        go.Sankey(
            node=dict(pad=3, thickness=10, line=dict(color="black", width=0.5), label=labels, color=node_colors),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,  # Application des couleurs aux flux
            ),
        )
    )

    # Afficher le graphique
    fig.show()


def streamlit_sankey(transition_matrix, main_node, scope=1, index_to_label=None, index_to_color=None):
    """
    Crée un diagramme de Sankey interactif sous Streamlit, affichant les flux
    entrants et sortants d'un nœud principal avec un affichage personnalisé au survol.
    """

    # Vérification des paramètres
    if index_to_label is None or index_to_color is None:
        st.error("Les mappings de labels et couleurs ne sont pas fournis.")
        return

    n_nodes = transition_matrix.shape[0]

    labels = []
    sources = []
    targets = []
    values = []
    node_colors = []
    link_colors = []
    node_hover_texts = []  # Liste pour stocker les tooltips des nœuds
    link_hover_texts = []  # Liste pour stocker les tooltips des flux

    # Fonction pour formater les valeurs en notation scientifique
    def format_scientific(value):
        return f"{value:.2e} ktN/yr"

    # Calculer le Throughflow pour chaque nœud
    throughflows = np.zeros(n_nodes)  # Initialiser le tableau des throughflows

    for i in range(n_nodes):
        # Somme des flux sortants si backward flow ou main node
        # Somme des flux entrants si forward flow
        if i == main_node or np.any(transition_matrix[:, i] > 0):
            throughflows[i] = np.sum(transition_matrix[i, :])  # Sortants
        else:
            throughflows[i] = np.sum(transition_matrix[:, i])  # Entrants

    # Génération des labels et couleurs des nœuds
    for i in range(n_nodes):
        if index_to_label[i] == "cereals (excluding rice) food nitrogen import-export":
            labels.append("cereals food export")
        elif index_to_label[i] == "cereals (excluding rice) feed nitrogen import-export":
            labels.append("cereals feed export")
        else:
            labels.append(index_to_label[i])

        node_colors.append(index_to_color[i])

        # Ajout des tooltips des nœuds avec Throughflow
        node_hover_texts.append(f"Node: {labels[i]}<br>Throughflow: {format_scientific(throughflows[i])}")

    # Ajout des flux sortants (cibles)
    def add_forward_flows(node, depth):
        if depth > scope:
            return
        for target_node in range(n_nodes):
            flow = transition_matrix[node, target_node]
            if flow > 0:
                sources.append(node)
                targets.append(target_node)
                values.append(flow)
                link_colors.append(index_to_color[target_node])
                link_hover_texts.append(
                    f"Source: {labels[node]}<br>Target: {labels[target_node]}<br>Value: {format_scientific(flow)}"
                )
                add_forward_flows(target_node, depth + 1)

    # Ajout des flux entrants (sources)
    def add_backward_flows(node, depth):
        if depth > scope:
            return
        for source_node in range(n_nodes):
            flow = transition_matrix[source_node, node]
            if flow > 0:
                sources.append(source_node)
                targets.append(node)
                values.append(flow)
                link_colors.append(index_to_color[source_node])
                link_hover_texts.append(
                    f"Source: {labels[source_node]}<br>Target: {labels[node]}<br>Value: {format_scientific(flow)}"
                )
                add_backward_flows(source_node, depth + 1)

    add_forward_flows(main_node, 1)
    add_backward_flows(main_node, 1)

    # Création du Sankey avec hovertemplate pour les nœuds et les liens
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=7,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
                customdata=node_hover_texts,  # Données pour le survol des nœuds
                hovertemplate="%{customdata}<extra></extra>",  # Affichage des tooltips personnalisés des nœuds
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                customdata=link_hover_texts,  # Données pour le survol des flux
                hovertemplate="%{customdata}<extra></extra>",  # Affichage des tooltips personnalisés des flux
            ),
        )
    )

    # Affichage du Sankey dans Streamlit
    st.plotly_chart(fig, use_container_width=True)


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


def streamlit_sankey_fertilization(
    model,
    cultures,
    legumineuses,
    prairies,
    merges={
        "population": ["urban", "rural"],
        "livestock": ["bovines", "ovines", "equine", "poultry", "porcines", "caprines"],
        "industry": ["Haber-Bosch", "other sectors"],
    },
    THRESHOLD=1e-1,
):
    """
    Crée un diagramme de Sankey montrant la distribution des backward flows
    pour les cultures, légumineuses et prairies après fusion de certains nœuds
    (ex: "urban"+"rural" -> "population"), en éliminant les nœuds et flux
    dont le throughflow/valeur sont inférieurs à THRESHOLD (1e-1).
    """

    if model is None:
        st.error("❌ Le modèle n'est pas encore exécuté. Lancez d'abord le modèle.")
        return

    # -- 1) Fusion des nœuds -----------------------------------
    adjacency_matrix = model.adjacency_matrix
    labels = model.labels
    new_matrix, new_labels, old_to_new = merge_nodes(adjacency_matrix, labels, merges)

    new_label_to_index = {lbl: i for i, lbl in enumerate(new_labels)}
    n_new = len(new_labels)

    # -- 2) Identifier les cibles "target_categories" (backward flows) ---
    def old_label_to_new_index(old_label):
        if old_label not in labels:
            return None
        old_idx = labels.index(old_label)
        return old_to_new[old_idx]

    # On fusionne les indices d'origine => on obtient set(...) d'index fusionnés
    all_targets_merged = set()
    for lbl in cultures + legumineuses + prairies:
        new_i = old_label_to_new_index(lbl)
        if new_i is not None:
            all_targets_merged.add(new_i)

    target_categories = sorted(all_targets_merged)

    # -- 3) Couleur des nœuds fusionnés (ex. palette simple + couleurs d'origine) ---
    color_dict = {"population": "darkblue", "livestock": "lightblue", "industry": "purple"}
    # On récupère éventuellement certaines couleurs d'origine
    # On suppose model.node_color: dict(index->couleur) ou dict(label->couleur)
    for k, v in node_color.items():
        if index_to_label[k] in labels:  # k est un label ?
            color_dict[index_to_label[k]] = v
    default_color = "black"

    def get_color_for_label(lbl):
        return color_dict.get(lbl, default_color)

    # Re-créer un "new_node_colors" => couleur de chaque new_label
    new_node_colors = [get_color_for_label(lbl) for lbl in new_labels]

    # -- 4) Récupérer tous les flux backward vers target_categories ---
    sources_raw = []
    targets_raw = []
    values = []
    link_hover_texts = []
    link_colors = []

    def format_scientific(value):
        return f"{value:.2e} ktN/yr"

    for target_new_idx in target_categories:
        for source_new_idx in range(n_new):
            flow = new_matrix[source_new_idx, target_new_idx]
            if flow > 0:
                sources_raw.append(source_new_idx)
                targets_raw.append(target_new_idx)
                values.append(flow)

                link_color = new_node_colors[source_new_idx]  # couleur du lien = couleur source
                link_colors.append(link_color)

                link_hover_texts.append(
                    f"Source: {new_labels[source_new_idx]}<br>"
                    f"Target: {new_labels[target_new_idx]}<br>"
                    f"Value: {format_scientific(flow)}"
                )

    # -- 5) Calcul du throughflow pour filtrer les nœuds trop petits ---
    # Si un nœud est dans target_categories, on calcule la somme des flux entrants,
    # sinon la somme des flux sortants, par exemple.
    throughflows = np.zeros(n_new)
    for i in range(n_new):
        if i in target_categories:
            throughflows[i] = np.sum(new_matrix[:, i])  # flux entrants
        else:
            throughflows[i] = np.sum(new_matrix[i, :])  # flux sortants

    # -- 6) Filtrage des flux trop faibles --
    # On garde seulement ceux dont la value >= THRESHOLD
    kept_links = []
    for idx, val in enumerate(values):
        if val >= THRESHOLD:
            kept_links.append(idx)

    sources_raw = [sources_raw[i] for i in kept_links]
    targets_raw = [targets_raw[i] for i in kept_links]
    values = [values[i] for i in kept_links]
    link_hover_texts = [link_hover_texts[i] for i in kept_links]
    link_colors = [link_colors[i] for i in kept_links]

    # -- 7) Filtrage des nœuds trop faibles (throughflow < THRESHOLD) --
    kept_nodes = [i for i in range(n_new) if throughflows[i] >= THRESHOLD]

    # Filtrer les liens qui impliquent des nœuds non conservés
    final_links = []
    for idx in range(len(sources_raw)):
        s = sources_raw[idx]
        t = targets_raw[idx]
        if (s in kept_nodes) and (t in kept_nodes):
            final_links.append(idx)

    sources_raw = [sources_raw[i] for i in final_links]
    targets_raw = [targets_raw[i] for i in final_links]
    values = [values[i] for i in final_links]
    link_hover_texts = [link_hover_texts[i] for i in final_links]
    link_colors = [link_colors[i] for i in final_links]

    # -- 8) Re-map pour le Sankey final --
    all_nodes = sorted(set(sources_raw + targets_raw))
    node_map = {old_i: new_i for new_i, old_i in enumerate(all_nodes)}

    sankey_sources = [node_map[s] for s in sources_raw]
    sankey_targets = [node_map[t] for t in targets_raw]

    node_labels = []
    node_colors_final = []
    node_hover_data = []
    for old_idx in all_nodes:
        lbl = new_labels[old_idx]
        node_labels.append(lbl)
        node_colors_final.append(new_node_colors[old_idx])
        node_hover_data.append(f"Node: {lbl}<br>Throughflow: {format_scientific(throughflows[old_idx])}")

    # -- 9) Construction du Sankey final --
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors_final,
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
            arrangement="snap",  # pour respecter potentiellement un placement x,y si besoin
        )
    )

    fig.update_layout(width=1200, height=1000)

    st.plotly_chart(fig, use_container_width=False)


def streamlit_sankey_food_flows(
    model,
    cultures,
    legumineuses,
    prairies,
    trades,
    merges={
        "cereals (excluding rice) trade": [
            "cereals (excluding rice) food trade",
            "cereals (excluding rice) feed trade",
        ],
        "fruits and vegetables trade": ["fruits and vegetables food trade", "fruits and vegetables feed trade"],
        "leguminous trade": ["leguminous food trade", "leguminous feed trade"],
        "oleaginous trade": ["oleaginous food trade", "oleaginous feed trade"],
    },
    THRESHOLD=1e-1,
):
    """
    Crée un diagramme de Sankey montrant les flux FORWARD à partir de:
      - cultures + légumineuses + prairies + trades
    vers:
      - betail + population

    Les nœuds et flux inférieurs au THRESHOLD sont éliminés.
    """

    if model is None:
        st.error("❌ Le modèle n'est pas encore exécuté. Lancez d'abord le modèle.")
        return

    # 1) Fusion des nœuds
    adjacency_matrix = model.adjacency_matrix
    labels = model.labels
    new_matrix, new_labels, old_to_new = merge_nodes(adjacency_matrix, labels, merges)

    new_label_to_index = {lbl: i for i, lbl in enumerate(new_labels)}
    n_new = len(new_labels)

    # 2) Déterminer quels nœuds (fusionnés) correspondent aux sources / cibles

    # - A) Construit un helper pour passer de label d'origine à l'index fusionné
    def old_label_to_new_index(old_label):
        if old_label not in labels:
            return None
        old_idx = labels.index(old_label)
        return old_to_new[old_idx]

    # - B) Récupérer l'ensemble des sources fusionnées
    sources_merged = set()
    for lbl in cultures + legumineuses + prairies + trades:
        idx_merged = old_label_to_new_index(lbl)
        if idx_merged is not None:
            sources_merged.add(idx_merged)

    # - C) Récupérer l'ensemble des cibles fusionnées (population + livestock)
    #   Comme on a fait "population": ["urban", "rural"] et "livestock": [...]
    #   On veut juste trouver dans new_labels où se trouve "population" + "livestock"
    #   => On peut itérer sur new_labels
    targets_merged = set()
    for new_i, lbl_fused in enumerate(new_labels):
        if lbl_fused in Pop + betail:
            targets_merged.add(new_i)

    # 3) Couleur des nœuds fusionnés (on peut reprendre la même logique)
    color_dict = {"population": "darkblue", "livestock": "lightblue", "industry": "purple"}
    # Si le modèle a un node_color (optionnel)
    for k, v in node_color.items():
        if index_to_label[k] in labels:
            color_dict[index_to_label[k]] = v
    default_color = "gray"

    def get_color_for_label(lbl):
        return color_dict.get(lbl, default_color)

    new_node_colors = [get_color_for_label(lbl) for lbl in new_labels]

    # 4) Collecter tous les flux forward: source->target
    #    Seuls flux où source est dans sources_merged, target dans targets_merged
    sources_raw = []
    targets_raw = []
    values = []
    link_colors = []
    link_hover_texts = []

    def format_scientific(value):
        return f"{value:.2e} ktN/yr"

    for s_idx in sorted(sources_merged):
        for t_idx in sorted(targets_merged):
            flow = new_matrix[s_idx, t_idx]
            if flow > 0:
                sources_raw.append(s_idx)
                targets_raw.append(t_idx)
                values.append(flow)
                link_colors.append(new_node_colors[s_idx])  # Couleur du lien = couleur de la source
                link_hover_texts.append(
                    f"Source: {new_labels[s_idx]}<br>Target: {new_labels[t_idx]}<br>Value: {format_scientific(flow)}"
                )

    # 5) Calcul du throughflow (pour filtrer)
    #    - Les sources => somme sortante
    #    - Les cibles => somme entrante
    #    - Les autres => on peut choisir l'une ou l'autre, ou la somme globale
    throughflows = np.zeros(n_new)
    for i in range(n_new):
        if i in targets_merged:
            throughflows[i] = np.sum(new_matrix[:, i])  # flux entrants
        elif i in sources_merged:
            throughflows[i] = np.sum(new_matrix[i, :])  # flux sortants
        else:
            # nœud ni source ni cible -> flux sortant? ou 0?
            # on peut mettre 0 pour forcer à supprimer
            throughflows[i] = np.sum(new_matrix[i, :]) + np.sum(new_matrix[:, i])

    # 6) Filtrage des flux trop faibles
    kept_links = []
    for idx, val in enumerate(values):
        if val >= THRESHOLD:
            kept_links.append(idx)

    sources_raw = [sources_raw[i] for i in kept_links]
    targets_raw = [targets_raw[i] for i in kept_links]
    values = [values[i] for i in kept_links]
    link_colors = [link_colors[i] for i in kept_links]
    link_hover_texts = [link_hover_texts[i] for i in kept_links]

    # 7) Filtrage des nœuds trop faibles
    kept_nodes = [i for i in range(n_new) if throughflows[i] >= THRESHOLD]

    final_links = []
    for i, val in enumerate(sources_raw):
        s = sources_raw[i]
        t = targets_raw[i]
        if (s in kept_nodes) and (t in kept_nodes):
            final_links.append(i)

    sources_raw = [sources_raw[i] for i in final_links]
    targets_raw = [targets_raw[i] for i in final_links]
    values = [values[i] for i in final_links]
    link_colors = [link_colors[i] for i in final_links]
    link_hover_texts = [link_hover_texts[i] for i in final_links]

    # 8) Re-map indices pour Sankey
    all_nodes = sorted(set(sources_raw + targets_raw))
    node_map = {old_i: new_i for new_i, old_i in enumerate(all_nodes)}

    sankey_sources = [node_map[s] for s in sources_raw]
    sankey_targets = [node_map[t] for t in targets_raw]

    node_labels = []
    node_final_colors = []
    node_hover_data = []
    for old_idx in all_nodes:
        lbl = new_labels[old_idx]
        node_labels.append(lbl)
        node_final_colors.append(new_node_colors[old_idx])
        node_hover_data.append(f"Node: {lbl}<br>Throughflow: {format_scientific(throughflows[old_idx])}")

    # 9) Sankey final
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
        width=1200,
        height=1000,
    )

    st.plotly_chart(fig, use_container_width=False)
