import numpy as np
import plotly.graph_objects as go
import streamlit as st

from grafs_e.donnees import *
from grafs_e.N_class import *


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
    transition_matrix,
    main_node,
    scope=1,
    index_to_label=index_to_label,
    index_to_color=node_color,
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
        "gold",
        "olive",
        "orange",
        "lightyellow",
        "darkgreen",
        "lightgreen",
        "lightblue",
        "darkblue",
        "crimson",
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
    col2_nodes = [
        4,
        5,
        6,
    ]  # Nœuds "Fruits and vegetables", "Grasslands and forages", "Leguminous"
    col3_nodes = [
        7,
        8,
        9,
        10,
    ]  # Nœuds "Livestock", "Population", "Losses", "Trade", "Atmosphere"

    # Créer le diagramme de Sankey
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=3,
                thickness=10,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
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

    fig.update_layout(
        template="plotly_dark",  # Thème sombre de Plotly
        font_color="black",  # Couleur générale du texte (titres, axes, etc.)
        font_size=20,
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


def streamlit_sankey_app(model, mode_complet):
    if mode_complet:
        # A) Mode complet, pas de fusion
        transition = model.adjacency_matrix
        new_labels = model.labels  # on ne change rien
        old_to_new = {i: i for i in range(len(new_labels))}  # mapping trivial

        # Sélectionner un objet parmi les labels originaux du modèle
        main_node_label = st.selectbox("Select an object", new_labels)

    else:
        # B) Mode simplifié, avec merges
        merges = {
            "Livestock": [
                "bovines",
                "ovines",
                "equine",
                "poultry",
                "porcines",
                "caprines",
            ],
            "Population": ["urban", "rural"],
            "Industry": ["Haber-Bosch", "other sectors"],
            "Cereals": [
                "Wheat",
                "Oat",
                "Barley",
                "Grain maize",
                "Rye",
                "Other cereals",
                "Rice",
            ],
            "Forages": [
                "Straw",
                "Forage maize",
                "Forage cabbages",
            ],
            "Temporary meadows": ["Non-legume temporary meadow", "Alfalfa and clover"],
            "Oleaginous": ["Rapeseed", "Sunflower", "Hemp", "Flax"],
            "Leguminous": [
                "Soybean",
                "Other oil crops",
                "Horse beans and faba beans",
                "Peas",
                "Other protein crops",
                "Green peas",
                "Dry beans",
                "Green beans",
            ],
            "Fruits and vegetables": [
                "Dry vegetables",
                "Dry fruits",
                "Squash and melons",
                "Cabbage",
                "Leaves vegetables",
                "Fruits",
                "Olives",
                "Citrus",
            ],
            "Roots": ["Sugar beet", "Potatoes", "Other roots"],
            "Environment": [
                "atmospheric N2",
                "Atmospheric deposition",
                "NH3 volatilization",
                "N2O emission",
                "hydro-system",
                "other losses",
                "soil stock",
            ],
            "Trade": [
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
                "fishery products",
                "temporary meadows feed trade",
            ],
        }

        # Fusion
        transition, new_labels, old_to_new = merge_nodes(model.adjacency_matrix, model.labels, merges=merges)

        # Sélectionner un objet parmi les *nouveaux* labels (ceux du merge)
        main_node_label = st.selectbox("Select an object", new_labels)

    # Récupérer l'index du "main_node_label" dans la liste new_labels
    main_node_index = new_labels.index(main_node_label)

    # =======================================================================
    # GÉNÉRER index->label et index->color pour la nouvelle matrice
    # =======================================================================
    new_index_to_label = {i: new_labels[i] for i in range(len(new_labels))}
    new_label_to_index = {new_labels[i]: i for i in range(len(new_labels))}

    # Par exemple, si node_color est un dict label -> couleur :
    def color_for_label(lbl):
        if lbl == "Livestock":
            return "lightblue"
        if lbl == "Population":
            return "darkblue"
        if lbl == "Environment":
            return "crimson"
        if lbl == "Leguminous":
            return "lightgreen"
        if lbl == "Forages":
            return "limegreen"
        if lbl == "Temporary meadows":
            return "darkgreen"
        if lbl == "Cereals":
            return "gold"
        if lbl == "Roots":
            return "orange"
        if lbl == "Oleaginous":
            return "darkkhaki"
        if lbl == "Fruits and vegetables":
            return "limegreen"
        if lbl == "Industry":
            return "purple"
        if lbl == "Trade":
            return "silver"
        return node_color.get(new_label_to_index[lbl], "gray")

    new_index_to_color = {i: color_for_label(new_labels[i]) for i in range(len(new_labels))}

    # =======================================================================
    # On peut maintenant appeler ta fonction streamlit_sankey
    # =======================================================================
    streamlit_sankey(
        transition_matrix=transition,
        main_node=main_node_index,
        scope=1,
        index_to_label=new_index_to_label,  # <--- on passe le nouveau mapping
        index_to_color=new_index_to_color,  # <--- idem
    )


def streamlit_sankey_fertilization(
    model,
    cultures,
    legumineuses,
    prairies,
    merges={
        "Population": ["urban", "rural"],
        "Livestock": ["bovines", "ovines", "equine", "poultry", "porcines", "caprines"],
        "Industry": ["Haber-Bosch", "other sectors"],
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
        st.error("❌ The model has not run yet. Please use the run tab.")
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
    color_dict = {
        "Population": "darkblue",
        "Livestock": "lightblue",
        "Industry": "purple",
        "Leguminous": "lightgreen",
        "Roots": "orange",
        "Fruits and vegetables": "limegreen",
        "Grassland and forages": "darkgreen",
        "Cereals": "gold",
        "Oleaginous": "darkkhaki",
        "Livestock and human": "lightblue",
        "Temporary meadows": "forestgreen",
        "Forages": "limegreen",
    }
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

    fig.update_layout(
        template="plotly_dark",  # Thème sombre de Plotly
        font_color="white",  # Couleur générale du texte (titres, axes, etc.)
        width=1200,
        height=1000,
    )

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
        "fruits and vegetables trade": [
            "fruits and vegetables food trade",
            "fruits and vegetables feed trade",
        ],
        "leguminous trade": ["leguminous food trade", "leguminous feed trade"],
        "oleaginous trade": ["oleaginous food trade", "oleaginous feed trade"],
    },
    THRESHOLD=1e-1,
):
    """
    Crée un diagramme de Sankey en séparant chaque nœud de trade en deux :
       - "(import)" pour les flux entrants dans la région
       - "(export)" pour les flux sortants de la région

    Flux considérés :
      - sources = cultures + légumineuses + prairies + trades
      - cibles   = population + livestock + trades
    Les nœuds et flux inférieurs à THRESHOLD sont éliminés.
    """

    if model is None:
        st.error("❌ Le modèle n'est pas encore exécuté. Lancez d'abord le modèle.")
        return

    adjacency_matrix = model.adjacency_matrix
    labels = model.labels

    # 1) Fusion des nœuds via merges
    new_matrix, new_labels, old_to_new = merge_nodes(adjacency_matrix, labels, merges)
    n_new = len(new_labels)

    # Petit helper pour retrouver l'index fusionné d'un ancien label :
    def old_label_to_new_index(old_label):
        if old_label not in labels:
            return None
        old_idx = labels.index(old_label)
        return old_to_new[old_idx]

    # -------------------------------------------------------------------------
    # 2) Construire des ensembles de nœuds "sources" et "cibles" (fusionnés)
    # -------------------------------------------------------------------------
    sources_merged = set()
    for lbl in cultures + legumineuses + prairies + trades:
        idx_merged = old_label_to_new_index(lbl)
        if idx_merged is not None:
            sources_merged.add(idx_merged)

    targets_merged = set()
    for lbl in betail + Pop + trades:
        idx_merged = old_label_to_new_index(lbl)
        if idx_merged is not None:
            targets_merged.add(idx_merged)

    # -------------------------------------------------------------------------
    # 3) Identifier les nœuds "trade" dans la nouvelle matrice
    #    => On cherche ceux dont le label contient "trade" OU ceux issus
    #       du merges si l'on veut être plus explicite
    # -------------------------------------------------------------------------
    trade_merged = {i for i, lbl_fused in enumerate(new_labels) if "trade" in lbl_fused.lower()}

    # -------------------------------------------------------------------------
    # 4) On va construire un nouveau "réseau" pour Sankey,
    #    où chaque nœud "trade" est dupliqué en 2 : import / export
    # -------------------------------------------------------------------------
    #    => On crée une liste "all_sankey_nodes" = labels finaux dans Sankey
    #    => On crée des mappings pour chaque nœud original i
    #       - node_map_import[i]  pour la partie import
    #       - node_map_export[i]  pour la partie export
    #       - ou un node_map_normal[i] pour ceux qui ne sont pas trade
    # -------------------------------------------------------------------------
    all_sankey_nodes = []
    node_map_import = {}
    node_map_export = {}
    node_map_normal = {}

    for i in range(n_new):
        label_i = new_labels[i]
        if i in trade_merged:
            # Créer 2 nœuds : (import) et (export)
            idx_import = len(all_sankey_nodes)
            all_sankey_nodes.append(f"{label_i[:-6]} import")
            node_map_import[i] = idx_import

            idx_export = len(all_sankey_nodes)
            all_sankey_nodes.append(f"{label_i[:-6]} export")
            node_map_export[i] = idx_export

        else:
            # Nœud normal : un seul nœud
            idx_norm = len(all_sankey_nodes)
            all_sankey_nodes.append(label_i)
            node_map_normal[i] = idx_norm

    # -------------------------------------------------------------------------
    # 5) Pour construire la liste finale des flux (links) du Sankey,
    #    on balaye l'ancienne matrice new_matrix.
    #    Sauf qu'on ne veut pas TOUT, seulement :
    #      - s ∈ sources_merged
    #      - t ∈ targets_merged
    #    Et on scinde le flux en "import" ou "export" selon le côté trade.
    # -------------------------------------------------------------------------
    final_sources = []
    final_targets = []
    final_values = []
    final_hover_texts = []

    def format_scientific(value):
        return f"{value:.2e} ktN/yr"

    for s_idx in range(n_new):
        if s_idx not in sources_merged:
            continue
        for t_idx in range(n_new):
            if t_idx not in targets_merged:
                continue
            flow = new_matrix[s_idx, t_idx]
            if flow <= 0:
                continue

            # Cas 1) s_idx ∈ trade, t_idx ∉ trade => flux import
            if s_idx in trade_merged and t_idx not in trade_merged:
                sankey_s = node_map_import[s_idx]  # (import)
                sankey_t = node_map_normal[t_idx]  # normal
                flow_type = "Import"

            # Cas 2) s_idx ∉ trade, t_idx ∈ trade => flux export
            elif s_idx not in trade_merged and t_idx in trade_merged:
                sankey_s = node_map_normal[s_idx]
                sankey_t = node_map_export[t_idx]  # (export)
                flow_type = "Export"

            # Cas 3) ni s_idx ni t_idx n’est trade => flux interne
            elif s_idx not in trade_merged and t_idx not in trade_merged:
                sankey_s = node_map_normal[s_idx]
                sankey_t = node_map_normal[t_idx]
                flow_type = "Interne"

            # Cas 4) s_idx ∈ trade, t_idx ∈ trade
            #        => flux trade->trade (rare?). On peut l’ignorer ou décider d’une convention
            else:
                # Soit on l'ignore :
                # continue
                # Soit on le met en import->export, par ex.:
                sankey_s = node_map_import[s_idx]
                sankey_t = node_map_export[t_idx]
                flow_type = "Trade->Trade"

            final_sources.append(sankey_s)
            final_targets.append(sankey_t)
            final_values.append(flow)
            final_hover_texts.append(
                f"Source: {all_sankey_nodes[sankey_s]}<br>"
                f"Target: {all_sankey_nodes[sankey_t]}<br>"
                f"Type: {flow_type}<br>"
                f"Value: {format_scientific(flow)}"
            )

    # -------------------------------------------------------------------------
    # 6) Filtrer par THRESHOLD sur final_values
    # -------------------------------------------------------------------------
    kept_indices = [i for i, v in enumerate(final_values) if v >= THRESHOLD]

    final_sources = [final_sources[i] for i in kept_indices]
    final_targets = [final_targets[i] for i in kept_indices]
    final_values = [final_values[i] for i in kept_indices]
    final_hover_texts = [final_hover_texts[i] for i in kept_indices]

    # -------------------------------------------------------------------------
    # 7) Calcul du throughflow pour chaque nœud (dans la nouvelle liste)
    #    Puis on supprime les nœuds trop faibles
    # -------------------------------------------------------------------------
    nb_sankey_nodes = len(all_sankey_nodes)
    flow_in = [0.0] * nb_sankey_nodes
    flow_out = [0.0] * nb_sankey_nodes

    for i, val in enumerate(final_values):
        s = final_sources[i]
        t = final_targets[i]
        flow_out[s] += val
        flow_in[t] += val

    # on peut définir le "throughflow" comme entrée+sortie ou max(entrée,sortie)
    # Ici, prenons la somme totale
    throughflow = [flow_in[i] + flow_out[i] for i in range(nb_sankey_nodes)]

    kept_nodes = {i for i, thr in enumerate(throughflow) if thr >= THRESHOLD}

    # Filtrage des liens pour garder ceux dont source ET target sont conservés
    new_links_idx = []
    for i in range(len(final_sources)):
        if final_sources[i] in kept_nodes and final_targets[i] in kept_nodes:
            new_links_idx.append(i)

    final_sources = [final_sources[i] for i in new_links_idx]
    final_targets = [final_targets[i] for i in new_links_idx]
    final_values = [final_values[i] for i in new_links_idx]
    final_hover_texts = [final_hover_texts[i] for i in new_links_idx]

    # -------------------------------------------------------------------------
    # 8) Re-map les indices de nœuds pour le Sankey final
    # -------------------------------------------------------------------------
    sorted_kept = sorted(kept_nodes)
    node_map_sankey = {old_i: new_i for new_i, old_i in enumerate(sorted_kept)}

    sankey_sources = [node_map_sankey[s] for s in final_sources]
    sankey_targets = [node_map_sankey[t] for t in final_targets]

    # 9) Constructions des labels finaux et couleurs
    sankey_labels = []
    sankey_colors = []
    color_dict = {
        "Cereals": "gold",
        "Fruits and vegetables": "lightgreen",
        "Leguminous": "darkgreen",
        "Oleaginous": "lightgreen",
        "Grassland and forages": "green",
        "monogastrics": "lightblue",
        "Livestock": "lightblue",
        "Population": "darkblue",
        "Roots": "orange",
        "Temporary meadows": "forestgreen",
        "Forages": "limegreen",
    }
    for old_i in sorted_kept:
        lbl = all_sankey_nodes[old_i]
        sankey_labels.append(lbl)

        # Couleur simple : import en vert, export en rouge
        if "import" in lbl:
            sankey_colors.append("slategray")
        elif "export" in lbl:
            sankey_colors.append("silver")
        elif lbl in color_dict.keys():
            # Sinon on utilise ta palette custom ou gris par défaut
            # On peut essayer de retrouver l'index "i" d'origine
            # Cf. ci-dessous ou on applique juste un gris
            sankey_colors.append(color_dict[lbl])
        elif lbl in labels:
            sankey_colors.append(node_color[label_to_index[lbl]])
        else:
            sankey_colors.append("gray")

    # 10) Construction du Sankey
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=sankey_labels,
                color=sankey_colors,
                customdata=[
                    f"Node: {lbl}<br>Throughflow: {format_scientific(throughflow[old_i])}" for old_i in sorted_kept
                ],
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=sankey_sources,
                target=sankey_targets,
                value=final_values,
                color=[
                    sankey_colors[s]  # Couleur = couleur du noeud source, par ex
                    for s in sankey_sources
                ],
                customdata=final_hover_texts,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            arrangement="snap",
        )
    )

    fig.update_layout(
        template="plotly_dark",  # Thème sombre de Plotly
        font_color="white",  # Couleur générale du texte (titres, axes, etc.)
        width=1200,
        height=1000,
    )
    st.plotly_chart(fig, use_container_width=False)


def streamlit_sankey_systemic_flows(
    model,
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
    Crée un diagramme de Sankey systémique montrant tous les flux de la matrice d'adjacence du modèle.
    Les nœuds sont fusionnés selon les règles de `merges`, et les flux inférieurs à `THRESHOLD` sont éliminés.

    :param model: Modèle contenant la matrice d'adjacence (model.adjacency_matrix) et les labels (model.labels).
    :param merges: Dictionnaire définissant les fusions de nœuds.
    :param THRESHOLD: Seuil en dessous duquel les flux sont supprimés (par défaut : 1e-1).
    """
    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st

    if model is None:
        st.error("❌ Le modèle n'est pas encore exécuté. Lancez d'abord le modèle.")
        return

    # 1) Fusion des nœuds
    adjacency_matrix = model.adjacency_matrix
    labels = model.labels
    new_matrix, new_labels, old_to_new = merge_nodes(adjacency_matrix, labels, merges)
    THRESHOLD = (adjacency_matrix.sum() / 100,)
    n_new = len(new_labels)

    # 2) Définir les couleurs des nœuds fusionnés
    color_dict = {
        "cereals (excluding rice)": "gold",
        "fruits and vegetables": "lightgreen",
        "leguminous": "lime",
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
        "temporary meadows": "forestgreen",
    }
    # Ajouter les couleurs des labels d'origine si disponibles
    for k, v in node_color.items():
        if index_to_label[k] in labels:
            color_dict[index_to_label[k]] = v
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
            arrangement="freeform",
        )
    )

    fig.update_layout(
        template="plotly_dark",  # Thème sombre de Plotly
        font_color="white",  # Couleur générale du texte (titres, axes, etc.)
        # title="Systemic Sankey Diagram: All Flows",
        width=2000,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=False)
