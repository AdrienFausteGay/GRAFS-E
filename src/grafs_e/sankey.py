import plotly.graph_objects as go
from N_class import *
from donnees import *

def create_sankey():
    # Définir les labels des nœuds
    labels = [
        'Wheat', 'atmosphere', 'hydro-system', 'other losses', 'cereals food export', 'cereals feed export',
        'bovines', 'rural', 'urban', 'Sugar beet', 'Natural meadow'
    ]

    labels = [
        "Haber-Bosch", "Rapeseed", "Sugarbeet", "Natural meadow ", "Wheat", "Barley",
        "Roots food export", "Bovines", "Cereals food export"
    ]

    color_labels = [
        "red", "yellow", "yellow", "darkgreen", "yellow", "yellow",
        "gray", "lightblue", "gray"
    ]
    
    # Définir les liens entre les nœuds
    # source, target, value (proportions ou flux entre les nœuds)
    # sources = [0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6]
    # targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    # values = [4.15, 7.97, 3.42, 70.2, 8.56, 3.77, 2.32, 3.41, 1.84, 18.67, 10]
    
    sources = [0, 0, 0, 0, 2, 3, 4, 5, 7]
    targets = [1, 2, 3, 4, 6, 7, 8, 8, 3]
    values = [16.99, 15.26, 28, 81.47, 22.61, 13.96, 70.2, 11.28, 18.66]
    color_links = [
        "red", "red", "red", "red", "yellow",
        "darkgreen", "yellow", "yellow", "lightblue"
    ]

    # Créer le diagramme de Sankey
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=7,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=color_labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=color_links
        )
    ))
    
    # Ajouter un titre
    # fig.update_layout(title_text="Main flows from Wheat in Picardy 2010", font_size=10)
    
    # Afficher le graphique
    fig.show()

def create_sankey_from_transition_matrix(transition_matrix, main_node, scope=2, label_to_index=label_to_index):
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
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))
    
    # Ajouter un titre
    # fig.update_layout(title_text=f"Flow Diagram starting from Node {main_node}", font_size=10)
    
    # Afficher le graphique
    fig.show()

def create_sankey_from_transition_matrix_2(transition_matrix, main_node, scope=1, index_to_label=index_to_label, index_to_color=node_color):
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
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=7,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors  # Application des couleurs aux nœuds
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors  # Application des couleurs aux flux
        )
    ))
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
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=7,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=None,
            color=node_colors  # Application des couleurs aux nœuds
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors  # Application des couleurs aux flux
        )
    ))

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
        "Export": [52, 54, 55, 56, 57, 59, 60, 61, 62]
    }

    # Créer un dictionnaire de nouveaux labels pour les nœuds fusionnés
    merged_labels = {i: label for i, label in enumerate(groups.keys())}
    
    # Créer les labels et couleurs des nœuds fusionnés
    labels = list(groups.keys())  # Ajouter les labels des groupes fusionnés
    node_colors = [
        "purple", "yellow", "olive", "orange", "lightyellow", 
        "darkgreen", "lightgreen", "lightblue", "darkblue", "red", "cyan", "gray", "gray"
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
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=3,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
            ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors  # Application des couleurs aux flux
        )
    ))

    # Afficher le graphique
    fig.show()