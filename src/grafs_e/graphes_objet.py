import heapq

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch

from grafs_e.donnees import *


class Graphs_maker:
    def __init__(self):
        from grafs_e.N_class import DataLoader

        self.data = DataLoader()

    def get_graph(self, region, year, cut_isolated=True):
        nitrogen_model = NitrogenFlowModel(
            data=self.data,
            year=year,
            region=region,
            categories_mapping=categories_mapping,
            labels=labels_init,
            cultures=cultures,
            legumineuses=legumineuses,
            prairies=prairies,
            betail=betail,
            Pop=Pop,
            ext=ext,
        )
        adjacency_matrix = nitrogen_model.get_transition_matrix()
        # Convertir la matrice pondérée en un graphe orienté pondéré
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

        # Ajouter un attribut 'index' pour conserver l'indexation d'origine
        for node in G.nodes:
            G.nodes[node]["index"] = node  # Stocker l'indice d'origine du nœud

        # Ajouter les poids aux arêtes
        for i, j in zip(*np.nonzero(adjacency_matrix)):
            G[i][j]["weight"] = adjacency_matrix[i, j]
        if cut_isolated:
            # Supprimer les nœuds sans arêtes entrantes et sortantes
            isolated_nodes = [node for node in G.nodes if G.in_degree(node) == 0 and G.out_degree(node) == 0]
            G.remove_nodes_from(isolated_nodes)
        return G

    def get_matrix(self, year, region):
        nitrogen_model = NitrogenFlowModel(
            data=self.data,
            year=year,
            region=region,
            categories_mapping=categories_mapping,
            labels=labels_init,
            cultures=cultures,
            legumineuses=legumineuses,
            prairies=prairies,
            betail=betail,
            Pop=Pop,
            ext=ext,
        )
        return nitrogen_model.get_transition_matrix()


class GraphVisualizer:
    def __init__(self, G):
        self.G = G
        self.pos = None  # Positions des nœuds
        self.node_colors = None
        self.label_to_index = label_to_index  # Mapping des labels vers les indices de nœuds
        self.index_to_label = {v: k for k, v in label_to_index.items()}

    def draw_curved_edges(self, pos, ax, alpha=0.7, edge_color="white"):
        for edge in self.G.edges():
            src, dst = edge
            rad = 0.2  # Ajustez ce paramètre pour contrôler la courbure
            arrowprops = dict(arrowstyle="-", color=edge_color, lw=0.5, alpha=alpha, connectionstyle=f"arc3,rad={rad}")
            ax.annotate("", xy=pos[dst], xycoords="data", xytext=pos[src], textcoords="data", arrowprops=arrowprops)

    def draw_curved_edges_log(self, pos, ax, alpha=0.1, edge_color="white"):
        min_weight = 1e-2  # Plage min pour l'échelle logarithmique
        max_weight = 1e2  # Plage max pour l'échelle logarithmique
        for edge in self.G.edges(data=True):
            src, dst, data = edge
            weight = data["weight"]
            # Appliquer l'échelle logarithmique
            log_weight = np.log10(max(weight, min_weight))  # Éviter les poids nuls
            normalized_weight = (log_weight - np.log10(min_weight)) / (np.log10(max_weight) - np.log10(min_weight))
            # Définir l'épaisseur des arêtes sur une plage adaptée (par exemple 0.5 à 5)
            line_width = 0.5 + normalized_weight * 4.5  # Épaisseur min 0.5, max 5
            rad = 0.2  # Courbure des arêtes
            arrowprops = dict(
                arrowstyle="-|>",  # Représenter la direction avec une flèche
                color=edge_color,
                lw=line_width,  # Utiliser l'épaisseur ajustée
                alpha=alpha,
                connectionstyle=f"arc3,rad={rad}",
            )
            ax.annotate("", xy=pos[dst], xycoords="data", xytext=pos[src], textcoords="data", arrowprops=arrowprops)

    def draw_curved_edges_fancy(self, pos, ax, alpha=0.7):
        min_weight = 1e-2
        max_weight = 1e2

        # Obtenir la liste des poids des arêtes
        weights = [self.G[u][v]["weight"] for u, v in self.G.edges()]

        # Normaliser les poids pour l'échelle logarithmique
        norm = mcolors.LogNorm(vmin=min_weight, vmax=max_weight)
        cmap = plt.cm.rainbow  # Choix de la palette de couleurs adaptée à un fond noir

        for edge in self.G.edges(data=True):
            src, dst, data = edge
            weight = data["weight"]

            # Appliquer l'échelle logarithmique
            log_weight = np.log10(max(weight, min_weight))  # Éviter les poids nuls
            normalized_weight = (log_weight - np.log10(min_weight)) / (np.log10(max_weight) - np.log10(min_weight))

            # Épaisseur de l'arête en fonction du poids
            line_width = 0.5 + normalized_weight * 4.5  # Épaisseur min 0.5, max 5

            # Couleur de l'arête en fonction du poids
            edge_color = cmap(norm(weight))

            rad = 0.2  # Courbure des arêtes

            # Créer un patch courbé avec la couleur définie par le poids
            arrow = FancyArrowPatch(
                posA=pos[src],
                posB=pos[dst],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                color=edge_color,
                lw=line_width,
                alpha=alpha,
                mutation_scale=10,
            )
            ax.add_patch(arrow)
        return cmap, norm

    def draw_nodes_with_patches(self, pos, ax, node_size=300, node_colors=None):
        for i, (node, (x, y)) in enumerate(pos.items()):
            # Dessiner les cercles avec `plt.Circle`
            circle = Circle(
                (x, y), radius=node_size / 2000, color=node_colors[i], zorder=2
            )  # zorder=2 pour être au-dessus des arêtes
            ax.add_patch(circle)

    def regroup_nodes_layout(self, pos, node_colors, group_colors, ecart=0.05):
        # Récupérer les positions initiales
        new_pos = pos.copy()
        group_centers = {}  # Pour mémoriser les centres des groupes

        # Grouper les nœuds par couleur
        for color in group_colors:
            # Récupérer les nœuds appartenant à ce groupe
            group_nodes = [node for node, color_node in zip(self.G.nodes, node_colors) if color_node == color]
            # Calculer le barycentre des positions actuelles pour ces nœuds
            if group_nodes:
                avg_x = np.mean([pos[node][0] for node in group_nodes])
                avg_y = np.mean([pos[node][1] for node in group_nodes])
                group_centers[color] = (avg_x, avg_y)

                # Réassigner les positions proches du centre de groupe
                for node in group_nodes:
                    new_pos[node] = (avg_x + np.random.uniform(-ecart, ecart), avg_y + np.random.uniform(-ecart, ecart))

        # Placer les nœuds gris vers l'extérieur
        for node, color in zip(self.G.nodes, node_colors):
            if color == "gray":
                # Calculer une direction vers l'extérieur (rayon plus large)
                new_pos[node] = (new_pos[node][0] * 1, new_pos[node][1] * 1)  # Ajustez si nécessaire

        return new_pos

    def color_nodes(self, cultures, legumineuses, prairies, betail, Pop):
        n, m, k, p, q = (
            len(cultures),
            len(legumineuses),
            len(prairies),
            len(betail),
            len(Pop),
        )  # Nombre de nœuds à colorier

        # Assigner une couleur en fonction de l'index d'origine
        for node in self.G.nodes:
            original_index = self.G.nodes[node]["index"]  # Récupérer l'index d'origine
            if original_index < n:
                self.G.nodes[node]["color"] = "#FFA500"  # Orange doux pour Cultures
            elif original_index < n + m:
                self.G.nodes[node]["color"] = "#66C2A5"  # Vert clair pour Légumineuses
            elif original_index < n + m + k:
                self.G.nodes[node]["color"] = "#FC8D62"  # Rouge doux pour Prairies
            elif original_index < n + m + k + p:
                self.G.nodes[node]["color"] = "#8DA0CB"  # Bleu pastel pour Bétail
            elif original_index < n + m + k + p + q:
                self.G.nodes[node]["color"] = "#E5E5E5"  # Blanc cassé pour Population
            else:
                self.G.nodes[node]["color"] = "#BFBFBF"  # Gris pour les autres

        # Stocker les couleurs des nœuds dans une liste
        self.node_colors = [self.G.nodes[node]["color"] for node in self.G.nodes()]

    def plot_graph_base(self):
        if self.node_colors is None:
            # Couleurs par défaut si non assignées
            self.node_colors = ["gray"] * self.G.number_of_nodes()

        # Calculer les positions des nœuds avec `nx.spring_layout`
        self.pos = nx.spring_layout(self.G, k=0.8)

        # Appliquer `regroup_nodes_layout` pour regrouper les nœuds par couleur et éloigner les gris
        group_colors = ["yellow", "green", "red", "blue", "white"]  # Couleurs de groupe à regrouper
        self.pos = self.regroup_nodes_layout(self.pos, self.node_colors, group_colors, ecart=0.1)

        # Dessiner le graphe
        fig, ax = plt.subplots(figsize=(16, 12), dpi=500, facecolor="black")
        ax.set_facecolor("black")
        ax.set_aspect("equal")

        # Dessiner les arêtes courbées avec `FancyArrowPatch`
        cmap, norm = self.draw_curved_edges_fancy(self.pos, ax, alpha=0.2)

        # Dessiner les nœuds avec `plt.Circle` pour qu'ils apparaissent au-dessus des arêtes
        self.draw_nodes_with_patches(self.pos, ax, node_size=20, node_colors=self.node_colors)

        # Ajuster les limites des axes pour éviter que le graphe soit coupé
        x_values, y_values = zip(*self.pos.values())
        ax.set_xlim(min(x_values) - 0.1, max(x_values) + 0.1)
        ax.set_ylim(min(y_values) - 0.1, max(y_values) + 0.1)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Nécessaire pour que ScalarMappable fonctionne avec colorbar
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label("Poids des arêtes")

        # Modifier la couleur du label de la colorbar en blanc
        cbar.set_label("Poids des arêtes", color="white")

        # Modifier la couleur des graduations (ticks) de la colorbar en blanc
        cbar.ax.yaxis.set_tick_params(color="white")

        # Modifier la couleur des étiquettes de la colorbar en blanc
        plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")

        # Désactiver les axes
        plt.axis("off")
        plt.show()

    def ajouter_retour_ligne_si_long(self, text, longueur_max):
        # Fonction pour ajouter un retour à la ligne si le texte est trop long
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= longueur_max:
                current_line += " " + word
            else:
                lines.append(current_line.strip())
                current_line = word
        lines.append(current_line.strip())
        return "\n".join(lines)

    def draw_custom_networkx_edges(
        self,
        subgraph,
        pos,
        edgelist=None,
        width=1,
        edge_color="blue",
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
        node_size=300,
    ):
        """
        Dessine des arêtes avec des flèches ajustées pour qu'elles se terminent à la périphérie des nœuds.

        :param subgraph: Le sous-graphe NetworkX
        :param pos: Positions des nœuds calculées par un layout (ex: nx.spring_layout)
        :param edgelist: Liste des arêtes à dessiner (par défaut, toutes les arêtes du sous-graphe)
        :param width: Largeur des arêtes (peut être une liste correspondant à chaque arête)
        :param edge_color: Couleur des arêtes
        :param arrows: Afficher des flèches ou non
        :param arrowstyle: Style des flèches ('->', '-|>', etc.)
        :param arrowsize: Taille des flèches
        :param node_size: Taille des nœuds pour ajuster les points de début et de fin des arêtes
        """
        ax = plt.gca()  # Récupérer l'axe actuel de la figure

        if edgelist is None:
            edgelist = list(subgraph.edges())

        node_radius = np.sqrt(node_size) / 110  # Calculer le rayon approximatif des nœuds en fonction de leur taille

        # Si width est une seule valeur, transformer en liste pour compatibilité
        if not isinstance(width, list):
            width = [width] * len(edgelist)

        for i, (u, v) in enumerate(edgelist):
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # Calculer l'angle de la direction de l'arête
            angle = np.arctan2(y2 - y1, x2 - x1)

            # Décaler les points de départ et d'arrivée de l'arête pour éviter les chevauchements avec les nœuds
            x1_new = x1 + node_radius * np.cos(angle)
            y1_new = y1 + node_radius * np.sin(angle)
            x2_new = x2 - node_radius * np.cos(angle)
            y2_new = y2 - node_radius * np.sin(angle)

            # Dessiner une flèche avec la tête à l'extérieur du nœud
            arrow = FancyArrowPatch(
                (x1_new, y1_new),
                (x2_new, y2_new),
                connectionstyle="arc3,rad=0.0",
                color=edge_color,
                lw=width[i],
                arrowstyle=arrowstyle,
                mutation_scale=arrowsize,
            )
            ax.add_patch(arrow)

    def afficher_et_visualiser_10_noeuds_plus_proches(self, noeud_depart_label):
        """
        Affiche les 10 nœuds les plus proches du nœud de départ, leur flux total,
        et visualise le sous-graphe correspondant en excluant les nœuds intermédiaires spécifiés.
        L'épaisseur des liens est proportionnelle au flux (avec échelle logarithmique).

        :param noeud_depart_label: Label du nœud de départ
        """
        # Vérifier que le mapping des labels est disponible
        if self.label_to_index is None or self.index_to_label is None:
            print("Le mapping des labels vers les indices n'est pas défini.")
            return

        # Vérifier que le nœud de départ existe dans le dictionnaire
        if noeud_depart_label not in self.label_to_index:
            print(f"Nœud de départ '{noeud_depart_label}' introuvable dans label_to_index.")
            return

        index_depart = self.label_to_index[noeud_depart_label]
        ext_nodes = [self.label_to_index[label] for label in ext if label in self.label_to_index]
        exclude_nodes_set = set(ext_nodes)
        exclude_nodes_set.discard(index_depart)  # Autoriser le nœud de départ même s'il est dans ext_nodes

        # Calculer les distances et les chemins en utilisant l'algorithme de Dijkstra personnalisé
        analyzer = GraphAnalyzer(self.G, self.label_to_index)
        distances, paths = analyzer.dijkstra_exclusion(index_depart, weight="weight", exclude_nodes=exclude_nodes_set)

        # Enlever le nœud de départ des résultats
        distances.pop(index_depart, None)
        paths.pop(index_depart, None)

        if not distances:
            print(f"Aucun nœud accessible depuis '{noeud_depart_label}' sans passer par les nœuds exclus.")
            return

        # Calculer le flux total pour chaque chemin (Option 1: utiliser 1/distance)
        flux_totaux = {target: 1 / distance for target, distance in distances.items()}

        # Trier les nœuds cibles en fonction du flux total (ordre décroissant)
        flux_trie = sorted(flux_totaux.items(), key=lambda x: x[1], reverse=True)
        top_10 = flux_trie[:10]

        # Mapper les index vers les labels
        index_to_label = self.index_to_label

        # Afficher les résultats
        print(
            f"Les 10 nœuds les plus proches de '{noeud_depart_label}' (en excluant les nœuds intermédiaires spécifiés) :"
        )
        for i, (index_noeud, flux_total) in enumerate(top_10, 1):
            label_noeud = index_to_label.get(index_noeud, f"Nœud {index_noeud}")
            print(f"{i}. {label_noeud} - Flux total: {flux_total:.2f}")

        # Construire le sous-graphe
        nodes_in_subgraph = set()
        edges_in_subgraph = []

        for index_noeud in [index for index, flux in top_10]:
            if index_noeud in paths:
                path = paths[index_noeud]
                nodes_in_subgraph.update(path)
                edges_in_subgraph.extend([(path[i], path[i + 1]) for i in range(len(path) - 1)])

        nodes_in_subgraph.add(index_depart)

        # Créer le sous-graphe
        subgraph = self.G.subgraph(nodes_in_subgraph).edge_subgraph(edges_in_subgraph).copy()

        # Préparer les labels
        labels = {index: index_to_label.get(index, str(index)) for index in subgraph.nodes()}

        # Visualisation
        plt.figure(figsize=(12, 8), dpi=500)
        ax = plt.gca()

        # Positionnement des nœuds
        pos = nx.spring_layout(subgraph, pos=nx.planar_layout(subgraph))

        # Couleurs des nœuds
        node_colors = []
        for node in subgraph.nodes():
            if node == index_depart:
                node_colors.append("red")  # Nœud de départ en rouge
            elif node in [index for index, flux in top_10]:
                node_colors.append("lightblue")  # Top 10 nœuds en bleu clair
            else:
                node_colors.append("gray")  # Nœuds intermédiaires en gris

        # Dessiner les nœuds avec labels
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=6000)

        # Ajouter les étiquettes avec un fond de la même couleur que les nœuds
        for node, (x, y) in pos.items():
            label = self.ajouter_retour_ligne_si_long(labels[node], 10)
            if label == noeud_depart_label:
                node_color = "red"
            else:
                node_color = "lightblue"
            ax.annotate(
                label,
                (x, y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.3", facecolor=node_color, edgecolor=node_color, alpha=0),
            )

        # Dessiner les arêtes en fonction du flux (poids) avec des épaisseurs proportionnelles (échelle logarithmique)
        edges_with_flux = list(subgraph.edges())
        flux = [self.G[u][v]["weight"] for u, v in edges_with_flux]

        # Échelle logarithmique pour l'épaisseur des arêtes
        flux_scaled = [np.log(f + 1) for f in flux]  # Ajouter 1 pour éviter log(0)
        max_flux_scaled = max(flux_scaled)
        edge_widths = [2 + (f / max_flux_scaled) * 10 for f in flux_scaled]  # Épaisseur entre 2 et 12

        # Dessiner les arêtes avec la méthode de la classe
        self.draw_custom_networkx_edges(
            subgraph,
            pos,
            edgelist=edges_with_flux,
            width=edge_widths,
            edge_color="blue",
            arrows=True,
            arrowstyle="->",
            arrowsize=50,
            node_size=300,
        )

        # Annoter les arêtes avec les flux
        edge_labels = {(u, v): f"{self.G[u][v]['weight']:.2f}" for u, v in subgraph.edges()}
        nx.draw_networkx_edge_labels(
            subgraph, pos, edge_labels=edge_labels, font_color="black", font_weight="bold", font_size=8
        )

        plt.axis("off")
        plt.margins(x=0.1, y=0.1)
        plt.show()

    def afficher_et_visualiser_10_noeuds_plus_proches_2(self, noeud_depart_label):
        """
        Affiche les 10 nœuds les plus proches du nœud de départ, leur flux total,
        et visualise le sous-graphe correspondant en excluant les nœuds intermédiaires spécifiés.
        L'épaisseur des liens est proportionnelle au flux (avec échelle logarithmique).

        :param noeud_depart_label: Label du nœud de départ
        """
        # Vérifier que le mapping des labels est disponible
        if self.label_to_index is None or self.index_to_label is None:
            print("Le mapping des labels vers les indices n'est pas défini.")
            return

        # Vérifier que le nœud de départ existe dans le dictionnaire
        if noeud_depart_label not in self.label_to_index:
            print(f"Nœud de départ '{noeud_depart_label}' introuvable dans label_to_index.")
            return

        index_depart = self.label_to_index[noeud_depart_label]
        ext_nodes = [self.label_to_index[label] for label in ext if label in self.label_to_index]
        exclude_nodes_set = set(ext_nodes)
        exclude_nodes_set.discard(index_depart)  # Autoriser le nœud de départ même s'il est dans ext_nodes

        # Calculer les distances et les chemins en utilisant l'algorithme de Dijkstra personnalisé
        analyzer = GraphAnalyzer(self.G, self.label_to_index)
        distances, paths = analyzer.dijkstra_exclusion(index_depart, weight="weight", exclude_nodes=exclude_nodes_set)

        # Enlever le nœud de départ des résultats
        distances.pop(index_depart, None)
        paths.pop(index_depart, None)

        if not distances:
            print(f"Aucun nœud accessible depuis '{noeud_depart_label}' sans passer par les nœuds exclus.")
            return

        # Calculer le flux total pour chaque chemin (Option 1: utiliser 1/distance)
        flux_totaux = {target: 1 / distance for target, distance in distances.items()}

        # Trier les nœuds cibles en fonction du flux total (ordre décroissant)
        flux_trie = sorted(flux_totaux.items(), key=lambda x: x[1], reverse=True)
        top_10 = flux_trie[:10]

        # Mapper les index vers les labels
        index_to_label = self.index_to_label

        # Afficher les résultats
        print(
            f"Les 10 nœuds les plus proches de '{noeud_depart_label}' (en excluant les nœuds intermédiaires spécifiés) :"
        )
        for i, (index_noeud, flux_total) in enumerate(top_10, 1):
            label_noeud = index_to_label.get(index_noeud, f"Nœud {index_noeud}")
            print(f"{i}. {label_noeud} - Flux total: {flux_total:.2f}")

        # Construire le sous-graphe
        nodes_in_subgraph = set()
        edges_in_subgraph = []

        for index_noeud in [index for index, flux in top_10]:
            if index_noeud in paths:
                path = paths[index_noeud]
                nodes_in_subgraph.update(path)
                edges_in_subgraph.extend([(path[i], path[i + 1]) for i in range(len(path) - 1)])

        nodes_in_subgraph.add(index_depart)

        # Créer le sous-graphe
        subgraph = self.G.subgraph(nodes_in_subgraph).edge_subgraph(edges_in_subgraph).copy()

        # Préparer les labels
        labels = {index: index_to_label.get(index, str(index)) for index in subgraph.nodes()}

        # Visualisation
        plt.figure(figsize=(12, 8), dpi=500)
        ax = plt.gca()

        # Positionnement des nœuds
        pos = nx.spring_layout(subgraph, iterations=50, k=0.07, pos=nx.planar_layout(subgraph))
        # pos = nx.planar_layout(subgraph, scale=10)
        # Couleurs des nœuds
        node_colors = []
        for node in subgraph.nodes():
            if node == index_depart:
                node_colors.append("red")  # Nœud de départ en rouge
            elif node in [index for index, flux in top_10]:
                node_colors.append("lightblue")  # Top 10 nœuds en bleu clair
            else:
                node_colors.append("gray")  # Nœuds intermédiaires en gris

        # Dessiner les nœuds sous forme de rectangles
        for node, (x, y) in pos.items():
            node_color = (
                "red" if node == index_depart else "lightblue" if node in [index for index, flux in top_10] else "gray"
            )
            ax.scatter(x, y, s=500, c=node_color, edgecolors="black", marker="s")  # 's' pour carré/rectangle

        # Ajouter les étiquettes avec un fond de la même couleur que les nœuds (boîtes rectangulaires)
        for node, (x, y) in pos.items():
            label = self.ajouter_retour_ligne_si_long(labels[node], 10)
            node_color = "red" if label == noeud_depart_label else "lightblue"
            ax.annotate(
                label,
                (x, y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.3", facecolor=node_color, edgecolor=node_color, alpha=0.5),
            )

        # Dessiner les arêtes en fonction du flux (poids) avec des épaisseurs proportionnelles (échelle logarithmique)
        edges_with_flux = list(subgraph.edges())
        flux = [self.G[u][v]["weight"] for u, v in edges_with_flux]

        # Échelle logarithmique pour l'épaisseur des arêtes
        flux_scaled = [np.log(f + 1) for f in flux]  # Ajouter 1 pour éviter log(0)
        max_flux_scaled = max(flux_scaled)
        edge_widths = [2 + (f / max_flux_scaled) * 10 for f in flux_scaled]  # Épaisseur entre 2 et 12

        # Dessiner les arêtes avec la méthode de la classe
        self.draw_custom_networkx_edges(
            subgraph,
            pos,
            edgelist=edges_with_flux,
            width=edge_widths,
            edge_color="blue",
            arrows=True,
            arrowstyle="->",
            arrowsize=50,
            node_size=300,
        )

        # Annoter les arêtes avec les flux
        edge_labels = {(u, v): f"{self.G[u][v]['weight']:.2f}" for u, v in subgraph.edges()}
        nx.draw_networkx_edge_labels(
            subgraph, pos, edge_labels=edge_labels, font_color="black", font_weight="bold", font_size=8
        )

        plt.axis("off")
        plt.margins(x=0.1, y=0.1)
        plt.show()

    def afficher_chemins_graphiquement_tot(self, n):
        """
        Affiche graphiquement les chemins trouvés dans le graphe, en mettant en évidence les nœuds et les arêtes
        impliqués dans les chemins sélectionnés.

        :param n: Nombre total de plus courts chemins à afficher
        """
        # Initialiser l'analyseur pour obtenir les chemins
        analyzer = GraphAnalyzer(self.G, self.label_to_index)
        chemins = analyzer.obtenir_n_plus_courts_chemins(n)

        pos = nx.spring_layout(self.G, seed=42)  # Disposition des nœuds avec un seed pour la reproductibilité

        plt.figure(figsize=(12, 8))

        # Dessiner tous les nœuds du graphe avec des couleurs standard
        nx.draw_networkx_nodes(self.G, pos, node_color="lightgray", node_size=500)

        # Dessiner toutes les arêtes du graphe avec des couleurs gris clair
        nx.draw_networkx_edges(self.G, pos, edge_color="lightgray", style="solid")

        # Dessiner les labels des nœuds
        nx.draw_networkx_labels(self.G, pos, font_size=10, labels=self.index_to_label)

        # Mettre en évidence les chemins sélectionnés
        for source, target, cost, chemin in chemins:
            # Colorer les nœuds impliqués dans les chemins sélectionnés
            nx.draw_networkx_nodes(self.G, pos, nodelist=chemin, node_color="lightblue", node_size=600)

            # Dessiner les arêtes correspondant aux chemins sélectionnés
            edges_in_path = [(chemin[i], chemin[i + 1]) for i in range(len(chemin) - 1)]
            nx.draw_networkx_edges(self.G, pos, edgelist=edges_in_path, edge_color="blue", width=2.5)

            # Annoter les distances le long des arêtes du chemin
            edge_labels = {
                (chemin[i], chemin[i + 1]): f"{1 / self.G[chemin[i]][chemin[i + 1]]['weight']:.2f}"
                for i in range(len(chemin) - 1)
            }
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color="red")

        plt.title("Visualisation des plus courts chemins trouvés", fontsize=14)
        plt.axis("off")  # Cacher les axes
        plt.show()

    def afficher_chemins_graphiquement_sous_graph(self, n):
        """
        Affiche graphiquement les chemins trouvés dans un graphe, en mettant en évidence uniquement les nœuds et les arêtes
        impliqués dans les chemins sélectionnés, avec les noms des nœuds affichés plutôt que leurs numéros.

        :param n: Nombre total de plus courts chemins à afficher
        """
        # Obtenir les 'n' plus courts chemins en utilisant l'analyseur
        analyzer = GraphAnalyzer(self.G, self.label_to_index)
        chemins = analyzer.obtenir_n_plus_courts_chemins(n)

        # Créer un sous-graphe contenant uniquement les nœuds et arêtes impliqués dans les 'n' plus courts chemins
        sous_graphe = nx.DiGraph()

        # Ajouter les nœuds et arêtes des chemins sélectionnés dans le sous-graphe
        for source, target, cost, chemin in chemins:
            sous_graphe.add_nodes_from(chemin)
            edges_in_path = [(chemin[i], chemin[i + 1]) for i in range(len(chemin) - 1)]
            sous_graphe.add_edges_from(edges_in_path)

        # Créer la position des nœuds pour le sous-graphe
        pos = nx.spring_layout(sous_graphe, seed=42)  # Layout pour le sous-graphe

        plt.figure(figsize=(12, 8))

        # Convertir les nœuds du sous-graphe en leurs noms
        labels = {node: self.index_to_label[node] for node in sous_graphe.nodes()}

        # Dessiner les nœuds du sous-graphe
        nx.draw_networkx_nodes(sous_graphe, pos, node_color="lightblue", node_size=600)

        # Dessiner les arêtes du sous-graphe
        nx.draw_networkx_edges(sous_graphe, pos, edge_color="blue", width=2.5)

        # Ajouter les labels des nœuds (affichant les noms plutôt que les indices)
        nx.draw_networkx_labels(sous_graphe, pos, labels, font_size=10)

        # Annoter les distances le long des arêtes du sous-graphe
        edge_labels = {(u, v): f"{self.G[u][v]['weight']:.2f}" for u, v in sous_graphe.edges()}
        nx.draw_networkx_edge_labels(sous_graphe, pos, edge_labels=edge_labels, font_color="red")

        plt.axis("off")  # Cacher les axes
        plt.show()

    def afficher_chemins_graphiquement(self, n):
        """
        Affiche graphiquement les chemins trouvés dans le graphe, en ajustant les arêtes pour qu'elles se terminent
        à la périphérie des nœuds et non au centre. Utilise des labels avec un fond de la même couleur que les nœuds.

        :param n: Nombre total de plus courts chemins à afficher
        """
        # Vérifier que le mapping des labels est disponible
        if self.label_to_index is None or self.index_to_label is None:
            print("Le mapping des labels vers les indices n'est pas défini.")
            return

        # Obtenir les 'n' plus courts chemins en utilisant l'analyseur
        analyzer = GraphAnalyzer(self.G, self.label_to_index)
        chemins = analyzer.obtenir_n_plus_courts_chemins(n)

        index_to_label = self.index_to_label

        # Créer un sous-graphe
        sous_graphe = nx.DiGraph()
        for source, target, cost, chemin in chemins:
            sous_graphe.add_nodes_from(chemin)
            edges_in_path = [(chemin[i], chemin[i + 1]) for i in range(len(chemin) - 1)]
            sous_graphe.add_edges_from(edges_in_path)

        # Position des nœuds
        pos = nx.shell_layout(sous_graphe)
        # Vous pouvez essayer d'autres layouts en décommentant l'une des lignes ci-dessous
        # pos = nx.planar_layout(sous_graphe)
        # pos = nx.random_layout(sous_graphe)
        # pos = nx.spiral_layout(sous_graphe)
        # pos = nx.spring_layout(sous_graphe, k=2)
        plt.figure(figsize=(10, 10))

        labels = {node: index_to_label[node] for node in sous_graphe.nodes()}

        # Taille et couleur des nœuds
        node_size = 8000
        node_radius = np.sqrt(node_size) / 500  # Rayon des nœuds (approximé)
        node_color = "orange"

        # Dessiner les nœuds
        nx.draw_networkx_nodes(sous_graphe, pos, node_color=node_color, node_size=node_size, edgecolors="black")

        # Ajuster les arêtes pour qu'elles ne se terminent pas au centre des nœuds
        ax = plt.gca()
        for u, v in sous_graphe.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # Calculer l'angle de la direction de l'arête
            angle = np.arctan2(y2 - y1, x2 - x1)

            # Décaler les points de départ et d'arrivée de l'arête pour éviter que la flèche ne se termine sous les nœuds
            x1_new = x1 + node_radius * np.cos(angle)
            y1_new = y1 + node_radius * np.sin(angle)
            x2_new = x2 - node_radius * np.cos(angle)
            y2_new = y2 - node_radius * np.sin(angle)

            # Dessiner une arête courbée avec la flèche ajustée
            if (v, u) in sous_graphe.edges():
                arrow = FancyArrowPatch(
                    (x1_new, y1_new),
                    (x2_new, y2_new),
                    connectionstyle="arc3,rad=0.15",
                    color="darkblue",
                    lw=6,
                    arrowstyle="-|>",
                    mutation_scale=40,
                )
            else:
                arrow = FancyArrowPatch(
                    (x1_new, y1_new),
                    (x2_new, y2_new),
                    connectionstyle="arc3,rad=0.05",
                    color="darkblue",
                    lw=6,
                    arrowstyle="-|>",
                    mutation_scale=40,
                )
            ax.add_patch(arrow)

        # Ajouter les étiquettes avec un fond de la même couleur que les nœuds
        for node, (x, y) in pos.items():
            label = self.ajouter_retour_ligne_si_long(labels[node], 10)
            ax.annotate(
                label,
                (x, y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=node_color, edgecolor=node_color, alpha=0),
            )

        # Annoter les flux sur les arêtes
        edge_labels = {(u, v): f"{self.G[u][v]['weight']:.2f}" for u, v in sous_graphe.edges()}
        nx.draw_networkx_edge_labels(
            sous_graphe, pos, edge_labels=edge_labels, font_color="black", font_size=10, font_weight="bold"
        )

        plt.axis("off")
        plt.margins(x=0.15, y=0.15)
        plt.show()

    def _hexagonal_layout_fixed(self, node_labels, scale=1.0):
        """
        Crée un layout hexagonal compact pour les nœuds du graphe en suivant un ordre précis.

        :param node_labels: Liste des labels des nœuds à placer dans cet ordre
        :param scale: Échelle de la disposition
        :return: Dictionnaire des positions des nœuds (hexagonal)
        """
        pos = {}
        width = np.ceil(np.sqrt(len(node_labels))).astype(int)  # Calcul de la largeur nécessaire

        x_spacing = scale
        y_spacing = np.sqrt(3) / 2 * scale

        for i, label in enumerate(node_labels):
            if label in self.label_to_index:
                node_index = self.label_to_index[label]
                row = i // width
                col = i % width
                x = col * x_spacing
                y = row * y_spacing
                if row % 2 == 1:
                    x += x_spacing / 2  # Décalage pour créer une disposition hexagonale
                pos[node_index] = (x, y)
            else:
                # Le nœud avec ce label n'est pas présent, on passe
                print(f"Problème avec le label '{label}'")
                continue

        return pos

    def draw_hexagonal_graph(self, node_property, year=None, node_labels=None, scale=1.0, filename=None):
        """
        Dessine un graphe avec un layout hexagonal et une échelle logarithmique dynamique,
        en tenant compte d'un ordre fixe des nœuds spécifié par node_labels.

        :param node_property: Propriété scalaire des nœuds (dict) à représenter avec une échelle logarithmique pour la taille des nœuds
        :param year: L'année à afficher en haut à gauche de l'image
        :param node_labels: Liste des labels des nœuds pour fixer leur ordre (par défaut, tous les labels)
        :param scale: Échelle du layout hexagonal
        :param filename: Nom du fichier si vous souhaitez sauvegarder l'image
        """
        if node_labels is None:
            node_labels = list(self.label_to_index.keys())

        node_property_full = {label: node_property.get(self.label_to_index[label], 0) for label in node_labels}
        # Définir les limites log pour la propriété des nœuds
        log_min = 0  # Correspond à 10^0
        log_max = 2.2  # Correspond à 10^2.2

        # Limiter les valeurs des propriétés des nœuds dans la plage [10^log_min, 10^log_max]
        node_property_full = {
            node: min(max(value, 10**log_min), 10**log_max) for node, value in node_property_full.items()
        }
        # Normaliser les tailles des nœuds sur une échelle logarithmique
        min_size = 10  # Taille minimale d'un nœud
        max_size = 10000  # Taille maximale d'un nœud
        log_node_property = np.log10([node_property_full[node] for node in node_labels])

        # Convertir les propriétés log-scaled en tailles de nœuds, encadrées entre log_min et log_max
        node_sizes = [
            min_size + (max_size - min_size) * (value - log_min) / (log_max - log_min) for value in log_node_property
        ]

        # Utiliser le layout hexagonal
        pos = self._hexagonal_layout_fixed(node_labels, scale=scale)

        # Créer une figure avec un fond noir
        plt.figure(figsize=(14, 14), facecolor="black")
        ax = plt.gca()
        ax.set_facecolor("black")

        # Tracer les nœuds avec les tailles ajustées
        node_colors = [self.G.nodes[node]["color"] for node in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, node_color=node_colors, alpha=1)

        # Tracer les arêtes avec des couleurs plus neutres (blanches) et une largeur fine pour les liaisons
        nx.draw_networkx_edges(self.G, pos, edge_color="white", alpha=0.1, width=0.5)

        # Afficher les labels pour les nœuds les plus gros uniquement (par exemple, les nœuds au-dessus d'une taille seuil)
        size_threshold = 1500  # Seuil pour afficher les labels (les plus gros nœuds uniquement)
        labels = {
            node: f"{self.ajouter_retour_ligne_si_long(self.index_to_label[node], 10)}"
            for node, size in zip(self.G.nodes(), node_sizes)
            if size > size_threshold
        }

        # Dessiner les labels
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10, font_color="white")

        # Supprimer les axes pour une meilleure lisibilité
        plt.axis("off")

        # Ajouter l'année en haut de l'image si spécifiée
        if year is not None:
            plt.text(
                0.02,
                0.98,
                f"Année = {year}",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="left",
                color="white",
            )

        if filename is None:
            # Afficher le graphe
            plt.show()
        else:
            plt.savefig(filename, format="png", dpi=350)
            plt.close()  # Fermer la figure pour ne pas l'afficher


class GraphAnalyzer:
    def __init__(self, G):
        self.G = G
        self.label_to_index = label_to_index

    @staticmethod
    def calculate_Neff(matrix):
        """
        Calcule N = C.F à partir d'une matrice NumPy ou d'un graphe NetworkX.
        """
        # Si c'est un graphe NetworkX, on extrait la matrice d'adjacence
        if isinstance(matrix, nx.Graph):
            matrix = nx.to_numpy_array(matrix)

        # Total T : somme de tous les éléments de la matrice
        T = np.sum(matrix)

        # Somme des lignes T_{i.} et des colonnes T_{.j}
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)

        # Calcul du produit selon la formule
        product = 1.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:  # Éviter de calculer si T_{ij} est nul
                    factor = (T**2) / (row_sums[i] * col_sums[j])
                    exponent = matrix[i, j] / (2 * T)
                    product *= factor**exponent

        return product

    @staticmethod
    def calculate_Feff(matrix):
        """
        Calcule F à partir d'une matrice NumPy ou d'un graphe NetworkX.
        """
        # Si c'est un graphe NetworkX, on extrait la matrice d'adjacence
        if isinstance(matrix, nx.Graph):
            matrix = nx.to_numpy_array(matrix)

        # Total T : somme de tous les éléments de la matrice
        T = np.sum(matrix)

        # Calcul du produit selon la formule
        product = 1.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                T_ij = matrix[i, j]
                if T_ij > 0:  # Éviter les divisions par zéro ou les logs de zéro
                    factor = T_ij / T
                    exponent = -T_ij / T
                    product *= factor**exponent

        return product

    @staticmethod
    def calculate_Ceff(matrix):
        return GraphAnalyzer.calculate_Feff(matrix) / GraphAnalyzer.calculate_Neff(matrix)

    @staticmethod
    def calculate_Reff(matrix):
        return GraphAnalyzer.calculate_Neff(matrix) ** 2 / GraphAnalyzer.calculate_Feff(matrix)

    def plot_flux_distribution_semilog(self):
        # Extraire les poids des arêtes
        edge_weights = [data["weight"] for u, v, data in self.G.edges(data=True)]

        # Créer l'histogramme sans transformation logarithmique
        plt.figure(figsize=(8, 6))
        plt.hist(edge_weights, bins=100, color="skyblue", edgecolor="black", log=True)

        # Ajouter un titre et des étiquettes
        plt.title("Distribution des poids des arêtes (Flux) - Échelle linéaire", fontsize=16)
        plt.xlabel("Poids des arêtes (Flux)", fontsize=14)
        plt.ylabel("Nombre de réalisations", fontsize=14)

        # Ajouter une grille
        plt.grid(True, alpha=0.5)

        # Afficher le graphique
        plt.show()

    def plot_flux_distribution_log(self):
        # Extraire les poids des arêtes
        edge_weights = [data["weight"] for u, v, data in self.G.edges(data=True)]
        # Appliquer une transformation logarithmique aux poids des arêtes
        log_weights = np.log10(edge_weights)
        # Créer l'histogramme
        plt.figure(figsize=(8, 6))
        plt.hist(log_weights, bins=50, color="deepskyblue", edgecolor="k", log=True)
        # plt.title('Distribution des poids des arêtes (Flux) - Échelle logarithmique', fontsize=16)
        plt.xlabel("Flow size (log scale)", fontsize=14)
        plt.ylabel("Number of occurrences (Log scale)", fontsize=14)
        plt.grid(True, alpha=0.5)
        plt.show()

    def plot_flux_distribution_log_x(self):
        # Extraire les poids des arêtes
        edge_weights = [data["weight"] for u, v, data in self.G.edges(data=True)]

        # Créer l'histogramme
        plt.figure(figsize=(8, 6))
        plt.hist(edge_weights, bins=10000, color="skyblue", edgecolor="black")

        # Appliquer une échelle logarithmique sur l'axe X uniquement
        plt.xscale("log")  # Échelle log pour l'axe X
        # L'axe Y reste linéaire par défaut

        # Ajouter un titre et des étiquettes
        plt.title("Distribution des poids des arêtes (Flux) - Échelle logarithmique sur X", fontsize=16)
        plt.xlabel("Poids des arêtes (Flux) (log)", fontsize=14)
        plt.ylabel("Nombre de réalisations", fontsize=14)

        # Ajouter une grille
        plt.grid(True, alpha=0.5)

        # Afficher le graphique
        plt.show()

    def plot_degree_distribution_power_law(self):
        # Extraire les degrés des nœuds
        degrees = [degree for node, degree in self.G.degree()]
        # Créer un histogramme des degrés
        plt.figure(figsize=(8, 6))
        plt.hist(degrees, bins=np.logspace(np.log10(1), np.log10(max(degrees)), 10), color="skyblue", edgecolor="black")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Distribution des degrés (Loi de puissance)", fontsize=16)
        plt.xlabel("Degré (Échelle log)", fontsize=14)
        plt.ylabel("Fréquence (Échelle log)", fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.show()

    def plot_weight_probability_distribution(self, N_bins=150):
        # Extraire les poids des arêtes
        edge_weights = [data["weight"] for u, v, data in self.G.edges(data=True)]
        # Créer un histogramme des poids avec normalisation
        plt.figure(figsize=(8, 6))
        counts, bins, patches = plt.hist(
            edge_weights,
            bins=np.logspace(np.log10(min(edge_weights)), np.log10(max(edge_weights)), N_bins),
            density=True,
            cumulative=False,
            color="deepskyblue",
            edgecolor="black",
        )
        plt.xscale("log")  # Échelle logarithmique sur l'axe des poids
        plt.yscale("log")  # Optionnel
        plt.xlabel("Edge Weight (Log Scale)", fontsize=14)
        plt.ylabel("Probability Density (Log Scale)", fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.show()

    def obtenir_n_plus_courts_chemins(self, n):
        """
        Obtenir les 'n' plus courts chemins dans tout le graphe en appliquant l'algorithme dijkstra_exclusion,
        c'est-à-dire en excluant certains nœuds comme nœuds intermédiaires.

        :param G: Graphe NetworkX avec un attribut 'weight' pour les arêtes
        :param n: Nombre total de chemins à retourner
        :return: Liste des 'n' plus courts chemins [(source, cible, coût, chemin)]
        """
        # Liste pour stocker tous les chemins trouvés
        tous_les_chemins = []
        all_nodes = list(self.G.nodes())  # Liste de tous les nœuds du graphe
        ext_nodes = [self.label_to_index[label] for label in ext]
        exclude_nodes_set = set(ext_nodes)  # Ensemble des nœuds à exclure comme nœuds intermédiaires

        # Parcourir chaque nœud comme nœud de départ
        for source in all_nodes:
            # Calculer les plus courts chemins depuis 'source' en appliquant dijkstra_exclusion
            distances, paths = self.dijkstra_exclusion(source, weight="weight", exclude_nodes=exclude_nodes_set)

            # Ajouter tous les chemins trouvés dans la liste
            for target, cost in distances.items():
                if source != target:
                    chemin = paths[target]  # Chemin de 'source' vers 'target'
                    tous_les_chemins.append((source, target, cost, chemin))

        # Trier tous les chemins trouvés par le coût (ordre croissant)
        tous_les_chemins_triees = sorted(tous_les_chemins, key=lambda x: x[2])

        # Retourner les 'n' plus courts chemins (ou moins si moins de chemins sont trouvés)
        return tous_les_chemins_triees[:n]

    def calculer_longueur_moyenne_chemins(self):
        longueurs_chemins = []  # Liste pour stocker les longueurs des plus courts chemins
        # Calculer les plus courts chemins depuis chaque nœud source
        for source, longueurs_cibles in nx.all_pairs_dijkstra_path_length(self.G, weight="weight"):
            for cible, longueur in longueurs_cibles.items():
                if source != cible:
                    longueurs_chemins.append(longueur)
        # Vérifier s'il y a des chemins pour éviter la division par zéro
        if len(longueurs_chemins) == 0:
            return float("inf")  # Ou retourner une valeur appropriée si aucun chemin n'existe
        longueur_moyenne = sum(longueurs_chemins) / len(longueurs_chemins)
        return longueur_moyenne

    def calculer_et_tracer_distribution_distances(
        self,
        nb_bins=50,
        titre="Distribution des distances entre les nœuds",
        sauvegarder=False,
        nom_fichier="histogramme_distances.png",
    ):
        """
        Calcule les plus courts chemins entre toutes les paires de nœuds dans un graphe pondéré dirigé,
        en utilisant la distance définie comme la somme des inverses des poids,
        et trace un histogramme de la distribution des distances avec une échelle logarithmique sur l'axe des abscisses.

        :param nb_bins: Nombre de barres dans l'histogramme
        :param titre: Titre du graphique
        :param sauvegarder: Booléen pour indiquer si le graphique doit être sauvegardé
        :param nom_fichier: Nom du fichier pour la sauvegarde du graphique
        """
        # Vérifier que le graphe a des poids sur les arêtes
        if not nx.get_edge_attributes(self.G, "weight"):
            raise ValueError("Le graphe n'a pas d'attribut 'weight' sur les arêtes.")

        # Définir la fonction de poids inversé
        def inverse_weight(u, v, d):
            poids = d.get("weight", 1)
            if poids > 0:
                return 1 / poids
            else:
                return float("inf")  # Gérer les poids nuls ou négatifs

        # Calculer les plus courts chemins en utilisant la fonction de poids inversé
        # Cela renvoie un dictionnaire de dictionnaires : lengths[source][target] = distance
        longueurs_dict = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=inverse_weight))

        # Collecter les distances dans une liste
        longueurs_chemins = []
        for source in longueurs_dict:
            for cible, longueur in longueurs_dict[source].items():
                if source != cible:
                    longueurs_chemins.append(longueur)

        # Filtrer les distances positives pour l'échelle logarithmique
        longueurs_positives = [l for l in longueurs_chemins if l > 0]

        # Tracer l'histogramme
        plt.figure(figsize=(10, 6))
        sns.histplot(longueurs_positives, bins=nb_bins, log_scale=(True, False))
        plt.xlabel("Longueur du plus court chemin (échelle logarithmique)")
        plt.ylabel("Nombre de paires de nœuds")
        plt.title(titre)
        if sauvegarder:
            plt.savefig(nom_fichier)
        plt.show()

        # Calculer la longueur moyenne des chemins
        if longueurs_chemins:
            longueur_moyenne = sum(longueurs_chemins) / len(longueurs_chemins)
        else:
            longueur_moyenne = np.nan
        return longueur_moyenne

    def calculer_et_tracer_distribution_distances_V2(
        self,
        nb_bins=50,
        titre="Distribution des distances entre les nœuds",
        sauvegarder=False,
        nom_fichier="histogramme_distances.png",
    ):
        """
        Calcule les plus courts chemins en excluant les chemins passant par des nœuds intermédiaires spécifiques,
        puis trace l'histogramme des longueurs des plus courts chemins.

        :param nb_bins: Nombre de barres dans l'histogramme
        :param titre: Titre du graphique
        :param sauvegarder: Booléen indiquant si le graphique doit être sauvegardé
        :param nom_fichier: Nom du fichier pour la sauvegarde du graphique
        :return: Longueur moyenne des chemins
        """

        longueurs_chemins = []
        all_nodes = list(self.G.nodes())

        # Exclure les nœuds en tant que nœuds intermédiaires, mais les autoriser en tant que source ou cible
        ext_nodes = [self.label_to_index[label] for label in ext]
        exclude_nodes_set = set(ext_nodes)

        for source in all_nodes:
            # Exclure le nœud source des nœuds à exclure
            exclude_nodes = exclude_nodes_set - {source}
            distances, paths = self.dijkstra_exclusion(source, weight="weight", exclude_nodes=exclude_nodes)
            for target, length in distances.items():
                if source != target:
                    longueurs_chemins.append(length)

        # Filtrer les longueurs positives pour l'échelle logarithmique
        longueurs_positives = [l for l in longueurs_chemins if l > 0]

        # Tracer l'histogramme
        plt.figure(figsize=(10, 6))
        sns.histplot(longueurs_positives, bins=nb_bins, log_scale=(True, False))
        plt.xlabel("Longueur du plus court chemin (échelle logarithmique)")
        plt.ylabel("Nombre de paires de nœuds")
        plt.title(titre)

        if sauvegarder:
            plt.savefig(nom_fichier)

        plt.show()

        # Calculer et retourner la longueur moyenne des chemins
        if len(longueurs_positives) == 0:
            return float("inf")
        longueur_moyenne = sum(longueurs_positives) / len(longueurs_positives)
        return longueur_moyenne

    def dijkstra_exclusion(self, source, weight="weight", exclude_nodes=set()):
        """
        Algorithme de Dijkstra personnalisé pour calculer les plus courts chemins dans un graphe,
        en excluant les nœuds intermédiaires appartenant à 'exclude_nodes', tout en permettant que
        les nœuds exclus soient utilisés comme nœud de départ ou nœud final.

        :param source: Le nœud de départ à partir duquel les chemins doivent être calculés
        :param weight: Nom de l'attribut des arêtes représentant les poids (par défaut 'weight')
        :param exclude_nodes: Ensemble des nœuds à exclure en tant que nœuds intermédiaires
        :return: distances, paths
                 distances: dictionnaire {nœud: coût du plus court chemin depuis le nœud de départ}
                 paths: dictionnaire {nœud: liste représentant le chemin le plus court depuis le nœud de départ}
        """

        distances = {}
        paths = {}
        queue = [(0, source, [source])]

        while queue:
            cost, u, path = heapq.heappop(queue)

            # Si nous avons déjà trouvé un chemin vers 'u', nous passons
            if u in distances:
                continue

            distances[u] = cost
            paths[u] = path

            for v, data in self.G[u].items():
                # Éviter les cycles
                if v in path:
                    continue

                edge_weight = data.get(weight, 1)
                if edge_weight == 0:
                    continue  # Évite la division par zéro

                new_cost = cost + 1 / edge_weight
                new_path = path + [v]

                # Vérifier si le nouveau chemin contient des nœuds exclus en tant que nœuds intermédiaires
                # Exclure le premier et le dernier nœud du chemin
                intermediates = new_path[1:-1]
                has_excluded_intermediate = any(node in exclude_nodes for node in intermediates)

                if has_excluded_intermediate:
                    continue  # Ne pas ajouter 'v' si le chemin passe par un nœud exclu intermédiaire

                # Enregistrer le chemin si c'est un meilleur coût ou s'il n'a pas encore été visité
                if v not in distances or new_cost < distances[v]:
                    heapq.heappush(queue, (new_cost, v, new_path))
        return distances, paths

    def in_centrality(self):
        """
        Calcule la centralité d'entrée pondérée (somme des poids entrants) pour chaque nœud du graphe.
        :return: Dictionnaire {nœud: centralité d'entrée pondérée}
        """
        in_centrality_dict = {index_to_label[node]: 0 for node in self.G.nodes()}
        for node in self.G.nodes():
            # Pour chaque nœud, on somme les poids des arêtes entrantes
            for predecessor, _, data in self.G.in_edges(node, data=True):
                weight = data.get("weight", 0)  # On récupère le poids de l'arête (défaut : 0)
                in_centrality_dict[index_to_label[node]] += weight
        return in_centrality_dict

    def out_centrality(self):
        """
        Calcule la centralité de sortie pondérée (somme des poids sortants) pour chaque nœud du graphe.
        :return: Dictionnaire {nœud: centralité de sortie pondérée}
        """
        out_centrality_dict = {index_to_label[node]: 0 for node in self.G.nodes()}
        for node in self.G.nodes():
            # Pour chaque nœud, on somme les poids des arêtes sortantes
            for _, successor, data in self.G.out_edges(node, data=True):
                weight = data.get("weight", 0)  # On récupère le poids de l'arête (défaut : 0)
                out_centrality_dict[index_to_label[node]] += weight
        return out_centrality_dict

    def weighted_assortativity(self):
        # Calculer les sommes des poids des arêtes pour les nœuds
        degree_dict = dict(self.G.degree(weight="weight"))

        # Extraire les arêtes et leurs poids
        edges = list(self.G.edges(data=True))
        weight_sum = 0
        degree_product_sum = 0
        degree_diff_sum = 0

        for u, v, data in edges:
            weight = data["weight"]
            deg_u = degree_dict[u]
            deg_v = degree_dict[v]

            weight_sum += weight
            degree_product_sum += deg_u * deg_v
            degree_diff_sum += (deg_u - deg_v) ** 2

        # Calcul de l'assortativité pondérée (une approximation)
        return weight_sum / degree_product_sum if degree_product_sum != 0 else 0

    # Implémentez les autres méthodes d'analyse selon vos besoins
