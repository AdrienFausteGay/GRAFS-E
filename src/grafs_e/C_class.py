import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings

from grafs_e.N_class import DataLoader, NitrogenFlowModel, FluxGenerator


# --- Classe Dataloader ---
class Dataloader_Carbon:
    def __init__(self, project_path, data_path, region, year, prospective=False):
        """Charger les données nécessaires pour le modèle"""
        # Instancier NitrogenFluxModel avec l'année et la région
        self.data = DataLoader(project_path, data_path)
        self.df_data = self.data.df_data
        self.year = year
        self.region = region
        self.prospective = prospective
        self.nitrogen_model = NitrogenFlowModel(
            self.data, region, year, prospective=prospective
        )  # Modèle de flux d'azote

        fixed_compartments = [
            "Haber-Bosch",
            "atmospheric CO2",
            "atmospheric CH4",
            "hydrocarbures",
            "fishery products",
            "other sectors",
            "waste",
            "soil stock",
            "seeds",
        ]

        trade = [i + " trade" for i in set(self.data.init_df_prod["Sub Type"])]

        mechanization = [
            i + " machines"
            for i in list(self.data.init_df_elevage.index)
            + list(self.data.init_df_cultures.index)
        ]

        suffixes = [" manure", " slurry", " grasslands excretion"]
        # Une seule list comprehension pour tout faire
        excretion_compartments = [
            i + suffix for i in self.data.init_df_elevage.index for suffix in suffixes
        ]

        self.labels = (
            list(self.data.init_df_cultures.index)
            + list(self.data.init_df_elevage.index)
            + excretion_compartments
            + mechanization
            + list(self.data.init_df_prod.index)
            + list(self.data.init_df_pop.index)
            + list(self.data.init_df_energy.index)
            + fixed_compartments
            + trade
        )

        self.n = len(self.labels)  # Nombre total de secteurs
        self.adjacency_matrix_N = (
            self.nitrogen_model.get_transition_matrix()
        )  # Récupérer la matrice des flux depuis N_class

        self.label_to_index = {
            self.label: index for index, self.label in enumerate(self.labels)
        }
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

    # Utilitaire : conversion robuste de "value" (virgules → points, etc.)
    @staticmethod
    def _to_num(x):
        return pd.to_numeric(
            pd.Series(x, dtype="object").astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).astype(float)

    def get_df_cultures(self):
        supp_columns = [
            "Carbon Mechanisation Intensity (ktC/ha)",
            "Residue Humification Coefficient (%)",
            "Root Humification Coefficient (%)",
        ]

        self.df_cultures = self.data.init_df_cultures.copy()

        self.df_cultures = self.data.get_columns(
            self.region, self.year, self.df_cultures, supp_columns
        )

        self.df_cultures = self.nitrogen_model.df_cultures.combine_first(
            self.df_cultures,
        )

        return self.df_cultures

    def get_df_prod(self):
        supp_columns = [
            "Carbon Content (%)",
        ]

        self.df_prod = self.data.init_df_prod.copy()

        self.df_prod = self.data.get_columns(
            self.region, self.year, self.df_prod, supp_columns
        )

        self.df_prod = self.nitrogen_model.df_prod.combine_first(
            self.df_prod,
        )

        return self.df_prod

    def get_df_elevage(self):
        supp_columns = [
            "C-CH4 enteric/LU (kgC)",
            "Infrastructure CO2 emissions/LU (kgC)",
        ]

        self.df_elevage = self.data.init_df_elevage.copy()

        self.df_elevage = self.data.get_columns(
            self.region, self.year, self.df_elevage, supp_columns
        )

        self.df_elevage = self.nitrogen_model.df_elevage.combine_first(
            self.df_elevage,
        )

        return self.df_elevage

    def get_df_excr(self):
        supp_columns = [
            "C/N",
            "CH4 EM (%)",
            "Humification coefficient (%)",
        ]

        self.df_excr = self.data.init_df_excr.copy()

        self.df_excr_supp = self.data.get_columns(
            self.region, self.year, self.df_excr, supp_columns
        )

        self.df_excr = self.nitrogen_model.df_excr.combine_first(
            self.df_excr,
        )

        return self.df_excr

    def get_df_pop(self):
        supp_columns = [
            "C/N",
            "CH4 EM (%)",
            "Humification coefficient (%)",
        ]

        self.df_pop = self.data.init_df_pop.copy()

        self.df_pop = self.data.get_columns(
            self.region, self.year, self.df_pop, supp_columns
        )

        self.df_pop = self.nitrogen_model.df_pop.combine_first(
            self.df_pop,
        )

        return self.df_pop

    def get_df_energy(self):
        supp_columns = [
            "Share CO2 (%)",
        ]

        self.df_energy = self.data.init_df_energy.copy()

        self.df_energy = self.data.get_columns(
            self.region, self.year, self.df_energy, supp_columns
        )

        self.df_energy = self.nitrogen_model.df_energy.combine_first(
            self.df_energy,
        )

        return self.df_energy


# --- Classe FluxModif ---
class CarbonFlowModel:
    def __init__(self, data):
        self.year = data.year
        self.region = data.region
        self.data_loader = data
        self.labels = data.labels

        self.df_cultures = self.data_loader.get_df_cultures()
        self.df_elevage = self.data_loader.get_df_elevage()
        self.df_prod = self.data_loader.get_df_prod()
        self.df_excr = self.data_loader.get_df_excr()
        self.df_pop = self.data_loader.get_df_pop()
        self.data_loader.data.get_global_metrics(
            self.region, self.year, carbon=True, prospect=False
        )
        self.df_global = self.data_loader.data.global_df
        self.df_energy = self.data_loader.get_df_energy()
        self.df_energy_display = self.data_loader.nitrogen_model.df_energy_display

        self.flux_generator = FluxGenerator(self.labels)
        self.carbon_matrix = self.flux_generator.adjacency_matrix

        self.compute_fluxes()

    def change_flow(self, source, target, factor):
        self.carbon_matrix[
            self.data_loader.label_to_index[source],
            self.data_loader.label_to_index[target],
        ] = (
            self.data_loader.nitrogen_model.flux_generator.get_coef(source, target)
            * factor
        )

    def plot_heatmap_interactive(self):
        """
        Generates an interactive heatmap using Plotly to visualize the nitrogen flux transition matrix.

        The heatmap has the following features:
        - Logarithmic scale (simulated via log10(z)) to handle wide-ranging values.
        - A horizontal colorbar placed at the bottom of the plot.
        - A legend that maps matrix indices to sector labels, positioned on the right, ensuring no overlap.
        - The X-axis is displayed at the top of the plot, and the title is centered above the plot.

        This visualization helps to understand the relative magnitudes of the nitrogen fluxes between sectors
        in a clear and interactive manner.

        Returns:
            plotly.graph_objects.Figure: An interactive Plotly figure containing the heatmap.
        """

        # 1) Préparation des labels numériques
        x_labels = list(range(1, len(self.labels) + 1))
        y_labels = list(range(1, len(self.labels) + 1))

        # Si vous ignorez la dernière ligne/colonne comme dans votre code :
        # adjacency_subset = self.adjacency_matrix[: len(self.labels), : len(self.labels)]

        adj = np.array(self.carbon_matrix)  # ou .copy()
        adjacency_subset = adj[: len(self.labels), : len(self.labels)].copy()

        # 2) Gestion min/max et transformation log10
        cmin = max(1e-2, np.min(adjacency_subset[adjacency_subset > 0]))
        cmax = 1e4  # np.max(adjacency_subset)
        log_matrix = np.where(adjacency_subset > 0, np.log10(adjacency_subset), np.nan)

        # 3) Construire un tableau 2D de chaînes pour le survol
        #    Même dimension que log_matrix
        strings_matrix = []
        for row_i, y_val in enumerate(y_labels):
            row_texts = []
            for col_i, x_val in enumerate(x_labels):
                # Valeur réelle (non log) => adjacency_subset[row_i, col_i]
                real_val = adjacency_subset[row_i, col_i]
                if np.isnan(real_val):
                    real_val_str = "0"
                else:
                    real_val_str = f"{real_val:.2e}"  # format décimal / exposant
                # Construire la chaîne pour la tooltip
                # y_val et x_val sont les indices 1..N
                # self.labels[y_val] = nom de la source, self.labels[x_val] = nom de la cible
                tooltip_str = f"Source : {self.labels[y_val - 1]}<br>Target : {self.labels[x_val - 1]}<br>Value  : {real_val_str} ktC/yr"
                row_texts.append(tooltip_str)
            strings_matrix.append(row_texts)

        # 3) Tracé Heatmap avec go.Figure + go.Heatmap
        #    On règle "zmin" et "zmax" en valeurs log10
        #    pour contrôler la gamme de couleurs
        trace = go.Heatmap(
            z=log_matrix,
            x=x_labels,
            y=y_labels,
            colorscale="Plasma_r",
            zmin=np.log10(cmin),
            zmax=np.log10(cmax),
            text=strings_matrix,  # tableau 2D de chaînes
            hoverinfo="text",  # on n'affiche plus x, y, z bruts
            # Colorbar horizontale
            colorbar=dict(
                title="ktC/year",
                orientation="h",
                x=0.5,  # centré horizontalement
                xanchor="center",
                y=-0.15,  # en dessous de la figure
                thickness=15,  # épaisseur
                len=1,  # longueur en fraction de la largeur
            ),
            # Valeurs de survol -> vous verrez log10(...) par défaut
            # Pour afficher la valeur réelle, on peut plus tard utiliser "customdata"
        )

        # Créer la figure et y ajouter le trace
        fig = go.Figure(data=[trace])

        # 4) Discrétisation manuelle des ticks sur la colorbar
        #    On veut afficher l'échelle réelle (et pas log10)
        #    => calcul de tickvals en log10, et ticktext en 10^(tickvals)
        tickvals = np.linspace(np.floor(np.log10(cmin)), np.ceil(np.log10(cmax)), num=7)
        ticktext = [
            10**x for x in range(int(np.log(cmin)), int(np.log(cmax)), 1)
        ]  # [f"{10**v:.2e}" for v in tickvals]
        # Mettre à jour le trace pour forcer l'affichage
        fig.data[0].update(
            colorbar=dict(
                title="ktC/year",
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.15,
                thickness=25,
                len=1,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
            )
        )

        # 5) Configuration de la mise en page
        fig.update_layout(
            width=1000,
            height=1000,
            margin=dict(t=0, b=0, l=0, r=220),  # espace à droite pour la légende
        )
        fig.update_layout(yaxis_scaleanchor="x")

        # Axe X en haut
        fig.update_xaxes(
            title="Target",
            side="top",  # place les ticks en haut
            tickangle=90,  # rotation
            tickmode="array",
            tickfont=dict(size=10),
            tickvals=x_labels,  # forcer l'affichage 1..N
            ticktext=[str(x) for x in x_labels],
        )

        # Axe Y : inverser l'ordre pour un style "matriciel" standard
        fig.update_yaxes(
            title="Source",
            autorange="reversed",
            tickmode="array",
            tickfont=dict(size=10),
            tickvals=y_labels,
            ticktext=[str(y) for y in y_labels],
        )

        # 6) Ajouter la légende à droite
        #    Format : "1: label[0]" ... vertical
        legend_text = "<br>".join(
            f"{i + 1} : {lbl}" for i, lbl in enumerate(self.labels)
        )
        fig.add_annotation(
            x=1.25,  # un peu à droite
            y=0.45,  # centré en hauteur
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

    def compute_fluxes(self):
        df_elevage = self.df_elevage
        df_excr = self.df_excr
        df_cultures = self.df_cultures
        df_pop = self.df_pop
        df_global = self.df_global
        df_prod = self.df_prod
        df_energy = self.df_energy
        df_energy_display = self.df_energy_display

        flux_generator = self.flux_generator

        ##flux des produits vers le reste
        df_prod["C/N"] = np.where(
            df_prod["Nitrogen Content (%)"] != 0,
            df_prod["Carbon Content (%)"] / df_prod["Nitrogen Content (%)"],
            0,
        )

        df_prod["Carbon Production (ktC)"] = (
            df_prod["Production (kton)"] * df_prod["Carbon Content (%)"] / 100
        )

        # Flux des cultures/livestocks vers les produits
        for i, row in df_prod.iterrows():
            source = {row["Origin compartment"]: 1}
            target = {i: row["Carbon Production (ktC)"]}
            flux_generator.generate_flux(source, target)

        # Flux des produits vers waste
        df_prod["Carbon Wasted (ktC)"] = (
            df_prod["Nitrogen Wasted (ktN)"] * df_prod["C/N"]
        )
        source = df_prod["Carbon Wasted (ktC)"].to_dict()
        target = {"waste": 1}
        flux_generator.generate_flux(source, target)

        # Flux des produits vers other sectors
        df_prod["Carbon for Other uses (ktC)"] = (
            df_prod["Nitrogen for Other uses (ktN)"] * df_prod["C/N"]
        )
        source = df_prod["Carbon for Other uses (ktC)"].to_dict()
        target = {"other sectors": 1}
        flux_generator.generate_flux(source, target)

        # Flux des produits exportés
        df_prod["Carbon Exported (ktC)"] = (
            df_prod["Nitrogen Exported (ktN)"] * df_prod["C/N"]
        )
        exports = df_prod.loc[
            df_prod["Carbon Exported (ktC)"] > 1e-9,
            ["Sub Type", "Carbon Exported (ktC)"],
        ]
        for sub_type, subdf in exports.groupby("Sub Type"):
            source = subdf["Carbon Exported (ktC)"].to_dict()
            target = {f"{sub_type} trade": 1}
            flux_generator.generate_flux(source, target)

        ## Générer les flux du modèle d'allocation

        # C/N des sources
        CN_waste = float(self.df_global.loc["Green waste C/N", "value"])

        def _cn_of_source(src: str) -> float:
            if src in self.df_prod.index:
                return float(self.df_prod.loc[src, "C/N"])
            if src in self.df_excr.index:
                return float(self.df_excr.loc[src, "C/N"])
            if src.lower().strip() == "waste":
                return CN_waste
            # Les flux “trade → facility” sont traités plus bas via importations_df, donc pas ici.
            raise KeyError(f"C/N inconnu pour la source énergie: {src}")

        alloc_df = self.data_loader.nitrogen_model.allocations_df

        for index, row in alloc_df.iterrows():
            p = row["Product"]
            cons = row["Consumer"]
            v = row["Allocated Nitrogen"]
            typ = str(row["Type"])

            # Récupérer le C/N du produit
            CN = _cn_of_source(p)

            # Cas import : on reroute depuis la "boîte trade" de la sous-catégorie du produit
            if typ.startswith("Imported"):
                # Ici on suppose que seuls des PRODUITS sont importés (pas d'excréta/waste)
                sub = str(self.df_prod.loc[p, "Sub Type"])
                source_label = f"{sub} trade"
            else:
                # Flux domestique (produit/excréta/waste) depuis la source elle-même
                source_label = p

            source = {source_label: v * CN}
            target = {cons: 1}
            flux_generator.generate_flux(source, target)

        ## Flux des graines

        df_cultures = df_cultures.join(df_prod["C/N"], on="Main Production", how="left")

        # 2. Créer la colonne "Seeds Input (ktC)"
        df_cultures["Seeds Input (ktC)"] = (
            df_cultures["C/N"] * df_cultures["Seeds Input (ktN)"]
        )

        target = df_cultures["Seeds Input (ktC)"].to_dict()
        source = {"seeds": 1}
        flux_generator.generate_flux(source, target)

        ## Flux de sortie des infrastructures énergétiques

        # Hypothèses physico-énergétiques pour CH4 (constants)
        NCV_kWh_per_m3 = 10.0  # kWh/m3 (pouvoir calorifique inf. du CH4)
        rho_CH4_kg_m3 = 0.717  # kg/m3
        Cfrac_CH4 = 12.0 / 16.0  # fraction massique de C dans CH4
        # ktC par kWh de CH4
        KT_C_PER_KWH_CH4 = (1.0 / NCV_kWh_per_m3) * rho_CH4_kg_m3 * Cfrac_CH4 / 1e6

        # somme des intrants C vers chaque facility
        C_in_by_fac = {fac: 0.0 for fac in df_energy.index}

        for _, r in alloc_df.iterrows():
            cons = str(r["Consumer"])
            if cons not in C_in_by_fac:
                continue  # pas une infrastructure énergie
            src = str(r["Product"])
            N_kt = float(r["Allocated Nitrogen"])
            if N_kt <= 0:
                continue
            C_in_by_fac[cons] += N_kt * _cn_of_source(src)

        # 2) Routage des sorties par facility
        for fac in df_energy.index:
            fac_type = str(df_energy.loc[fac, "Type"])
            share_co2 = float(df_energy.loc[fac, "Share CO2 (%)"]) / 100.0
            E_kWh = float(df_energy.loc[fac, "Energy Production (GWh)"])
            C_in = float(C_in_by_fac.get(fac, 0.0))  # ktC

            # CO2 direct (fraction des intrants)
            C_to_CO2 = max(0.0, share_co2 * C_in)

            if fac_type == "Methanizer":
                # Carbone vers hydrocarbures = C_CH4 dérivé de l'énergie
                C_to_HC = max(0.0, E_kWh * KT_C_PER_KWH_CH4)

                # Digestat = le reste (contrôle de cohérence)
                C_digest = C_in - C_to_CO2 - C_to_HC
                if C_digest < -1e-6:
                    warnings.warn(
                        f"[{fac}] Negative digestate carbon ({C_digest:.3f} ktC). "
                        f"Energy demand too high w.r.t. inputs or wrong 'Share CO2 (%)'. "
                        f"Clamped to 0."
                    )
                C_digest = max(0.0, C_digest)

                # Flows
                if C_to_HC > 0:
                    self.flux_generator.generate_flux(
                        {fac: C_to_HC}, {"hydrocarbures": 1}
                    )
                if C_to_CO2 > 0:
                    self.flux_generator.generate_flux(
                        {fac: C_to_CO2}, {"atmospheric CO2": 1}
                    )
                if C_digest > 0:
                    self.flux_generator.generate_flux(
                        {fac: C_digest}, {"soil stock": 1}
                    )

            elif fac_type == "Bioraffinery":
                # Pas de digestat: le reste des intrants (hors CO2) = hydrocarbures
                C_to_HC = max(0.0, C_in - C_to_CO2)

                if C_to_HC > 0:
                    self.flux_generator.generate_flux(
                        {fac: C_to_HC}, {"hydrocarbures": 1}
                    )
                if C_to_CO2 > 0:
                    self.flux_generator.generate_flux(
                        {fac: C_to_CO2}, {"atmospheric CO2": 1}
                    )

            else:
                raise ValueError(
                    f"Unknown energy facility Type for '{fac}': {fac_type}"
                )

        ## Flux des excretions aux cultures (soil stock)

        # Humification

        df_excr["Humification (ktC)"] = (
            df_excr["Excretion to soil (ktN)"] * df_excr["C/N"]
        )
        df_excr["Excretion (ktC)"] = df_excr["Humification (ktC)"] / (
            df_excr["Humification coefficient (%)"] / 100
        )
        source = df_excr["Humification (ktC)"].to_dict()
        target = {"soil stock": 1}

        flux_generator.generate_flux(source, target)

        # CH4

        df_excr["Excretion to CH4 (ktC)"] = (
            df_excr["Excretion (ktC)"] * df_excr["CH4 EM (%)"] / 100
        )

        source = df_excr["Excretion to CH4 (ktC)"].to_dict()
        target = {"atmospheric CH4": 1}
        flux_generator.generate_flux(source, target)

        # CO2

        df_excr["Excretion to Energy (ktC)"] = (
            df_excr["Excretion to Energy (ktN)"] * df_excr["C/N"]
        )

        df_excr["Excretion to CO2 (ktC)"] = (
            df_excr["Excretion (ktC)"]
            * (
                1
                - df_excr["CH4 EM (%)"] / 100
                - df_excr["Humification coefficient (%)"] / 100
                - df_excr["Excretion to Energy (ktC)"] / df_excr["Excretion (ktC)"]
            )  # On enlève la part envoyer aux méthaniseurs
        )

        source = df_excr["Excretion to CO2 (ktC)"].to_dict()
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(source, target)

        ## Flux des animaux aux excretions
        for i in df_elevage.index:
            source = {i: 1}
            target = df_excr.loc[
                df_excr["Origin compartment"] == i, "Excretion (ktC)"
            ].to_dict()
            flux_generator.generate_flux(source, target)

        ##flux excretion humaine (même chose sauf que les excretions humaines sont gérées dans df_pop)
        # Humification
        df_pop["Humification (ktC)"] = (
            df_pop["Excretion after volatilization (ktN)"] * df_pop["C/N"]
        )
        df_pop["Excretion (ktC)"] = df_pop["Humification (ktC)"] / (
            df_pop["Humification coefficient (%)"] / 100
        )
        source = df_pop["Humification (ktC)"].to_dict()
        target = {"soil stock": 1}

        flux_generator.generate_flux(source, target)

        # CH4
        df_pop["Excretion to CH4 (ktC)"] = (
            df_pop["Excretion (ktC)"] * df_pop["CH4 EM (%)"] / 100
        )

        source = df_pop["Excretion to CH4 (ktC)"].to_dict()
        target = {"atmospheric CH4": 1}
        flux_generator.generate_flux(source, target)

        # CO2
        df_pop["Excretion to CO2 (ktC)"] = (
            df_pop["Excretion (ktC)"]
            * (100 - df_pop["CH4 EM (%)"] - df_pop["Humification coefficient (%)"])
            / 100
        )

        source = df_pop["Excretion to CO2 (ktC)"].to_dict()
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(source, target)

        # Flux de carbone liés à Haber Bosch
        # Prendre les données nécessaires dans df_global

        HB_CH4_intensity = df_global.loc[
            "Total Haber-Bosch methan input (kgC/kgN)"
        ].item()
        HB_prod = self.data_loader.nitrogen_model.flux_generator.get_coef(
            "atmospheric N2", "Haber-Bosch"
        )

        source = {"hydrocarbures": 1}
        target = {"Haber-Bosch": HB_prod * HB_CH4_intensity}
        flux_generator.generate_flux(source, target)

        source = {"Haber-Bosch": 1}
        target = {"atmospheric CO2": HB_prod * HB_CH4_intensity}
        flux_generator.generate_flux(source, target)

        # Flux de carbone liés à la respiration et à la fermentation entérique

        # CH4 Entérique
        df_elevage["CH4 enteric (ktC)"] = (
            df_elevage["LU"] * df_elevage["C-CH4 enteric/LU (kgC)"] / 1e6
        )
        source = df_elevage["CH4 enteric (ktC)"].to_dict()
        target = {"atmospheric CH4": 1}
        flux_generator.generate_flux(source, target)

        # Respiration CO2
        df_elevage["Ingestion (ktC)"] = df_elevage.index.map(
            lambda i: self.carbon_matrix[:, self.data_loader.label_to_index[i]].sum()
        )

        excretion_sum_aligned = (
            df_excr.groupby("Origin compartment")["Excretion (ktC)"]
            .sum()
            .reindex(df_elevage.index, fill_value=0)
        )

        prod_sum_aligned = (
            df_prod.groupby("Origin compartment")["Carbon Production (ktC)"]
            .sum()
            .reindex(df_elevage.index, fill_value=0)
        )

        df_elevage["Respiration (ktC)"] = (
            df_elevage["Ingestion (ktC)"]
            - df_elevage["CH4 enteric (ktC)"]
            - excretion_sum_aligned
            - prod_sum_aligned
        ).clip(lower=0)
        source = df_elevage["Respiration (ktC)"].to_dict()
        target = {"atmospheric CO2": 1}

        flux_generator.generate_flux(source, target)

        def _calculate_cn_critical(df, index_key):
            """Calcule C/N critique et gère les erreurs."""
            try:
                row = df.loc[df.index == index_key]
                if row.empty:
                    return np.nan

                # Extrait les valeurs scalaires
                N_emissions_percent = (
                    row[["N-NH3 EM (%)", "N-N2 EM (%)", "N-N2O EM (%)"]].iloc[0].sum()
                )
                Humif_coeff = row["Humification coefficient (%)"].iloc[0]
                CN_ratio = row["C/N"].iloc[0]

                if Humif_coeff == 0:
                    return np.nan

                return (
                    (100 - N_emissions_percent) / 100 / (Humif_coeff / 100) * CN_ratio
                )
            except Exception:
                return np.nan

        for index, row in df_elevage[df_elevage["Respiration (ktC)"] == 0].iterrows():
            excretion_value = excretion_sum_aligned.loc[index]

            products_df = df_prod.loc[df_prod["Origin compartment"] == index]
            production = products_df["Carbon Production (ktC)"].sum()

            somme_sorties = row["CH4 enteric (ktC)"] + excretion_value + production

            cn_mean_calc = (
                products_df["C/N"] * products_df["Carbon Production (ktC)"]
            ).sum() / production
            cn_mean_str = f"{cn_mean_calc:.4f}"

            cn_data = {
                "Manure (Fumier)": _calculate_cn_critical(df_excr, index + " manure"),
                "Slurry (Lisiers)": _calculate_cn_critical(df_excr, index + " slurry"),
                "Grasslands excretion (Pâture)": _calculate_cn_critical(
                    df_excr, index + " grasslands excretion"
                ),
            }

            cn_details = "\n".join(
                [
                    f"    {key}: {val:.4f}" if not np.isnan(val) else f"    {key}: N/A"
                    for key, val in cn_data.items()
                ]
            )

            cn_details.join(f"    C/N Production Moy. : {cn_mean_str}")

            message = (
                f"La respiration de '{index}' forcée à zéro (Calcul brut: {row['Ingestion (ktC)'] - somme_sorties:.4f} ktC). \n"
                f"  Détails des flux (ktC) : Ingestion: {row['Ingestion (ktC)']:.4f} | CH4: {row['CH4 enteric (ktC)']:.4f} | Excrétion: {excretion_value:.4f} | Produits animaux : {production:.4f} \n"
                f"  C/N feed moyen critique: \n{cn_details}"
            )
            warnings.warn(message, category=UserWarning)

        # Respiration humaine
        df_pop["Ingestion (ktC)"] = df_pop.index.map(
            lambda i: self.carbon_matrix[:, self.data_loader.label_to_index[i]].sum()
        )

        df_pop["Respiration (ktC)"] = (
            df_pop["Ingestion (ktC)"] - df_pop["Excretion (ktC)"]
        ).clip(lower=0)
        source = df_pop["Respiration (ktC)"].to_dict()
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(source, target)

        for index, row in df_pop[df_pop["Respiration (ktC)"] == 0].iterrows():
            message = (
                f"La respiration du groupe de population '{index}' a été forcée à zéro (Respiration (ktC) <= 0). "
                f"Détails des flux (ktC) : "
                f"Ingestion (Entrée): {row['Ingestion (ktC)']:.4f} | "
                f"Excrétion (Sortie): {row['Excretion (ktC)']:.4f} | "
            )
            warnings.warn(message, category=UserWarning)

        ## Flux de carbone liés aux machines / mécanisation

        # Flux liés aux machines des cultures

        df_cultures["Mecanisation Emission (ktC)"] = (
            df_cultures["Carbon Mechanisation Intensity (ktC/ha)"]
            * df_cultures["Area (ha)"]
        )
        dict_mecanisation = {
            index + " machines": value
            for index, value in df_cultures["Mecanisation Emission (ktC)"].items()
        }
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(dict_mecanisation, target)
        source = {"hydrocarbures": 1}
        flux_generator.generate_flux(source, dict_mecanisation)

        # Flux liés aux machines des élevages

        df_elevage["Mecanisation Emission (ktC)"] = (
            df_elevage["Infrastructure CO2 emissions/LU (kgC)"] * df_elevage["LU"]
        )
        dict_mecanisation = {
            index + " machines": value
            for index, value in df_elevage["Mecanisation Emission (ktC)"].items()
        }
        target = {"atmospheric CO2": 1}
        flux_generator.generate_flux(dict_mecanisation, target)
        source = {"hydrocarbures": 1}
        flux_generator.generate_flux(source, dict_mecanisation)

        ## Cultures, photosynthèse

        # La production des produits est faite plus haut.
        # Résidus :

        carbon_prod_agg = df_prod.groupby("Origin compartment")[
            "Carbon Production (ktC)"
        ].sum()

        # Jointure de la production de carbone agrégée avec df_cultures
        df_cultures = df_cultures.join(carbon_prod_agg, how="left")

        df_cultures["Main Carbon Production (ktC)"] = df_cultures[
            "Main Nitrogen Production (ktN)"
        ] * df_cultures["Main Production"].map(df_prod["C/N"])

        # Calcul des Résidus
        df_cultures["Residue Production (ktC)"] = 0.0
        mask_hi = df_cultures["Harvest Index"] != 0
        df_cultures.loc[mask_hi, "Residue Production (ktC)"] = (
            (
                df_cultures.loc[mask_hi, "Carbon Production (ktC)"]
                / df_cultures.loc[mask_hi, "Harvest Index"]
                - df_cultures["Carbon Production (ktC)"]
            )
            .clip(lower=0)
            .astype(float)
        )

        df_cultures["Root Production (ktC)"] = 0.0
        df_cultures.loc[mask_hi, "Root Production (ktC)"] = (
            df_cultures.loc[mask_hi, "Carbon Production (ktC)"]
            / df_cultures.loc[mask_hi, "Harvest Index"]
            * (df_cultures.loc[mask_hi, "BGN"] - 1).clip(lower=0.0)
        ).astype(float)

        target = (
            df_cultures["Carbon Production (ktC)"]
            + df_cultures["Residue Production (ktC)"]
            + df_cultures["Root Production (ktC)"]
            - df_cultures["Seeds Input (ktC)"]
        ).to_dict()
        source = {"atmospheric CO2": 1}

        flux_generator.generate_flux(source, target)

        source = (
            df_cultures["Residue Production (ktC)"]
            * df_cultures["Residue Humification Coefficient (%)"]
            / 100
            + df_cultures["Root Production (ktC)"]
            * df_cultures["Root Humification Coefficient (%)"]
            / 100
        ).to_dict()
        target = {"soil stock": 1}

        flux_generator.generate_flux(source, target)

        source = (
            df_cultures["Residue Production (ktC)"]
            * (100 - df_cultures["Residue Humification Coefficient (%)"])
            / 100
            + df_cultures["Root Production (ktC)"]
            * (100 - df_cultures["Root Humification Coefficient (%)"])
            / 100
        ).to_dict()
        target = {"atmospheric CO2": 1}

        flux_generator.generate_flux(source, target)

        self.df_elevage = df_elevage
        self.df_excr = df_excr
        self.df_cultures = df_cultures
        self.df_pop = df_pop
        self.df_prod = df_prod

    def get_transition_matrix(self):
        """
        Returns the full nitrogen transition matrix.

        This matrix represents all nitrogen fluxes between sectors, including core and external processes.

        :return: A 2D NumPy array representing nitrogen fluxes between all sectors.
        :rtype: numpy.ndarray
        """
        return self.carbon_matrix

    def check_balance(self):
        """
        Vérifie la balance des flux (sommes lignes et colonnes) de la matrice de transition M.
        Optimisé pour un affichage rapide en utilisant une seule instruction print finale.
        """
        M = self.get_transition_matrix()

        # Prépare les calculs en une seule fois (vectorisation)
        # Calcule la somme de chaque ligne (flux sortant)
        row_sums = M.sum(axis=1)
        # Calcule la somme de chaque colonne (flux entrant)
        col_sums = M.sum(axis=0)

        output_lines = []

        for i in range(len(M)):
            label = self.data_loader.index_to_label[i]

            # Ajout des informations à la liste
            output_lines.append(label)
            output_lines.append(f"Flux sortant (Somme Ligne): {row_sums[i]:.6f}")
            output_lines.append(f"Flux entrant (Somme Colonne): {col_sums[i]:.6f}")
            output_lines.append("===")

        # Effectue un seul appel d'impression avec toutes les lignes jointes par un saut de ligne
        print("\n".join(output_lines))
