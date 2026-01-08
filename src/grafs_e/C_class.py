import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings

from grafs_e.N_class import DataLoader, NitrogenFlowModel, FluxGenerator
from grafs_e.sankey import merge_nodes


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
        self.adjacency_matrix = self.flux_generator.adjacency_matrix

        self.compute_fluxes()

    def change_flow(self, source, target, factor):
        self.adjacency_matrix[
            self.data_loader.label_to_index[source],
            self.data_loader.label_to_index[target],
        ] = (
            self.data_loader.nitrogen_model.flux_generator.get_coef(source, target)
            * factor
        )

    def plot_heatmap_interactive(
        self, detailed_view=False, group_axes=True, legend_max_rows="auto"
    ):
        """
        Heatmap interactive :
        - detailed_view=True  → pas d’agrégation, pas de labels sur axes, tooltips OK,
                                liste complète des labels à droite (multi-colonnes).
        - detailed_view=False → agrégation par catégories (merges comme dans l’app),
                                pas de labels sur axes, tooltips OK (noms agrégés),
                                liste des labels agrégés à droite (multi-colonnes).
        """
        import math
        import numpy as np
        import plotly.graph_objects as go
        import pandas as pd

        # ---------- 0) Préparer labels + matrice ----------
        labels = list(self.labels)
        matrix = np.asarray(self.adjacency_matrix, dtype=float)

        # Helper: transformer tous les éléments de merges en labels (robuste indices/strings)
        def _coerce_merges_to_labels(merges_dict, all_labels):
            name_set = set(all_labels)
            coerced = {}
            for gname, members in merges_dict.items():
                out = []
                for m in members:
                    if isinstance(m, (int, np.integer)):
                        if 0 <= int(m) < len(all_labels):
                            out.append(all_labels[int(m)])
                    else:
                        m = str(m)
                        if m in name_set:
                            out.append(m)
                if out:
                    coerced[gname] = sorted(set(out))
            return coerced

        # ---------- Agrégation (sauf si detailed_view) ----------
        do_group = (not detailed_view) and bool(group_axes)
        if do_group:
            merges = {}

            # Crops par Category
            if (
                hasattr(self, "df_cultures")
                and isinstance(self.df_cultures, pd.DataFrame)
                and not self.df_cultures.empty
            ):
                for cat, idxs in self.df_cultures.groupby("Category").groups.items():
                    merges[str(cat)] = list(
                        idxs
                    )  # indices du DF (version qui "marchait" chez toi)

            # Livestock
            if (
                hasattr(self, "df_elevage")
                and isinstance(self.df_elevage, pd.DataFrame)
                and not self.df_elevage.empty
            ):
                merges["Livestock"] = list(self.df_elevage.index)

            # Population
            if (
                hasattr(self, "df_pop")
                and isinstance(self, pd.DataFrame) is False
                and hasattr(self, "df_pop")
                and not self.df_pop.empty
            ):
                merges["Population"] = list(self.df_pop.index)

            # Trade
            trade_labels = [lbl for lbl in labels if "trade" in str(lbl).lower()]
            if trade_labels:
                merges["Trade"] = trade_labels

            # Industry (Haber-Bosch + autres secteurs + énergie + *machines*)
            industry_candidates = [lbl for lbl in ["Haber-Bosch", "other sectors"]]

            # Énergie (si la couche expose un DF énergie)
            if (
                hasattr(self, "df_energy")
                and self.df_energy is not None
                and not self.df_energy.empty
            ):
                industry_candidates += [
                    lbl
                    for lbl in self.df_energy.index
                    if lbl in getattr(self, "labels", [])
                ]

            # 🔹 Nouveau : tout label contenant "machine" (insensible à la casse)
            machine_like = [
                lbl
                for lbl in getattr(self, "labels", [])
                if isinstance(lbl, str) and ("machine" in lbl.lower())
            ]
            industry_candidates = sorted(set(industry_candidates + machine_like))

            if industry_candidates:
                merges["Industry"] = industry_candidates

            # Environnement
            env_candidates = [
                "atmospheric NH3",
                "atmospheric N2O",
                "atmospheric N2",
                "soil stock",
                "hydro-system",
                "other losses",
                "atmospheric CO2",
                "atmospheric CH4",
            ]
            env_kept = [lbl for lbl in env_candidates if lbl in labels]
            if env_kept:
                merges["Environment"] = env_kept

            # Produits -> groupe du crop d'origine
            if (
                hasattr(self, "df_prod")
                and isinstance(self.df_prod, pd.DataFrame)
                and not self.df_prod.empty
            ):
                for prod_label, row in self.df_prod.iterrows():
                    origin = row.get("Origin compartment")
                    if isinstance(origin, str):
                        dest = None
                        for gname, members in merges.items():
                            # on compare à des labels (pas indices)
                            if origin in [str(x) for x in members]:
                                dest = gname
                                break
                        if dest is None:
                            dest = origin
                            merges.setdefault(dest, []).append(origin)
                        merges.setdefault(dest, []).append(prod_label)

            # Excrétion -> groupe du bétail d’origine
            if (
                hasattr(self, "df_excr")
                and isinstance(self.df_excr, pd.DataFrame)
                and not self.df_excr.empty
            ):
                for ex_label, row in self.df_excr.iterrows():
                    origin = row.get("Origin compartment")
                    if isinstance(origin, str):
                        dest = None
                        for gname, members in merges.items():
                            if origin in [str(x) for x in members]:
                                dest = gname
                                break
                        if dest is None and "Livestock" in merges:
                            dest = "Livestock"
                        if dest is None:
                            dest = origin
                            merges.setdefault(dest, []).append(origin)
                        merges.setdefault(dest, []).append(ex_label)

            # → coercition vers des LISTES DE LABELS (robuste) avant merge_nodes
            merges = _coerce_merges_to_labels(merges, labels)
            matrix, labels, _ = merge_nodes(matrix, labels, merges)

        mat = matrix[: len(labels), : len(labels)].copy()

        # ═══════════════════════════════════════════════════════════════
        # Suppression des lignes/colonnes vides (lignes ET colonnes nulles)
        # ═══════════════════════════════════════════════════════════════
        row_sums = np.abs(mat).sum(axis=1)  # Somme des valeurs absolues par ligne
        col_sums = np.abs(mat).sum(axis=0)  # Somme des valeurs absolues par colonne

        # Indices à conserver : ligne i ou colonne i doivent être non-nulles
        keep_indices = np.where((row_sums > 0) | (col_sums > 0))[0]

        # Filtrer matrice et labels
        mat = mat[np.ix_(keep_indices, keep_indices)]
        labels = [labels[i] for i in keep_indices]

        # Recalculer max_abs_val après filtrage (au cas où)
        max_abs_val = np.max(np.abs(mat)) if mat.size > 0 else 1.0
        if max_abs_val == 0:
            max_abs_val = 1

        # ═══════════════════════════════════════════════════════════════
        # 2) Préparation de la matrice (log10)
        # ═══════════════════════════════════════════════════════════════
        positive = mat > 0

        if not np.any(positive):
            positive[0, 0] = True
            mat[0, 0] = 1e-6

        cmin = max(1e-4, float(mat[positive].min()))
        cmax = float(mat[positive].max())
        log_matrix = np.full_like(mat, np.nan, dtype=float)
        log_matrix[positive] = np.log10(mat[positive])

        # ═══════════════════════════════════════════════════════════════
        # 3) Heatmap + tooltips (comme plot_heatmap_interactive)
        # ═══════════════════════════════════════════════════════════════
        n = len(labels)
        x_idx = list(range(1, n + 1))
        y_idx = list(range(1, n + 1))

        # customdata: [source_label, target_label, real_value]
        custom = [
            [[labels[i], labels[j], mat[i, j]] for j in range(n)] for i in range(n)
        ]

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=log_matrix,
                    x=x_idx,
                    y=y_idx,
                    colorscale="Plasma_r",
                    zmin=np.log10(cmin),
                    zmax=np.log10(cmax),
                    customdata=custom,
                    hovertemplate=(
                        "Source : %{customdata[0]}<br>"
                        "Target : %{customdata[1]}<br>"
                        "Value  : %{customdata[2]:.2e} ktN/yr<extra></extra>"
                    ),
                    colorbar=dict(
                        title="ktN/year",
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=-0.12,
                        thickness=18,
                        len=0.95,
                        tickmode="array",
                        tickvals=np.arange(
                            np.floor(np.log10(cmin)), np.ceil(np.log10(cmax)) + 1
                        ),
                        ticktext=[
                            f"{(10**v):.2e}"
                            for v in np.arange(
                                np.floor(np.log10(cmin)), np.ceil(np.log10(cmax)) + 1
                            )
                        ],
                    ),
                )
            ]
        )

        # Pas de labels sur les axes (illisibles)
        fig.update_xaxes(
            side="top", showticklabels=False, ticks="", title_text="Target"
        )
        fig.update_yaxes(
            autorange="reversed", showticklabels=False, ticks="", title_text="Source"
        )

        # ═══════════════════════════════════════════════════════════════
        # 4) Légende multi-colonnes à DROITE (même logique que référence)
        # ═══════════════════════════════════════════════════════════════
        def wrap_label(label, max_chars=20):
            """Coupe un label en plusieurs lignes si trop long."""
            if len(label) <= max_chars:
                return label
            words = label.split()
            lines, current = [], []
            for w in words:
                if sum(len(x) for x in current) + len(current) + len(w) <= max_chars:
                    current.append(w)
                else:
                    lines.append(" ".join(current))
                    current = [w]
            if current:
                lines.append(" ".join(current))
            return "<br>".join(lines)

        def _distribute_labels_multicol(
            labels, max_lines_per_col: int = 70, max_chars: int = 20
        ):
            """
            Distribue les labels en colonnes en respectant une hauteur max exprimée
            en nombre de lignes effectives (après wrapping).
            Retourne une liste de colonnes, chaque colonne = liste de (index, label_wrapped).
            """
            wrapped = [wrap_label(lbl, max_chars=max_chars) for lbl in labels]
            line_counts = [w.count("<br>") + 1 for w in wrapped]
            total_lines = sum(line_counts)

            # Nombre de colonnes minimal pour respecter la hauteur max
            n_cols = max(1, int(np.ceil(total_lines / max_lines_per_col)))

            cols = []
            cur_col, cur_lines = [], 0
            for i, (w, lc) in enumerate(zip(wrapped, line_counts)):
                # Si on dépasse la hauteur max et qu'on a encore des colonnes dispo, on passe à la suivante
                if cur_lines + lc > max_lines_per_col and len(cols) < n_cols - 1:
                    cols.append(cur_col)
                    cur_col, cur_lines = [], 0
                cur_col.append((i, w))
                cur_lines += lc
            if cur_col:
                cols.append(cur_col)

            # Si pour une raison quelconque on a moins de colonnes que prévu, ce n'est pas grave.
            return cols

        max_lines_per_col = 70  # Si pas de retours à la ligne, équivaut à “70 items”
        cols = _distribute_labels_multicol(
            labels, max_lines_per_col=max_lines_per_col, max_chars=20
        )
        n_cols = len(cols)

        # Mise en page: largeur supplémentaire en px selon nb de colonnes
        fig_h = 1200
        fig_w = 1000
        col_w_px = 200
        right_pad_px = 60
        extra_w = n_cols * col_w_px + right_pad_px
        total_w = max(1400, fig_w + extra_w)

        # Annotations colonnes
        # Chaque colonne = liste de (index_filtré, label_wrapped)
        for col_idx, col_items in enumerate(cols):
            # Construit le bloc en conservant la numérotation 1..n sur la base des labels filtrés
            block = "<br>".join(f"{i + 1} : {wrapped}" for (i, wrapped) in col_items)
            fig.add_annotation(
                x=1.2 + 0.20 * col_idx,  # espace entre colonnes à droite
                y=1.0,
                xref="paper",
                yref="paper",
                text=block,
                showarrow=False,
                align="left",
                yanchor="top",
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
            )

        fig.update_layout(
            width=total_w,
            height=fig_h,
            margin=dict(t=50, b=60, l=20, r=extra_w),
            title_text=f"Heatmap of nitrogen fluxes {'(detailed view)' if detailed_view else '(aggregated view)'} for {self.region} in {self.year}",
            title_x=0.5,
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
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

        # df_excr["Humification (ktC)"] = (
        #     df_excr["Excretion to soil (ktN)"] * df_excr["C/N"]
        # )
        # df_excr["Excretion (ktC)"] = df_excr["Humification (ktC)"] / (
        #     df_excr["Humification coefficient (%)"] / 100
        # )
        # source = df_excr["Humification (ktC)"].to_dict()
        # target = {"soil stock": 1}

        # flux_generator.generate_flux(source, target)

        df_excr["Excretion (ktC)"] = df_excr["Excretion (ktN)"] * df_excr["C/N"]

        df_excr["Humification (ktC)"] = (
            df_excr["Excretion (ktC)"] * df_excr["Humification coefficient (%)"] / 100
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
            - df_excr["Humification (ktC)"]
            - df_excr["Excretion to CH4 (ktC)"]
            - df_excr[
                "Excretion to Energy (ktC)"
            ]  # On enlève la part envoyée aux méthaniseurs
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
        # df_pop["Humification (ktC)"] = (
        #     df_pop["Excretion after volatilization (ktN)"] * df_pop["C/N"]
        # )
        # df_pop["Excretion (ktC)"] = df_pop["Humification (ktC)"] / (
        #     df_pop["Humification coefficient (%)"] / 100
        # )

        df_pop["Excretion (ktC)"] = (
            df_pop["Excretion after volatilization (ktN)"] * df_pop["C/N"]
        )
        df_pop["Humification (ktC)"] = (
            df_pop["Excretion (ktC)"] * df_pop["Humification coefficient (%)"] / 100
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
            "Total Haber-Bosch methan input (kgC/kgN)", "value"
        ]
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
            lambda i: self.adjacency_matrix[:, self.data_loader.label_to_index[i]].sum()
        )

        excretion_sum_aligned = (
            df_excr.groupby("Origin compartment")["Excretion (ktC)"]
            .sum()
            .reindex(df_elevage.index, fill_value=0)
        )

        df_elevage["Excretion (ktC)"] = 0.0
        df_elevage["Excretion (ktC)"] = excretion_sum_aligned

        prod_sum_aligned = (
            df_prod.groupby("Origin compartment")["Carbon Production (ktC)"]
            .sum()
            .reindex(df_elevage.index, fill_value=0)
        )

        df_elevage["Production (ktC)"] = 0.0
        df_elevage["Production (ktC)"] = prod_sum_aligned

        df_elevage["Respiration (ktC)"] = (
            df_elevage["Ingestion (ktC)"]
            - df_elevage["CH4 enteric (ktC)"]
            - df_elevage["Production (ktC)"]
            - df_elevage["Excretion (ktC)"]
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

            # --- C/N moyens ingestion (alignement robuste) ---
            alloc = self.data_loader.nitrogen_model.allocations_df
            mask = alloc["Consumer"] == index
            a_sel = alloc.loc[mask, ["Product", "Allocated Nitrogen"]].copy()

            # aligner sur l'index produits de df_prod
            cn_map = df_prod["C/N"]  # index = produits
            a_sel = a_sel.join(cn_map.rename("C/N"), on="Product")

            # avertir si des produits n'ont pas de C/N
            missing = a_sel["C/N"].isna()
            if missing.any():
                warnings.warn(
                    f"[C/N] Produits sans C/N pour '{index}': "
                    + ", ".join(map(str, a_sel.loc[missing, "Product"].unique())),
                    category=UserWarning,
                )

            # moyenne pondérée (sûre aux NaN et /0)
            a_sel = a_sel.dropna(subset=["Allocated Nitrogen", "C/N"])
            num_ing = float(np.nansum(a_sel["Allocated Nitrogen"] * a_sel["C/N"]))
            den_ing = float(np.nansum(a_sel["Allocated Nitrogen"]))
            mean_ingestion_cn = 0.0 if den_ing == 0.0 else num_ing / den_ing

            # --- C/N moyen excrétion (inchangé, sûr /0) ---
            mask_ex = df_excr["Origin compartment"] == index
            w_ex = df_excr.loc[mask_ex, "Excretion (ktN)"].to_numpy()
            cn_ex = df_excr.loc[mask_ex, "C/N"].to_numpy()
            H_ex = df_excr.loc[mask_ex, "Humification coefficient (%)"].to_numpy() / 100
            num_ex = float(np.nansum(w_ex * cn_ex / H_ex))
            den_ex = float(np.nansum(w_ex))
            mean_excretion_cn = 0.0 if den_ex == 0.0 else num_ex / den_ex

            # --- Warning mis à jour ---
            message = (
                f"La respiration de '{index}' forcée à zéro (Calcul brut: {row['Ingestion (ktC)'] - somme_sorties:.4f} ktC).\n"
                f"  Détails (ktC) : Ingestion {row['Ingestion (ktC)']:.4f} | CH4 {row['CH4 enteric (ktC)']:.4f} | Excrétion {excretion_value:.4f} | Produits {production:.4f}\n"
                f"  C/N moyens : Ingestion={mean_ingestion_cn:.3f} | Excrétion={mean_excretion_cn:.3f} | Δ={mean_ingestion_cn - mean_excretion_cn:+.3f}"
            )
            warnings.warn(message, category=UserWarning)

        # Respiration humaine
        df_pop["Ingestion (ktC)"] = df_pop.index.map(
            lambda i: self.adjacency_matrix[:, self.data_loader.label_to_index[i]].sum()
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
            df_elevage["Infrastructure CO2 emissions/LU (kgC)"] * df_elevage["LU"] / 1e6
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

        self.df_elevage_display = df_elevage
        self.df_excr_display = df_excr
        self.df_cultures_display = df_cultures
        self.df_prod_display = df_prod

    def get_transition_matrix(self):
        """
        Returns the full nitrogen transition matrix.

        This matrix represents all nitrogen fluxes between sectors, including core and external processes.

        :return: A 2D NumPy array representing nitrogen fluxes between all sectors.
        :rtype: numpy.ndarray
        """
        return self.adjacency_matrix

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

    def compute_CO2_eq(self):
        CO2_eq = {}

        N2O_EM = self.data_loader.nitrogen_model.N2O_em()
        CO2_eq["N2O"] = N2O_EM * 273

        M = self.get_transition_matrix()
        CO2_EM = (
            (
                self.df_elevage["Mecanisation Emission (ktC)"].sum()
                + self.df_cultures["Mecanisation Emission (ktC)"].sum()
            )
            * (16 * 2 + 12)
            / 12
        )
        CO2_eq["CO2"] = CO2_EM * 1

        CH4_EM = (
            M[:, self.data_loader.label_to_index["atmospheric CH4"]].sum()
            * (4 + 12)
            / 12
        )
        CO2_eq["CH4"] = CH4_EM * 27.9

        return CO2_eq
