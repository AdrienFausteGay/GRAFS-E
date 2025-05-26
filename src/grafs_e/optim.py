import numpy as np
import pandas as pd


class OptimizationSetup:
    def __init__(
        self,
        df_cultures,
        df_cons_vege,
        regimes,
        N_synth_crop,
        N_synth_grass,
        net_import,
        w_spread_fixed,
        w_diet,
        w_Nsyn,
        w_imp,
        prod_func="Ratio",
    ):
        self.prod_func = prod_func
        self.df_cultures = df_cultures  # Utilisez le DataFrame étendu si nécessaire
        self.df_cons_vege = df_cons_vege
        self.regimes = regimes
        self.N_synth_crop = N_synth_crop
        self.N_synth_grass = N_synth_grass
        self.net_import = net_import
        self.w_spread_fixed = w_spread_fixed
        self.w_diet = w_diet
        self.w_Nsyn = w_Nsyn
        self.w_imp = w_imp

        self.CROPS = list(self.df_cultures.index)
        self.CONSUMERS = list(self.df_cons_vege.index)

        # Mappings numériques pour un accès rapide aux tableaux NumPy
        self.crop_to_idx = {c: i for i, c in enumerate(self.CROPS)}
        self.consumer_to_idx = {k: i for i, k in enumerate(self.CONSUMERS)}
        self.num_crops = len(self.CROPS)
        self.num_consumers = len(self.CONSUMERS)

        # Convertir les dictionnaires en tableaux NumPy/Pandas Series pour la performance
        self.area_np = pd.Series(self.df_cultures["Area (ha)"]).loc[self.CROPS].values
        self.nonSynthFert_np = (
            pd.Series(self.df_cultures["Surface Non Synthetic Fertilizer Use (kgN/ha)"]).loc[self.CROPS].values
        )

        if self.prod_func == "Ratio":
            self.Ymax_np = pd.Series(self.df_cultures["Ymax (kgN/ha)"]).loc[self.CROPS].values
        elif self.prod_func == "Linear":
            self.a_np = pd.Series(self.df_cultures["a"]).loc[self.CROPS].values
            self.b_np = pd.Series(self.df_cultures["b"]).loc[self.CROPS].values

        self.ingestion_np = pd.Series(self.df_cons_vege).loc[self.CONSUMERS].values

        # Filter out any (c,k) pairs not in the diet to build allowed_ck
        self.allowed_ck = []
        # Store index pairs for allowed_ck for vectorization
        self.allowed_ck_indices = []  # list of (c_idx, k_idx) tuples

        for k in self.CONSUMERS:
            authorized_crops = set()
            for p_ideal, c_list in self.regimes[k].items():
                # Ensure all crops in c_list actually exist in self.CROPS
                for c in c_list:
                    if c in self.CROPS:  # Only add if the crop exists in our master list
                        authorized_crops.add(c)
                    else:
                        print(f"Warning: Crop '{c}' in regime for '{k}' not found in df_cultures.index. Skipping.")

            for c in self.CROPS:  # Iterate over all master crops
                if c in authorized_crops:
                    self.allowed_ck.append((c, k))
                    self.allowed_ck_indices.append((self.crop_to_idx[c], self.consumer_to_idx[k]))

        # --- Indexing for decision variables in x ---
        self.idx_synth_slice = slice(0, self.num_crops)  # x[0] to x[num_crops-1]

        offset = self.num_crops
        self.idx_alloc_slice = slice(offset, offset + len(self.allowed_ck))

        offset += len(self.allowed_ck)
        self.idx_import_slice = slice(offset, offset + len(self.allowed_ck))

        self.n_vars = offset + len(self.allowed_ck)  # Total number of decision variables

        # --- Pre-processing for fixed_availability_proportion and regimes ---
        # This is where we create matrices/tensors for faster lookups
        self.epsilon = 1e-9  # Define epsilon here

        # 1. Map all unique crop groups to numerical indices
        self.unique_crop_groups = []  # list of tuple(sorted(crops))
        self.group_to_idx = {}
        for k_name in self.CONSUMERS:
            for group_crops_list in self.regimes[k_name].values():
                if isinstance(group_crops_list, (list, tuple, set)):
                    # Ensure all crops in group_crops_list exist in CROPS before sorting/hashing
                    valid_crops_in_group = [c for c in group_crops_list if c in self.CROPS]
                    if valid_crops_in_group:
                        group_key = tuple(sorted(valid_crops_in_group))
                        if group_key not in self.group_to_idx:
                            self.group_to_idx[group_key] = len(self.unique_crop_groups)
                            self.unique_crop_groups.append(group_key)

        self.num_groups = len(self.unique_crop_groups)

        # 2. Pre-calculate fixed_availability_proportion into a tensor
        # tensor: [consumer_idx, group_idx, crop_idx] -> proportion
        self.fixed_proportion_tensor = np.zeros((self.num_consumers, self.num_groups, self.num_crops))

        for k_name in self.CONSUMERS:
            k_idx = self.consumer_to_idx[k_name]
            for group_crops_list_raw in self.regimes[k_name].values():
                if not isinstance(group_crops_list_raw, (list, tuple, set)):
                    continue

                valid_crops_in_group = [c for c in group_crops_list_raw if c in self.CROPS]
                if not valid_crops_in_group:
                    continue

                group_key = tuple(sorted(valid_crops_in_group))
                group_idx = self.group_to_idx[group_key]

                group_total_area = sum(self.area_np[self.crop_to_idx[c2]] for c2 in valid_crops_in_group)

                if group_total_area < self.epsilon:
                    continue

                for c_name in valid_crops_in_group:
                    c_idx = self.crop_to_idx[c_name]
                    crop_area = self.area_np[c_idx]
                    proportion = crop_area / group_total_area
                    self.fixed_proportion_tensor[k_idx, group_idx, c_idx] = proportion

        # 3. Pre-process regimes for diet_deviation
        # We need a matrix that, for each consumer, for each crop, tells us its 'p_ideal'
        # This is tricky because one crop can belong to multiple groups with different p_ideal.
        # Instead, we'll create a list of (consumer_idx, group_idx, p_ideal) for diet deviation.
        self.regimes_indexed = []  # List of (k_idx, group_idx, p_ideal)
        self.regimes_group_crop_mask = np.zeros(
            (self.num_consumers, self.num_groups, self.num_crops), dtype=bool
        )  # Mask: True if crop c_idx is in group_idx for consumer k_idx

        for k_name in self.CONSUMERS:
            k_idx = self.consumer_to_idx[k_name]
            for p_ideal, group_crops_list_raw in self.regimes[k_name].items():
                if not isinstance(group_crops_list_raw, (list, tuple, set)):
                    continue

                valid_crops_in_group = [c for c in group_crops_list_raw if c in self.CROPS]
                if not valid_crops_in_group:
                    continue

                group_key = tuple(sorted(valid_crops_in_group))
                group_idx = self.group_to_idx[group_key]

                self.regimes_indexed.append((k_idx, group_idx, p_ideal))
                for c_name in valid_crops_in_group:
                    c_idx = self.crop_to_idx[c_name]
                    self.regimes_group_crop_mask[k_idx, group_idx, c_idx] = True

    # Helper for Y_th functions to handle arrays and division by zero
    def Y_th_lin_vec(self, f, a, b):
        return np.minimum(a * f, b)

    def Y_th_ratio_vec(self, f, ymax):
        # Avoid division by zero: where f+ymax is zero, result is zero
        denominator = f + ymax
        # Create a result array, set to 0 where denominator is 0, otherwise calculate
        result = np.zeros_like(f, dtype=float)
        non_zero_denom_mask = denominator != 0
        result[non_zero_denom_mask] = (f[non_zero_denom_mask] * ymax[non_zero_denom_mask]) / denominator[
            non_zero_denom_mask
        ]
        return result

    # --------------------------------------------------------------------------
    # 3) Define the vectorized objective function
    # --------------------------------------------------------------------------
    def objective_vectorized(self, x):
        x = np.asarray(x)

        # Extract relevant slices for clarity
        x_synth = x[self.idx_synth_slice]  # synthetic fertilizers per crop (indexed by crop)

        # Reshape allocation and import to (crop_idx, consumer_idx) matrices
        # We need a matrix that maps (crop_idx, consumer_idx) to the correct position in x_alloc/x_import.
        # This requires reconstructing the matrix from allowed_ck_indices.
        x_alloc_matrix = np.zeros((self.num_crops, self.num_consumers))
        x_import_matrix = np.zeros((self.num_crops, self.num_consumers))

        # Fill the matrices based on allowed_ck_indices and slices
        for i, (c_idx, k_idx) in enumerate(self.allowed_ck_indices):
            x_alloc_matrix[c_idx, k_idx] = x[self.idx_alloc_slice.start + i]
            x_import_matrix[c_idx, k_idx] = x[self.idx_import_slice.start + i]

        # 3.a) diet_deviation
        total_dev = 0.0

        # Calculate total allocation per consumer (denom_k) for all consumers at once
        total_alloc_per_consumer = np.sum(x_alloc_matrix, axis=0) + np.sum(x_import_matrix, axis=0)

        # Iterate over pre-processed regimes_indexed
        for k_idx, group_idx, p_ideal in self.regimes_indexed:
            denom_k = total_alloc_per_consumer[k_idx]

            # Use the pre-computed mask to sum allocations for this group and consumer
            # This selects allocations for crops within the specific group for consumer k_idx
            group_alloc = np.sum(x_alloc_matrix[self.regimes_group_crop_mask[k_idx, group_idx, :], k_idx])
            group_alloc += np.sum(x_import_matrix[self.regimes_group_crop_mask[k_idx, group_idx, :], k_idx])

            if denom_k < self.epsilon:  # Use self.epsilon from pre-processing
                proportion_real = 0.0
            else:
                proportion_real = group_alloc / denom_k

            total_dev += (proportion_real - p_ideal) ** 2

        # 3.b) fertilizer_deviation
        total_synth = np.sum(x_synth * self.area_np)  # Use pre-computed area_np

        total_synth_kt = total_synth / 1e6

        desired_total_N_synth = self.N_synth_crop + self.N_synth_grass
        scale = 1 if desired_total_N_synth < 1 else desired_total_N_synth

        fert_dev = np.maximum(0, (total_synth_kt - desired_total_N_synth) / scale) ** 2

        # 3.c) import_export_deviation
        sum_imp = np.sum(x_import_matrix)

        # Production for all crops at once
        current_synth_fert_vals = x_synth + self.nonSynthFert_np

        if self.prod_func == "Ratio":
            production_c_all = self.Y_th_ratio_vec(current_synth_fert_vals, self.Ymax_np) * self.area_np / 1e6
        elif self.prod_func == "Linear":
            production_c_all = self.Y_th_lin_vec(current_synth_fert_vals, self.a_np, self.b_np) * self.area_np / 1e6
        else:
            production_c_all = np.zeros(self.num_crops)

        allocated_c_all = np.sum(x_alloc_matrix, axis=1)  # Sum over consumers for each crop

        leftover_c_all = production_c_all - allocated_c_all
        export_total = np.sum(leftover_c_all)

        net_import_model = sum_imp - export_total

        # net_import needs to be a class member or passed
        net_import = self.net_import

        if abs(net_import) < 1:
            imp_dev = (net_import_model - net_import) ** 2
        else:
            imp_dev = ((net_import_model - net_import) / (net_import + self.epsilon)) ** 2

        # --- NEW: Allocation Spread Penalty based on Fixed Proportions ---
        spread_penalty_fixed = 0.0

        # Calculate group_alloc_kG for all consumers and all groups first
        # group_alloc_kG_matrix[k_idx, group_idx]
        group_alloc_kG_matrix = np.zeros((self.num_consumers, self.num_groups))

        for k_idx in range(self.num_consumers):
            for group_idx in range(self.num_groups):
                # Sum allocations for crops in this group for this consumer
                group_alloc_kG_matrix[k_idx, group_idx] = np.sum(
                    x_alloc_matrix[self.regimes_group_crop_mask[k_idx, group_idx, :], k_idx]
                )
                group_alloc_kG_matrix[k_idx, group_idx] += np.sum(
                    x_import_matrix[self.regimes_group_crop_mask[k_idx, group_idx, :], k_idx]
                )

        # Now, calculate penalty using vectorised operations where possible
        # This part iterates over (k_idx, group_idx, c_idx) for which fixed_proportion_tensor has non-zero values

        # Filter for relevant entries in fixed_proportion_tensor where proportion > epsilon
        relevant_proportions_mask = self.fixed_proportion_tensor > self.epsilon

        # Extract relevant consumer, group, crop indices
        k_indices, group_indices, c_indices = np.where(relevant_proportions_mask)

        # Get the corresponding target proportions
        target_proportions = self.fixed_proportion_tensor[relevant_proportions_mask]

        # Calculate target_alloc_c for all relevant (k, group, c) at once
        # Need to gather group_alloc_kG for these specific k_idx, group_idx pairs
        gathered_group_alloc_kG = group_alloc_kG_matrix[k_indices, group_indices]

        target_alloc_c_all = target_proportions * gathered_group_alloc_kG

        # Calculate actual_alloc_ck for all relevant (k, group, c) at once
        # Note: c_indices here refer to the crop's index, not its position in allowed_ck_indices
        # Need to look up x[idx_alloc[(c_name, k_name)]] + x[idx_import[(c_name, k_name)]]
        # This is complex because x_alloc_matrix and x_import_matrix have (crop_idx, consumer_idx)
        # We need to map (c_indices, k_indices) back to the flat x_alloc/x_import based on allowed_ck_indices

        # This part requires re-mapping from (c_idx, k_idx) to the flat index if not directly from x_alloc_matrix
        # A simpler way is to just index x_alloc_matrix and x_import_matrix directly with the gathered indices
        actual_alloc_ck_all = x_alloc_matrix[c_indices, k_indices] + x_import_matrix[c_indices, k_indices]

        # Calculate deviations and sum of squares
        deviations = actual_alloc_ck_all - target_alloc_c_all
        spread_penalty_fixed = np.sum(deviations**2)

        # --- Final Objective Value ---
        return (
            self.w_diet * total_dev
            + self.w_Nsyn * fert_dev
            + self.w_imp * imp_dev
            + self.w_spread_fixed * spread_penalty_fixed
        )
