# Output

This page documents the **outputs** produced by the Nitrogen (N) layer of GRAFS‑E. Outputs are identical in **Historical** and **Prospective** modes; differences between modes affect **how** some values are computed (via the engine) but **not** the structure of outputs.

Outputs are delivered as:
1) a **transition matrix** (territory‑scale nitrogen flows, ktN·yr⁻¹), and  
2) a set of **dataframes** reporting per‑compartment results (some columns echo inputs for context, others are computed by the model):
- **df_cultures**: related to crops
- **df_elevage**: related to livestock
- **df_pop**: related to populations
- **df_prod**: related to products
- **df_global**: provides the global input data.
- **allocations_df**: shows the output of the allocation model between products and consumers (humans and animals)
- **deviations_df**: shows the deviations between the allocation model output and the target (Diet tab from the data Excel sheet)
- **df_energy_flows**: shows the inflows of energy infrastructures.

**df_cultures**, **df_elevage**, **df_pop**, **df_prod**, and **df_energy** contain both outputs and input data. **df_global** simply records the global variables provided as input.

## Transition Matrix

The most comprehensive output from the model is a transition matrix. This square matrix shows the flows between each compartment. It has the same size as the number of compartments. Each coefficient \(c_{ij}\) represents the nitrogen flow in ktN from compartment i to compartment j. The sum of column j gives all the inputs of a compartment, while the sum of row i gives all the outputs of a compartment.

Compartments in the sets **crops**, **livestock**, **products**, **population**, **excretion**, **energy** are **internal and balanced** by construction (sum in = sum out).

![Transition Matrix from example data](_static/matrix.png)

## Nitrogen Layer — Output Columns

This section enumerates the **columns created by the Nitrogen layer** in each output dataframe.  
We list **only model‑generated outputs** (not input fields). Units are indicated inline.

### `df_cultures` (crops)

| Column Name | Description |
|---|---|
| **Nitrogen Harvest Index** | N‑based harvest index computed at crop level. |
| **Main Crop Production (kton)** | Fresh‑weight production of the main product mapped back to the crop. |
| **Main Nitrogen Production (ktN)** | Nitrogen in the main product harvested. |
| **Yield (qtl/ha)** | Crop yield per hectare in quintals. |
| **Yield (kgN/ha)** | Harvested nitrogen per hectare. |
| **Seeds Input (ktN)** | Nitrogen associated with sowing seeds at crop level. |
| **Seeds Input (kgN/ha)** | Seeds nitrogen expressed per hectare. |
| **BNF (kgN/ha)** | Biological nitrogen fixation per hectare. |
| **BNF (ktN)** | Area‑scaled biological nitrogen fixation. |
| **Atmospheric deposition (ktN)** | Atmospheric nitrogen deposition attributed to the crop. |
| **Adjusted Total Synthetic Fertilizer Use (ktN)** | Territory‑consistent synthetic nitrogen assigned to the crop (pre‑field). |
| **Adjusted Surface Synthetic Fertilizer Use (kgN/ha)** | Surface‑normalized synthetic N usage. |
| **Volatilized Nitrogen N‑NH3 (ktN)** | Aggregated NH₃ emissions associated with synthetic fertiliser. |
| **Volatilized Nitrogen N‑N2O (ktN)** | Aggregated N₂O emissions associated with synthetic fertiliser. |
| **Synthetic to field (ktN)** | Synthetic nitrogen effectively reaching the field after aggregated losses. |
| **Harvested Production (ktN)** | Total harvested nitrogen from all products originating from the crop. |
| **Inputs to field (ktN)** | Lumped field inputs used in the balance (dep + BNF + excreta + digestate + seeds + synthetic_to_field). |
| **Balance (ktN)** | Signed balance before partition: `Inputs to field − Harvested Production`. |
| **Surplus (ktN)** | Positive part of the balance (otherwise 0). |
| **Leached to hydro‑system (ktN)** | Share of positive surplus routed to leaching. |
| **Surplus N2O (ktN)** | Share of positive surplus routed to N₂O emissions. |
| **Soil storage from surplus (ktN)** | Share of positive surplus stored in soil (prairie first‑store logic included). |
| **Mining from soil (ktN)** | Positive part of the opposite balance (harvest exceeding inputs). |
| **Soil stock (ktN)** | Net soil stock change: `Soil storage from surplus − Mining from soil`. |
| **Total Non Synthetic Fertilizer Use (ktN)** | Sum of non‑synthetic field inputs (dep + BNF + excreta + digestate + seeds). |
| **Surface Non Synthetic Fertilizer Use (kgN/ha)** | Previous aggregate normalized by area. |
| **Surface Inputs to field (kgN/ha)** | Field inputs normalized by area. |
| **Surface Surplus (kgN/ha)** | Surplus normalized by area. |

---

### `df_prod` (products)

| Column Name | Description |
|---|---|
| **Nitrogen Production (ktN)** | Product nitrogen production after internal consistency checks. |
| **Nitrogen Wasted (ktN)** | Nitrogen routed to the waste node prior to allocation. |
| **Nitrogen for Other uses (ktN)** | Nitrogen routed to the “other sectors” node prior to allocation. |
| **Available Nitrogen Production (ktN)** | Availability after waste and other uses. |
| **Nitrogen For Feed (ktN)** | Total allocation of this product to livestock (local + imports). |
| **Nitrogen For Food (ktN)** | Total allocation of this product to population (local + imports). |
| **Nitrogen For Energy (ktN)** | Allocation of this product to energy infrastructures. |
| **Nitrogen Exported (ktN)** | Residual exported/stored/unused nitrogen after domestic uses. |
| **Production (kton)** | Fresh‑weight production recomputed in prospective mode from nitrogen and content data. |

---

### `df_elevage` (livestock)

| Column Name | Description |
|---|---|
| **Excreted indoor as slurry (%)** | Completed indoor management share for excretion (convenience field). |
| **Excreted on grassland (%)** | Outdoor excretion share (complement to indoor). |
| **Edible Nitrogen (ktN)** | Edible animal product nitrogen. |
| **Non‑Edible Nitrogen (ktN)** | Non‑edible animal co‑product nitrogen. |
| **Dairy Nitrogen (ktN)** | Dairy/egg nitrogen outputs. |
| **Excreted nitrogen (ktN)** | Total nitrogen excreted by the livestock (pre‑routing). |
| **Ingestion (ktN)** | livestock nitrogen ingestion computed from excretion and animal production. |
| **Consumed nitrogen from local feed (ktN)** | Nitrogen from local products allocated to the livestock. |
| **Consumed nitrogen from imported feed (ktN)** | Nitrogen from imported products allocated to the livestock. |
| **Net animal nitrogen exports (ktN)** | Net exports of animal products for the livestock. |

---

### `df_excr` (animal excretions)

| Column Name | Description |
|---|---|
| **Excretion after volatilization (ktN)** | Excreta nitrogen available after gaseous losses. |
| **Excretion to Energy (ktN)** | Portion of post‑volatilization excreta sent to energy. |
| **Excretion to soil (ktN)** | Portion of post‑volatilization excreta applied to fields. |

---

### `df_pop` (population)

| Column Name | Description |
|---|---|
| **Ingestion (ktN)** | Total nitrogen ingestion by the population node. |
| **Fishery Ingestion (ktN)** | Nitrogen ingestion from fishery products. |
| **Excretion after volatilization (ktN)** | Human excreta nitrogen available after gaseous losses. |

---

### `allocations_df` (product → consumer flows)

Allocation table produced by the LP/MILP solver; one row per product–consumer flow.

| Column Name | Description |
|---|---|
| **Product** | Product name. |
| **Consumer** | Consumer name (human group, livestock group or energy facility). |
| **Allocated Nitrogen (ktN)** | Nitrogen mass allocated in the solution. |
| **Type** | Semantic label of the allocation (e.g., Local Feed, Imported Food, Energy). |

---

### `deviations_df` (diet)

Diagnostics used by the objective for soft constraints.

| Column Name | Description |
|---|---|
| **Consumer** | Identifier of the consumer or facility. |
| **Type** | Consumer type (Human, Animal or Energy). |
| **Expected Proportion (%)** | Target diet proportion. |
| **Proportion Allocated (%)** | Realized proportion from allocations. |
| **Deviation (%)** | Deviation between realized and target. |
| **Product** | List of products of diet group. |

---

## 8) `df_energy` (energy infrastructures)

| Column Name | Description |
|---|---|
| **Energy Production (GWh)** | Total energy produced by the facility (all inputs combined). |
| **Nitrogen Input to Energy (ktN)** | Total nitrogen mass consumed by the facility. |

---

## 9) `df_energy_flows` (sources → facility)

Per‑source breakdown of inputs and attributable energy.

| Column Name | Description |
|---|---|
| **source** | Name of the source node (product, excreta, waste). |
| **target** | Facility name. |
| **allocation (ktN)** | Nitrogen allocated from source to facility. |
| **energy production (GWh)** | Energy attributable to this allocation. |
| **allocation share (%)** | Share of the source allocation in facility total N input. |
| **energy production share (%)** | Share of the source’s energy in facility total energy output. |
| **Type** | Category of the source (product / excreta / waste). |
| **Diet** | Facility diet tag. |