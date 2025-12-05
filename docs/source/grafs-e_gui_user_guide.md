# Graphical Interface

You can access the online interface here: **[E-GRAFS.streamlit.app](https://E-GRAFS.streamlit.app/)** 

---

## What is the GRAFS‑E UI?

The **GRAFS‑E** UI is a **Streamlit** application that explores nitrogen flows across compartments (crops, livestock, population, environment, industry, trade). It offers:
- **Sankey diagrams** (detailed or simplified/merged),
- a **Fertilization** view (sources → crops, with an aggregated **Seeds** box),
- **Food/Feed & Trade** view (trade nodes split into **import/export**),
- **Historical** plots (annual series by territory and metric),
- data tables and **multi‑sheet Excel/CSV downloads**.

---

## Getting started

### Run locally

```bash
streamlit run grafs_e/app.py
```

Open the local URL displayed by Streamlit.

---

## Using it from Python (without Streamlit)

```python
from grafs_e.model import NitrogenFlowModel
from grafs_e.sankey import app_sankey, get_unique_colors

model = NitrogenFlowModel(data=dataloader, area="France", year=2020)

A = model.get_transition_matrix()
labels = model.labels
colors = get_unique_colors(labels, fixed_colors_map={})

main_node = labels.index("cereals")  # example
fig = app_sankey(A, labels, main_node=main_node, index_to_color=colors, scope=1)
fig.show()
```

> `app_sankey` returns a **Plotly figure** (usable in notebooks or scripts). Streamlit‑specific functions (e.g. `streamlit_sankey_*`) draw directly with `st.plotly_chart`.

**Merged view logic.** The “simplified”/merged view is constructed in the app: crops are grouped by category, **products** are attached to their **origin crop** (`df_prod['Origin compartment']`), and **excretions** to their **origin livestock** (or `Livestock`). You can produce a `merges: dict` in Python and pass it to the plotting functions.

---

## UI tour

### Documentation
General description of E-GRAFS model and E-GRAFS UI.

### Data Uploading
Upload project/data files formated as explained in Input section of the documentation. You will see load status.

### Run E-GRAFS
This tab allow you to run E-GRAFS model for a selected Area and Year. It displays the transition matrix after computation.

### Sankey
This tab display several types of sankeys. Each sankey has a toggle switch for detailed/global view. Detailed view mode projects flows from the transition matrix, global view merge the flow belonging to the same categories.

#### Compartment sankey

This sankey just shows inflow and outflows for selected compartment. Selectable compartments exclude **products** and **excretions** (they are added downstream when relevant).  

#### Fertilization
This sankey display all fertilization vectors used in the system.
- Shows **incoming flows to crops** only.    
- Threshold slider shows the **value** and corresponding **% of total flow**.

#### Food/Feed & Trade
This sankey display the flow between products, import, consummers (livestock and populations) and export.
Trade nodes are split into `(... import)` and `(... export)`.

### Detailed data & Download
- Summary tables (crops, livestock, excretions, products, allocations, diet deviations).  
- **Download**: one Excel **with a sheet per dataframe** and/or individual CSVs.

### Historic evolution
This tab displays historic evolution of several metrics from the dataset file loaded.
- **Scalars**: net imports, total plant/animal production, livestock density, etc.  
- **Stacked series**:
  - **Crop production by category** (ktN/yr) and **Relative crop production by category** (%).  
  - **Fertilization by source** (ktN/yr) and **Relative fertilization by source** (%).  
  - **Area** aggregated **by category** (optionally add a relative version).

#### Environmental Foot print — Land footprint (ha) by component

**Goal.** Convert nitrogen amounts (ktN) associated with **consumption/flows** into **equivalent crop area** (ha) using **per‑crop yields** and areas.

**Notation.** Let, for each crop `c` (index aligned with `df_cultures.index`):
- `prod_k[c]` = nitrogen production (ktN) for crop `c` (from `_yield_area_map()`),
- `area_ha[c]` = area (ha) for crop `c` (from `_yield_area_map()`),
- `N_flow[c]` = nitrogen amount (ktN) attributed to crop `c` for the component considered.

**Equivalent area per crop:**

```{math}
\mathrm{surface}_{\mathrm{eq}}[c] =
\begin{cases}
\displaystyle
\frac{N_{\mathrm{flow}}[c]}{\mathrm{prod}_k[c]} \;\cdot\; \mathrm{area}_{\mathrm{ha}}[c], & \text{if } \mathrm{prod}_k[c] > 0,\\[6pt]
0, & \text{otherwise.}
\end{cases}
```

We then **sum over all crops** to get the footprint (ha).

#### Total fertilization by source (ktN)

`tot_fert()` aggregates **flows into crops** from major source families:
- **Haber‑Bosch** (mineral fertilizers),
- **Atmospheric deposition** = N₂O + NH₃ → crops,
- **Atmospheric N₂** (biological nitrogen fixation),
- **Animal excretion** = livestock (`df_elevage.index`) → crops,
- **Human excretion** = population (`df_pop.index`) → crops,
- **Seeds** (seeds → crops),
- **Mining** (soil stock → crops).

Implementation uses helper lookups (`_find_label_indices`, `_indices_for_crops`, `_sum_from_to`) and sums incoming flows per family.

---

## 5. User guide (step‑by‑step)

1. **Upload data** on Data Uploading tab.  
2. **Pick territory and year**; run the model, select prospective mode is you want to use it.  
3. **Explore Sankey**:
   - *Detailed* for fine‑grained inspection,
   - *Simplified* for category‑level overview,
   - adjust the **threshold** to declutter weak links.
4. **Analyze fertilization**: check how much each **source** (Haber‑Bosch, deposition, excretions, Seeds, …) feeds **crops**.  
5. **Inspect Food/Feed & Trade**: see imports/exports explicitly.  
6. **Compare over time** (Historic tab): choose a metric and plot.  
7. **Export** the tables for post‑processing.

---

## 8. Troubleshooting

- **Excel export not available:** install `openpyxl` or fall back to **ZIP of CSVs**.