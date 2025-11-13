# Carbon Layer

This document describes the **Carbon** layer of GRAFS‑E. This layer represents Carbon fluxes between compartments. This layer aims to deepen the analysis of agricultural metabolism, enlarge public (policy makers, engeeneers) and propose easy to use Carbon footprint evaluation for global system and each compartment.
The Carbon layer is **entirely built on the Nitrogen (N) layer**: it reuses the N transition matrix and allocation outcomes to derive C flows. Therefore, it behaves the **same** in Historical and Prospective modes; any differences between modes stem from the underlying Nitrogen layer.

The purpose is to represent coherent **carbon fluxes** (ktC) across the same compartments as the Nitrogen layer (crops, livestock, products, population, excreta, energy infrastructures, external nodes), enabling system‑wide C accounting and compartment‑level footprints.

## How to use

The carbon layer can be called from C_class.py. It includes 2 classes :
- DataLoader_Carbon(project_path, data_path, region, year)
- CarbonFlowModel(data, year, region)

With DataLoader_Carbon() the class to load data. Project and Data files are structured the same way as for Nitrogen layer (see Input Data page). Yet these files must include supplementary data described in section 'Input Data (additional to the Nitrogen layer)' below. Year and Region must be given at the data loading stage.

CarbonFlowModel() is the class using input data to produce output data (see section 'Outputs' below). Methods are similar to NitrogenFlowModel().

Once input data are correctly formated, simply use:

```python
from grafs_e import Dataloader_Carbon, CarbonFlowModel

# Paths to project and data files (same structure as Nitrogen; add C-specific columns below)
project = "path/to/project_file.xlsx"
data    = "path/to/data_file.xslx"

region = "MyRegion"
year   = 2022

# Load (the loader internally builds the Nitrogen model and its transition matrix)
dlc = Dataloader_Carbon(project, data, region, year, prospective=False)

# Build Carbon flows
C = CarbonFlowModel(dlc)

# Retrieve full Carbon transition matrix
M_C = C.get_transition_matrix()
```

Output can be get as for Nitrogen layer.

## Methodology

**Units & conventions.**
- Flows are in **ktC** unless explicitly stated.
- Percentages in inputs are stored as **percent** and converted to **fractions** in formulas.
- $NCV$ is the **lower heating value** of CH₄ in kWh·m⁻³ (default 10 kWh·m⁻³), $\rho_{CH_4}$ is the CH₄ density (default 0.717 kg·m⁻³).

### Products

**C content and C/N.** For product $p$:
```{math}
(C/N)_p \;=\; \frac{\%C_p}{\%N_p}\,.
```
**Carbon production.**
```{math}
C^{\mathrm{prod}}_p \;=\; \mathrm{Production}_p^{(\mathrm{kton})}\;\frac{\%C_p}{100}\,.
```
**Wastes, other uses, exports.** When a Nitrogen flow $N$ (ktN) is known:
```{math}
C \;=\; N \cdot (C/N)_p \,.
```
Hence, $C^{\mathrm{waste}}_p = N^{\mathrm{waste}}_p (C/N)_p$, same for **other uses** and **exports**. Exports are aggregated by product **Sub Type** to dedicated “trade” nodes.

### Seeds (to crops)
For crop $c$ (with main product $p_c$):
```{math}
C^{\mathrm{seed}}_c \;=\; \bigl(C/N\bigr)_{p_c}\cdot N^{\mathrm{seed}}_{c}\,.
```
### Crops: photosynthesis, residues, roots

**Photosynthesis uptake** from atmospheric CO₂ balances the formation of harvested products, residues, and roots, net of seeds:
```{math}
C^{\mathrm{photo}}_c \;=\; C^{\mathrm{prod}}_c \;+\; C^{\mathrm{res}}_c \;+\; C^{\mathrm{root}}_c \;-\; C^{\mathrm{seed}}_c\,.
```
**Residues and roots.** Let $HI_c$ be the harvest index. With $C^{\mathrm{prod}}_c$ the **total** product carbon from crop $c$:
```{math}
C^{\mathrm{res}}_c \;=\; \frac{C^{\mathrm{prod}}_c}{HI_c} - C^{\mathrm{prod}}_c,
\qquad
C^{\mathrm{root}}_c \;=\; C^{\mathrm{res}}_c \cdot \bigl(BGN_c - 1).
```
**Humification vs CO₂** for residues & roots (fractions $h^{\mathrm{res}}_c$, $h^{\mathrm{root}}_c$):
```{math}
C^{\mathrm{soil}}_{c,\mathrm{res+root}}
\;=\;
h^{\mathrm{res}}_c\,C^{\mathrm{res}}_c
\;+\;
h^{\mathrm{root}}_c\,C^{\mathrm{root}}_c,
\qquad
C^{\mathrm{CO_2}}_{c,\mathrm{res+root}}
\;=\;
(1-h^{\mathrm{res}}_c)\,C^{\mathrm{res}}_c
\;+\;
(1-h^{\mathrm{root}}_c)\,C^{\mathrm{root}}_c.
```

### Livestock: enteric CH₄ and respiration

**Enteric CH₄** (per LU):
```{math}
C^{\mathrm{CH_4}}_{\mathrm{enteric},\ell} \;=\; \frac{\mathrm{LU}_\ell\cdot \mathrm{C{\text-}CH4\ enteric/LU}_\ell}{10^6}\,.
```
**Respiration CO₂** is the mass balance of C in the compartment:
```{math}
C^{\mathrm{resp}}_\ell
\;=\;
C^{\mathrm{ingest}}_\ell
\;-\;
C^{\mathrm{CH_4}}_{\mathrm{enteric},\ell}
\;-\;
C^{\mathrm{excr}}_\ell
\;-\;
C^{\mathrm{prod}}_\ell
\quad \text{(clipped at } \ge 0\text{)}.
```
Here $C^{\mathrm{ingest}}_\ell$ is the **incoming** C to livestock $\ell$ from all allocated sources (sum of the Carbon transition matrix column).

### Excreta (animals)

From **Excretion to soil** nitrogen (ktN) and the C/N ratio:
```{math}
C^{\mathrm{humif}}_e \;=\; N^{\mathrm{to\ soil}}_e \cdot (C/N)_e,\qquad
C^{\mathrm{excr}}_e \;=\; \frac{C^{\mathrm{humif}}_e}{h^{\mathrm{excr}}_e}\,,
```
with $h^{\mathrm{excr}}_e$ the **humification coefficient** (fraction). Partition:
```{math}
C^{\mathrm{CH_4}}_{e} \;=\; C^{\mathrm{excr}}_e \cdot \eta^{\mathrm{CH_4}}_e,\qquad
C^{\mathrm{CO_2}}_{e} \;=\; C^{\mathrm{excr}}_e \cdot \bigl(1 - \eta^{\mathrm{CH_4}}_e - h^{\mathrm{excr}}_e - \theta^{\mathrm{E}}_e\bigr),
```
where $\eta^{\mathrm{CH_4}}_e$ is the CH₄ emission factor and $\theta^{\mathrm{E}}_e$ the **share sent to energy** (in C terms).

### Population (human excreta & respiration)

Analogous definitions apply with population‑specific parameters $(C/N)$, $\eta^{\mathrm{CH_4}}$, and $h$:
```{math}
C^{\mathrm{resp}}_{u} \;=\; \max\!\bigl(C^{\mathrm{ingest}}_{u} - C^{\mathrm{excr}}_{u},\,0\bigr)\,.
```

### Energy infrastructures (generic)

Let $f$ denote a facility and $\mathcal{I}_f$ its admissible inputs (products, excreta, waste). The **carbon input** to $f$ is the sum of allocated Nitrogen inputs times their source‑specific C/N:
```{math}
C^{\mathrm{in}}_f \;=\; \sum_{i \in \mathcal{I}_f} N^{\mathrm{alloc}}_{i \to f}\cdot (C/N)_i\,.
```
A **process CO₂ share** $s^{\mathrm{CO_2}}_f$ (from input data) sends
```{math}
C^{\mathrm{CO_2}}_f \;=\; s^{\mathrm{CO_2}}_f \, C^{\mathrm{in}}_f
\quad \text{to atmospheric CO₂.}
```

#### Methanizer
Energy output $E_f$ (GWh) is given from the Nitrogen layer. The **carbon in methane** produced is:
```{math}
C^{\mathrm{CH_4}}_f\ [\mathrm{ktC}]
\;=\;
E_f\ [\mathrm{GWh}]\;
\left(\frac{\rho_{CH_4}}{NCV}\right)\;
\frac{12}{16}\,.
```
The **digestate carbon** is the remainder:
```{math}
C^{\mathrm{digest}}_f \;=\; \max\!\bigl(C^{\mathrm{in}}_f - C^{\mathrm{CO_2}}_f - C^{\mathrm{CH_4}}_f,\;0\bigr)\,,
```
which is routed to **soil stock** (consistent with the Nitrogen layer assumption that all N entering methanizers exits in digestate and is spread under human‑excreta rules). The **hydrocarbures** node receives $C^{\mathrm{CH_4}}_f$.

#### Bioraffinery
There is **no digestate**. The CO₂ share is emitted as above and the remainder is directed to **hydrocarbures**:
```{math}
C^{\mathrm{HC}}_f \;=\; \max\!\bigl(C^{\mathrm{in}}_f - C^{\mathrm{CO_2}}_f,\;0\bigr)\,.
```

### Haber–Bosch methane consumption (upstream)

With $I_{\mathrm{HB}}$ the total methane carbon intensity (kgC/kgN) and $N_{\mathrm{HB}}$ the Nitrogen throughput (ktN):
```{math}
C^{\mathrm{HB}} \;=\; \frac{I_{\mathrm{HB}}}{10^6}\;N_{\mathrm{HB}}.
```
This carbon is drawn from **hydrocarbures** to the **Haber–Bosch** node and released to **atmospheric CO₂**.

### Mechanization (crops & livestock)

Emissions associated with machinery and infrastructures are parameterized as **surface** (crops) and **LU** (livestock) intensities:
```{math}
C^{\mathrm{mech}}_{\mathrm{crop},c} \;=\; \mathrm{CI}^{\mathrm{mech}}_{c}\cdot A_c,\qquad
C^{\mathrm{mech}}_{\mathrm{liv},\ell} \;=\; \mathrm{CI}^{\mathrm{infra}}_{\ell}\cdot \mathrm{LU}_\ell\,.
```
Flows are routed **from hydrocarbures** to dedicated “machines” nodes, then to **atmospheric CO₂**.

---

## Input data (additional to the Nitrogen layer)

The Carbon layer as been conceived to necessitate as little as possible supplementary data. Here is the description of data required to run the CarbonFlowModel. These data must be included either as defaut value in project file or interannual values in data file. See 'GRAFS_project_example_C.xlsx' and 'GRAFS_data_example_C.xslx' in example file.

### Crops
| Column | Description | Type |
|---|---|---|
| **Carbon Mechanisation Intensity (ktC/ha)** | Machinery/infrastructure C emissions for cultivation | float (≥0) |
| **Residue Humification Coefficient (%)** | Fraction of residue C stabilized in soil | [0,100] |
| **Root Humification Coefficient (%)** | Fraction of root C stabilized in soil | [0,100] |

Unlike Nitrogen layer, in Carbon layer, the Harvest Index and BGN is required for all crops. Otherwise, roots and residues productions will be set to 0.

### Livestock
| Column | Description | Type |
|---|---|---|
| **C-CH4 enteric/LU (kgC)** | Enteric methane C per livestock unit | float (≥0) |
| **Infrastructure CO2 emissions/LU (kgC)** | Machinery/infrastructure C for husbandry | float (≥0) |

### Products
| Column | Description | Type |
|---|---|---|
| **Carbon Content (%)** | Product C content (used with N content to build C/N) | float (≥0) |

### Excretion (animals)
| Column | Description | Type |
|---|---|---|
| **C/N** | C to N ratio of excreta (after C and N losses) | float (≥0) |
| **CH4 EM (%)** | CH₄ emission factor | [0,100] |
| **Humification coefficient (%)** | Fraction stabilized in soil | [0,100] |

### Population

| Column | Description | Type |
|---|---|---|
| **C/N** | C to N ratio of excreta (after C and N losses) | float (≥0) |
| **CH4 EM (%)** | CH₄ emission factor | [0,100] |
| **Humification coefficient (%)** | Fraction stabilized in soil | [0,100] |

### Energy facilities
| Column | Description | Type |
|---|---|---|
| **Type** | Facility kind (`Methanizer`, `Bioraffinery`) | categorical |
| **Share CO2 (%)** | Share of input carbon emitted as process CO₂ | [0,100] |
| **Energy Production (GWh)** | Final energy assigned to the facility (from N layer results) | float (≥0) |

### Global data
| Variable | Description | Type |
|---|---|---|
| **Total Haber-Bosch methan input (kgC/kgN)** | Total methane carbon per kgN in HB process (direct + indirect) | float (≥0) |
| **Green waste C/N** | C/N of green waste | float (≥0) |

---

## 5. Outputs

Here are display the output columns specific to carbon layer.

### Crops — `df_cultures`
| Column | Unit | Definition / Role | Provenance / Formula |
|---|---|---|---|
| **C/N** | – | Carbon-to-nitrogen ratio of the main product | `(%C / %N)` of main product |
| **Carbon Production (ktC)** | ktC | Total carbon harvested from all products of crop $c$ | `Production (kton) * Carbon Content (%) / 100` summed by crop |
| **Main Carbon Production (ktC)** | ktC | Carbon in the main product of crop $c$ | `Main N Production * (C/N)` |
| **Residue Production (ktC)** | ktC | Above‑ground residues carbon | `max( (MainCarbon/HI) - CarbonProduction , 0 )` |
| **Root Production (ktC)** | ktC | Below‑ground carbon (roots) | `ResidueProduction * (1 - BGN_c)` with $BGN_c$ from N layer |
| **Seeds Input (ktC)** | ktC | Carbon in seeds applied | `(C/N)_main * Seeds Input (ktN)` |
| **Mecanisation Emission (ktC)** | ktC | Carbon emitted due to mechanization | `Area * Carbon Mechanisation Intensity` |

> Residues and roots are used for **C routing** (soil vs CO₂) via humification fractions; they do **not** alter the N surface balance.

### Livestock — `df_elevage`
| Column | Unit | Definition / Role | Provenance / Formula |
|---|---|---|---|
| **Ingestion (ktC)** | ktC | Carbon ingested by livestock (sum of allocated C) | From C transition (post‑allocation) |
| **CH4 enteric (ktC)** | ktC | Enteric methane emissions (C) | `LU * (C-CH4 enteric/LU) / 10^6` |
| **Respiration (ktC)** | ktC | CO₂ from respiration (mass balance) | `max(Ingestion - CH4 enteric - Excretion - Products, 0)` |
| **Mecanisation Emission (ktC)** | ktC | Carbon emitted due to husbandry infra | `LU * Infrastructure CO2 emissions/LU / 10^6` |

> `Excretion` and `Products` terms for livestock come from the corresponding C flows built from the N layer (excretion C via C/N; product C via `%C`).

### Products — `df_prod`
| Column | Unit | Definition / Role | Provenance / Formula |
|---|---|---|---|
| **C/N** | – | Carbon-to-nitrogen ratio | `Carbon Content (%) / Nitrogen Content (%)` |
| **Production (kton)** | kton | Fresh mass production | From N layer |
| **Carbon Production (ktC)** | ktC | Product carbon | `Production * Carbon Content / 100` |
| **Carbon Wasted (ktC)** | ktC | C wasted | `Nitrogen Wasted * (C/N)` |
| **Carbon for Other uses (ktC)** | ktC | C to “other sectors” | `Nitrogen for Other uses * (C/N)` |
| **Carbon Exported (ktC)** | ktC | C exported (trade) | `Nitrogen Exported * (C/N)` |

> Exports are aggregated **by `Sub Type`** to dedicated trade nodes in the transition matrix.

### Animal Excreta — `df_excr`
| Column | Unit | Definition / Role | Provenance / Formula |
|---|---|---|---|
| **Excretion (ktC)** | ktC | Total excreta carbon (implied by humification) | `Humification / h`, with $h$ = Humification coefficient (fraction) |
| **Humification (ktC)** | ktC | Carbon stabilized in soil | `Excretion to soil (ktN) * (C/N)` |
| **Excretion to CH4 (ktC)** | ktC | Methane emissions from excreta | `Excretion * CH4 EM (fraction)` |
| **Excretion to CO2 (ktC)** | ktC | CO₂ from excreta | `Excretion * (1 - CH4 EM - h - θ^E)` |
| **Excretion to Energy (ktC)** | ktC | Carbon routed to energy inputs | `Excretion to Energy (ktN) * (C/N)` |

> $θ^E$ is the share sent to energy in **C terms** (inherited from the N routing and C/N conversions).

### Population — `df_pop`
| Column | Unit | Definition / Role | Provenance / Formula |
|---|---|---|---|
| **Humification (ktC)** | ktC | Carbon stabilized in soil | `Excretion after vol. (ktN) * (C/N)` |
| **Excretion to CH4 (ktC)** | ktC | Methane emissions from human excreta | `Excretion * CH4 EM (fraction)` |
| **Excretion to CO2 (ktC)** | ktC | CO₂ from human excreta | `Excretion * (1 - CH4 EM - h)` |
| **Respiration (ktC)** | ktC | CO₂ from human respiration (mass balance) | `max(Ingestion - Excretion, 0)` |

### Energy facilities — `df_energy`

The Carbon layer derives per‑facility **carbon input** (via allocated N and C/N), **CH₄ carbon** (for methanizers using $E_f$), and **digestate carbon** (methanizers only). These are routed in the **transition matrix** rather than stored as columns.

---

## 6. Diagnostics & safeguards

- **Respiration clipping.** Negative respiration values (due to inconsistent inputs) are clipped to zero with a **warning** detailing flows and a **critical C/N** hint per excreta stream.  
- **Balance check.** `check_balance()` prints incoming/outgoing sums by sector to verify matrix consistency.

## 7. Notes on scope and assumptions

- Emission factors (humification, CH₄, process CO₂ shares) are **aggregated**: users may assemble finer‑scope sub‑factors upstream.  
- For methanizers, **all N** entering exits as **digestate N** in the Nitrogen layer; the Carbon layer mirrors this assumption by routing **digestate C** (input C - methan production - CO2 losses) to **soil stock**.  
- For bioraffineries, no digestate is produced: all non‑CO₂ C goes to **hydrocarbures**.