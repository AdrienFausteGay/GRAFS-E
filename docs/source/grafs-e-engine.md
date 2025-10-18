# GRAFS-E Methodology

Here are described GRAFS-E mecanisms to create a coherent flow vision of agro-food systems. Flow building are described by main flow families.

## Input Products flows

Input flows from products are get with product data. For each product $i$ with 'Origin compartment' $j$, the inflow is: 
```{math}
F_{ij} = \text{Production}_i * \text{Nitrogen Content}_i
```

This flow is now nammed $\text{Nitrogen Production}_i$
## Crops input fertilization flows

Crops can be fertilized by several vector depending of the context : synthetic fertilizer, excretion, atmospheric deposition, seeds...

### Seeds input

Each yearly crops required an initial seeds input to grow. This is computed as follow for a crop $i$: 
```{math}
F_{ii} = \text{Seed input}_i * \text{Nitrogen Production}_i
```

This is the only self loop in GRAFS-E.

### Atmosperic deposition

Atmospheric deposition is modelled in a simple maner with a surface coefficient. The origin of this flow is 90 % from atmospheric NH3 and 10 % from atmospheric N2O as proposed by IPCC {cite:p}`intergovernmentalpanelonclimatechangeClimateChangeLand2022`.
For a crop $i$, the input flow is: 
```{math}
F_i = \text{Atmospheric deposition coef}_i * \text{Area}_i
```

### Human excretion

Human excretion is the sum of all Nitrogen Ingested including fischery product.

#### Sludge spread

The excretion is partially recycled (Excretion recycling input parameter) to be spread on crops. This part is shared among crops according to their area and spreading rate.
Spreading designate the action of manually spreading manure on a field (by opposition to direct excretion on grasslands). A streading rate of 0% means that no hectares benefited from sludges, manure and slurry (whatever is the amount). Spreading rate of 100% means that all hectares of this crop benefited from spreeding. The available amount of sludge Nitrogen is spread using this distribution: 
```{math}
{\rho}_i = \frac{\text{Area}_i*\text{Spreading rate_i}}{{\sum}_{j\in \text{crops}, \notin \text{natural meadow}} \text{Area_j}*\text{Spreading rate_j}}
```

With $i$ a crop. Therefore Spreading rate encompass how much a crop will get access to manure, sludge and slurry spreading. 

#### Other fate of human excretion

The rest of human excretion get various fate. These are studied in {cite:p}`starckFateNitrogenFrench2023a`. A part is volatilized as ammoniac ($NH_3$), a part is volatilized as nitrous oxyde ($N_2O$), a part is volilized as nitrous gas ($N_2$). Each of these ends are taken account by 'N-X EM excretion' with X the name of the volatilized molecule in input data.
All human excretion which is neither volatilized or recycled is lost in hydro-system compartment.

### Animal excretion

Animal excretion is the sum of ingestion (feed) and animal production (edible, non edible and dairy products). Animal spend time indoor (Excreted indoor parameter) or in grasslands (natural meadows and temporary meadows categories).

#### Animal indoor excretion

The mecanisms for Indoor animal excretion are very similar than human excretion. The same distribution is used for manure and slurry spreading.
Two excretion managments are considered in GRAFS-E : slurry and manure. Their have the same mecanisms but with differents technical coefficients. Manure has its own volatilization coefficient, as slurry. 
Another difference with human excretion is that there is no losses in hydro-system compartment. All available manure and slurry is used on crops after volatilization.

### Animal excretion on grasslands

The proportion of time spend outdoor (1 - Excreted indoor) is the proportion of excretion on grasslands crops. This kind of ecretion has its own volatilization coefficients. The distribution of outdoor excretion is simply proportional to grasslands area.

### Biological Nitrogen Fixation

We compute BNF following Anglade et al.{cite:p}`angladeRelationshipsEstimatingN2`:
```{math}
\mathrm{BNF}\;=\;\bigl(\alpha_{\text{cult}}\cdot \frac{Y}{HI} + \beta_{\text{cult}}\bigr)\cdot BGN
```

With ${\alpha}_{\text{cult}}$ and ${\beta}_{\text{cult}}$ are crop specific slope and intercept of the affine function (see Anglade for values), Y the yield (kgN/ha), HI the Harvest Index and BGN the Bellowground contribution multiplicative factor. For more details and  values for ${\alpha}_{\text{cult}}$ and ${\beta}_{\text{cult}}$ and BNG, check Anglade paper.

For reference, the total plant N (harvest + residues + roots) is:
```{math}
Y_{\mathrm{NPP}} \;=\; Y\cdot\frac{BGN}{HI}
```
These flows are recorded from atmospheric $N_2$ to the corresponding legume crop nodes.

### Residues, roots and "legume legacy"

#### Non-legume crops
We assume a neutral residue balance from one year to the next (residues left ≈ residues received). Consequently, residues and roots are not explicitly exchanged between crops in the N balance (they cancel out at yearly scale).

#### Legume crops
Legumes enrich the system. We compute residue Nitrogen and root Nitrogen from the harvested Nitrogen $Y$:
```{math}
N_{\text{residues}} \;=\; \frac{Y}{HI} - Y,
\qquad
N_{\text{roots}} \;=\; \frac{Y}{HI}\cdot(BGN-1).
```

Priority of fertilisation uptake is organic N first (seed + BNF + organic inputs), then mineral N to meet remaining needs.

Roots are routed to soil stock (slow pool).
Shoot residues (and any organic surplus) form a legume legacy that is transferred to the pool of cereals crops (e.g., allocated by cereal area share).

If BNF plus other non-symbiotic inputs (seeds, excretions etc.) are insufficient to close the plant N balance (harvest + residues + roots), we add an explicit soil uptake flow to close mass balance (representing the use of background soil mineral N).

Note: Permanent grasslands do not pass a legacy to the next crop; their residues and organic surplus go to soil stock (storage), consistent with their role as long-lived pools.

Second Note: Haber Bosch inputs for grasslands (natural and temporary) are computed before Residues and Roots managment but after for non legume crops.

### Synthetic Fertilization

#### Concept

Synthetic fertilization flows are the flows from Haber-Bosch compartment to each crop using synthetic fertilization. Crop category 'leguminous' are excluded from this mecanism.
Synthetic fertilization computation has two alternative mecanisms : 
- Nitrogen balance
- Standard Use

#### Computation by Nitrogen Crop Balance
Synthetic fertilization can be computed based with the gap between fertilization needs and non-synthetic fertilizer used. This gap is normalized using the total amount of synthetic fertilizer used on the territory.  Non-synthetic fertilizer regroup all fertilizing vectors presented above. 

The philosophy behind the flows of synthetic fertilizer use is to proceed to an allocation from the stock of global use of synthetic fertilizer on the territory for a given year to each crops. The distribution of synthetic fertilizer is proportional to the gap between fertilization needs and non-synthetic fertilizer use. 
```{math}
F_{\text{synth tot}} = \sum_{i \in \text{crops}}f_{HB,i} \quad
F_{HB,i} \propto \text{fertilization need}_i - \sum_{j \in \text{non synth fertilization}} F_{ji}
```

##### Mecanism

Total fertilization needs (synthetic + non synthetic) are computed by surface unit. They can be given directly as a constant. This is 'Surface Fertilizer Need (kgN/ha)' in input data. Fertilization needs can also be given by production unit. This is 'Fertilization Need (kgN/qtl)'. The fertilization need is obtain by multiplying by the Yield (qtl/ha). The gap between fertilizer need and surface non synthetic fertilizer use gives the raw synthetic fertilizer use. 
```{math}
N^{i}_{\text{synth input}}=Y^i\rho^{i}_{\text{input}}-N_{\text{other input}} \quad
N^{i}_{\text{synth input}}=N^{i}_{\text{input}}-N_{\text{other inputs}}
```

#### Computation with Standard Use

Synthetic Fertilization can be based on synthetic fertilizer use data as 'Raw Surface Synthetic Fertilizer Use (kgN/ha)' input data of crops tab. The final use of synthetic fertilizer is then normalized to fit 'Total Synthetic Fertilizer Use on crops (ktN)' and 'Total Synthetic Fertilizer Use on grasslands (ktN)' in global input tab (see nest section).

#### Normalization

With $N^{i}_{\text{synth input}}$ denotes the raw compuation of synthetic fertilizer input per hectare for crop $i$, $Y^i$ signifies the yield of crop i, $\rho^{i}_{\text{input}}$ represents fertilization d need per unit of yield (kgN/qtl) and $N_{\text{other inputs}}$ refers to the surface nitrogen input from other sources.
This is a theorical value get with a agronomist computation. Yet the raw synthetic fertilizer must fit the total synthetic fertilization use given in input data. Therefore, the raw values are normalized using this compuation:
```{math}
\Gamma \sum_i N^{i}_{\text{synt input}}\frac{S_i}{S}=N^{\text{input data}}_{\text{synth input}}
```
Where $\Gamma$ is the normalization constant to go from raw values to adjusted values. Then we have:
```{math}
f^{\text{adj}}_{HB,i} = f^{\text{raw}}_{HB,i}\Gamma
```
The flows get after the normalization are called 'Adjusted Total Synthetic Fertilizer Use (ktN)'. The computation steps for synthetic fertilizer use are summed up in this figure

![Computation steps for synthetic fertilizer flows](_static/calcul_gamma.jpg)

#### Volatilization Losses

In addition to flows from Haber-Bosch process to crops, GRAFS-E compute Nitrogen leaks from crops to atmospheric ammoniac and nitrous oxyde. These flows are computed using 'coefficient N-NH3 volatilization synthetic fertilization (%)' and 'coefficient N-N2O volatilization synthetic fertilization (%)' input data.

#### $N_2$ and the Haber–Bosch compartment

All **synthetic fertiliser** supplied to crops originates from the **Haber–Bosch** compartment. Every fertiliser flow to a crop has its source in this compartment.

To conserve nitrogen, the Haber–Bosch compartment is itself **mass‑balanced** by an inflow from **atmospheric $N_2$**. In other words, the total outflow of nitrogen from Haber–Bosch equals the nitrogen extracted from atmospheric $N_2$ for ammonia synthesis including volatilization losses.

## Feed–Food–Methanizer Allocation Model (GRAFS‑E)

### Philosophy

Linking **food/feed demand** to a **physically coherent** system of flows (local production, imports, exports, human/animal use, methanizer) requires more than a simple mass balance. Detailed information on *who uses which product*, and *gross imports/exports by product*, is often **incomplete** or **heterogeneous**.  
GRAFS‑E closes this gap with a **linear optimization model** (PuLP){cite:p}mitchellPuLPLinearProgramming that allocates nitrogen (N) from products to consumers **under hard constraints** (balances, availabilities, prohibitions) while **minimizing** **diet deviations**, **distribution imbalances**, **gross imports**, and **energy target deviation** for the methanizer.

---

### Notation (mapping to code)

| Concept | Meaning | In the code |
|---|---|---|
| Products $p$ | Rows of `df_prod` (e.g., *Wheat grain*, *Soya beans grain*) | `df_prod.index` |
| Consumers $c$ | Rows of `df_cons` (livestock & population) | `df_cons.index` |
| Diets | Table `diets` by `Consumer`, with product groups and proportions | `pairs` |
| Local allocation | N from product $p$ to $c$ | `x_vars[(p,c)] ≥ 0` |
| Imports | N imported of product $p$ to $c$ | `I_vars[(p,c)] ≥ 0` |
| Diet deviation | Slack per diet group of $c$ | `delta_vars[(c, group)] ≥ 0` |
| Intra‑group penalty | Keep distribution within a group balanced | `penalite_culture_vars[(c,prop,p)] ≥ 0` |
| Fair share across consumers | For each product, stay near a target split | `gamma_fair_abs[(p,c)] ≥ 0` |
| Export surplus | Unused local N (not to methanizer) | `U_vars[p] ≥ 0` |
| No‑swap (optional MILP) | Forbid importing when exporting same product | `y_vars[p] ∈ {0,1}` |
| Methanizer (products) | N from products to methanizer | `x_meth_prod[p] ≥ 0` |
| Methanizer (excreta) | N from excreta to methanizer | `x_meth_excr[e] ≥ 0` |
| Methanizer (waste) | Aggregated waste N to methanizer | `N_waste_meth ≥ 0` |
| Diet deviation (meth) | Slack per methanizer diet group | `delta_meth[(Meth, group)] ≥ 0` |
| Fair share (meth, products) | Per‑product target vs allocation | `gamma_fair_abs_meth[p] ≥ 0` |
| Intra‑group (meth) | Balanced distribution within meth groups | `penalite_culture_meth[(Meth,prop,it)] ≥ 0` |
| Energy deviation | Distance to energy target | `meth_energy_dev ≥ 0` |

---

### Inputs

- **Local production by product** (ktN): `"Available Nitrogen Production (ktN)"` after losses and other uses.
- **Consumers**: `df_cons["Type"] ∈ {Human, Livestock}`, `df_cons["Ingestion (ktN)"]`.
- **Diets**: `diets` with columns `Consumer`, `Proportion`, `Products` (list). Preprocessed as  
  `pairs = [(consumer, prop, tuple(products)), ...]`.
- **Global parameters/weights** (from `df_global`):  
  `Weight diet`, `Weight distribution`, `Weight import`, `Weight fair local split`,  
  `Methanizer Energy Production (GWh)`, `Weight methanizer production`, `Weight methanizer inputs`.
- **Excreta**: `df_excr` (e.g., `Excretion after volatilization (ktN)`, `Nitrogen Content (%)`, `Methanization power (MWh/tFW)`).
- **Waste→methanizer**: `df_global["Green waste methanization power (MWh/ktN)"]`.

---

### Decision variables (main blocks)

- **Local allocations**: $x_{p,c} \ge 0$.  
- **Imports**: $I_{p,c} \ge 0$.  
- **Diet slacks**: $\delta_{c,G} \ge 0$.  
- **Intra‑group slacks**: $\pi_{c,G,p} \ge 0$.  
- **Fair‑share slacks**: $\gamma_{p,c} \ge 0$.  
- **Export surplus**: $U_p \ge 0$.  
- **No‑swap binary**: $y_p \in \{0,1\}$.

**Methanizer:**  
- $x^{meth}_{p} \ge 0$ (products), $x^{meth}_{e} \ge 0$ (excreta), $N^{meth}_{waste} \ge 0$.  
- $\delta^{meth}_G \ge 0$, $\gamma^{meth}_p \ge 0$, $\pi^{meth}_{G,it} \ge 0$, $meth\_energy\_dev \ge 0$.

---

### Hard constraints

#### (H1) Consumer balance
Each consummer (except methanizer) has a fixed amount of Nitrogen intake. For each consumer $c$:
```{math}
\sum_{p \in \text{diet}(c)} x_{p,c} \;+\; \sum_{p \in \text{diet}(c)} I_{p,c} \;=\; \text{Ingestion}_c.
```

#### (H2) Product availability (balance)
It is not possible to use more product than available. For each product $p$:
```{math}
\sum_{c} x_{p,c} \;+\; x^{meth}_{p} \;+\; U_p \;=\; \text{AvailableProd}_p.
```

#### (H3) Excreta availability
It is not possible to use more excretat than available. For each excreta $e$:
```{math}
x^{meth}_{e} \;\le\; \text{ExcretionAfterVol}_e.
```

#### (H4) No imports of natural meadows
Products with 'grazing' as Sub Type cannont be traded (imported of exported).
```{math}
\sum_{c \in \text{Animals}} \sum_{p \in \text{grazing}} I_{p,c} \;=\; 0.
```

#### (H5) Enforce animal share (optional)
If enabled, the model cannont substitute animal product and plant product. For each $c$:
```{math}
\sum_{p \in \text{diet}(c)\cap \text{Type=animal}} (x_{p,c}+I_{p,c})
= \text{Ingestion}_c \times \text{share_animal}(c).
```

#### (H6) Optional anti “import & surplus” (MILP)
With product‑specific $M_p$:
```{math}
\sum_{c} I_{p,c} \le M_p\,y_p,\qquad U_p \le M_p\,(1-y_p).
```
> Either **import** or **export surplus** for product $p$, not both.

---

### Soft constraints (all linearized)

#### (S1) Diet deviation per group (consumers)
This soft constraint limit the gap between the diet of input data and the diet made by the allocation model. For any group $G$ in $c$'s diet (target share $prop_{c,G}$):
```{math}
\Big|\frac{\sum_{p\in G} (x_{p,c}+I_{p,c})}{\text{Ingestion}_c} - prop_{c,G}\Big| \;\le\; \delta_{c,G}.
```

#### (S2) Intra‑group balance (consumers)
Group $G$ of size $|G|$. Target per item:
```{math}
\text{target}_{c,G,p}=\frac{prop_{c,G}\cdot \text{Ingestion}_c}{|G|},\qquad
\big| (x_{p,c}+I_{p,c}) - \text{target}_{c,G,p} \big| \le \pi_{c,G,p}.
```
*Avoids corner solutions (one item taking all).*

#### (S3) Fair‑share across consumers (per product)
Build normalized shares $s^{ref}_{p,c}$ from all diets (by product). Target:
```{math}
x^\star_{p,c} = s^{ref}_{p,c}\cdot \text{Prod}_p,\qquad
|x_{p,c} - x^\star_{p,c}| \le \gamma_{p,c}.
```
*Prevents a product from being monopolized by one consumer.*

---

### Methanizer

#### Energy conversion

For nitrogen $N$ (in **ktN**) sent to the methanizer:

- **Products**:  
  $\text{MWh/ktN}=\dfrac{\text{MWh/tFW}\times 1000}{\%N}$,  
  so $E_{\text{GWh}} = N \times \text{MWh/ktN} / 1000$.
- **Excreta**: same formula using entries from `df_excr`.
- **Waste**: use `Green waste methanization power (MWh/ktN)` (already per ktN).

Total:
```{math}
E_{\text{GWh}}=\frac{1}{1000}\!\left(\sum_{p} x^{meth}_{p}\!\cdot\!\frac{\text{MWh/tFW}\cdot 1000}{\%N_p}
+ \sum_{e} x^{meth}_{e}\!\cdot\!\frac{\text{MWh/tFW}\cdot 1000}{\%N_e}
+ N^{meth}_{waste}\!\cdot\!\text{MWh/ktN}_{waste}\right).
```

#### Diet & balance constraints for methanizer

- **Diet groups (linear, no division)**  
  For any meth group $G$ with target $prop^{meth}_G$:
  ```{math}
  \big|N^{group} - prop^{meth}_{G}\cdot N^{total}_{meth}\big| \le \delta^{meth}_{G},
  ```
  where $N^{group}$ is the sum of allocations in $G$, and
  $N^{total}_{meth} = \sum_p x^{meth}_p + \sum_e x^{meth}_e + N^{meth}_{waste}$.

- **Fair‑share on products (meth vs other consumers)**  
  For each product $p$:
  ```{math}
  |x^{meth}_{p} - s^{ref,meth}_{p}\cdot \text{Prod}_p| \le \gamma^{meth}_{p}.
  ```

- **Intra‑group balance (meth)**  
  For any meth group $G$ of size $|G|$:
  ```{math}
  \big| \text{alloc}_{it} - \frac{prop^{meth}_G \cdot N^{total}_{meth}}{|G|} \big| \le \pi^{meth}_{G,it}.
  ```

- **Energy target (relative deviation, linear)**  
  With target $E^{target}$:
  ```{math}
  meth\_energy\_dev \ge \frac{|E_{\text{GWh}} - E^{target}|}{E^{target}}.
  ```

- **Excreta cap (hard)**  
  $x^{meth}_{e} \le \text{ExcretionAfterVol}_e$.

> **Mass balance remark**: the methanizer **does not change** the total N returned to the system (excreta+digestate stay constant); it only **changes the origin** of these flows. Reporting distinguishes `Excretion to Methanizer` and `Excretion to soil`.

---

### Import penalty (normalized)

To steer the model away from **bulk imports**, we penalize **gross imports** with a **normalized** term:
```{math}
\text{ImportTerm} \;=\; \frac{\sum_{p,c} I_{p,c}}{N^{scale}},
```
where $N^{scale}$ is a **constant** (e.g., a proxy of territory‑level deficit computed from inputs).  
This keeps the model **linear** and makes the weight comparable to other terms.

---

### Objective function

We minimize a non‑negative weighted sum:
```{math}
\min \;
\underbrace{\omega_{dev}\,\sum \delta}_{\text{diet deviations}}
+\underbrace{\omega_{cult}\,\sum \pi}_{\text{intra‑group balance}}
+\underbrace{\omega_{imp}\,\text{ImportTerm}}_{\text{normalized gross imports}}
+\underbrace{\omega_{fair}\,\tfrac{\sum \gamma + \sum \gamma^{meth}}{|\text{Products}|}}_{\text{fair‑share across consumers}}
+\underbrace{\omega_{methE}\,meth\_energy\_dev}_{\text{meth energy target}}
+\underbrace{\omega_{methD}\,\tfrac{\sum \delta^{meth}}{|\text{Meth groups}|}}_{\text{meth diet}}
+\underbrace{\tfrac{\omega_{cult}}{|\text{Meth items}|}\,\sum \pi^{meth}}_{\text{meth intra‑group}}.
```

Normalizing by counts (products, groups) stabilizes weights across data granularity.

---

### Choosing weights (guidelines)

- `Weight import` ↑ → favors **local autonomy** (e.g., 19th century scenarios).  
- `Weight diet` ↑ → keeps diets close to targets (less substitution).  
- `Weight distribution` ↑ → prevents corner solutions within groups.  
- `Weight fair local split` ↑ → fair split of each product across consumers.  
- `Weight methanizer production` ↑ → meet energy target.  
- `Weight methanizer inputs` ↑ → respect meth diet.

`Weight distribution` and `Weight fair local split` are technical constraints weights. They should be kept lower than others. Thez can be not given in input data. A value will be automatically assigned to them based on the other weights (see input data page).
Start with something like `(1.0, 1.0, 0.1, 0.1, 1.0, 1.0)` and run small sensitivity sweeps ×{0.5, 1, 2, 5}.  
Inspect the **debug report** (shares %) to see which term dominates and adjust.

---

### Debug & diagnostics (post‑solve)

If debug = True when NitrogenFlowModel is called, the code reconstructs each objective component and prints:
- **weighted contributions** per term,
- **shares %**,
- **solver status**.

Use this to:
- spot unexpected dominance of one term,
- verify soft constraints are active (non‑zero slacks),
- tune weights.

---

### Implementation notes

- Sets/pairs built from `diets` ensure only **valid** `(product,consumer)` variables exist.
- Fair‑share targets per product computed from **normalized** diet shares across all consumers.

---

### Minimal example

A minimal runnable example with synthetic data is available in **`example/`**:
- `example/GRAFS_project_example_N.xlsx` — Project file,
- `example/GRAFS_data_example_N.xlsx` — Data file.

> You can replicate/extend the example to test different weight sets (e.g., high `Weight import` to emulate 19th‑century import frictions).

## Export

Export is deduced with mass balance on each product $i$:
```{math}
N_{\text{export}}^i = N_{\text{local production}}^i + N_{\text{import}}^i - N_{\text{consumption}}^i
```
Each term on the right side of the equation is given by input data or optimization model.

## Final checks

GRAFS‑E performs **mass‑balance checks** for every crop compartment. For all other compartments, balance is enforced by the model’s mechanistic definitions.

- **Deficit (outputs > inputs):**  
  If the total nitrogen outputs of a crop compartment exceed its inputs, the deficit is taken from the **soil stock** compartment. This may indicate (i) an unsustainable agro‑food system, (ii) difficulties in closing loops at the system scale, or (iii) inconsistencies in the input data.

- **Surplus (inputs > outputs):**  
  If the total nitrogen inputs of a crop compartment exceed its outputs, GRAFS‑E distributes the nitrogen surplus is partitioned as:  
     - **99.25 %** to **hydrosystem losses**,   
     - **0.75 %** to **atmospheric $N_2O$**.

All **crop, livestock, population, and product** compartments are balanced by construction. Once these checks pass, the GRAFS‑E nitrogen‑flow representation of the agro‑food system is **closed and physically coherent**.

Then, for each crop compartment, inward and outward flows to “soil stock” are netted out; only the resulting net exchange with “soil stock” is retained in the accounting.

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
:labelprefix: M