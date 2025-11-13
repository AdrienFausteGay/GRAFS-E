# GRAFS‑E — Nitrogen Model (2025) — Scientific Reference (v4)

This page documents the nitrogen mechanisms used in GRAFS‑E, with a rigorous, pedagogical progression:
1) **Symbols & mapping** to input/output tables,  
2) **Fertilisation vectors** (closed‑form formulas),  
3) **Crop nitrogen balance** at territory level,  
4) **Compartment mechanisms** (products, livestock, excreta, population, energy),  
5) **Historical vs Prospective crop sub‑modules**,  
6) **Central LP/MILP allocation model (Pulp)**: variables, constraints (hard/soft), and objective.

All equations use ```{math}``` blocks. Greek letters in text are written as LaTeX inline (e.g., $\gamma$).

---

## 0 — Symbols and mapping to inputs/outputs

| Symbol | Meaning | Input sheet / column (example) | Unit | Output column (example) |
|---|---|---|---|---|
| $A_c$ | Crop area | `crops` / `Area (ha)` or `Input data`→`Area (ha)` | ha | — |
| $y_c$ | Yield (fresh mass per ha) | [H] observed; [P] endogenous via curve | tFW/ha | — |
| $HI_c$ | Harvest index | `crops` / `Harvest Index` | — | — |
| $\alpha_c,\beta_c$ | BNF parameters | `crops` / `BNF alpha`, `BNF beta` | kgN/tFW, kgN/ha | — |
| $BGN_c$ | Below‑ground factor | `crops` / `BGN` | — | — |
| $N^{\mathrm{dep}}_c$ | Atmospheric deposition (to field) | `global` / `Atmospheric deposition coef (kgN/ha)` | ktN (after ha→ktN) | `Atmospheric deposition (ktN)` |
| $N^{\mathrm{seed}}_c$ | Seeds N to field | `crops` / `Seed input (ktN/ktN)` × harvested ktN | ktN | `Seeds Input (ktN)` |
| $N^{\mathrm{excr}}_c$ | Excreta to field (effective) | from livestock/excreta module | ktN | `Excreta Fertilization (ktN)` |
| $N^{\mathrm{digest}}_c$ | Digestate to field (effective) | from energy module | ktN | `Digestat Fertilization (ktN)` |
| $N^{\mathrm{syn}}_c$ | Total synthetic N allocated to crop $c$ (territory) | computed per mode (H/P) | ktN | `Adjusted Total Synthetic Fertilizer Use (ktN)` |
| $k_{\mathrm{NH3}}$ | Aggregated NH$_3$ factor for synthetic | `global` / `coefficient N-NH3 volatilization synthetic fertilization (%)` | — (fraction) | `Volatilized Nitrogen N-NH3 (ktN)` |
| $k_{\mathrm{N2O}}$ | Aggregated direct N$_2$O factor for synthetic | `global` / `coefficient N-N2O emission synthetic fertilization (%)` | — (fraction) | `Volatilized Nitrogen N-N2O (ktN)` |
| $N^{\mathrm{syn,field}}_c$ | Synthetic effective to field | formula below | ktN | `Synthetic to field (ktN)` |
| $N^{\mathrm{BNF}}_c$ | Symbiotic fixation | formula below | ktN | `BNF (ktN)` |
| $N^{\mathrm{harv}}_c$ | N harvested (sum of products from crop $c$) | `prod` via origin mapping | ktN | `Harvested Production (ktN)` |
| $S_c$ | Surplus (positive part) | defined below | ktN | `Surplus (ktN)` |
| $M_c$ | Mining (positive part) | defined below | ktN | `Mining from soil (ktN)` |
| $L_c$ | Leaching to hydro‑system | territory parameter dependent | ktN | `Leached to hydro-system (ktN)` |
| $E^{\mathrm{surp}}_c$ | N$_2$O from surplus | territory parameter dependent | ktN | `Surplus N2O (ktN)` |
| $\Delta S^{\mathrm{soil}}_c$ | Net soil stock change | storage from surplus minus mining | ktN | `Soil stock (ktN)` |

> Notes. Fractions such as “share of surplus leached on arable (%)”, “share of surplus to N2O on arable (%)”, meadow‑specific fractions, and the threshold $T$ (kgN/ha stored first on meadows) are set in the `global` items at territory level.

---

## 1 — Fertilisation vectors (closed‑form)

All flows are computed at **territory** level for crop $c$ in ktN, except when a per‑ha intermediate is explicit.

### 1.1 Atmospheric deposition
```{math}
N^{\mathrm{dep}}_{c} \;=\; \mathrm{Dep}\_{\text{kg/ha}}\; \frac{A_c}{10^{6}}\,.
```

### 1.2 Seeds
Let $\rho^{\mathrm{seed}}_{c}$ be the seed coefficient in ktN per ktN of harvested N for the crop’s **main product**.
```{math}
N^{\mathrm{seed}}_{c} \;=\; \rho^{\mathrm{seed}}_{c}\;N^{\mathrm{harv}}_{c}\,.
```

### 1.3 Symbiotic fixation (historical and prospective)
Per ha, then scaled by area:
```{math}
N^{\mathrm{BNF}}\_{c,\mathrm{kg/ha}} \;=\; \Bigl(\alpha_{c}\,\frac{y_c}{HI_c} + \beta_c\Bigr)\,BGN_c\,,
\qquad
N^{\mathrm{BNF}}_{c} \;=\; N^{\mathrm{BNF}}\_{c,\mathrm{kg/ha}} \cdot \frac{A_c}{10^{6}}\,.
```

### 1.4 Synthetic fertiliser — effective to field and emissions (aggregated scope)
Let $k_{\mathrm{NH3}}$ and $k_{\mathrm{N2O}}$ be the **aggregated** factors chosen by the user for the desired scope (application, plus optionally upstream, etc.). Then:
```{math}
N^{\mathrm{syn,field}}_{c} \;=\; N^{\mathrm{syn}}_{c}\,\bigl(1 - k_{\mathrm{NH3}} - k_{\mathrm{N2O}}\bigr)\,,
```
```{math}
E^{\mathrm{NH_3}}_{\mathrm{syn},c} \;=\; N^{\mathrm{syn}}_{c}\,k_{\mathrm{NH3}}\,,\qquad
E^{\mathrm{N_2O}}_{\mathrm{syn},c} \;=\; N^{\mathrm{syn}}_{c}\,k_{\mathrm{N2O}}\,.
```

### 1.5 Excreta and digestate to field (effective)
The corresponding modules compute effective ktN flows to field (after their own aggregated factors) as $N^{\mathrm{excr}}_c$ and $N^{\mathrm{digest}}_c$; see Sections 3.2 and 3.4.

---

## 2 — Crop nitrogen balance (territory level)

### 2.1 Inputs and outputs
```{math}
N^{\mathrm{in}}_{c} \;=\; N^{\mathrm{dep}}_{c} + N^{\mathrm{BNF}}_{c} + N^{\mathrm{excr}}_{c}
+ N^{\mathrm{digest}}_{c} + N^{\mathrm{seed}}_{c} + N^{\mathrm{syn,field}}_{c}\,.
```
```{math}
N^{\mathrm{out}}_{c} \;=\; N^{\mathrm{harv}}_{c}\,.
```

### 2.2 Surplus and mining (positive parts)
```{math}
S_c \;=\; \max\!\bigl(N^{\mathrm{in}}_{c} - N^{\mathrm{out}}_{c},\,0\bigr)\,,\qquad
M_c \;=\; \max\!\bigl(N^{\mathrm{out}}_{c} - N^{\mathrm{in}}_{c},\,0\bigr)\,.
```

### 2.3 Partition of positive surplus
**Arable (non‑meadows):**
```{math}
L_c \;=\; f^{\mathrm{leach}}_{\mathrm{arable}}\,S_c,\qquad
E^{\mathrm{surp}}_c \;=\; f^{\mathrm{N_2O}}_{\mathrm{arable}}\,S_c,\qquad
\mathrm{SoilStore}^{\mathrm{surp}}_c \;=\; S_c - L_c - E^{\mathrm{surp}}_c\,.
```

**Meadows:** first store up to $T$ kgN/ha, then split the remainder:
```{math}
s^{\mathrm{kg/ha}}_c = \frac{S_c\cdot 10^6}{A_c},\quad
\mathrm{FirstStore}_c = \min(s^{\mathrm{kg/ha}}_c,\,T)\,\frac{A_c}{10^6},\quad
R_c = S_c - \mathrm{FirstStore}_c,
```
```{math}
L_c \;=\; f^{\mathrm{leach}}_{\mathrm{meadow}}\,R_c,\qquad
E^{\mathrm{surp}}_c \;=\; f^{\mathrm{N_2O}}_{\mathrm{meadow}}\,R_c,\qquad
\mathrm{SoilStore}^{\mathrm{surp}}_c \;=\; \mathrm{FirstStore}_c + R_c - L_c - E^{\mathrm{surp}}_c\,.
```

### 2.4 Net soil stock change and surface indicators
```{math}
\Delta S^{\mathrm{soil}}_c \;=\; \mathrm{SoilStore}^{\mathrm{surp}}_c - M_c\,,
```
```{math}
\mathrm{Inputs}^{\mathrm{kg/ha}}_{c} \;=\; \frac{N^{\mathrm{in}}_{c}\cdot 10^6}{A_c},\qquad
\mathrm{Surplus}^{\mathrm{kg/ha}}_{c} \;=\; \frac{S_c\cdot 10^6}{A_c}\,.
```

---

## 3 — Compartment mechanisms

### 3.1 Products
For any product $p$ originating from crop $c$ or livestock $l$:
```{math}
N^{\mathrm{prod}}_{p} \;=\; \mathrm{FM}_{p}\cdot \%N_{p}/10^6\,,
```
Availability after pre‑allocation losses (waste and other uses):
```{math}
N^{\mathrm{avail}}_{p} \;=\; N^{\mathrm{prod}}_{p}\,(1 - w^{\mathrm{waste}}_{p} - w^{\mathrm{other}}_{p})\,.
```

### 3.2 Livestock and excreta
Excreta production uses livestock units (LU), excretion factors per LU and indoor/outdoor splits to form the **streams** (e.g., manure, slurry, pasture excretion). Each stream $e$ is assigned aggregated emission factors; the **effective** ktN that can be spread or sent to energy is:
```{math}
E^{\mathrm{eff}}_{e} \;=\; E_{e}\,\bigl(1 - k^{e}_{\mathrm{NH3}} - k^{e}_{\mathrm{N_2O}} - k^{e}_{\mathrm{N_2}}\bigr)\,.
```
The allocation to crops defines $N^{\mathrm{excr}}_c = \sum_{e} a_{ec}\,E^{\mathrm{eff}}_{e}$ with area‑based eligibility (e.g., spreading rate).

### 3.3 Population
For each population node $u$, ingestion requirement (territory level) is:
```{math}
R_u \;=\; \mathrm{Inhabitants}_u \times \mathrm{Total\ ingestion\ per\ capita}_u \,.
```
Optional fish share and other splits are applied via Diets (Section 6). Human excretion recycling contributes to excreta streams analogously to livestock.

### 3.4 Energy facilities
For facility $f$, admissible inputs $i$ (products or excreta) have powers $P_{f i}$ in MWh/ktN (converted internally from MWh/tFW via $\%N_i$). Facility‑level production is:
```{math}
E_f \;=\; \sum_{i} X^{E}_{i f}\,P_{f i}\,,
```
where $X^{E}_{i f}$ are ktN sent to facility $f$. Facility diets (if specified) are treated like consumer diets but at the facility level (soft).

---

## 4 — Crop sub‑modules (Historical vs Prospective)

### 4.1 Historical (H): distributing a fixed synthetic N **for the territory**
Two alternative logics lead to $N^{\mathrm{syn}}_{c}$ while matching territory‑level totals for crops and meadows.

**H1 — Balance‑based needs (kgN/ha or per unit yield).**
```{math}
N^{\mathrm{syn,res}}_{c} \;=\; \frac{A_c}{10^6}\,b^{\mathrm{ha}}_{c} - \bigl(N^{\mathrm{dep}}_{c}+N^{\mathrm{BNF}}_{c}+N^{\mathrm{excr}}_{c}+N^{\mathrm{digest}}_{c}+N^{\mathrm{seed}}_{c}\bigr)\,,
```
then scale by $\kappa$ to satisfy crop/grassland totals $S^{\mathrm{tot}}_{\mathrm{crop}},S^{\mathrm{tot}}_{\mathrm{grass}}$:
```{math}
N^{\mathrm{syn}}_{c} \;=\; \kappa\,\max(N^{\mathrm{syn,res}}_{c},\,0)\,.
```

**H2 — Standard per‑ha usage.**
```{math}
N^{\mathrm{syn,raw}}_{c} \;=\; u_c\,\frac{A_c}{10^6},\qquad
N^{\mathrm{syn}}_{c} \;=\; \kappa\,N^{\mathrm{syn,raw}}_{c}\,.
```

### 4.2 Prospective (P): endogenous yields via convex response (non‑legumes)
```{math}
Y_c(F) \;=\; Y^{\max}_{c}\,\bigl(1 - e^{-F/F^\star_{c}}\bigr)\,,
\qquad
f_c=\sum_i \lambda_{i c} F_{i c},\quad
y_c=\sum_i \lambda_{i c} Y_{i c},\quad
\sum_i \lambda_{i c}=1,\ \lambda_{i c}\ge0\ \text{contiguous.}
```
Legumes enforce $N^{\mathrm{syn}}_{c}=0$. $N^{\mathrm{BNF}}_{c}$ uses the same expression as in H (with endogenous $y_c$).

---

## 5 — Central LP/MILP allocation model (Pulp)

We define separate symbols for **imports** ($M$) and **ingestion requirements** ($R$) to avoid ambiguity. Consumers are $u$ (livestock groups or population nodes).

### 5.1 Decision variables
- $X_{p u}\ge 0$: ktN of product $p$ allocated to consumer $u$.
- $M_{p u}\ge 0$: ktN of product $p$ **imported** to consumer $u$ (territory boundary).
- $Z_{p}\ge 0$: ktN of product $p$ exported/unused.
- $X^{E}_{i f}\ge 0$: ktN of input $i$ to facility $f$.
- $\delta_{uG}\ge 0$: deviation from target share for consumer $u$, group $G$.
- $\pi_{uGp}\ge 0$: intra‑group smoothing deviation.
- $\gamma_{p u}\ge 0$: fair‑share deviation.
- [P] $d^{\mathrm{syn}}\ge 0$: excess synthetic ktN beyond budget (territory).
- [P] $d^{F^\star}_c\ge 0$: per‑crop deviation from $f_c/F^\star_c=1$ (non‑legumes).

### 5.2 Hard constraints (with interpretation)

#### (H‑1) Product mass balance (territory)
```{math}
\sum_{u} X_{p u} + \sum_{f} X^{E}_{p f} + Z_p \;\le\; N^{\mathrm{avail}}_{p}\,,\qquad \forall p.
```
All outgoing allocations plus exports cannot exceed availability.

#### (H‑2) Consumer ingestion requirement
```{math}
\sum_{G}\sum_{p\in\mathcal{P}_G} \bigl(X_{p u}+M_{p u}\bigr) \;=\; R_u\,,\qquad \forall u.
```
Each consumer receives exactly its territory‑level requirement $R_u$.

#### (H‑3) No imports for grazing products
```{math}
M_{p u} \;=\; 0 \quad \text{if } p\in\texttt{grazing}.
```
Grazing is local to the territory’s grasslands.

#### (H‑4) Excreta capacity to energy
```{math}
\sum_{f} X^{E}_{e f} \;\le\; E^{\mathrm{eff}}_{e}\,,\qquad \forall e\in\mathcal{E}.
```
Inputs to energy from each excreta stream are capped by effective availability.

#### (H‑5) Facility energy production identity
```{math}
E_f \;=\; \sum_{i} X^{E}_{i f}\,P_{f i}\,,\qquad \forall f.
```
Energy output equals mass inputs times their powers.

*(Optional modeling aids such as anti‑swap binaries are omitted from the hard set for clarity.)*

### 5.3 Soft constraints (definitions feeding the objective)

#### (S‑1) Diet share respect (consumers)
For each consumer $u$ and group $G$ with target share $\pi^{\star}_{uG}$,
```{math}
\left|\frac{\sum_{p\in \mathcal{P}_G}(X_{p u}+M_{p u})}{R_u} - \pi^{\star}_{uG}\right| \;\le\; \delta_{uG}\,.
```

#### (S‑2) Intra‑group smoothing
```{math}
\left|X_{p u} - \frac{1}{|\mathcal{P}_G|}\sum_{r\in\mathcal{P}_G} X_{r u}\right| \;\le\; \pi_{u G p}\,,\qquad p\in\mathcal{P}_G.
```
Prevents extreme concentration on a single item when several are nutritionally interchangeable.

#### (S‑3) Fair‑share by product
```{math}
\left|X_{p u} - s^{\mathrm{ref}}_{p u}\,N^{\mathrm{avail}}_{p}\right| \;\le\; \gamma_{p u}\,.
```
Encourages equitable distribution relative to reference patterns $s^{\mathrm{ref}}_{p u}$ (e.g., historical).

#### (S‑4) [P] Synthetic budget (territory)
Let $S^{\mathrm{tot}}_{\mathrm{crop}}$ be the crop synthetic budget. Define
```{math}
d^{\mathrm{syn}} \;\ge\; \sum_{c\in\mathcal{C}} N^{\mathrm{syn}}_{c} - S^{\mathrm{tot}}_{\mathrm{crop}}\,,\qquad d^{\mathrm{syn}}\ge 0.
```

#### (S‑5) [P] Distribution around $F^\star$ (non‑legumes)
```{math}
d^{F^\star}_c \;\ge\; \left|\frac{f_c}{F^\star_{c}} - 1\right|\,,\qquad c\notin\mathcal{C}_{\mathrm{legume}}.
```

### 5.4 Objective (annotated)
We minimize a non‑negative weighted sum of deviations and costs:
```{math}
\begin{aligned}
\min\;\; 
&\underbrace{\sum_{u,G} w^{\mathrm{diet}}\;\delta_{uG}}_{\text{respect of diet shares}} \;+\;
\underbrace{\sum_{u,G,p} w^{\mathrm{intra}}\;\pi_{uGp}}_{\text{intra-group smoothing}} \;+\;
\underbrace{\sum_{p,u} w^{\mathrm{fair}}\;\gamma_{p u}}_{\text{fair-share vs reference}} \\[4pt]
&+\; \underbrace{\sum_{p,u} w^{\mathrm{imp}}\;\frac{M_{p u}}{N^{\mathrm{avail}}_{p}+\varepsilon}}_{\text{imports (normalized)}} \;+\;
\underbrace{\sum_{f}\Bigl( w^{E}\,\mathrm{dev}^E_f + \sum_{G} w^{E\text{-diet}}\,\delta^{f}_{G} + \sum_{G,i} w^{E\text{-intra}}\,\pi^{f}_{G,i} \Bigr)}_{\text{energy targets and facility diets}} \\[4pt]
&+\; \underbrace{w^{\mathrm{syn}}\;d^{\mathrm{syn}}}_{\text{[P] synthetic budget}} \;+\;
\underbrace{w^{\mathrm{syn\text{-}dist}}\;\sum_{c\notin \mathcal{C}_{\mathrm{legume}}} d^{F^\star}_c}_{\text{[P] distribution around }F^\star}\,.
\end{aligned}
```
**Imports term.** Normalizing by $N^{\mathrm{avail}}_p+\varepsilon$ avoids penalizing scarce items disproportionately; the weight $w^{\mathrm{imp}}$ expresses a preference for internal sourcing within the territory while keeping feasibility when internal supply is insufficient.

---

## 6 — Notes on emissions scope

Emission coefficients ($k_{\mathrm{NH3}}$, $k_{\mathrm{N2O}}$, etc.) are **aggregated**: users can compose them from sub‑factors (storage, application, upstream manufacturing) to match a chosen scope. The balances and the allocation model only require the **effective** factors; reporting can still disaggregate emissions ex‑post if needed.
