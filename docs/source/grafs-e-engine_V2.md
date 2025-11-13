# GRAFS-E Methodology

This document details the **nitrogen mechanisms** of GRAFS-E with explicit **assumptions** and **rationale for the chosen formulations**. We distinguish features that apply to **Historical [H]**, **Prospective [P]**, or **Both [B]**. Carbon is documented elsewhere.

---

## 0. Notation

- Indices: $a$ (Area), $t$ (Year), $c$ (crop), $p$ (product), $l$ (livestock), $e$ (excreta), $h$ (population), $f$ (facility), $G$ (diet group).
- Sets: $\mathcal{C}$ (crops), $\mathcal{P}$ (products), $\mathcal{L}$ (livestock), $\mathcal{E}$ (excreta), $\mathcal{F}$ (facilities), $\mathcal{G}$ (diet groups).
- Units: ktN (nitrogen masses), kgN/ha (areal terms), tFW (fresh mass). Percentages are fractions in $[0,1]$ unless stated otherwise.

---

## 1. Unified crop nitrogen balance (GRAFS‑style)

The balance is computed **per crop** and **per year**. Organic and mineral fertilization are **not distinguished** in the closure itself; roots and residues are tracked internally but **not counted as outputs** when computing the surplus. All flows are reported in ktN unless stated otherwise.

### 1.1 Components

- Synthetic fertilizer to field after pre‑field losses:
```{math}
N^{\mathrm{syn,field}}_{c} \;=\; N^{\mathrm{syn}}_{c}\,\bigl(1 - \phi^{\mathrm{loss}}_{\mathrm{syn}}\bigr)
```
where $\phi^{\mathrm{loss}}_{\mathrm{syn}}$ is an **aggregated** loss fraction (e.g., volatilization + direct N$_2$O + any other scope the user decides to include). Emissions can be reported with dedicated aggregated factors, e.g.
```{math}
E^{\mathrm{NH_3}}_{\mathrm{syn}} \;=\; N^{\mathrm{syn}}_{c}\,\phi_{\mathrm{NH_3}}\,,\qquad
E^{\mathrm{N_2O}}_{\mathrm{syn}} \;=\; N^{\mathrm{syn}}_{c}\,\phi_{\mathrm{N_2O}}\,.
```
with user‑chosen $\phi$ matching the desired scope.

- Nitrogen harvested from all plant **products** linked to crop $c$:
```{math}
N^{\mathrm{harv}}_{c} \;=\; \sum_{p\in\mathcal{P}_c} N^{\mathrm{prod}}_{p}, 
\qquad 
N^{\mathrm{prod}}_{p} \;=\; \mathrm{FM}_{p}\cdot \%N_{p}/10^6\,,
```
where $\mathcal{P}_c$ is the set of products whose origin is $c$, $\mathrm{FM}_{p}$ is fresh‑mass, and $\%N_{p}$ its nitrogen content in \%.

- Inputs to field (lumped for the balance):
```{math}
N^{\mathrm{in}}_{c} \;=\; N^{\mathrm{dep}}_{c} + N^{\mathrm{BNF}}_{c} + N^{\mathrm{excr}}_{c}
+ N^{\mathrm{digest}}_{c} + N^{\mathrm{seed}}_{c} + N^{\mathrm{syn,field}}_{c}\,.
```

- Symbiotic fixation (same formula in Historical and Prospective):
```{math}
N^{\mathrm{BNF}}_{c,\,\mathrm{kg/ha}} \;=\; 
\Bigl(\alpha_{c}\,\frac{y_{c}}{HI_{c}} + \beta_{c}\Bigr)\,BGN_{c}\,,
\qquad
N^{\mathrm{BNF}}_{c} \;=\; N^{\mathrm{BNF}}_{c,\,\mathrm{kg/ha}}\cdot A_{c}/10^6\,.
```

### 1.2 Surplus, mining, storage and leakages

- Surplus at crop scale:
```{math}
\mathrm{Surplus}_{c} \;=\; N^{\mathrm{in}}_{c} - N^{\mathrm{harv}}_{c}\,.
```

- Mining (soil impoverishment when inputs are insufficient):
```{math}
\mathrm{Mining}_{c} \;=\; \max\!\bigl(-\mathrm{Surplus}_{c},\,0\bigr)\,.
```

- Partition of **positive** surplus. Let $\mathcal{P}^{\mathrm{meadow}}\subset\mathcal{C}$ be meadows/grasslands and $A_{c}$ the area (ha).
    - **Arable (non‑meadows):** with fractions $f^{\mathrm{leach}}_{\mathrm{arable}}$ and $f^{\mathrm{N_2O}}_{\mathrm{arable}}$,
    ```{math}
    \begin{aligned}
    \mathrm{Leach}_{c} &= f^{\mathrm{leach}}_{\mathrm{arable}}\cdot \max(\mathrm{Surplus}_{c},0),\\[2pt]
    \mathrm{N_2O}^{\mathrm{surp}}_{c} &= f^{\mathrm{N_2O}}_{\mathrm{arable}}\cdot \max(\mathrm{Surplus}_{c},0),\\[2pt]
    \mathrm{SoilStore}^{\mathrm{surp}}_{c} &= \max(\mathrm{Surplus}_{c},0) - \mathrm{Leach}_{c} - \mathrm{N_2O}^{\mathrm{surp}}_{c}\,.
    \end{aligned}
    ```
    - **Meadows:** first store up to a threshold $T$ (kgN/ha), then split the **remainder** with meadow‑specific fractions:
    ```{math}
    \begin{aligned}
    s^{\mathrm{kg/ha}}_c &= \frac{\max(\mathrm{Surplus}_{c},0)\cdot 10^6}{A_c},\qquad
    s^{\mathrm{first}}_c = \min(s^{\mathrm{kg/ha}}_c,\,T),\\[2pt]
    \mathrm{FirstStore}_{c} &= s^{\mathrm{first}}_c \cdot A_c / 10^6,\\[2pt]
    \mathrm{Rem}_{c} &= \max(\mathrm{Surplus}_{c},0) - \mathrm{FirstStore}_{c},\\[2pt]
    \mathrm{Leach}_{c} &= f^{\mathrm{leach}}_{\mathrm{meadow}}\cdot \mathrm{Rem}_{c},\qquad
    \mathrm{N_2O}^{\mathrm{surp}}_{c} = f^{\mathrm{N_2O}}_{\mathrm{meadow}}\cdot \mathrm{Rem}_{c},\\[2pt]
    \mathrm{SoilStore}^{\mathrm{surp}}_{c} &= \mathrm{FirstStore}_{c} + \mathrm{Rem}_{c} - \mathrm{Leach}_{c} - \mathrm{N_2O}^{\mathrm{surp}}_{c}\,.
    \end{aligned}
    ```

- Net soil stock change:
```{math}
\Delta S^{\mathrm{soil}}_{c} \;=\; \mathrm{SoilStore}^{\mathrm{surp}}_{c} - \mathrm{Mining}_{c}\,.
```

- Surface indicators (kgN/ha), with $A_c>0$:
```{math}
\mathrm{Inputs}^{\mathrm{kg/ha}}_{c} \;=\; \frac{N^{\mathrm{in}}_{c}\cdot 10^6}{A_c},\qquad
\mathrm{Surplus}^{\mathrm{kg/ha}}_{c} \;=\; \frac{\mathrm{Surplus}_{c}\cdot 10^6}{A_c}\,.
```

A dedicated reporting can attribute emissions to NH$_3$, N$_2$O, etc., using the user‑defined aggregated factors $\phi$; this allows consistent accounting regardless of the emission scope retained.

---

## 2. Crop sub‑modules (Historical vs Prospective)

### 2.1 Historical (H): distributing a fixed national stock of synthetic N

Two alternative logics are available; both end by matching the **given** stocks for crops and grasslands.

- **H1 — Balance‑based needs (per ha or per unit yield).** Let $b^{\mathrm{ha}}_{c}$ be a need in kgN/ha (either provided, or converted from a need per yield unit using observed $y_c$). Synthetic is used as the **residual** to meet the total need after deposition, BNF, seeds and organic/digestate inputs:
```{math}
N^{\mathrm{syn,res}}_{c} \;=\; \frac{A_c}{10^6}\,b^{\mathrm{ha}}_{c} - \bigl(N^{\mathrm{dep}}_{c}+N^{\mathrm{BNF}}_{c}+N^{\mathrm{excr}}_{c}+N^{\mathrm{digest}}_{c}+N^{\mathrm{seed}}_{c}\bigr)\,.
```
A global scaling $\kappa$ is then applied to meet the national stocks:
```{math}
N^{\mathrm{syn}}_{c} \;=\; \kappa\;\max\!\bigl(N^{\mathrm{syn,res}}_{c},\,0\bigr),\qquad
\sum_{c\in\mathcal{C}_{\mathrm{crop}}} N^{\mathrm{syn}}_{c} \;=\; S^{\mathrm{tot}}_{\mathrm{crop}}\,,\quad
\sum_{c\in\mathcal{C}_{\mathrm{grass}}} N^{\mathrm{syn}}_{c} \;=\; S^{\mathrm{tot}}_{\mathrm{grass}}\,.
```

- **H2 — Standard per‑ha usage.** Given observed $u_c$ in kgN/ha:
```{math}
N^{\mathrm{syn,raw}}_{c} \;=\; u_c \cdot A_c/10^6,\qquad
N^{\mathrm{syn}}_{c} \;=\; \kappa\,N^{\mathrm{syn,raw}}_{c}
```
with $\kappa$ chosen so that crop and grassland totals match $S^{\mathrm{tot}}_{\mathrm{crop}}$ and $S^{\mathrm{tot}}_{\mathrm{grass}}$.

In both H1 and H2, BNF uses the same formula as in Prospective (but $y_c$ is observed). There is **no explicit legume→cereal legacy**; balances close via $\Delta S^{\mathrm{soil}}_{c}$.

### 2.2 Prospective (P): endogenous yields, convex response

For non‑legumes ($c\notin \mathcal{C}_{\mathrm{legume}}$), the yield response is:
```{math}
Y_c(F) \;=\; Y^{\max}_{c}\,\bigl(1 - e^{-F/F^\star_{c}}\bigr)\,.
```
It is enforced by a piecewise‑linear envelope (SOS2 or convex combination) with breakpoints $(F_{i c}, Y_{i c})$:
```{math}
\sum_i \lambda_{i c}=1,\qquad 
f_c=\sum_i \lambda_{i c} F_{i c},\qquad 
y_c=\sum_i \lambda_{i c} Y_{i c},\qquad 
\lambda_{i c}\ge 0 \text{ and contiguous.}
```
Legumes have $N^{\mathrm{syn}}_{c}=0$ by construction. The BNF expression is **identical** to Historical, but now $y_c$ is endogenous.

---

## 3. Central LP/MILP allocation model (Pulp)

This model allocates available nitrogen from plant and animal products to **consumers** (livestock, population) and **energy facilities**, possibly using **imports**; it balances **diet fidelity**, **energy targets**, **imports**, and **fairness/smoothing** with tunable weights.

### 3.1 Data

- Availability of each product $p$: $N^{\mathrm{avail}}_{p}$ (ktN) after waste and other uses.  
- Consumer ingestion targets $I_q$ (ktN) and group shares $\pi_{qG}$ from Diets.  
- Facility $f$ target $E^{\mathrm{tar}}_f$ (GWh) and admissible inputs with powers $P_{f,i}$ (MWh/ktN).  
- Excreta available for energy: $E^{\mathrm{remain}}_e$ (ktN).

### 3.2 Variables

Allocations and logistics (all $\ge 0$ unless stated otherwise):
- $x_{p q}$: product $p$ to consumer $q$.  
- $I_{p q}$: **imports** of $p$ to consumer $q$.  
- $U_p$: exports/unused surplus of $p$.  
- $\delta_{qG}$: diet share deviations.  
- $\pi_{qGp}$: intra‑group smoothing deviations.  
- $\gamma_{p q}$: fair‑share deviations.  
- Optional anti‑swap binary $y_p\in\{0,1\}$.

Energy side:
- $x^f_{p}$, $x^f_{e}$, $N^f_{\mathrm{waste}}$, $I^f_{p}$ (imports allowed only for bioraffineries).  
- $\delta^{f}_{G},\;\pi^{f}_{G,it}$: facility diet deviations.  
- $\mathrm{dev}^E_f$: relative deviation from energy target.

### 3.3 Hard constraints

Consumer diets by groups (softly enforced via $\delta_{qG}$):
```{math}
\left|\frac{\sum_{p\in \mathcal{P}_G}\bigl(x_{p q}+I_{p q}\bigr)}{I_q} - \pi_{qG}\right| \;\le\; \delta_{qG}\,,\qquad \forall q,G\,.
```

Consumer totals:
```{math}
\sum_{G}\sum_{p\in \mathcal{P}_G} \bigl(x_{p q}+I_{p q}\bigr) \;=\; I_q\,,\qquad \forall q\,.
```

Availability:
```{math}
\sum_{q} x_{p q} + \sum_{f} x^{f}_{p} + U_{p} \;\le\; N^{\mathrm{avail}}_{p}\,,\qquad \forall p\,.
```

Excreta capacity for energy:
```{math}
\sum_{f} x^{f}_{e} \;\le\; E^{\mathrm{remain}}_{e}\,,\qquad \forall e\,.
```

No imports for grazing products:
```{math}
I_{p q} \;=\; 0 \quad \text{if } p \in \texttt{grazing}\,.
```

Energy production identity:
```{math}
E_f \;=\; \sum_{p} x^{f}_{p}\,P_{f p} + \sum_{e} x^{f}_{e}\,P_{f e} + N^{f}_{\mathrm{waste}}\,P_{f,\mathrm{waste}}\,,\qquad \forall f\,.
```

Anti‑swap (optional MILP):
```{math}
\sum_q I_{p q} \le M_p\,y_p,\qquad U_p \le M_p\,(1-y_p),\qquad y_p\in\{0,1\}\,.
```

### 3.4 Soft constraints (absolute‑value linearizations)

Fair‑share by product (with references $s^{\mathrm{ref}}_{p q}$):
```{math}
\bigl|x_{p q} - s^{\mathrm{ref}}_{p q}\,N^{\mathrm{avail}}_{p}\bigr| \;\le\; \gamma_{p q}\,.
```

Intra‑group smoothing:
```{math}
\left|x_{p q} - \frac{1}{|\mathcal{P}_G|}\sum_{r\in\mathcal{P}_G} x_{r q}\right| \;\le\; \pi_{q G p}\,.
```

Facility diet constraints mirror the consumer ones with superscript $f$.

### 3.5 Objective (with annotated terms)

We minimize a non‑negative weighted sum; braces label each contribution:
```{math}
\begin{aligned}
\min\;\; 
&\underbrace{\sum_{q,G} w^{\mathrm{diet}}\;\delta_{qG}}_{\text{diet share deviations (consumers)}}
+ \underbrace{\sum_{q,G,p} w^{\mathrm{intra}}\;\pi_{qGp}}_{\text{within-group smoothing}}
+ \underbrace{\sum_{p,q} w^{\mathrm{fair}}\;\gamma_{p q}}_{\text{fair-share vs references}}
+ \underbrace{\sum_{p,q} w^{\mathrm{imp}}\;\frac{I_{p q}}{N^{\mathrm{avail}}_{p}+\varepsilon}}_{\text{normalized imports}} \\[4pt]
&+ \underbrace{\sum_{f}\Bigl( w^{E}\,\mathrm{dev}^E_f + \sum_{G} w^{E\text{-diet}}\,\delta^{f}_{G} + \sum_{G,it} w^{E\text{-intra}}\,\pi^{f}_{G,it} \Bigr)}_{\text{energy targets and facility diets}} \\[4pt]
&+ \underbrace{w^{\mathrm{syn}}\;\max\!\Bigl(0, \sum_{c\in\mathcal{C}} N^{\mathrm{syn}}_{c} - S^{\mathrm{tot}}_{\mathrm{crop}}\Bigr)}_{\text{[P] crop synthetic budget penalty}} 
+ \underbrace{w^{\mathrm{syn\text{-}dist}}\;\sum_{c\notin \mathcal{C}_{\mathrm{legume}}}\left|\frac{f_c}{F^\star_{c}}-1\right|}_{\text{[P] distribution around }F^\star}
\end{aligned}
```

### 3.6 Weights — role and practical guidance

The weights prioritize goals that may conflict: feasibility, fidelity to diets, achievement of energy targets, import minimization, equitable and smooth allocations, and—only in Prospective—compliance with synthetic budgets and reasonable per‑crop fertilization around $F^\star$.  
Typical practice is to keep the **stabilizers** ($w^{\mathrm{intra}}$, $w^{\mathrm{fair}}$ and energy analogs) at least one order of magnitude below the primary priorities (diet fidelity, energy targets, imports). A quick diagnostic is to print the contribution of each objective term (absolute value and share of the optimum): large $\delta_{qG}$ or $\mathrm{dev}^E_f$ signal scarcity or tight targets; large $\gamma_{p q}$ indicates inequitable splits; high imports show reliance on external supply. In Prospective, also monitor $\sum_c N^{\mathrm{syn}}_{c}$ vs $S^{\mathrm{tot}}_{\mathrm{crop}}$ and $\sum_{c\notin \mathcal{C}_{\mathrm{legume}}}\bigl|f_c/F^\star_c-1\bigr|$.

---

## 4. Notes on emissions scope

Emission coefficients $\phi$ are **aggregated by design**. Users can compose them from sub‑factors (e.g., storage, application, upstream manufacturing) to match a chosen scope. The balance itself only needs the effective factor(s) and remains valid under any consistent decomposition.
