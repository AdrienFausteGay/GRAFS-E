# Input Data

In this section, we describe the **input data** required for the proper functioning of **E-GRAFS**. Data can use several sources but must be coherent. If data are lacking, you can use defaut value or hypothesis. For example, if a crop production is unavailable, use crop area and mean crop productivity. The data is organized into two Excel files: one project spreadsheet and one data spreadsheet.

Names are **case-sensitive** and must match **exactly**.

## Project Spreadsheet

The **project** spreadsheet contains the **metadata** for structuring the project, such as compartments and their characteristics and default values. This file consists of several sheets:

- **crops**: Data on crops
- **livestock**: Data on livestock
- **excretion**: Data on livestock excretions
- **energy**: Data on bioenergy facilities
- **pop**: Data on human populations
- **prod**: Data on agricultural products
- **global**: Data regarding a specific territory

Each sheet must contain at least the **List of Compartments** (names). Two compartments cannot have the same name.

In each sheet, you may add **Default Values**: If a default value for an item (e.g., Nitrogen Content for Wheat is 2 %) is provided in the project file, it will be used unless another value is available in the data file.

If a row or column is missing, it will be automatically added with data filled with 0.

The **project** file defines the basic structures for each category of data.

## Data Spreadsheet

The **data** spreadsheet consists of tree sheets:
1. **Input Data**: Details of data points for E-GRAFS
2. **Diet**: Dietary distribution information for livestock and human populations
3. **Energy power**: Potential of energy production from biomass in energy facilities

### Input Data

The **Input Data** sheet contains rows with specific data for each territory, year, category, and compartment. Each row contains the following information:
- **Area**: Territory concerned (e.g., France)
- **Year**: Year of the data (e.g., 2010)
- **Category**: Type of the data (e.g., Production (ktN))
- **Item**: Concerned compartment (e.g., Wheat) or global parameter
- **Value**: Value of the data point

Each row states: *“for (Area, Year), set (category) of (item) to (value)”*. These entries are **interannual values** and take **priority** over project-sheet defaults.

Example:

| Area   | Year | category                  | item | value |
| --------- | ---------- | ------------------------- | ----- | ----- |
| France | 1961       | Production (kton)    | barley | 15000 |

With this line in your dataset, if you call `NitrogenFlowModel("France", 1961)`, the production of Barley in France in 1961 will be 15 000 kton. See input data page for all input data names.

### Diet

The **Diet** sheet describes the ideal nitrogen distribution for feeding a consumer (livestock or human). Each diet is defined by:
- **Diet ID**: Identifier for the diet (e.g., `b_2023_fr`)
- **Products**: List of consumed products (e.g., Wheat, Barley, Oats)
- **Proportion**: Proportion of nitrogen allocated to each product (e.g., 65% for forage, 10% for wheat and barley, etc.)

Example:
| Diet ID   | Proportion | Products                  |
| --------- | ---------- | ------------------------- |
| b_2023_fr | 0.65       | Natural meadows forage    |
| b_2023_fr | 0.10       | Barley grain, Wheat grain |
| b_2023_fr | 0.25       | Soya beans cake           |

This indicates that consumers with the diet `b_2023_fr` consume:
- 65% of natural meadow forage
- 10% of wheat and barley grain
- 25% of soybeans cake

If for a Diet ID the sum of proportions is not 1, the proportion values are normalized.

For each consummers (population, livestock and methanizer), a diet ID must be given as Diet category in "Input data" tab of data file. Give the same diet for all territories for all year to a specific consummer is a valid use of E-GRAFS because the model adapt itself to local production and import availabilities. The weight values `Weight import` and `Weight diet` of optimization model make the model adapt diet to local context or import products for consummers. Here an example of how to give the diet name of a consummer :
| Area   | Year | category                  | item | value |
| --------- | ---------- | ------------------------- | ----- | ----- |
| France | 2023       | Diet    | bovines | b_2023_fr |

#### Bioenergy facilities diets

A diet must be given for each bioenergy ficilities on the territory. This is done like diets for populations and livestock with a diet name given in "Input data". Yet Methanizer type are allowed to consume products, excretion compartments and "waste" which represent green waste. 
Bioraffinery type can only use products compartments and waste as inputs.

### Energy power

The **Energy power** tab is used to give the power potential of each item (product, excretion, waste) for each bioenergy facility. It's structure is composed of 3 columns :

- **Facility**: Name of the bioenergy facility (from Facility column in energy tab)
- **Items**: list of items concerned by the definition of energy power. Each item must be coma separated
- **Energy Power (MWh/tFW)**: Energy production by ton of Fresh Weight in the facility.

Powers are converted internally to **MWh/ktN** using the item’s `%N` from `prod`/`excretion` (for `waste`, the global `%N` is used).

Example:
| Facility   | Items | Energy Power (MWh/tFW)                  |
| --------- | ---------- | ------------------------- |
| Bioreffinery G1 | Wheat grain, Barley Grain       | 1.2    |

## Input Data

In this section, we describe the **Input data** required for the proper functioning of **E-GRAFS**. The data can be provided by default in the project spreadsheet or in detail in the data spreadsheet (Input data tab). If there is a conflict for a given year and territory between a data point in the project spreadsheet and the data spreadsheet, the data spreadsheet takes precedence.

For example, in the **crops** tab of the project spreadsheet, it is indicated that the nitrogen content of wheat is 2%. But in the data spreadsheet, it is indicated that in France in 2010, the nitrogen content of wheat is 3%, so the value retained for the calculations will be 3%.

It is not possible to have two compartments with the same name. For example, it is forbidden to have a crop compartment called "Wheat" and a product called "Wheat".

Compartment name can use space (' ') but other special characters are forbiden.

> Legend for Required: `✔/✔` Historical / Prospective, `✔/—` Historical-only, `—/✔` Prospective-only, `—/—` optional.

### Crops

Here are the required crop-related input data. 

| Column | Required (Hist./Pros.) | Unit / Type | Allowed values | Default | Description |
|---|---|---|---|---|---|
| `Culture` | ✔ / ✔ | string (key) | unique | — | Primary crop ID used across sheets. |
| `Main Production` | ✔ / ✔ | string (Product) | must exist in `prod.Product` | — | Main harvested product for this crop. |
| `Category` | ✔ / ✔ | enum | — | — | Drives agronomic rules (grasslands vs arable; legumes). |
| `BNF alpha` | ✔ / ✔ | kgN/tFW | ≥ 0 | 0 for non-legumes | Linear term in symbiotic fixation (both modes). |
| `BNF beta` | ✔ / ✔ | kgN/ha | ≥ 0 | 0 for non-legumes | Constant term in symbiotic fixation (both modes). |
| `BGN` | ✔ / ✔ | ratio | ≥ 0 | — | Below-ground factor (roots). |
| `Harvest Index` | ✔ / ✔ | ratio (0–1) | 0–1 | — | Fraction of above-ground biomass harvested. |
| `Spreading Rate (%)` | ✔ / ✔ | % | 0–100 | 100 | Share of hectares eligible for spreading. |
| `Seed input (ktN/ktN)` | ✔ / ✔ | ktN per ktN | ≥ 0 | 0 | Seed N per ktN of harvested N. |
| `Fertilization Need (kgN/qtl)` | ✔ / — | kgN/ha | ≥ 0 | 0 | Historical-only: TOTAL fertilization need by yield unit. Optional if `Surface Fertilization Need (kgN/ha)` or `Raw Surface Synthetic Fertilizer Use (kgN/ha)` given |
| `Surface Fertilization Need (kgN/ha)` | ✔ / — | kgN/ha | ≥ 0 | 0 | Historical-only: TOTAL fertilization need by area unit. Optional if `Fertilization Need (kgN/qtl)` or `Raw Surface Synthetic Fertilizer Use (kgN/ha)` given |
| `Raw Surface Synthetic Fertilizer Use (kgN/ha)` | ✔ / — | kgN/ha | ≥ 0 | 0 | Historical-only: observed per-ha synthetic use. Optional if `Surface Fertilization Need (kgN/ha)` or `Fertilization Need (kgN/qtl)` given |
| `Area (ha)` | ✔ / ✔ | ha | ≥ 0 | — | Land area of the crop. |
| `Residue Nitrogen Content (%)` | ✔ / ✔ | % | ≥ 0 | 0.5 non-legumes / 1.5 legumes | Nitrogen content of aerial residues. |
| `Maximum Yield (tFW/ha)` | — / ✔ | tFW/ha | ≥ 0 | — | Prospective-only: Y_max of the yield curve. |
| `Characteristic Fertilisation (kgN/ha)` | — / ✔ | kgN/ha | ≥ 0 | — | Prospective-only: Caracteristic total fertilization of the yield curve. |

Some **Category** have special rules in E-GRAFS:
- **leguminous**: Leguminous crops are excluded from synthetic fertilizer distribution.
- **natural meadows** and **temporary meadows**: These crops benefit from outdoor animal excretion. The input Nitrogen surplus is fully stocked in soil up to 100 kgN/ha. After, same tratment of crops.

The user can define as many additional categories as desired, but these will not have special rules. Example of other common categories:
- Cereals
- Forage
- Roots
- Fruits and Legumes
- Rice

Fertilization input can be given by a mix of [**Fertilization Need (kgN/qtl)** and **Surface Fertilization Need (kgN/ha)**] or with **Raw Surface Synthetic Fertilizer Use (kgN/ha)** for all crops. Using **Fertilization Need (kgN/qtl)** and **Surface Fertilization Need (kgN/ha)** will compute a Nitrogen balance on crops but using **Raw Surface Synthetic Fertilizer Use (kgN/ha)** will give a proxy for synthetic fertilizer use. See E-GRAFS engine page for more details.

### Livestock

Here are the required input data related to **livestock**:
                  

| Column | Required (Hist./Pros.) | Unit / Type | Allowed values | Default | Description |
|---|---|---|---|---|---|
| `Livestock` | ✔ / ✔ | string (key) | unique | — | Livestock identifier. |
| `Excreted indoor (%)` | ✔ / ✔ | % | 0–100 | — | Time in housing; remainder on grassland. |
| `Excreted indoor as manure (%)` | ✔ / ✔ | % | 0–100 | — | Of indoor: share to manure (else slurry). |
| `LU` | ✔ / ✔ | Livestock Unit | ≥ 0 | — |  |
| `Excretion / LU (kgN)` | ✔ / ✔ | kgN | ≥ 0 | — | Amount of Nitrogen excreted by LU |
| `Diet` | ✔ / ✔ | string (Diet ID) | must exist in `Diet.Diet ID` | — | Input diet for facility. |

### Excretion

E-GRAFS manage 3 excretion types: manure, slurry and grassland excretion. The amount of each excretion type is given by livestock tab data, yet the induced flows are computed and represented with specific excretion compartments.
The excretion tab must be composed of 3 lines per livestock defined in livestock tab. With i the name of each livestock, the following lines must be in excretion sheet :
- i manure
- i slurry
- i grasslands excretion
*If some compulsory excreta rows (manure, slurry, grasslands) are missing for a livestock, the engine creates them with 0 and emits a warning.*

Required input data related to **animal excretion**:

| Column | Required (Hist./Pros.) | Unit / Type | Allowed values | Default | Description |
|---|---|---|---|---|---|
| `Excretion` | ✔ / ✔ | string (key) | unique | — | Excreta stream name. |
| `Origin compartment` | ✔ / ✔ | string | must match `livestock.Livestock` | — | Animal source. |
| `Type` | ✔ / ✔ | enum | grasslands excretion, manure, slurry | — | manure / slurry / grasslands excretion. |
| `N-NH3 EM (%)` | ✔ / ✔ | % | 0–100 | — | $NH_3$ volatilization share. |
| `N-N2 EM (%)` | ✔ / ✔ | % | 0–100 | — | $N_2$ emission share. |
| `N-N2O EM (%)` | ✔ / ✔ | % | 0–100 | — | Direct $N_2O$ emission share. |
| `Nitrogen Content (%)` | ✔ / ✔ | % | 0–100 | — | For N-based power conversion. |


### Population

Here are the required input data related to **human populations**:

| Column | Required (Hist./Pros.) | Unit / Type | Allowed values | Default | Description |
|---|---|---|---|---|---|
| `Population` | ✔ / ✔ | string (key) | unique | — | Population identifier. |
| `Inhabitants` | ✔ / ✔ | — | integer ≥ 0 | — | Number of inhabitants for this population. |
| `N-NH3 EM excretion (%)` | ✔ / ✔ | % | 0–100 | — | $NH_3$ excretion volatilization share. |
| `N-N2O EM excretion (%)` | ✔ / ✔ | % | 0–100 | — | $N_2O$ excretion volatilization share. |
| `N-N2 EM excretion (%)` | ✔ / ✔ | % | 0–100 | — | $N_2$ excretion volatilization share. |
| `Total ingestion per capita (kgN)` | ✔ / ✔ | kgN | ≥ 0 | — | Total ingestion (fishery products, animal products and plant products) |
| `Fishery ingestion per capita (kgN)` | ✔ / ✔ | kgN | ≥ 0 | — | Fischery only ingestion |
| `Excretion recycling (%)` | ✔ / ✔ | % | 0–100 | — | Share of Nitrogen excreted spread on croplands after volatilization. Sum of $NH_3$, $N_2O$, $N_2$ and recycling share cannot be higher than 100 %. |
| `Diet` | ✔ / ✔ | string (Diet ID) | must exist in `Diet.Diet ID` | — | Input diet for facility. |

### Product Data

Here is the data related to **products**:

| Column | Required (Hist./Pros.) | Unit / Type | Allowed values | Default | Description |
|---|---|---|---|---|---|
| `Product` | ✔ / ✔ | string (key) | unique | — | Product identifier. |
| `Origin compartment` | ✔ / ✔ | string | `crops.Culture` or `livestock.Livestock` | — | Source entity. |
| `Type` | ✔ / ✔ | enum | animal, plant | — | `plant` or `animal`. |
| `Sub Type` | ✔ / ✔ | enum | see typology below | — | Governs grouping and rules. |
| `Nitrogen Content (%)` | ✔ / ✔ | % | 0–100 | — | Used for ktN conversions and energy power per ktN. |
| `Waste (%)` | ✔ / ✔ | % | 0–100 | 0 | Loss before allocation. |
| `Other uses (%)` | ✔ / ✔ | % | 0–100 | 0 | Non food/energy uses removed from availability. |
| `Production (kton)` | ✔ / — | kton FW | ≥ 0 | — | Product production in kilo ton |
| `Co-Production Ratio (%)` | — / ✔ | % | 0–100 | — | Prospective-only: fresh-mass share by co-product. 100 if main product of the crop, other value if co-production (straw, oil, cake...)|

> `Waste (%)` and `Other uses (%)` are used to compute Available Nitrogen for allocation model. The sum of these proportion must be below 100 %.

#### Product Typology

To handle products and the flows associated with them, E-GRAFS uses a standardized two-level typology (Type and Sub Type).

##### Type

Two type of products are handle by E-GRAFS. Each product must be 'animal' or 'plant'. Product in none of these type might compromise E-GRAFS functionning.

##### Sub Type
Each Sub Type 'i' defines a "Trade i" compartment. This means that trade exchanges are categorized by Sub Type.
The Sub Type of plant products are freely defined by the user. Only the **grazing** Sub Type has a special rule: **grazing** products cannot be imported or exported. Avoid diet groups with only grazing products, add at least one tradable product to prevent abnormal behaviour of the model.

The Sub Types for **animal** products are fixed and include 3 categories.

- **Animal**: Includes all products derived from animals:
    - **edible meat**: Consumable meat
    - **dairy**: Dairy products and eggs
    - **non edible meat**: Non-consumable meat. All production in this sub-type is directed to the 'other sector' (used in other industries or burned). Cannot be traded or consummed.

```{warning}
**Warning**: The Sub types must be different than **crops** categories.
```

### Bioenergy Data

The tab 'energy' is used to define the bioenergy facilities related to the territory. This tab can be left empty (just columns name) if no facilities are linked to the territory. A bioenergy facility is considered as a consummer and must therefore have a diet defined in Diet tab of data file. Unlike other consummers, bioenergy facilities of type Methanizer can have products and excretion compartments and 'waste' compartment.

| Column | Required (Hist./Pros.) | Unit / Type | Allowed values | Default | Description |
|---|---|---|---|---|---|
| `Facility` | ✔ / ✔ | string (key) | unique | — | Facility identifier. |
| `Diet` | ✔ / ✔ | string (Diet ID) | must exist in `Diet.Diet ID` | — | Input diet for facility. |
| `Target Energy Production (GWh)` | ✔ / ✔ | GWh | ≥ 0 | — | Annual energy target. |
| `Type` | ✔ / ✔ | enum | Bioraffinery, Methanizer | — | `Methanizer` or `Bioraffinery`. |

#### Bioenergy Facilities Typology

Two Types are available for bioenergy facilities :
- 'Methanizer': The Methanizer can have inputs from product, excretion or waste compartments. All Nitrogen inputs goes to digestat and is spread on crops with the same rules as excretions or sludges (see E-GRAFS engine page). Methanizer cannont import their inputs flows.
- 'Bioreffinery': The Bioreffinery can have inputs only from products or waste. All input nitrogen is directed to 'hydrocarbures' compartment. Bioreffinery can import products.

    
### Global Data

E-GRAFS also relies on **global variables**, which apply to all compartments for a given year and territory.

| **Column Name** | **Description**                             | **Type**         | **Comment**                                                                                                           |
| ----------------| ------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| **item**        | Name of the global variable                 | str              | The number and name of the items are fixed (see next section)                                                        |
| **value**       | Value of the item                           | depends on item  | See next section                                                                                                       |

#### Global Values

The name and number of global variables are fixed.

| Item | Required (Hist./Pros.) | Unit / Type | Default (project) | Description |
|---|---|---|---|---|
| `Total Synthetic Fertilizer Use on crops (ktN)` | ✔ / ✔ | ktN | 2000 | National synthetic N stock for crops; scaling target in H and budget cap/penalty in P. |
| `Total Synthetic Fertilizer Use on grasslands (ktN)` | ✔ / ✔ | ktN | — | National synthetic N stock for grasslands; same logic as crops. |
| `Atmospheric deposition coef (kgN/ha)` | ✔ / ✔ | kgN/ha | — | Areal nitrogen input applied to crop surfaces. |
| `coefficient N-NH3 volatilization synthetic fertilization (%)` | ✔ / ✔ | % | 3 | Fraction of synthetic N volatilized as $NH_3$ at field application. |
| `coefficient N-N2O emission synthetic fertilization (%)` | ✔ / ✔ | % | 0.05 | Fraction of synthetic N emitted as direct N₂O at field application. |
| `coefficient N-N2O indirect emission synthetic fertilization (%)` | ✔ / ✔ | % | 0.2 | Fraction of volatilized NH₃ accounted as indirect N₂O. |
| `Weight diet` | ✔ / ✔ | non-negative scalar | 1 | Penalty on consumer diet share deviations. |
| `Weight import` | ✔ / ✔ | non-negative scalar | 1 | Penalty on normalized imports per product. |
| `Weight energy production` | ✔ / ✔ | non-negative scalar | 10 | Penalty on facility energy target deviation (relative). |
| `Weight energy inputs` | ✔ / ✔ | non-negative scalar | 0.1 | Penalty on deviations from facility input diet shares. |
| `Weight distribution` | ✔ / ✔ | non-negative scalar | — | Stabilizer for smoother within-group allocations. |
| `Weight fair local split` | ✔ / ✔ | non-negative scalar | — | Stabilizer towards reference fair-share by product. |
| `Weight synthetic fertilizer` | — / ✔ | non-negative scalar | 1 | Penalty when total crop synthetic N exceeds the budget. |
| `Weight synthetic distribution` | — / ✔ | non-negative scalar | 1 | Penalty to keep per-crop f/F* close to 1. |
| `Enforce animal share` | ✔ / ✔ | boolean | True | If true, enforce the target animal share in diets. |
| `Green waste nitrogen content (%)` | ✔ / ✔ | % | 1 | %N used to convert MWh/tFW of `waste` into MWh/ktN. |