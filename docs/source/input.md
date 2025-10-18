# Input Data

In this section, we describe the **input data** required for the proper functioning of **GRAFS-E**. Data can use several sources but must be coherent. If data are lacking, you can use defaut value or hypothesis. For example, if a crop production is unavailable, use crop area and mean crop productivity. The data is organized into two Excel files: one project spreadsheet and one data spreadsheet.

## Project Spreadsheet

The **project** spreadsheet contains the **metadata** for structuring the project, such as compartments and their characteristics and default values. This file consists of several sheets:

- **crops**: Data on crops
- **livestock**: Data on livestock
- **excretion**: Data on livestock excretions
- **pop**: Data on human populations
- **prod**: Data on agricultural products
- **global**: Data regarding a specific territory

Each sheet must contain the **List of Compartments** (names): Each compartment must be clearly defined (e.g., crops, livestock, etc.). Two compartments must not have the same name.

In each sheet, you may add **Default Values**: If a default value for an item (e.g., Nitrogen Content for Wheat is 2 %) is provided in the project file, it will be used unless another value is available in the data file.

GRAFS-E manage 3 excretion types: manure, slurry and grassland excretion. The amount of each excretion type is given by livestock tab data, yet the induced flows are computed and represented with specific excretion compartments.
The excretion tab must be composed of 3 lines per livestock defined in livestock tab. With i the name of each livestock, the following lines must be in excretion sheet :
- i manure
- i slurry
- i grasslands excretion

If a row is missing, it will be automatically added with data filled with 0.

The **project** file defines the basic structures for each category of data.

## Data Spreadsheet

The **data** spreadsheet consists of two main sheets:
1. **Input Data**: Details of data points for GRAFS-E
2. **Diet**: Dietary distribution information for livestock and human populations

### Input Data

The **Input Data** sheet contains rows with specific data for each territory, year, category, and compartment. Each row contains the following information:
- **Area**: Territory concerned (e.g., France)
- **Year**: Year of the data (e.g., 2010)
- **Category**: Type of the data (e.g., Production (ktN))
- **Item**: Concerned compartment (e.g., Wheat) or global parameter
- **Value**: Value of the data point

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

For each consummers (population, livestock and methanizer), a diet ID must be given as Diet category in "Input data" tab of data file. Give the same diet for all territories for all year to a specific consummer is a valid use of GRAFS-E because the model adapt itself to local production and import availabilities. The weight values `Weight import` and `Weight diet` of optimization model make the model adapt diet to local context or import products for consummers. Here an example of how to give the diet name of a consummer :
| Area   | Year | category                  | item | value |
| --------- | ---------- | ------------------------- | ----- | ----- |
| France | 2023       | Diet    | bovines | b_2023_fr |

#### Methanizer diet

A diet must be given for methanizer on the territory. This is done like diets for populations and livestock with a diet name given in "Input data". Yet Methanizer are allowed to consume excretion compartments and "waste" which represent green waste. 
If the is no methanizer in the territory, gave it this minimal diet :
| Diet ID   | Proportion | Products                  |
| --------- | ---------- | ------------------------- |
| Methanizer | 1       | waste    |


## Input Data

In this section, we describe the **Input data** required for the proper functioning of **GRAFS-E**. The data can be provided by default in the project spreadsheet or in detail in the data spreadsheet (Input data tab). If there is a conflict for a given year and territory between a data point in the project spreadsheet and the data spreadsheet, the data spreadsheet takes precedence.

For example, in the **crops** tab of the project spreadsheet, I indicated that the nitrogen content of wheat is 2%. But in the data spreadsheet, I indicated that in France in 2010, the nitrogen content of wheat is 3%, so the value retained for the calculations will be 3%.

It is not possible to have two compartments with the same name. For example, it is forbidden to have a crop compartment called "Wheat" and a product called "Wheat".

### Crops

Here are the required crop-related input data.

| Column Name                         | Description                                                                                          | Type             | Comment                                                                                                            |
| -----------------------------------  | ---------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Crops**                            | Name of the crops                                                                                   | str              |                                                                                                                    |
| **Main Production**                  | Name of the main production of this crop                                                             | str              | Generally, the main product is the commercial product of this crop                                                |
| **Category**                         | Type of crop                                                                                        | str              | See the list of available types and their specificities below                                                     |
| **Fertilization Need (kgN/qtl)**     | Nitrogen needs of the crop based on yield                                                           | float (>=0)      | Optional. See the complete description in the methodology section                                                          |
| **Surface Fertilization Need (kgN/ha)** | Nitrogen needs of the crop per unit area                                                            | float (>=0)      | Optional. See the complete description in the methodology section                                                          |
| **Raw Surface Synthetic Fertilizer Use (kgN/ha)** | Synthetic Nitrogen Fertilizer needs of the crop per unit area                                                            | float (>=0)      | Optional. See the complete description in the methodology section                                                          |
| **BNF alpha**                        | Alpha coefficient for symbiotic nitrogen fixation                                                   | float            | 0 if the crop does not fix nitrogen symbiotically. Otherwise, refer to data from Anglade et al. (2015)            |
| **BNF beta**                         | Beta coefficient for symbiotic nitrogen fixation                                                    | float            | 0 if the crop does not fix nitrogen symbiotically. Otherwise, refer to data from Anglade et al. (2015)            |
| **BNG**                              | Contribution of roots to symbiotic nitrogen fixation                                               | float (>=1)      | 0 if the crop does not fix nitrogen symbiotically. Otherwise, refer to data from Anglade et al. (2015)            |
| **Harvest index**                    | Part of the crop harvested. Useful only for calculating symbiotic nitrogen fixation                | float ([0,1])    |                                                                                                                    |
| **Area (ha)**                        | Area occupied by this crop in the considered territory                                              | float (>=0)      |                                                                                                                    |
| **Spreading Rate (%)**               | Proportion of the area of this crop benefiting from manure, slurry, fertilizer, etc.                | float ([0, 100]) | If no data is available, use 100 by default for all crops.                                                        |
| **Seed input (kt seeds/kt Ymax)**    | Amount of seeds to be sown per unit of yield                                                        | float (>=0)      | Used to calculate the portion of production reinvested into the crop for the next year.                           |

The **Categories** available in GRAFS-E are:
- **cereal (excluding rice)**: This category groups all cereals except rice. This distinction is made to determine which crops to direct excess symbiotic fixation toward.
- **leguminous**: Legumes are excluded from synthetic fertilizer distribution and are the source of symbiotic nitrogen fixation.
- **natural meadows**: Permanent natural meadows. It is not possible to export or import the products from this type of crop. Surplus production returns to the soil.
- **temporary meadows**: Meadows that can be mowed and whose products can be sold. Any surplus from symbiotic nitrogen fixation is directed toward the **cereal (excluding rice)** category.

The user can define as many additional categories as desired, but these will not have special rules. Example of other common categories:
- Forage
- Roots
- Fruits and Legumes
- Rice

### Livestock

Here are the required input data related to **livestock**:

| Column Name                        | Description                                                                                        | Type             | Remark                                                                                                                                          |
| ----------------------------------  | -------------------------------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Livestock**                       | Name of the livestock (e.g., cattle, sheep, etc.)                                                   | str              | Livestock can be defined by animal type (e.g., cattle, sheep) and differentiated by specific types of livestock (e.g., beef cattle, dairy cattle) |
| **Excreted indoor (%)**             | Proportion of time spent indoors in a building                                                     | float ([0, 100]) | 100 - **Excreted indoor (%)** gives the proportion of time spent grazing                                                                      |
| **Excreted indoor as manure (%)**   | Proportion of excretions indoors converted into manure                                              | float ([0, 100]) | The rest is converted into slurry.                                                                                                                |
| **LU**                              | Number of Livestock Units                                                                           | float (>0)       |                                                                                                                                               |
| **Excretion / LU (kgN)**            | Amount of nitrogen excreted per LU                                                                  | float (>0)       |                                                                                                                                               |                  
| **Diet**        | Diet ID to use for this population                                                           | str | Must have a corresponding Diet ID in Diet tab                                                                            |

The management of excretions distinguishes between three types:
- **Manure**
- **Slurry**
- **Outdoor (non-recoverable)**

For each type of excretion management **X**, the difference 100 - (N-NH3 EM X (%) + N-N2O EM X (%) + N-N2 EM manure (%)) is considered lost to continental water (compartment "hydro-system").

### Excretion

As stated in Here are the required input data related to **animal excretion**:

| Column Name                        | Description                                                                                        | Type             | Remark                                                                                                                                          |
| ----------------------------------  | -------------------------------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Origin compartment** | Name of the livestock compartment producing this excretion. | str              | It must match a defined  livestock compartment.      |
| **Type**              | Type of excretion (manure, slurry or grasslands excretion)       | str              |                                                                                                         |
| **N-NH3 EM (%)**             | Ammonia emission factor                                                                  | float ([0, 100]) |
| **N-N2O EM (%)**            | Nitrous oxide emission factor                                             | float ([0, 100]) | 
| **N-N2 EM (%)**             | Nitrogen N2 emission factor                                               | float ([0, 100]) |
| **Methanization power (MWh/tFW)**          | Energy production potential in methanizer by ton of input fresh weight                     | float ([0, 100])              | Put 0 if this is not intended to methanizer                                                                                                          |
| **Nitrogen Content (%)** | Nitrogen content of the excretion               | float ([0, 100]) | Nitrogen content of the fresh excretion, not in dry matter. Can be set to 0 for grasslands excretion. Only used for manure and slurry.                                                                 |


### Population

Here are the required input data related to **human populations**:

| Column Name                        | Description                                                                                         | Type             | Comment                                                                                                           |
| ----------------------------------  | --------------------------------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Population**                      | Name of the population (e.g., vegetarians, omnivores)                                                | str              |                                                                                                                   |
| **Inhabitants**                     | Number of inhabitants in the population                                                              | int              | Ideally, consider the "real" population (official population + tourism + non-counted population)                 |
| **N-NH3 EM excretion (%)**          | Ammonia emission factor for excretion                                                                | float ([0, 100]) |                                                                                                                   |
| **N-N2O EM excretion (%)**          | Nitrous oxide emission factor for excretion                                                          | float ([0, 100]) |                                                                                                                   |
| **N-N2 EM excretion (%)**           | Nitrogen N2 emission factor for excretion                                                            | float ([0, 100]) |                                                                                                                   |
| **Total ingestion per capita (kgN)**| Total nitrogen intake from all sources (plant products, animals, and the sea)                       | float (>0)       |                                                                                                                   |
| **Fishery ingestion per capita (kgN)**| Nitrogen intake from sea products                                                                   | float (>0)       |                                                                                                                   |
| **Excretion recycling (%)**        | Rate of nitrogen recycling from excretions                                                           | float ([0, 100]) | Forms the slurry to be spread on crops                                                                             |
| **Diet**        | Diet ID to use for this population                                                           | str | Must have a corresponding Diet ID in Diet tab                                                                            |

### Product Data

Here is the data related to **products** necessary for GRAFS-E to simulate nitrogen flows in agricultural systems:

| **Column Name**       | **Description**                                 | **Type**         | **Comment**                                                                                                           |
| --------------------- | ----------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Production (kton)** | Gross production in kilotons                    | float (>0)       | Be careful not to provide the production in dry matter                                                                |
| **Nitrogen Content (%)** | Nitrogen content of the product               | float ([0, 100]) | Nitrogen content of the raw product, not in dry matter                                                                 |
| **Origin compartment** | Name of the compartment producing the product. | str              | It must match a defined compartment. Typically, the name of the crop (crops table) or livestock (livestock table)      |
| **Type**              | Type of production ("plant" or "animal")       | str              | See Typology                                                                                                          |
| **Sub Type**          | Subtype of the production                      | str              | See Typology                                                                                                          |
| **Waste (%)**          | Share of Nitrogen Production wasted                      | float ([0, 100])              | Depending of the focus of the study, this can include transport waste, processing waste, distribution and domestic waste... This describe a flow from product to waste compartment.                                                                                                        |
| **Other uses (%)**          | Share of Nitrogen Production used outside of Agro-Food system                     | float ([0, 100])              | This can include product for energy production, building material... This describe a flow from product to other sectors.                                                                                                          |
| **Methanization power (MWh/tFW)**          | Energy production potential in methanizer by ton of input fresh weight                     | float ([0, 100])              | Put 0 if this is not intended to methanizer                                                                                                          |

#### Product Typology

To handle products and the flows associated with them, GRAFS-E uses a standardized two-level typology (Type and Sub Type). Each Sub Type 'i' defines a "Trade i" compartment. This means that trade exchanges are categorized by Sub Type. The Sub Type of plants products are freely defined by the user. Only the **grazing** Sub Type has a special rule. **grazing** products cannot be imported or exported.

The Sub Types for **animals** products are fixed and include 3 categories.

- **Animal**: Includes all products derived from animals:
    - **edible meat**: Consumable meat
    - **dairy**: Dairy products and eggs
    - **non edible meat**: Non-consumable meat. All production in this sub-type is directed to the 'other sector' (used in other industries or burned).

```{warning}
**Warning**: The Sub types must be different than **crops** categories.
```
    
### Global Data

GRAFS-E also relies on **global variables**, which apply to all compartments for a given year and territory.

| **Column Name** | **Description**                             | **Type**         | **Comment**                                                                                                           |
| ----------------| ------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| **item**        | Name of the global variable                 | str              | The number and name of the items are fixed (see next section)                                                        |
| **value**       | Value of the item                           | depends on item  | See next section                                                                                                       |

#### Global Values

The name and number of global variables are fixed. Any differences from this will raise an error.

| **Variable Name**                                            | **Description**                                                                                                                                                                                                                                    | **Type**         | **Comment**                                                                                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Methanizer Energy Production (GWh)**                                              | Energy production objective for methanizer.                                                                                                                                                                                          | float (>0)       |                                        |
| **Weight diet**                                              | Weight of the optimization model for the diet constraint                                                                                                                                                                                          | float (>0)       | Adjusts the importance of respecting the proportions defined in the diets of populations and livestock                                        |
| **Weight import**                                              | Weight of the optimization model for the import constraint                                                                                                                                                                                          | float (>0)       | The higher is this weight, the more the model will try to limit import and change diet to consumme local production                                         |
| **Weight methanizer production**                                              | Weight of the optimization model for the energy methanizer production constraint                                                                                                                                                                                          | float (>0)       | Adjusts the importance of respecting the energy production goal                             |
| **Weight methanizer inputs**                                              | Weight of the optimization model for the constraint on methanizer diet                                                                                                                                                                                        | float (>0)       | Adjusts the importance of respecting the proportions defined in the diet of methanizer. If no value is given in project or data file, the methanizer input weight is Weight diet.                                        |
| **Weight distribution**                                       |Weight of the optimization model for the constraint on products allocation in diet groups.                                                                                                                                       | float (>0)       | See optimization model section for full definition. This input data is optional. If no value is given in project or data files, the distribution weight is $\text{weight diet}/10$                                                                |
| **Weight fair local split**                                            | Weight of the optimization model for the constraint on products distribution to consummers.                                                                                                                                                                                           | float (>0)       | See optimization model section for full definition. This input data is optional. If no value is given in project or data files, the distribution weight is $\text{weight diet}/20$                                                             |
| **Total Synthetic Fertilizer Use on crops (ktN)**            | Total synthetic nitrogen use on crops not in the following categories: "natural meadow", "leguminous", "temporary meadows"                                                                                                                     | float (>0)       | Used to normalize the synthetic fertilizer usage after calculating needs.                                           |
| **Total Synthetic Fertilizer Use on grasslands (ktN)**       | Total synthetic nitrogen use on grasslands not in the following categories: "natural meadow", "temporary meadows"                                                                                                                                | float (>0)       | Used to normalize the synthetic fertilizer usage after calculating needs.                                           |
| **Atmospheric deposition coef (kgN/ha)**                     | Atmospheric deposition coefficient per unit of area                                                                                                                                                                                               | float (>0)       | This flux is considered to originate from ammonia and nitrous oxide present in the atmosphere                         |
| **coefficient N-NH3 volatilization synthetic fertilization (%)** | Ammonia volatilization coefficient during the application of synthetic fertilizers                                                                                                                                                                 | float ([0, 100]) | Then 1% of this volatilization recombines into nitrous oxide in the atmosphere                                      |
| **coefficient N-N2O volatilization synthetic fertilization (%)** | Nitrous oxide volatilization coefficient during the application of synthetic fertilizers                                                                                                                                                           | float ([0, 100]) |                                                                                                                      |
| **Enforce animal share**                                      | If True, the proportions of animal and plant consumption defined in the diets will be set as a hard constraint by the model. The model will not propose substitutions of animal and plant proteins to balance the flows. | Bool             | Set to True if diet data is solid. False will be particularly useful for scenario analysis.                          |
| **Green waste methanization power (MWh/ktN)**                                      | Methanization energy potential for green waste by ktN of input. | float (>0)             |                          |