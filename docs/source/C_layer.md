# Carbon Layer

GRAFS-E includes a Carbon layer in beta mode. This layer represents Carbon fluxes between compartments. This layer aims to deepen the analysis of agricultural metabolism, enlarge public (policy makers, engeeneers) and propose easy to use Carbon footprint evaluation for global system and each compartment.

## How to use

The carbon layer can be called from C_class.py. It includes 2 classes :
- DataLoader_Carbon(project_path, data_path, region, year)
- CarbonFlowModel(data, year, region)

With DataLoader_Carbon() the class to load data. Project and Data files are structured the same way as for Nitrogen layer (see input.md). Yet these files must include supplementary data described in section 'Supplementary Input Data'. Year and Region must be given at the data loading stage.

CarbonFlowModel() is the class using input data to produce output data (see section 'Carbon Output Data'). Methods are similar to NitrogenFlowModel().

Once input data are correctly formated, simply use:

```python
# 1) Import
from grafs_e import DataLoader_Carbon, CarbonFlowModel

# 2) Load inputs (paths to your project config and your data file)
input_project_file_path = "path/to/project.yml"   # or .json/.toml as applicable
input_data_file_path    = "path/to/data.xlsx"     # or .csv/.parquet/.sqlite etc.
territory = "MyRegion"
year = 2022

data = DataLoader_Carbon(input_project_file_path, input_data_file_path, territory, year)

# 3) Instantiate and run
c = CarbonFlowModel(data, territory, year)
```

Output can be get as for Nitrogen layer.

## Methodology

Carbon Layer is build based on Nitrogen Layer. DataLoader_Carbon() uses Dataloader() and NitrogenFlowModel() to build Nitrogen transition matrix. 

For products fluxes, a ration $\frac{C}{N}$ is used to compute carbon flows. For excretion, humification coefficient and $CH_4$ emission factors are used to compute carbon fluxes from excretion compartments to atmospheric $CO_2$, atmospheric $CH_4$ and soil stock.

Crops Carbon comes from photosynthesis (atmospheric $CO_2$ compartment) which balance crops production : products, roots and residues. Roots and residues have a humification coefficient to get carbon soil stock fluxes.

Livestock compartments have a $CH_4$ enteric coefficient to compute these emissions. 

Population and Livestock compartments respiration is computed by the difference between input and output. Humification coefficients and Carbon contents must be chosen carefully to avoid negative respiration. With correctly build dataset, this shouldn't happen.

Crops and Livestock compartment have a Machine and livestock Carbon intensity factor reflecting the fluxes from Hydrocarbures to atmospheric $CO_2$. 

Haber-Bosch process include a methan consumption factor for synthetic fertilizer production. This include direct ($CH_4$ for chemical reaction) and indirect ($CH_4$ to heat and pressure chemical reactor) consumption.

## Input data

The Carbon layer as been conceived to necessitate as little as possible supplementary data. Here is the description of data required to run the CarbonFlowModel. These data must be included either as defaut value in project file or interannual values in data file. See 'GRAFS_project_example_C.xlsx' and 'GRAFS_data_example_C.xslx' in example file.

### Crops

| Column Name                         | Description                                                                                          | Type             | Comment                                                                                                            |
| -----------------------------------  | ---------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Carbon Mechanisation Intensity (ktC/ha)**                            | Carbon emissions from the use of machinery (mobile and infrastructure) for cultivation                                                                                   | float (>0)              |                                                                                                                   |
| **Residue Humification Coefficient (%)**                  | Share of Carbon residue being stabilized in soil                                                             | float ([0,100])              | Draw the flows from crops to soil stock and to atmospheric $CO_2$                                                |
| **Surface Root Production (kgC/ha)**                            | Surface Carbon roots production of the crops                                                                                   | float (>0)              |    The root production relies on a simplistic assumption of uniform root production. This might be upgraded by making it proportional to yield.                                                                                                               |
| **Root Humification Coefficient (%)**                  | Share of Carbon root being stabilized in soil                                                             | float ([0,100])              | Draw the flows from crops to soil stock and to atmospheric $CO_2$                                                |

### Livestock

| Column Name                         | Description                                                                                          | Type             | Comment                                                                                                            |
| -----------------------------------  | ---------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **C-CH4 enteric/LU (kgC)**                            | Methan emmision from enteric processes per livestock unit                                                                                   | float (>0)              | Must be computed based on agronomic data (e.g. MCF, P-$CH_4$, VSE..)                                                                                                                   |
| **Infrastructure CO2 emissions/LU (kgC)**                  | Carbon emissions from the use of machinery (mobile and infrastructure) for livestock breeding                                                             | float (>0)              |                                                 |

### Product

| Column Name                         | Description                                                                                          | Type             | Comment                                                                                                            |
| -----------------------------------  | ---------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Carbon Content (%)**                            | Carbon content of product                                                                                   | float (>0)              | Usually 45-50% for plant based product.                                                                                                                   |

### Excretion

| Column Name                         | Description                                                                                          | Type             | Comment                                                                                                            |
| -----------------------------------  | ---------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **C/N**                            | Carbon to Nitrogen content ratio. This ration must be given after carbon losses ($CO_2$ and $CH_4$) and Nitrogen losses ($NH_3$, $N_2O$ and $N_2$ losses).                                                                                   | float ([0,100])              |                                                                                                                   |
| **CH4 EM (%)**                            | Methan emission factor for animal excretion                                                                                   | float ([0,100])              |                                                                                                                   |
| **Humification Coefficient (%)**                  | Share of Carbon excreted being stabilized in soil                                                             | float ([0,100])              | Draw the flows from excretion to soil stock and to atmospheric $CO_2$                                                |

### Population

Similar data than Excretion compartments.

| Column Name                         | Description                                                                                          | Type             | Comment                                                                                                            |
| -----------------------------------  | ---------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **C/N**                            | Carbon to Nitrogen content ratio of excretion. This ration must be given after carbon losses (1 - humification share) and Nitrogen losses ($NH_3$, $N_2O$ and $N_2$ losses).                                                                                   | float ([0,100])              |                                                                                                                   |
| **CH4 EM (%)**                            | Methan emission factor for animal excretion                                                                                   | float ([0,100])              |                                                                                                                   |
| **Humification Coefficient (%)**                  | Share of Carbon excreted being stabilized in soil                                                             | float ([0,100])              | Draw the flows from excretion to soil stock and to atmospheric $CO_2$                                                |

### Global data

| **Variable Name**                                            | **Description**                                                                                                                                                                                                                                    | **Type**         | **Comment**                                                                                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Total Haber-Bosch methan input (kgC/kgN)**                                         | Total methan consumption of methan (direct and indirect) for Haber bosch process                                                                                                                                                                                                     | float (>0)       |                                                                  |