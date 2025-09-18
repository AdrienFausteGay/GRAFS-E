ChatGPT a dit :

# Output

The output data is organized into a transition matrix showing the flows between each compartment, along with 7 dataframes presenting the input data, intermediate calculations, and output data:

- **df_cultures**: related to crops
- **df_elevage**: related to livestock
- **df_pop**: related to populations
- **df_prod**: related to products
- **df_global**: provides the global input data.
- **allocations_df**: shows the output of the allocation model between products and consumers (humans and animals)
- **deviations_df**: shows the deviations between the allocation model output and the target (Diet tab from the data Excel sheet)

**df_cultures**, **df_elevage**, **df_pop**, **df_prod**, and **df_global** contain both outputs and input data. **df_global** simply records the global variables provided as input.

## Transition Matrix

The most comprehensive output from the model is a transition matrix. This square matrix shows the flows between each compartment. It has the same size as the number of compartments. Each coefficient \(c_{ij}\) represents the nitrogen flow in ktN from compartment i to compartment j. The sum of column j gives all the inputs of a compartment, while the sum of row i gives all the outputs of a compartment.

For all compartments in the following sets:
- **crops**
- **livestock**
- **products**
- **population**

The sum of the inputs equals the sum of the outputs. These compartments are considered **internal**, and they are balanced.

![Transition Matrix from example data](_static/matrix.png)

## Crops Outputs

The output data related to crops is recorded in **df_cultures**. In addition to the input columns for crops, the following columns are created by GRAFS-E:

| **Column Name**                                 | **Description**                                                                                      | **Type** | **Comment**                                                                                                           |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| **Main Nitrogen Production (ktN)**              | Nitrogen production in ktN from the main commercial product of the crop                               |          |                                                                                                                       |
| **Yield (qtl/ha)**                              | Yield in quintals per hectare                                                                         |          |                                                                                                                       |
| **Yield (kgN/ha)**                              | Yield in ktN per hectare                                                                               |          |                                                                                                                       |
| **Symbiotic fixation (ktN)**                    | Symbiotic nitrogen fixation performed by the crop                                                     |          |                                                                                                                       |
| **Total Non-Synthetic Fertilizer Use (ktN)**     | Total fertilizer use excluding synthetic fertilizers                                                 |          | Used to calculate the need for synthetic fertilizers                                                                  |
| **Surface Non-Synthetic Fertilizer Use (kgN/ha)**| Total fertilizer use per unit area excluding synthetic fertilizers                                    |          |                                                                                                                       |
| **Leguminous Nitrogen Surplus (ktN)**           | Difference between symbiotic fixation and crop nitrogen needs                                         |          | Only for crops that perform symbiotic nitrogen fixation                                                               |
| **Leguminous heritage (ktN)**                   | Nitrogen from the surplus of symbiotic nitrogen fixation distributed to other crops                   |          | Only benefits crops categorized as 'cereal (excluding rice)'                                                          |
| **Raw Surface Synthetic Fertilizer Use (kgN/ha)**| Theoretical need for synthetic fertilizer per unit area                                               |          | Calculated by subtracting Surface Fertilization Need (kgN/ha) and Surface Non-Synthetic Fertilizer Use (kgN/ha)       |
| **Raw Total Synthetic Fertilizer Use (ktN)**    | Theoretical total need for synthetic fertilizer for the crop                                          |          |                                                                                                                       |
| **Adjusted Total Synthetic Fertilizer Use (ktN)**| Adjusted need for synthetic fertilizer based on the total synthetic fertilizer consumption of the territory |          | See calculation for \(\Gamma\)                                                                                         |
| **Adjusted Surface Synthetic Fertilizer Use (kgN/ha)**| Adjusted need per unit area based on the total synthetic fertilizer consumption of the territory      |          |                                                                                                                       |
| **Volatilized Nitrogen N-NH3 (ktN)**            | Portion of synthetic fertilizer volatilized as ammonia                                                |          |                                                                                                                       |
| **Volatilized Nitrogen N-N2O (ktN)**            | Portion of synthetic fertilizer volatilized as nitrous oxide                                          |          | Includes indirect emissions due to the partial recombination of ammonia into nitrous oxide in the atmosphere            |
| **Total Nitrogen Production (ktN)**             | Sum of all the products produced by this crop                                                         |          |                                                                                                                       |
| **Balance (ktN)**                               | Difference (positive or negative) between the inputs (fertilization) and nitrogen removal from the crop |          |                                                                                                                       |

For detailed calculations, refer to the **GRAFS-E Engine** section. Unless stated otherwise, the type of each column is a positive real number. The **Surface Fertilization Need (kgN/ha)** column is completed by multiplying the **Fertilization Need (kgN/qtl)** and **Yields (qtl/ha)** columns where the **Fertilization Need (kgN/qtl)** is non-zero.

## Livestock Outputs

The output data related to livestock is recorded in **df_elevage**. In addition to the input columns for livestock, the following columns are created by GRAFS-E:

| **Column Name**                            | **Description**                                                                            | **Type** | **Comment**                                      |
| -----------------------------------------  | ------------------------------------------------------------------------------------------ | -------- | ------------------------------------------------ |
| **Edible Nitrogen (ktN)**                 | Total nitrogen production from edible products in the livestock                           |          |                                                  |
| **Non-Edible Nitrogen (ktN)**             | Total nitrogen production from non-edible products in the livestock                       |          |                                                  |
| **Dairy Nitrogen (ktN)**                  | Total nitrogen production from dairy and egg products in the livestock                    |          |                                                  |
| **Excreted nitrogen (ktN)**               | Total nitrogen excreted, all types of management combined                                 |          |                                                  |
| **Ingestion (ktN)**                       | Total ingestion by the livestock                                                           |          | Sum of production and excretion                  |
| **Consumed nitrogen from local feed (ktN)**| Nitrogen consumed from local products                                                      |          | Result of the optimization model                 |
| **Consumed Nitrogen from imported feed (ktN)** | Nitrogen consumed from imported products                                                 |          | Result of the optimization model                 |
| **Net animal nitrogen exports (ktN)**     | Net export of products from this livestock type                                            |          |                                                  |
| **Conversion factor (%)**                 | Ratio between production from livestock and ingestion                                     |          |                                                  |

## Population Outputs

The output data related to populations is recorded in **df_pop**. In addition to the input columns for population, the following columns are created by GRAFS-E:

| **Column Name**         | **Description**                                      | **Type** | **Comment** |
| ---------------------- | ---------------------------------------------------- | -------- | ----------- |
| **Ingestion (ktN)**     | Total consumption (animal and plant) by the population |          |             |
| **Fishery Ingestion (ktN)**| Total consumption of sea products                  |          |             |

## Products Outputs

The output data related to products is recorded in **df_prod**. In addition to the input columns for products, the following columns are created by GRAFS-E:

| **Column Name**                           | **Description**                                                   | **Type** | **Comment**                                                                                              |
| ----------------------------------------- | --------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| **Nitrogen Production (ktN)**             | Nitrogen production in ktN for the product                       |          |                                                                                                          |
| **Available Nitrogen After Feed and Food (ktN)** | Nitrogen available after allocating consumption to livestock and humans |          | Output from the optimization model. For the 'grasslands' sub-type, this remainder is exported           |
| **Nitrogen For Feed (ktN)**               | Nitrogen allocated to livestock                                  |          | For detailed allocation, see **allocations_df**                                                            |
| **Nitrogen For Food (ktN)**               | Nitrogen allocated to populations                                |          | For detailed allocation, see **allocations_df**                                                            |

## Allocations Outputs

The output data related to the allocations proposed by the allocation model is recorded in **allocations_df**. Each row in this dataframe represents a flow between a product (or import) and a consumer. The table has four columns:
- **Product**: Name of the product
- **Consumer**: Name of the consumer
- **Allocated Nitrogen**: Amount of nitrogen allocated
- **Type**: Combination of [Imported, local] + [feed, food] based on whether the product is locally produced or imported, and whether the consumer is a population or livestock.

## Deviations Output

The output data related to the deviations between the allocations proposed by the optimization model and the ideal diet defined in the input data is recorded in **deviations_df**. Each row represents an entry in the ideal diet. The table has six columns:
- **Consumer**: The consumer
- **Type**: Animal or human
- **Expected Proportion (%)**: Proportion defined in the ideal diet
- **Deviation (%)**: Deviation between the target allocation and the allocation proposed by the optimization model (0% if the model perfectly respects the ideal diet)
- **Proportion Allocated (%)**: Proportion allocated by the optimization model
- **Product**: List of products for this diet entry.