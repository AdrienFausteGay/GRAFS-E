# GRAFS-Extended: Comprehensive Analysis of Nitrogen Flux in Agricultural Systems

## Overview

The GRAFS-extended model serves as an advanced tool designed to analyze and map the evolution of nitrogen utilization within agricultural systems, with a particular focus on 33 regions of France from 1852 to 2014. This model builds upon the GRAFS/PVAR framework and integrates graph theory to provide a detailed analysis of nitrogen flows in agriculture, identifying key patterns, transformations, and structural invariants. The model enables researchers to construct robust prospective scenarios and examine the global structure of nitrogen flows in agricultural ecosystems.

## 🚀 How to Run the Model

To run the model, follow these steps:

1. **Install the package** using `pip`:

   ```bash
   pip install grafs_e
   ```

2. **Launch the interface** with the following command:

   ```bash
   grafs-e
   ```

## Features

- Historical Data: Covers nitrogen flow analysis for the period from 1852 to 2014 across 33 French regions.
 - Graph Theory Integration: Utilizes graph theory to examine and analyze nitrogen flow networks in agriculture.
 - Comprehensive Nitrogen Flux Model: Includes 36 varieties of crops, 6 livestock categories, and multiple industrial sectors.
 - Prospective Scenario Building: Enables the development of robust, future-oriented models based on historical trends.

## Methods
### GRAFS Development

The GRAFS model is designed to encapsulate the nitrogen utilization process in agricultural systems by considering historical transformations in French agricultural practices. It captures the transition from traditional crop-livestock agriculture to more intensive, specialized systems.

- Uniform Allocation Strategy: Used for distributing nitrogen across crops, livestock, and other agricultural categories.
- Disaggregation of Flows: Allows detailed examination of nitrogen interactions across 51 categories, including crops, livestock, industrial sectors, and ecosystems.

### Network Analysis

The GRAFS-extended model creates a transition matrix for nitrogen flows between various categories. It utilizes network theory to analyze critical nodes, resilience, and structural invariants. Flow probability distributions are also studied to assess the concentration and specialization of agricultural activities.

## Results

The model generates extensive transition matrices representing nitrogen flows between different agricultural components. Analysis reveals that a small number of high-impact flows drive much of the nitrogen distribution in agricultural systems, particularly through key processes such as synthetic fertilizer application and crop-livestock interactions.

## Future Work

- Network Resilience: Further analysis using Ecological Network Analysis (ENA) can help improve the model's understanding of resilience in nitrogen flows.
- Multi-Layer Models: Future versions may include additional structural flows such as energy, water, and financial transfers.

## Usage

To replicate the results or analyze other regions using the GRAFS-extended model, ensure the following data are available:

- Crops: Cultivated area, production volume, nitrogen content, and nitrogen fixation coefficients.
- Livestock: Number of animals, nitrogen excreted, and dietary composition.
- Population: Size and dietary patterns for plant and animal nitrogen intake.

## Data Requirements

This model requires regional data on crops, livestock, population, and nitrogen-related variables, which are typically available from national databases.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the LICENSE file for details.
Contact

For any questions or contributions, feel free to reach out to Adrien Fauste-Gay at [adrien.fauste-gay@univ-grenoble-alpes.fr].