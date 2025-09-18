# Overview

## Nitrogen as physical core of Agri-Food systems

Nitrogen plays a central role in agricultural systems. First, it is a crucial element for the functioning of biological systems, especially for the synthesis of proteins {cite:p}`lehninger2017`. As a consequence, this is a major macro-nutriment for plant growth {cite:p}`hawkesford2023functions`. Moreover, nitrogen is a good environmental indicator of agricultural atmospheric pollution (volatilization of $NH_3$ and $NO_2$, {cite:p}`terman1980volatilization`) and of groundwater pollution (for example, eutrophication, {cite:p}`conley2009controlling`).

Tracking nitrogen dynamics offers to specify the links between agricultural practices, international trade {cite:p}`dupasTimeDynamicsInvariant2019`, industrial sectors {cite:p}`harchaoui2019energy`, nutrient use efficiency {cite:p}`nowakNutrientRecyclingOrganic2015`.

## GRAFS : the precursor

To do so, the Generalized Representation of Agro-Food Systems (GRAFS) model has been developed by Gilles Billen and Josette Garnier {cite:p}`lassalettaFoodFeedTrade2014` and further developed by Julia Le Noë {cite:p}`le2018long, lenoeHowStructureAgrofood2017, le2018biogeochemical`. This model is based on the evolution of nitrogen utilization fluxes across four compartments of agro-food systems: arable land, permanent grassland, livestock, population.

## GRAFS-E

### General description

GRAFS-E (GRAFS-Extended) incorporates a more detailed and flexible representation of nitrogen flows. This extension consists of disaggregating GRAFS flows. 
From 4 objects which exchange nitrogen, GRAFS-E use compartment defined for user data. GRAFS-E integrates optimization techniques to dynamically allocate nitrogen resources, enabling more realistic simulations of agricultural systems under diverse scenarios.

GRAFS-E has 2 modes :
- Historical mode : 
- Prospective mode : Using similar mecanisms as historic mode, the prospective mode build possible physical functionning of Agri-food systems in French territories from 2025 to 2050.  

### Core Concepts

GRAFS-E relies on a physics description of agricultural systems. X core rules guided the conception of GRAFS-E :

- Nothing come from 

### Difference from other Nitrogen Flow models

Unlike static models like biomass balance model GlobAgri-AgT {cite:p}`le2018globagri`, GRAFS-E accommodates complex interactions between over 66 disaggregated objects, including specific crops, livestock, and population groups. Globagri approach uses fixed production processes. GRAFS-E relies on dynamic biomass allocation to meet all demands. This approach allows for better adaptation to external factors, making GRAFS-E a well-suited tool for investigating past agro-food trajectories and designing future scenarios. 

It retains GRAFS’s accessibility while expanding its applicability to broader temporal and spatial scales, from farm-level assessments to national analyses. We used data enabling the construction of nitrogen flow matrices across 33 territories as defined by {cite:p}`lenoePlaceTransportDenrees2016` in France from the 19th century to the present. This approach considers environment and human managed processes which enable a detailed socio-metabolic analysis of nitrogen flows. 
This higher resolution is a significant improvement over existing Social Metabolism models as the energy-nitrogen model developed by Chatzimpiros {cite:p}`harchaoui2019energy` or the holistic approach of Spanish metabolism {cite:p}`gonzalez2020social`.

## References

```{bibliography}
:filter: docname in docnames