# Overview

## E-GRAFS purpose

E-GRAFS (Generalized Representation of Agro-Food Systems Extended) is a python model to represent Nitrogen flows in agriculturals systems. The model has been created to be adapted to many contexts (scale, territory and period). E-GRAFS can model agriculture flows is a small French rural city in 1870 or at a country scale as Ivory Coast in 2020. 
E-GRAFS relies on common agricultural statistics (e.g.: area, production) and technical coefficients (e.g.: NH3 volatilisation) which make it easily useable.

## Nitrogen as physical core of Agri-Food systems

Nitrogen plays a central role in agricultural systems. First, it is a crucial element for the functioning of biological systems, especially for the synthesis of proteins {cite:p}`lehninger2017`. As a consequence, this is a major macro-nutriment for plant growth {cite:p}`hawkesford2023functions`. Moreover, nitrogen is a good environmental indicator of agricultural atmospheric pollution (volatilization of $NH_3$ and $NO_2$, {cite:p}`terman1980volatilization`) and of groundwater pollution (for example, eutrophication, {cite:p}`conley2009controlling`).

Tracking nitrogen dynamics offers to specify the links between agricultural practices, international trade {cite:p}`dupasTimeDynamicsInvariant2019`, industrial sectors {cite:p}`harchaoui2019energy`, nutrient use efficiency {cite:p}`nowakNutrientRecyclingOrganic2015`.

## GRAFS : the precursor

E-GRAFS is based over a previous model : the Generalized Representation of Agro-Food Systems (GRAFS). This model has been developed by Gilles Billen and Josette Garnier {cite:p}`lassalettaFoodFeedTrade2014` and further developed by Julia Le Noë {cite:p}`le2018long, le2018biogeochemical`.
GRAFS is a general framework to account nitrogen, carbon and phophorus flows between 4 compartements : arable land, permanent grassland, livestock, population. It has been used in many context for example, French territories {cite:p}`lenoeHowStructureAgrofood2017`, Paris basin, Spanish food system from 19th century {cite:p}`rodraNestingNitrogenBudgets2023`.
Such tool for deep is precious to observe changes over a long range or compare different agricultural metabolisms over space and time. GRAFS has aslo been used for prospective scenarisation {cite:p}`TransAgriSce`.

## E-GRAFS

### Why E-GRAFS ?

GRAFS is a precious tool and it's core concepts have shown their operational efficiency for system analysis. Yet its implementation suffer from lacks of simplicity and adaptivity. GRAFS is basicaly an excel model which make difficult to create a new project, identify errors and read somebody else work. Moreover, the flow representation is fixed (4 compartments) and is too simple to capture interresting paterns, even if data are available.

E-GRAFS has been developped by Adrien Fauste Gay with the precious help of Julia Le Noë to adress these issues.

### General description

E-GRAFS produces a more detailed and flexible representation of nitrogen flows. E-GRAFS integrates optimization techniques to dynamically allocate nitrogen resources, enabling more realistic simulations of agricultural systems under diverse situations.

### Core Concepts

E-GRAFS relies on a physical description of agricultural systems. X core rules guided the conception of E-GRAFS :

- **Nothing comes from nothing:** all compartments are directly or indirectly connected to environmental, industrial, or trade reservoirs and outlets—no orphan stocks or flows.

- **It adds up:** conservation (mass/energy) is enforced; closed physical loops are explicit to keep accounting coherent across scales.

- **The system is self-coherent:** units, time steps, boundaries, and sign conventions are consistent across modules; every inflow has a counter-party outflow and balances close within numerical tolerances.

- **The system is more than the sum of its components:** feedbacks and indirect effects are modeled through networked interactions, allowing combined interventions to reveal emergent behavior at system scale.

### Questions adressed by E-GRAFS

E-GRAFS is a model. As all models, it relies on hypothesis valid in a specific range. Because a global modelisation at 1:1 scale is impossible, there will never be a model fit for all questions. E-GRAFS does not escape this limit despite it's great adaptability. E-GRAFS has been designed to physically loop representations of agricultural systems. More especially :

- Which flows make agricultural production and reproduction for a given context?
- Why some systems are not likely to last?
- What levers are available to change metabolism? To what extend? With what effect ?

Moreover, E-GRAFS aims at representing agricultural systems, yet systems do not have physical existence {cite:p}`SystemeDelevageDunea`. E-GRAFS is designed to give a physically coeherent global vision of agricultural functionning from a social metabolism perspective.

### Difference from other Nitrogen Flow models

Unlike static models like biomass balance model GlobAgri-AgT {cite:p}`le2018globagri`, E-GRAFS accommodates complex interactions between compartments, including specific crops, livestock, population groups, trade, industries and environment. Globagri approach uses fixed production and allocation processes. E-GRAFS relies on dynamic biomass allocation to meet all demands. This approach allows for better adaptation to external factors, making E-GRAFS a well-suited tool for investigating past agro-food trajectories and designing future scenarios. 

This higher compartment resolution is a significant improvement over existing Social Metabolism models as the energy-nitrogen model developed by Chatzimpiros {cite:p}`harchaoui2019energy` or the holistic approach of Spanish metabolism {cite:p}`gonzalez2020social` which use a simplistic representation of agro-food systems.

## Next development steps

The current version of E-GRAFS is the stable version for Nitrogen and Carbon flows representation which are critical to understand the functionning of agricultural systems. Yet other flows have critical importance as Phosphorus.

This version of E-GRAFS has been designed to represent historical agro-food systems. It can already be used to propose a physical view of a scenarized system. But for more control over the scenarisation and to simplificate the use of E-GRAFS in prospective view, a E-GRAFS prospective mode is under current development.

E-GRAFS is developped in Python which make it accessible for a large part of scientific community. Yet, to spread E-GRAFS among politicians, associations and the general public, a graphical interface is under current development. 

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
:labelprefix: OV