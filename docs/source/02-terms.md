# Terminology and Units

## Units
- All flows are expressed in **ktN / yr** (kilo ton of Nitrogen per year) unless stated otherwise. ktN refer to the amount of pure Nitrogen. For example, a flow of 1 ktN/yr of ammonia (NH3) correspond to a real flow of 1.21 (17/14) ktNH3/yr.

## Compartments

### What is a compartment ?

E-GRAFS describe the flows between compartments. A compartment, is the minimal relational unit in the model. Flows might exist inside a compartment but are not represented by E-GRAFS. Only flows between compartment are considered. A compartment is a black box corresponding usually to a process a production or a reservoir.
These compartments belongs to differents type : crops, livestock, product, population, trade, environment, industry. Some compartments are defined fy the input data, other are fixed in E-GRAFS framework. Usually a E-GRAFS modelization has these kinds of compartments : 

- **Crops:** These compartments refer to agricultural lands. These are process unit consumming fertilizer and producing vegetal products and losses. In E-GRAFS terminology, crops are yearly crops, permanents crops and grasslands. They are defined in the 'crops' tab in input data. It is possible to define crops by system production (e.g., Wheat standard, Wheat biological...).
- **Livestock:** These compartments refer to livestock farms. These are process unit consumming feed and producing animal products, excretions and losses. They are defined in the 'livestock' tab in input data. It is possible to define livestock by system production (e.g., meat bovine, milk bovine...).
- **Excretion:** These compartments refer to excretion fluxes. These compartments are fixed by livestock compartments. E-GRAFS manages 3 types of excretion: manure, slurry and grasslands excretion. For each livestock compartment, there must be 3 corresponding excretion compartments (see input page for more information). They are defined in the 'excretion' tab in input data.
- **Population:** These compartments refer to populations living on the modeled territory. There can be one population as several depending on input data, size and cultural uniformity of people living in the considered territory. Population compartments are process unit consumming food and producing excretion and losses. Populations are defined in the 'pop' tab in input data.
- **Product:** These compartments refer to agricultural products. This type of compartment is general enough to add plant and animal products, raw and processed product. Yet E-GRAFS does not represent (yet) the stages of food processing. Products are defined in the 'prod' tab in input data.
- **Environment:** These compartments refer the environmental reservoir and outlet. No living systems can function by itself. This is true for agro-food systems which require nitrogen input and need an outlet for waste and losses. These compartments represents atmosphere, land and water. They are defined by E-GRAFS. The user cannot add or remove environmental compartments. These compartments are:
    - hydro-system: represents continental water as ground water and surface water
    - atmospheric N2: represents nitrogen gas ($N_2$) composing 70% of atmosphere.
    - atmospheric NH3: represents ammoniac ($NH_3$) volatilization.
    -  atmospheric N2O: represents nitroux oxyde ($N_2O$) emissions.
    - soil stock: represents land reservoir and outlet.
    - other losses: other sources and outlet for flows with undefined source or target outside agricultural system.
- **Trade:** These compartments refer to trade processes, i.e. flows to other territories. Trade are indirectly defined by the user. Each 'Sub Type' of Product define a corresponding trade category. E-GRAFS include a fixed compartment 'fischery products' for the consumption of seafood products.
- **Industry:** These compartments refer to insdustrial processes. Industry compartments are fixed in E-GRAFS framework. E-GRAFS use a simple yet functionnal representation of industry for agro-food systems. Compartments are:
    - Haber-Bosch which produce as much synthetic fertilizer as used. Relies on atmospheric N2 reservoir.
    - other sectors which represent all other sectors using or producing nitrogen input.

## Acronyms and Terminology
- **BNF**: Biological Nitrogen Fixation.
- **HB**: Haber–Bosch (synthetic fertilizers).
- **prod**: Product
- **pop**: Population
- **Consummer**: designates an livestock type or a population.
- **Grasslands**: designates teporary meadow and natural meadow.