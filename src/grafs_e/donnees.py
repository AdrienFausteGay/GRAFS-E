# Fichier pour regroupper les données
annees_disponibles = [
    "1852",
    "1885",
    "1906",
    "1929",
    "1946",
    "1955",
    "1965",
    "1970",
    "1974",
    "1978",
    "1981",
    "1985",
    "1989",
    "1993",
    "1997",
    "2000",
    "2004",
    "2006",
    "2008",
    "2010",
    "2012",
    "2014",
]

cultures = [
    "Wheat",
    "Rye",
    "Barley",
    "Oat",
    "Grain maize",
    "Rice",
    "Other cereals",
    "Straw",
    "Forage maize",
    "Forage cabbages",
    "Rapeseed",
    "Sunflower",
    "Other oil crops",
    "Sugar beet",
    "Potatoes",
    "Other roots",
    "Dry vegetables",
    "Dry fruits",
    "Squash and melons",
    "Cabbage",
    "Leaves vegetables",
    "Fruits",
    "Olives",
    "Citrus",
    "Hemp",
    "Flax",
]

legumineuses = [
    "Horse beans and faba beans",
    "Peas",
    "Other protein crops",
    "Green peas",
    "Dry beans",
    "Green beans",
    "Soybean",
    "Alfalfa and clover",
]
# Note: j'ai laissé other protein crop dans la liste des legumineuses
# même si cette culture n'est pas considérée comme
# une légumineuse dans pvar (cf fixation N2)

prairies = [
    "Natural meadow ",
    "Non-legume temporary meadow",
]

categories_mapping = {
    # Céréales (hors riz)
    "Wheat": "cereals (excluding rice)",
    "Rye": "cereals (excluding rice)",
    "Barley": "cereals (excluding rice)",
    "Oat": "cereals (excluding rice)",
    "Grain maize": "cereals (excluding rice)",
    "Other cereals": "cereals (excluding rice)",
    # Riz
    "Rice": "rice",
    # Oléagineux
    "Rapeseed": "oleaginous",
    "Sunflower": "oleaginous",
    "Flax": "oleaginous",
    "Hemp": "oleaginous",
    "Other oil crops": "leguminous",
    # Légumineuses
    "Soybean": "leguminous",
    "Horse beans and faba beans": "leguminous",
    "Peas": "leguminous",
    "Other protein crops": "leguminous",
    "Green peas": "leguminous",
    "Dry beans": "leguminous",
    "Green beans": "leguminous",
    # Racines
    "Sugar beet": "roots",
    "Potatoes": "roots",
    "Other roots": "roots",
    # Fourrages
    "Forage maize": "forages",
    "Forage cabbages": "forages",
    "Straw": "forages",
    # Prairies temporaires
    "Non-legume temporary meadow": "temporary meadows",
    "Alfalfa and clover": "temporary meadows",
    # Prairies permanentes
    "Natural meadow ": "natural meadows ",
    # Fruits et légumes
    "Dry vegetables": "fruits and vegetables",
    "Dry fruits": "fruits and vegetables",
    "Squash and melons": "fruits and vegetables",
    "Cabbage": "fruits and vegetables",
    "Leaves vegetables": "fruits and vegetables",
    "Fruits": "fruits and vegetables",
    "Olives": "fruits and vegetables",
    "Citrus": "fruits and vegetables",
}


Pop = ["urban", "rural"]

betail = [
    "bovines",
    "ovines",
    "caprines",
    "porcines",
    "poultry",
    "equine",
    # TODO Remettre ces catégories pour définir des régimes alimentaires plus fin
    # "eggs",
    # "cow milk",
    # "sheep and goat milk"
]

# ext = ["environnement", "Haber-Bosch", "autres secteurs"] + ["import-export azote végétal", "import-export azote animal", "import-export azote feed"]

# ext = ["atmosphere", "hydro-system", "other losses", "Haber-Bosch", "other sectors"] + ["plant nitrogen import-export", "animal nitrogen import-export", "feed nitrogen import-export"]
# Nouveau ext, plus détaillé au niveau des import/export
# Extraire les catégories uniques
# categories = sorted(set(categories_mapping.values()))

# # Construire les chaînes spécifiques pour chaque catégorie
# ext = [
#     "atmosphere", "hydro-system", "other losses", "Haber-Bosch", "other sectors",
#     "plant nitrogen import-export", "animal nitrogen import-export", "feed nitrogen import-export"
# ] + [f"{category} food nitrogen import-export" for category in categories] \
#   + [f"{category} feed nitrogen import-export" for category in categories]

ext = [
    "NH3 volatilization",
    "N2O emission",
    "hydro-system",
    "other losses",
    "soil stock",
    "atmospheric N2",
    "Haber-Bosch",
    "methaniser",
    "green waste",
    "other sectors",
    "animal trade",
    "fishery products",
    "cereals (excluding rice) food trade",
    "fruits and vegetables food trade",
    "leguminous food trade",
    "oleaginous food trade",
    "roots food trade",
    "rice food trade",
    "cereals (excluding rice) feed trade",
    "leguminous feed trade",
    "oleaginous feed trade",
    "forages feed trade",
    "temporary meadows feed trade",
]

labels_init = cultures + legumineuses + prairies + betail + Pop + ext

regions = [
    "Nord Pas de Calais",
    "Picardie",
    "Seine Maritime",
    "Calvados-Orne",
    "Eure",
    "Manche",
    "Bretagne",
    "Eure-et-Loir",
    "Ile de France",
    "Champ-Ard-Yonne",
    "Grande Lorraine",
    "Bourgogne",
    "Alsace",
    "Grand Jura",
    "Loire Amont",
    "Loire Centrale",
    "Loire aval",
    "Vendée-Charente",
    "Gironde",
    "Dor-Lot",
    "Landes",
    "Pyrénées occid",
    "Garonne",
    "Savoie",
    "Alpes",
    "Ain-Rhône",
    "Aveyron-Lozère",
    "Cantal-Corrèze",
    "Isère-Drôme Ardèche",
    "Gard-Hérault",
    "Côte d'Azur",
    "Grand Marseille",
    "Pyrénées Orient",
]

regions_extented = [
    "Nord Pas de Calais",
    "Picardie",
    "Seine Maritime",
    "Calvados-Orne",
    "Eure",
    "Manche",
    "Bretagne",
    "Eure-et-Loir",
    "Ile de France",
    "Champ-Ard-Yonne",
    "Grande Lorraine",
    "Bourgogne",
    "Alsace",
    "Grand Jura",
    "Loire Amont",
    "Loire Centrale",
    "Loire Aval",
    "Vendée-Charente",
    "Gironde",
    "Dor-Lot",
    "Landes",
    "Pyrénées occidentales",
    "Garonne",
    "Savoie",
    "Alpes",
    "Ain-Rhône",
    "Aveyron-Lozère",
    "Cantal-Corrèze",
    "Isère-Drôme Ardèche",
    "Gard-Hérault",
    "Côte d'Azur",
    "Grand Marseille",
    "Pyrénées orientales",
    "Belgique Luxembourg",
    "Pays-Bas",
    "Danemark",
    "Allemagne",
    "Autriche",
    "Royaume Uni",
    "Irlande",
    "Italie",
    "Espagne",
    "Portugal",
    "Pologne",
    "Bulgarie",
    "Roumanie",
    "Hongrie",
    "autres EU28",
    "Russie",
    "Amérique du Nord",
    "Amérique centrale",
    "Amérique latine",
    "Nouvelle Zélande et Australie",
    "Magreb et Proche Orient",
    "Afrique sub Saharienne",
    "Asie",
    "Océanie",
]


# Labels used in this work:
labels = cultures + legumineuses + prairies + betail + Pop + ext

label_to_index = {label: index for index, label in enumerate(labels)}
index_to_label = {v: k for k, v in label_to_index.items()}
node_color = {
    label_to_index[label]: (
        "gold"
        if label in categories_mapping
        and categories_mapping[label]
        in [
            "cereals (excluding rice)",
        ]
        else "lightgreen"
        if label in categories_mapping and categories_mapping[label] == "fruits and vegetables"
        else "orange"
        if label in categories_mapping and categories_mapping[label] == "roots"
        else "darkkhaki"
        if label in categories_mapping and categories_mapping[label] == "oleaginous"
        else "darkgreen"
        if label in categories_mapping and categories_mapping[label] == "leguminous"
        else "lime"
        if label in categories_mapping and categories_mapping[label] in ["temporary meadows", "forages"]
        else "darkgreen"
        if label in categories_mapping and categories_mapping[label] == "natural meadows "
        else "lightblue"
        if label in betail
        else "darkblue"
        if label in ["urban", "rural"]
        else "red"
        if label in ["atmospheric volatilization", "hydro-system", "other losses", "NH3 volatilization", "N2O emission"]
        else "sienna"
        if label in ["soil stock"]
        else "cyan"
        if label in ["Atmospheric deposition"]
        else "purple"
        if label in ["Haber-Bosch", "other sectors"]
        else "seagreen"
        if label in ["atmospheric N2"]
        else "gray"  # Default color if the label is not in any of the above categories
    )
    for label in labels
}

nickname_dict = {
    "Wheat": "Wht",
    "Rye": "Rye",
    "Barley": "Bly",
    "Oat": "Oat",
    "Grain maize": "G-Mz",
    "Rice": "Rice",
    "Other cereals": "Oth-Cer",
    "Straw": "Straw",
    "Rapeseed": "Rpsd",
    "Sunflower": "Sunf",
    "Other oil crops": "Oth-Oil",
    "Sugar beet": "Sgr-Bt",
    "Potatoes": "Pot",
    "Other roots": "Oth-Rt",
    "Dry vegetables": "Dry-Veg",
    "Dry fruits": "Dry-Frt",
    "Squash and melons": "Sq-Mel",
    "Cabbage": "Cab",
    "Leaves vegetables": "Lf-Veg",
    "Fruits": "Frt",
    "Olives": "Oliv",
    "Citrus": "Cit",
    "Hemp": "Hemp",
    "Flax": "Flax",
    "Forage maize": "F-Mz",
    "Forage cabbages": "F-Cab",
    "Alfalfa and clover": "Alf-Clv",
    "Non-legume temporary meadow": "N-Leg-Mdw",
    "Horse beans and faba beans": "H-Bns",
    "Peas": "Peas",
    "Other protein crops": "Oth-Prot",
    "Green peas": "G-Peas",
    "Dry beans": "D-Bns",
    "Green beans": "G-Bns",
    "Soybean": "Soy",
    "Natural meadow ": "Nat-Mdw",
    "bovines": "Bov",
    "ovines": "Ovn",
    "caprines": "Cap",
    "porcines": "Porc",
    "poultry": "Pltry",
    "equine": "Equ",
    "urbain": "Urb",
    "rural": "Rur",
    "environnement": "Env",
    "Haber-Bosch": "H-Bosch",
    "autres secteurs": "Oth-Sec",
    "import-export azote végétal": "Imp-Exp Veg N",
    "import-export azote animal": "Imp-Exp Ani N",
    "import-export azote feed": "Imp-Exp Feed N",
}

# Coefficients LU par catégorie
# Source : https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Livestock_unit_(LSU)
lu_coefficients = {
    "Milking cows": 1.0,
    "Suckler cows": 0.8,
    "Heifer for milking herd renewal over 2 yrs old": 0.8,
    "Heifer for suckler cows renewal over 2 yrs old": 0.8,
    "Heifer for slaughter over 2 yrs old": 0.8,
    "Males of milking type over 2 yrs old": 1,
    "Males of butcher type over 2 yrs old": 1,
    "Heifer for milking herd renewal between 1 and 2 yrs old": 0.7,
    "Heifer for suckler cows renewal between 1 and 2 yrs old": 0.7,
    "Heifer for slaughter between 1 and 2 yrs old": 0.7,
    "Males of milking type between 1 and 2 yrs old": 0.7,
    "Males of butcher type between 1 and 2 yrs old": 0.7,
    "Veal calves": 0.4,
    "Other females under 1 yr": 0.4,
    "Other males under 1 yr": 0.4,
    "kid goats": 0.1,
    "female goats": 0.1,
    "Other caprines (including male goats)": 0.1,
    "ewe lambs": 0.1,
    "Sheep": 0.1,
    "other ovines (incl. rams)": 0.1,
    "piglets": 0.027,
    "young pigs between 20 and 50 kg": 0.3,
    "Sows over 50 kg": 0.5,
    "Boars over 50 kg ": 0.3,
    "fattening pigs over 50 kg ": 0.3,
    "Laying hens for hatching eggs": 0.014,
    "Laying hens for consumption eggs": 0.014,
    "young hens": 0.014,
    "chickens": 0.007,
    "Duck for 'foie gras'": 0.01,
    "Ducks for roasting": 0.01,
    "Turkeys": 0.03,
    "Gooses": 0.02,
    "Guinea fowls": 0.001,
    "quails": 0.001,
    "mother rabbits": 0.02,
    "horses ": 0.8,
    "donkeys etc": 0.8,
}

animal_type_mapping = {
    "Milking cows": "bovines",
    "Suckler cows": "bovines",
    "Heifer for milking herd renewal over 2 yrs old": "bovines",
    "Heifer for suckler cows renewal over 2 yrs old": "bovines",
    "Heifer for slaughter over 2 yrs old": "bovines",
    "Males of milking type over 2 yrs old": "bovines",
    "Males of butcher type over 2 yrs old": "bovines",
    "Heifer for milking herd renewal between 1 and 2 yrs old": "bovines",
    "Heifer for suckler cows renewal between 1 and 2 yrs old": "bovines",
    "Heifer for slaughter between 1 and 2 yrs old": "bovines",
    "Males of milking type between 1 and 2 yrs old": "bovines",
    "Males of butcher type between 1 and 2 yrs old": "bovines",
    "Veal calves": "bovines",
    "Other females under 1 yr": "bovines",
    "Other males under 1 yr": "bovines",
    "kid goats": "caprines",
    "female goats": "caprines",
    "Other caprines (including male goats)": "caprines",
    "ewe lambs": "ovines",
    "Sheep": "ovines",
    "other ovines (incl. rams)": "ovines",
    "piglets": "porcines",
    "young pigs between 20 and 50 kg": "porcines",
    "Sows over 50 kg": "porcines",
    "Boars over 50 kg ": "porcines",
    "fattening pigs over 50 kg ": "porcines",
    "Laying hens for hatching eggs": "poultry",
    "Laying hens for consumption eggs": "poultry",
    "young hens": "poultry",
    "chickens": "poultry",
    "Duck for 'foie gras'": "poultry",
    "Ducks for roasting": "poultry",
    "Turkeys": "poultry",
    "Gooses": "poultry",
    "Guinea fowls": "poultry",
    "quails": "poultry",
    "mother rabbits": "poultry",
    "horses ": "equine",
    "donkeys etc": "equine",
}

## Régimes

herbes = [
    "Natural meadow ",
    "Non-legume temporary meadow",
    "Alfalfa and clover",
    "Straw",
]

# ATTENTION, cette mise en forme des données interdit d'avoir des proportions égales dans une catégorie
regimes = {
    "bovines": {
        0.61: herbes,
        0.08: ["Forage maize"],
        0.1: ["Barley", "Wheat", "Rye", "Other cereals"],
        0.2: ["Soybean", "Rapeseed", "Peas", "Horse beans and faba beans"],
        0.01: ["Forage cabbages"],
    },
    "ovines": {
        0.67: herbes,
        0.05: ["Forage maize"],
        0.08: ["Wheat", "Barley", "Oat", "Rye", "Other cereals"],
        0.06: ["Other oil crops", "Peas", "Other protein crops", "Sunflower"],
        0.09: ["Soybean", "Horse beans and faba beans"],
        0.04: ["Rapeseed"],
        0.01: ["Forage cabbages"],
    },
    "caprines": {
        0.63: herbes,
        0.07: ["Forage maize"],
        0.08: ["Barley"],
        0.22: ["Soybean", "Rapeseed", "Peas", "Horse beans and faba beans"],
    },
    "equine": {0.87: herbes, 0.13: ["Oat"]},
    "poultry": {
        0.28: ["Wheat", "Other cereals"],
        0.10: ["Grain maize"],
        0.57: ["Soybean", "Horse beans and faba beans"],
        0.05: [
            "Rapeseed",
            "Sunflower",
            "Other oil crops",
            "Peas",
            "Other protein crops",
        ],
    },
    "porcines": {
        0.18: ["Wheat"],
        0.12: ["Grain maize"],
        0.13: ["Barley", "Other cereals"],
        0.23: ["Soybean", "Horse beans and faba beans"],
        0.07: ["Rapeseed"],
        0.27: ["Peas", "Green beans", "Dry beans", "Green peas"],
    },
    "urban": {
        0.605: ["Wheat"],
        0.033: ["Rice"],
        0.004: ["Barley"],
        0.052: ["Grain maize"],
        0.003: ["Rye"],
        0.005: ["Oat"],
        0.006: ["Other cereals"],
        0.076: ["Potatoes"],
        0.002: ["Sugar beet"],
        0.001: ["Other roots"],
        0.01: ["Dry beans", "Green beans", "Horse beans and faba beans"],
        0.0045: ["Green peas", "Peas"],
        0.016: ["Other protein crops"],
        0.042: ["Dry fruits"],
        0.0008: ["Soybean"],
        0.0005: ["Rapeseed"],
        0.0003: ["Other oil crops"],
        0.105: ["Dry vegetables", "Cabbage", "Leaves vegetables"],
        0.031: ["Fruits", "Squash and melons"],
        0.0009: ["Citrus"],
        0.0015: ["Olives"],
        0.0325: ["Sunflower"],
    },
    "rural": {
        0.605: ["Wheat"],
        0.033: ["Rice"],
        0.004: ["Barley"],
        0.052: ["Grain maize"],
        0.003: ["Rye"],
        0.005: ["Oat"],
        0.006: ["Other cereals"],
        0.076: ["Potatoes"],
        0.002: ["Sugar beet"],
        0.001: ["Other roots"],
        0.01: ["Dry beans", "Green beans", "Horse beans and faba beans"],
        0.0045: ["Green peas", "Peas"],
        0.016: ["Other protein crops"],
        0.042: ["Dry fruits"],
        0.0008: ["Soybean"],
        0.0005: ["Rapeseed"],
        0.0003: ["Other oil crops"],
        0.105: ["Dry vegetables", "Cabbage", "Leaves vegetables"],
        0.031: ["Fruits", "Squash and melons"],
        0.0009: ["Citrus"],
        0.0015: ["Olives"],
        0.0325: ["Sunflower"],
    },
}


#     "urban": {
#         0.45: [key for key, value in categories_mapping.items() if value == "cereals (excluding rice)"],
#         0.09: [key for key, value in categories_mapping.items() if value == "rice"],
#         0.11: [key for key, value in categories_mapping.items() if value == "leguminous"],
#         0.05: [key for key, value in categories_mapping.items() if value == "fruits and vegetables"],
#         0.16: [key for key, value in categories_mapping.items() if value == "oleaginous"],
#         0.14: [key for key, value in categories_mapping.items() if value == "roots"],
#     },
#     "rural": {
#         0.45: [key for key, value in categories_mapping.items() if value == "cereals (excluding rice)"],
#         0.09: [key for key, value in categories_mapping.items() if value == "rice"],
#         0.11: [key for key, value in categories_mapping.items() if value == "leguminous"],
#         0.05: [key for key, value in categories_mapping.items() if value == "fruits and vegetables"],
#         0.16: [key for key, value in categories_mapping.items() if value == "oleaginous"],
#         0.14: [key for key, value in categories_mapping.items() if value == "roots"],
#     },
# }


# Source : FAO

# # TODO Trouver des données pour étayer/spatialiser/temporaliser ces données
# regime_humains = {
#     "cereals (excluding rice)": 0.35,
#     "rice": 0.20,
#     "leguminous": 0.10,
#     "fruits and vegetables": 0.05,
#     "oleaginous": 0.15,
#     "roots": 0.15,
# }


# regime_humains = {'céréales (hors riz)': 0.50, 'légumineuses': 0.25, 'fruits et légumes': 0.25}
