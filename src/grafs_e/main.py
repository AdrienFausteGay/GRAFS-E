import pandas as pd

from .donnees import *

# Pour éviter un warning :
pd.set_option("future.no_silent_downcasting", True)

# Afficher toutes les colonnes
pd.set_option("display.max_columns", None)

# Afficher toutes les lignes
pd.set_option("display.max_rows", None)

pvar_path = "C:/Users/faustega/Documents/These/Matérialité historique/Données Julia Le noe/GRAFS-15-09-2018.xlsx"
sheets_dict = pd.read_excel(pvar_path, sheet_name=None)
