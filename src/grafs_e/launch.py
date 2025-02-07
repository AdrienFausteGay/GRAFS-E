import os
import sys

import streamlit.web.cli

# Déterminer le chemin du fichier de config Streamlit
config_dir = os.path.expanduser("~/.streamlit")
config_path = os.path.join(config_dir, "config.toml")

# Vérifier si le dossier ~/.streamlit existe, sinon le créer
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

# Écrire ou modifier le fichier config.toml pour imposer le dark mode
with open(config_path, "w") as config_file:
    config_file.write("[theme]\nbase='dark'\n")


def run():
    """Lance Streamlit en utilisant le package installé"""
    package_path = os.path.dirname(__file__)  # Trouve le chemin de grafs_e
    app_path = os.path.join(package_path, "app.py")  # Chemin de app.py
    print(f"Launching Streamlit from: {app_path}")  # Debugging
    sys.argv = ["streamlit", "run", app_path]  # Simule une commande CLI
    streamlit.web.cli.main()  # Démarre Streamlit directement


if __name__ == "__main__":
    run()
