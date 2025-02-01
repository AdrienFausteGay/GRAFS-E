import os
import sys

import streamlit.web.cli


def run():
    """Lance Streamlit en utilisant le package installé"""
    package_path = os.path.dirname(__file__)  # Trouve le chemin de grafs_e
    app_path = os.path.join(package_path, "app.py")  # Chemin de app.py
    print(f"Launching Streamlit from: {app_path}")  # Debugging
    sys.argv = ["streamlit", "run", app_path]  # Simule une commande CLI
    streamlit.web.cli.main()  # Démarre Streamlit directement


if __name__ == "__main__":
    run()
