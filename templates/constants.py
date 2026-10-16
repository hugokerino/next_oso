import os
import pandas as pd
from matplotlib.colors import ListedColormap

is_cluster = "SLURM_JOB_ID" in os.environ
path_to_oso_name = (
    "/work/THEIA/CES/oso/hugok/nextoso-dataset/metadata/oso_classes.csv"
    if is_cluster
    else "/home/hkerino/Documents/nextoso-dataset/data/10m/metadata/oso_classes.csv"
)

names_class = (
    pd.read_csv(path_to_oso_name).get("name_class").values
)  # Doit contenir 25 entrées (0 à 24)
num_class = len(names_class)


# Colormap personnalisée
# Définition des couleurs personnalisées (RVB 0-255)
OSO_COLORS = {
    0: (255, 255, 255),  # Blanc pour 'nan'
    1: (255, 0, 255),  # Urbain dense
    2: (255, 85, 255),  # Urbain diffus
    3: (255, 170, 255),  # Zones industrielles et commerciales
    4: (0, 255, 255),  # Routes
    5: (255, 255, 0),  # Colza
    6: (208, 255, 0),  # Céréales à paille
    7: (161, 214, 0),  # Protéagineux
    8: (255, 170, 68),  # Soja
    9: (214, 214, 0),  # Tournesol
    10: (255, 85, 0),  # Maïs
    11: (197, 255, 255),  # Riz
    12: (170, 170, 97),  # Tubercules / Racines
    13: (170, 170, 0),  # Prairies
    14: (170, 170, 255),  # Vergers
    15: (85, 0, 0),  # Vignes
    16: (0, 156, 0),  # Forêts de feuillus
    17: (0, 50, 0),  # Forêts de conifères
    18: (170, 255, 0),  # Pelouse
    19: (85, 170, 127),  # Landes
    20: (255, 0, 0),  # Surfaces minérales
    21: (255, 184, 2),  # Plages et Dunes
    22: (190, 190, 190),  # Glaciers et neiges éternelles
    23: (0, 0, 255),  # Eau
    24: (68, 121, 200),  # Serres
}

# Normaliser les couleurs (diviser par 255)
normalized_colors = [
    (r / 255, g / 255, b / 255)
    for r, g, b in [OSO_COLORS[i] for i in sorted(OSO_COLORS.keys())]
]

cmap_oso = ListedColormap(normalized_colors)

eco_zone_titles = {
    1: "montagne",
    2: "semi-continental",  # "semi-continental marges montagnardes"
    3: "océanique dégradé",  # "océanique dégradé des plaines du Centre et du Nord"
    4: "océanique altéré",
    5: "océanique franc",
    6: "méditerranéen altéré",
    7: "Bassin du Sud-Ouest",
    8: "méditerranéen franc",
    9: "corse",
}

eco_zone_titles_english = {
    1: "Mountain",
    2: "Semi-continental",  # "semi-continental mountain margins"
    3: "Degraded oceanic",  # "degraded oceanic of the Central and Northern plains"
    4: "Altered oceanic",
    5: "Pure oceanic",
    6: "Altered mediterranean",
    7: "South-West Basin",
    8: "Pure mediterranean",
    9: "Corsica",
}

custom_colors = {
    1: "#0E1A89",  # montagne
    2: "#2F74B5",  # semi-continental
    3: "#85C6EA",  # océanique dégradé
    4: "#10E361",  # océanique altéré
    5: "#065918",  # océanique franc
    6: "#F0E359",  # méditerranéen altéré
    7: "#BE821B",  # Bassin du Sud-Ouest"
    8: "#C41919",  # méditerranéen franc
    9: "#FF00FF",  # Corse
}
