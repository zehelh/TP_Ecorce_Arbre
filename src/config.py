"""
Configuration globale du projet de classification d'écorces d'arbres.
"""

import os
from pathlib import Path

# Chemins des répertoires
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, "models")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Fichiers de données
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")

# Paramètres du dataset
NUM_CLASSES = 101
IMAGE_SIZE = (224, 224)  # Taille de redimensionnement des images
BATCH_SIZE = 32
NUM_WORKERS = 8

# Paramètres d'entraînement
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Paramètres des modèles
MODEL_TYPES = ["base", "advanced", "patch", "enhanced_patch"]

# Paramètres de l'approche par patches
PATCH_SIZE = 64
PATCH_STRIDE = 32
NUM_PATCHES = 9  # Nombre de patches à extraire par image

# Configuration des augmentations de données
DATA_AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_range": 15,
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
}

# Paramètres de visualisation
VISUALIZE_FREQUENCY = 10  # Fréquence de génération des visualisations (en époques) 