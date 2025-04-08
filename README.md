# Classification d'Écorces d'Arbres avec CNN

Ce projet implémente des réseaux de neurones convolutifs (CNN) pour la classification d'écorces d'arbres en utilisant le dataset Bark-101, qui contient 2587 images appartenant à 101 espèces d'arbres différentes.

## Objectifs

1. Explorer et comprendre le jeu de données Bark-101
2. Concevoir au moins trois architectures CNN différentes
3. Implémenter une approche basée sur des patches pour améliorer la classification
4. Optimiser les hyperparamètres pour chaque architecture
5. Évaluer et comparer les performances des modèles
6. Analyser l'impact des choix architecturaux sur les performances
7. Implémenter des techniques de visualisation pour interpréter les décisions du modèle

## Structure du Projet

```
TP_Ecorce_Arbre/
│
├── data/                    # Dossier contenant le dataset Bark-101
│   ├── 0-100/              # Images classées par dossiers
│   ├── train.txt           # Fichier de liste d'entraînement
│   ├── test.txt            # Fichier de liste de test
│   └── info.txt            # Informations sur le dataset
│
├── notebooks/              # Notebooks Jupyter pour l'exploration et la visualisation
│   ├── exploration.ipynb   # Exploration du dataset
│   └── visualisation.ipynb # Visualisation des résultats
│
├── src/                    # Code source Python
│   ├── data/               # Gestion des données
│   │   ├── dataset.py      # Classe pour charger le dataset
│   │   └── transforms.py   # Transformations et augmentation de données
│   │
│   ├── models/             # Modèles CNN
│   │   ├── base_model.py   # Architecture CNN de base
│   │   ├── advanced_model.py # Architecture CNN avancée
│   │   └── patch_model.py  # Modèle basé sur des patches
│   │
│   ├── utils/              # Utilitaires
│   │   ├── training.py     # Fonctions d'entraînement
│   │   ├── evaluation.py   # Fonctions d'évaluation
│   │   └── visualization.py # Fonctions de visualisation
│   │
│   └── config.py           # Configuration du projet
│
├── results/                # Résultats d'expériences
│   ├── models/             # Modèles sauvegardés
│   ├── logs/               # Logs d'entraînement
│   └── figures/            # Figures et visualisations
│
├── main.py                 # Point d'entrée principal
├── train.py                # Script d'entraînement
├── evaluate.py             # Script d'évaluation
├── requirements.txt        # Dépendances du projet
├── README.md               # Documentation du projet
└── CHANGELOG.md            # Journal des modifications
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Exploration du dataset

```bash
python main.py --action explore
```

### Entraînement d'un modèle

```bash
# Models names are: base, advanced, patch, enhanced_patch

# Utilisation via main.py
python main.py --action train --model base --cuda

# Utilisation directe via train.py
# use of --epochs x | --lr | --batch_size etc..
python train.py --model base --cuda
```

### Évaluation d'un modèle

```bash
# Utilisation via main.py
python main.py --action evaluate --model_path results/models/base_model_best.pth

# Utilisation directe via evaluate.py
python evaluate.py --model_path results/models/base_model_best.pth --cuda
```

### Visualisation de l'attention du modèle

```bash
python main.py --action visualize --model base --model_path results/models/base_model_best.pth --image_index 10 --cuda

# Pour les modèles à base de patches (patch et enhanced_patch), visualisation spécifique des patches
python main.py --action visualize --model enhanced_patch --model_path results/models/enhanced_patch_model_best.pth --image_index 10 --cuda
```

## Architectures CNN implémentées

1. **Modèle de base**: Architecture CNN simple avec des couches de convolution, pooling et dropout
2. **Modèle avancé**: Architecture plus profonde avec des blocs résiduels et batch normalization
3. **Modèle basé sur des patches**: Division des images en patches pour améliorer la classification
4. **Modèle amélioré basé sur des patches**: Architecture avancée avec extracteur de caractéristiques ResNet, mécanisme d'attention multi-têtes et modélisation du contexte inter-patches

## Résultats

Les résultats détaillés sont disponibles dans le dossier `results/` et dans les notebooks d'analyse.

## Licence

Ce projet est réalisé dans le cadre d'un TP de Deep Learning. 


Nombre total d'images d'entraînement: 1292
Nombre total d'images de test: 1295
Nombre total de classes: 101

Répartition des classes (entraînement):
  Classe 0: 16 images
  Classe 1: 4 images
  Classe 2: 2 images
  Classe 3: 10 images
  Classe 4: 13 images
  Classe 5: 24 images
  Classe 6: 55 images
  Classe 7: 3 images
  Classe 8: 16 images
  Classe 9: 31 images
  Classe 10: 2 images
  Classe 11: 2 images
  Classe 12: 19 images
  Classe 13: 20 images
  Classe 14: 13 images
  Classe 15: 55 images
  Classe 16: 18 images
  Classe 17: 8 images
  Classe 18: 3 images
  Classe 19: 6 images
  Classe 20: 10 images
  Classe 21: 3 images
  Classe 22: 3 images
  Classe 23: 11 images
  Classe 24: 6 images
  Classe 25: 7 images
  Classe 26: 6 images
  Classe 27: 2 images
  Classe 28: 1 images
  Classe 29: 1 images
  Classe 30: 15 images
  Classe 31: 13 images
  Classe 32: 1 images
  Classe 33: 21 images
  Classe 34: 21 images
  Classe 35: 14 images
  Classe 36: 9 images
  Classe 37: 30 images
  Classe 38: 4 images
  Classe 39: 5 images
  Classe 40: 12 images
  Classe 41: 7 images
  Classe 42: 6 images
  Classe 43: 12 images
  Classe 44: 1 images
  Classe 45: 17 images
  Classe 46: 17 images
  Classe 47: 24 images
  Classe 48: 13 images
  Classe 49: 3 images
  Classe 50: 25 images
  Classe 51: 9 images
  Classe 52: 10 images
  Classe 53: 35 images
  Classe 54: 9 images
  Classe 55: 15 images
  Classe 56: 2 images
  Classe 57: 5 images
  Classe 58: 3 images
  Classe 59: 69 images
  Classe 60: 19 images
  Classe 61: 13 images
  Classe 62: 5 images
  Classe 63: 17 images
  Classe 64: 16 images
  Classe 65: 4 images
  Classe 66: 3 images
  Classe 67: 6 images
  Classe 68: 20 images
  Classe 69: 1 images
  Classe 70: 6 images
  Classe 71: 5 images
  Classe 72: 8 images
  Classe 73: 3 images
  Classe 74: 64 images
  Classe 75: 1 images
  Classe 76: 9 images
  Classe 77: 12 images
  Classe 78: 16 images
  Classe 79: 35 images
  Classe 80: 13 images
  Classe 81: 2 images
  Classe 82: 2 images
  Classe 83: 5 images
  Classe 84: 41 images
  Classe 85: 10 images
  Classe 86: 9 images
  Classe 87: 10 images
  Classe 88: 12 images
  Classe 89: 9 images
  Classe 90: 33 images
  Classe 91: 7 images
  Classe 92: 6 images
  Classe 93: 3 images
  Classe 94: 11 images
  Classe 95: 6 images
  Classe 96: 12 images
  Classe 97: 12 images
  Classe 98: 1 images
  Classe 99: 6 images
  Classe 100: 22 images

Répartition des classes (test):
  Classe 0: 16 images
  Classe 1: 5 images
  Classe 2: 3 images
  Classe 3: 9 images
  Classe 4: 13 images
  Classe 5: 23 images
  Classe 6: 55 images
  Classe 7: 4 images
  Classe 8: 16 images
  Classe 9: 31 images
  Classe 10: 1 images
  Classe 11: 3 images
  Classe 12: 18 images
  Classe 13: 20 images
  Classe 14: 14 images
  Classe 15: 54 images
  Classe 16: 19 images
  Classe 17: 8 images
  Classe 18: 4 images
  Classe 19: 5 images
  Classe 20: 10 images
  Classe 21: 4 images
  Classe 22: 2 images
  Classe 23: 12 images
  Classe 24: 5 images
  Classe 25: 6 images
  Classe 26: 7 images
  Classe 27: 2 images
  Classe 28: 1 images
  Classe 29: 2 images
  Classe 30: 15 images
  Classe 31: 12 images
  Classe 32: 2 images
  Classe 33: 20 images
  Classe 34: 21 images
  Classe 35: 15 images
  Classe 36: 10 images
  Classe 37: 29 images
  Classe 38: 5 images
  Classe 39: 4 images
  Classe 40: 12 images
  Classe 41: 8 images
  Classe 42: 5 images
  Classe 43: 12 images
  Classe 44: 1 images
  Classe 45: 16 images
  Classe 46: 17 images
  Classe 47: 24 images
  Classe 48: 13 images
  Classe 49: 3 images
  Classe 50: 26 images
  Classe 51: 8 images
  Classe 52: 10 images
  Classe 53: 35 images
  Classe 54: 10 images
  Classe 55: 14 images
  Classe 56: 3 images
  Classe 57: 5 images
  Classe 58: 2 images
  Classe 59: 69 images
  Classe 60: 20 images
  Classe 61: 12 images
  Classe 62: 5 images
  Classe 63: 18 images
  Classe 64: 16 images
  Classe 65: 4 images
  Classe 66: 4 images
  Classe 67: 5 images
  Classe 68: 19 images
  Classe 69: 1 images
  Classe 70: 6 images
  Classe 71: 4 images
  Classe 72: 9 images
  Classe 73: 2 images
  Classe 74: 65 images
  Classe 75: 1 images
  Classe 76: 8 images
  Classe 77: 12 images
  Classe 78: 16 images
  Classe 79: 36 images
  Classe 80: 14 images
  Classe 81: 2 images
  Classe 82: 2 images
  Classe 83: 5 images
  Classe 84: 41 images
  Classe 85: 10 images
  Classe 86: 10 images
  Classe 87: 10 images
  Classe 88: 11 images
  Classe 89: 10 images
  Classe 90: 32 images
  Classe 91: 7 images
  Classe 92: 5 images
  Classe 93: 3 images
  Classe 94: 12 images
  Classe 95: 6 images
  Classe 96: 12 images
  Classe 97: 12 images
  Classe 98: 2 images
  Classe 99: 5 images
  Classe 100: 23 images

Analyse des dimensions des images...
Largeur min: 93, max: 704, moyenne: 388.00, médiane: 376.5
Hauteur min: 261, max: 800, moyenne: 611.57, médiane: 600.0
Ratio d'aspect min: 0.21, max: 1.50, moyenne: 0.66 

Base model:
Époque 45/50
Train Loss: 3.3559, Train Acc: 0.1757
Val Loss: 3.6084, Val Acc: 0.1792
EarlyStopping: 10/10

Advanced model:
Époque 20/20 : train_loss=2.8832, train_acc=0.2763, val_loss=3.6411, val_acc=0.2054
EarlyStopping: patience 5/10

Patch model:
Époque 50/50 : train_loss=3.7636, train_acc=0.1354, val_loss=3.8673, val_acc=0.1398
Validation loss améliorée (3.868634 --> 3.867259). Sauvegarde du modèle...
Entraînement terminé. Meilleur modèle sauvegardé à D:\Dev\DL\TP_Ecorce_Arbre\results\models\patch_model_best.pth
Entraînement terminé en 16433.27 secondes

Enhanced patch model:
