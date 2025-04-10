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
├── src/                    # Code source Python
│   ├── data/               # Gestion des données
│   │   ├── dataset.py      # Classe pour charger le dataset
│   │   └── transforms.py   # Transformations et augmentation de données
│   │
│   ├── models/             # Modèles CNN
│   │   ├── base_model.py   # Architecture CNN de base
│   │   ├── advanced_model.py # Architecture CNN avancée
│   │   ├── patch_model.py  # Modèle basé sur des patches
│   │   └── enhanced_patch_model.py # Modèle avancé basé sur des patches
│   │
│   ├── utils/              # Utilitaires
│   │   ├── training.py     # Fonctions d'entraînement
│   │   ├── evaluation.py   # Fonctions d'évaluation
│   │   ├── visualization.py # Fonctions de visualisation
│   │   ├── callbacks.py    # Callbacks d'entraînement
│   │   └── metrics.py      # Calcul des métriques
│   │
│   └── config.py           # Configuration du projet
│
├── results/                # Résultats d'expériences
│   ├── models/             # Modèles sauvegardés
│   ├── logs/               # Logs d'entraînement
│   ├── figures/            # Figures et visualisations
│   └── hyperopt/           # Résultats d'optimisation d'hyperparamètres
│
├── main.py                 # Point d'entrée principal
├── train.py                # Script d'entraînement standard
├── train_2phases.py        # Script d'entraînement en deux phases
├── evaluate.py             # Script d'évaluation
├── evaluate_optimized.py   # Script d'évaluation pour modèle optimisé
├── hyperopt.py             # Optimisation d'hyperparamètres avec Ray Tune
├── preprocess_data.py      # Prétraitement et augmentation des données
├── enhanced_patch_optimized_model.py # Modèle optimisé avec ResNet50
├── QUESTIONS.md            # Réponses détaillées aux questions du TP
├── requirements.txt        # Dépendances du projet
├── README.md               # Documentation du projet
└── CHANGELOG.md            # Journal des modifications
```

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/TP_Ecorce_Arbre.git
cd TP_Ecorce_Arbre

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset Bark-101 (si nécessaire)
# Les instructions de téléchargement sont disponibles sur le site officiel
```

## Utilisation

### Préparation et exploration des données

Pour explorer le dataset et générer des visualisations :

```bash
python main.py --action explore
```

### Augmentation ciblée des données

Pour augmenter uniquement les classes sous-représentées du dataset et réduire le déséquilibre :

```bash
python preprocess_data.py --augment --threshold_percentile 25 --target_count 30
```

Options disponibles :
- `--threshold_percentile` : Percentile utilisé pour identifier les classes sous-représentées (défaut: 25)
- `--target_count` : Nombre cible d'images par classe après augmentation
- `--visualize` : Afficher des exemples d'augmentation
- `--max_per_class` : Nombre maximal d'images générées par classe

### Entraînement des modèles

Pour entraîner le modèle de base :

```bash
python main.py --action train --model base --cuda
```

Ou directement via le script d'entraînement :

```bash
python train.py --model base --cuda
```

Options disponibles pour l'entraînement :
- `--model` : Type de modèle (base, advanced, patch, enhanced_patch)
- `--epochs` : Nombre d'époques d'entraînement
- `--batch_size` : Taille du batch
- `--lr` : Taux d'apprentissage
- `--patience` : Nombre d'époques sans amélioration avant arrêt précoce
- `--cuda` : Utiliser CUDA si disponible

### Entraînement en deux phases (transfert learning + fine-tuning)

Pour entraîner le modèle optimisé en utilisant l'approche en deux phases :

```bash
python train_2phases.py --cuda --use_best_model --phase1_epochs 30 --phase2_epochs 50 --patch_size 96 --patch_stride 24 --num_patches 9 --lr 0.00044 --momentum 0.938 --weight_decay 0.00018
```

Options disponibles :
- `--phase1_epochs` : Nombre d'époques pour la phase 1 (entraînement des couches supérieures)
- `--phase2_epochs` : Nombre d'époques pour la phase 2 (fine-tuning du backbone)
- `--skip_phase1` : Pour passer directement à la phase 2 (nécessite `--phase1_checkpoint`)
- `--use_best_model` : Utiliser le modèle avec les meilleures performances de validation

### Évaluation des modèles

Pour évaluer un modèle spécifique :

```bash
python main.py --action evaluate --model_path results/models/base_model_best.pth
```

Ou directement via le script d'évaluation :

```bash
python evaluate.py --model_path results/models/enhanced_patch_2phases_final.pth --cuda
```

Pour le modèle EnhancedPatchOptimizedModel (basé sur ResNet50) :

```bash
python evaluate_optimized.py --model_path results/models/enhanced_patch_optimized_best.pth --cuda
```

Pour évaluer tous les modèles à la fois :

```bash
python evaluate.py --all --cuda
```

### Visualisation de l'attention

Pour visualiser les cartes d'attention du modèle sur une image spécifique :

```bash
python main.py --action visualize --model base --model_path results/models/base_model_best.pth --image_index 10 --cuda
```

Options disponibles :
- `--image_index` : Index de l'image à visualiser
- `--save_path` : Chemin pour sauvegarder la visualisation
- `--show` : Afficher la visualisation (nécessite interface graphique)

### Optimisation des hyperparamètres

Pour le modèle EnhancedPatchCNN, un script d'optimisation automatique des hyperparamètres est disponible:

```bash
# Optimisation avec Ray Tune (20 combinaisons d'hyperparamètres)
python hyperopt.py --num_trials 20 --epochs 10 --cuda

# Optimisation puis entraînement du modèle final avec les meilleurs hyperparamètres
python hyperopt.py --num_trials 15 --epochs 5 --train_final --final_epochs 50 --cuda

# Optimisation avec ressources personnalisées (multi-GPU)
python hyperopt.py --num_trials 30 --num_cpus 8 --num_gpus 2 --cuda
```

Les hyperparamètres optimisés incluent:
- Learning rate
- Momentum et régularisation
- Architecture du backbone (ResNet18/34/50)
- Nombre et taille des patches
- Activation de la modélisation du contexte

Les résultats de l'optimisation sont enregistrés dans `results/hyperopt/` et `results/logs/`.

## Architectures CNN implémentées

1. **Modèle de base**: Architecture CNN simple avec des couches de convolution, pooling et dropout
2. **Modèle avancé**: Architecture plus profonde avec des blocs résiduels et batch normalization
3. **Modèle basé sur des patches**: Division des images en patches pour améliorer la classification
4. **Modèle amélioré basé sur des patches**: Architecture avancée avec extracteur de caractéristiques ResNet, mécanisme d'attention multi-têtes et modélisation du contexte inter-patches
5. **Modèle optimisé en deux phases**: Version améliorée utilisant ResNet50 et entraînement progressif

## Résultats

| Modèle | Accuracy | Precision | Recall | F1-Score | Taille du modèle |
|--------|----------|-----------|--------|----------|------------------|
| Base CNN | 17.92% | 0.2357 | 0.1792 | 0.1743 | 199 MB |
| Advanced CNN | 20.54% | 0.2621 | 0.2054 | 0.2012 | 86 MB |
| Patch CNN | 13.98% | 0.1689 | 0.1398 | 0.1412 | 1.2 MB |
| EnhancedPatch CNN | 37.07% | 0.4694 | 0.3707 | 0.3736 | 62 MB |
| EnhancedPatch Optimized | 44.09% | 0.5327 | 0.4409 | 0.4502 | 395 MB |
| EnhancedPatch (2-phases) | 43.94% | 0.5289 | 0.4394 | 0.4471 | 243 MB |

Les résultats détaillés sont disponibles dans le dossier `results/` et le fichier `QUESTIONS.md`.

## Informations sur le dataset

- Nombre total d'images d'entraînement: 1292
- Nombre total d'images de test: 1295
- Nombre total de classes: 101

Caractéristiques des images :
- Largeur min: 93px, max: 704px, moyenne: 388.00px, médiane: 376.5px
- Hauteur min: 261px, max: 800px, moyenne: 611.57px, médiane: 600.0px
- Ratio d'aspect min: 0.21, max: 1.50, moyenne: 0.66

## Analyse des performances

Les modèles basés sur des patches avec mécanismes d'attention ont nettement surpassé les approches classiques, avec une amélioration de la précision de plus de 20 points de pourcentage. L'approche en deux phases (transfert learning puis fine-tuning) s'est révélée particulièrement efficace pour ce problème de classification avec données limitées.

Pour une analyse détaillée des performances par classe et des réponses aux questions du TP, consulter le fichier `QUESTIONS.md`.

## Auteurs

- [Votre Nom]

## Licence

Ce projet est réalisé dans le cadre d'un TP de Deep Learning.